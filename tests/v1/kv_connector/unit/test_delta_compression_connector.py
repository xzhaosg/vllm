# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vLLM V1 DeltaCompressionConnector (Path A Phase 1)."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

try:
    from vllm.g3_integration import cns_bridge, patchers

    cns_bridge.activate_cns_bridge()
    patchers.inject_wildcard_mocks()
    patchers.patch_metadata_versions()
    patchers.patch_online_serving_dependencies()
    patchers.redirect_caches_to_tmp()
    patchers.disable_usage_stats()
    patchers.apply_single_process_patches()
except ImportError:
    pass

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    DEFAULT_PAD_TOKEN,
    CompressionTier,
    DeltaBlockStore,
    DeltaBlockTransferSpec,
    DeltaCompressionConnector,
    LayerCompressionPolicy,
    _requant_dsv41_kv,
    align_structural_prompt_server,
    align_tokens_to_block,
    apply_relative_rope_rotation,
    compute_full_prefix_block_hash,
    compute_token_block_hash,
    dequantize_residual_packed,
    encode_file_viewer_bucketed_server,
    quantize_residual_packed,
    resolve_pad_token_id,
)


def _compute_sqnr_db(exact: torch.Tensor, recon: torch.Tensor) -> float:
    err = exact.to(torch.float32) - recon.to(torch.float32)
    sig_pow = torch.sum(exact.to(torch.float32) ** 2).item()
    err_pow = torch.sum(err**2).item()
    if err_pow <= 1e-20:
        return 160.0
    return 10.0 * torch.log10(torch.tensor(sig_pow / err_pow)).item()


class TestDeltaCompressionConnector(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        DeltaCompressionConnector.reset_shared_stores()

    def test_relative_rope_rotation_exactness(self) -> None:
        """Verifies R(delta_p) * R(p_ref) * K_raw == R(p_target) * K_raw."""
        block_size, num_heads, head_dim = 16, 4, 128
        k_raw = torch.randn(block_size, num_heads, head_dim, dtype=torch.bfloat16)

        p_ref = torch.arange(10, 10 + block_size)
        delta_p = 27
        p_target = p_ref + delta_p

        for is_neox in (True, False):
            for rotary_dim in (128, 64, 0):
                k_ref_post = apply_relative_rope_rotation(
                    k_raw,
                    delta_pos=p_ref,
                    rope_theta=10000.0,
                    rotary_dim=rotary_dim,
                    is_neox_style=is_neox,
                )
                k_target_post = apply_relative_rope_rotation(
                    k_raw,
                    delta_pos=p_target,
                    rope_theta=10000.0,
                    rotary_dim=rotary_dim,
                    is_neox_style=is_neox,
                )
                k_rotated = apply_relative_rope_rotation(
                    k_ref_post,
                    delta_pos=delta_p,
                    rope_theta=10000.0,
                    rotary_dim=rotary_dim,
                    is_neox_style=is_neox,
                )
                sqnr = _compute_sqnr_db(k_target_post, k_rotated)
                self.assertGreater(
                    sqnr,
                    50.0,
                    f"RoPE relative rotation SQNR too low ({sqnr:.2f} dB) for"
                    f" is_neox={is_neox}, rotary_dim={rotary_dim}",
                )

    def test_packed_int4_and_int2_quantizer_bytes_and_sqnr(self) -> None:
        """Verifies packed uint8 byte footprint and reconstruction SQNR for INT4 & INT2."""
        block_size, num_heads, head_dim = 16, 4, 128
        base_k = torch.randn(block_size, num_heads, head_dim, dtype=torch.bfloat16)
        # Simulate a realistic Layer >= 1 residual with eta_L2 ~ 0.15
        residual = (
            0.15 * torch.randn(block_size, num_heads, head_dim, dtype=torch.bfloat16)
        )
        target_k = base_k + residual
        raw_bytes = target_k.numel() * target_k.element_size()  # 16,384 bytes

        packed_int4 = quantize_residual_packed(residual, bits=4)
        # 8192 elems -> 4096 packed bytes + 64*2 scale bytes = 4224 bytes (3.88x smaller)
        self.assertEqual(packed_int4.packed_data.dtype, torch.uint8)
        self.assertEqual(
            packed_int4.packed_data.numel(), (block_size * num_heads * head_dim) // 2
        )
        self.assertLess(packed_int4.nbytes, raw_bytes // 3)

        recon_res_int4 = dequantize_residual_packed(packed_int4)
        recon_k_int4 = (
            base_k.to(torch.float32) + recon_res_int4.to(torch.float32)
        ).to(torch.bfloat16)
        sqnr_int4 = _compute_sqnr_db(target_k, recon_k_int4)
        self.assertGreater(sqnr_int4, 34.0)

        packed_int2 = quantize_residual_packed(residual, bits=2)
        # 8192 elems -> 2048 packed bytes + 128 scale bytes = 2176 bytes (7.53x smaller)
        self.assertEqual(
            packed_int2.packed_data.numel(), (block_size * num_heads * head_dim) // 4
        )
        self.assertLess(packed_int2.nbytes, raw_bytes // 7)

        recon_res_int2 = dequantize_residual_packed(packed_int2)
        recon_k_int2 = (
            base_k.to(torch.float32) + recon_res_int2.to(torch.float32)
        ).to(torch.bfloat16)
        sqnr_int2 = _compute_sqnr_db(target_k, recon_k_int2)
        self.assertGreater(sqnr_int2, 22.0)

    def test_layer0_zero_bit_reuse_and_rope_shift(self) -> None:
        """Verifies Layer 0 compresses identical token blocks to Tier 0 (0 tensor bytes)."""
        store = DeltaBlockStore(block_size=16)
        layer_name = "model.layers.0.self_attn.attn"
        store.register_layer_policy(
            layer_name, LayerCompressionPolicy(layer_idx=0, rope_theta=10000.0)
        )

        tokens = tuple(range(100, 116))
        k_raw = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v_raw = torch.randn(16, 4, 128, dtype=torch.bfloat16)

        # Prompt A: block at position 16
        pos_a = 16
        k_a = apply_relative_rope_rotation(
            k_raw, delta_pos=torch.arange(pos_a, pos_a + 16)
        )
        hash_a = compute_full_prefix_block_hash("PREFIX_A", tokens)
        rec_a = store.store_block(layer_name, hash_a, tokens, pos_a, k_a, v_raw)
        self.assertEqual(rec_a.tier, CompressionTier.UNCOMPRESSED_BASE)

        # Prompt B: divergent prefix, same token block shifted to position 64 (delta_p = +48)
        pos_b = 64
        k_b = apply_relative_rope_rotation(
            k_raw, delta_pos=torch.arange(pos_b, pos_b + 16)
        )
        hash_b = compute_full_prefix_block_hash("PREFIX_B_DIVERGENT", tokens)
        rec_b = store.store_block(layer_name, hash_b, tokens, pos_b, k_b, v_raw)

        self.assertEqual(rec_b.tier, CompressionTier.TIER0_ZERO_BIT)
        self.assertEqual(rec_b.delta_pos, 48)
        self.assertEqual(rec_b.nbytes, 16)  # Only 16-byte metadata pointer!

        k_recon, v_recon = store.reconstruct_block(layer_name, hash_b)
        assert v_recon is not None
        # Values at Layer 0 are bitwise identical
        self.assertTrue(torch.equal(v_raw, v_recon))
        # Keys at Layer 0 match up to 1-ULP bfloat16 rotation (> 50 dB SQNR)
        self.assertGreater(_compute_sqnr_db(k_b, k_recon), 50.0)

    def test_sliding_window_zero_bit_dedup_at_divergence_distance(self) -> None:
        """Verifies LOCAL_SLIDING layers switch from INT4 Delta to Tier 0 once d_div >= 64."""
        store = DeltaBlockStore(block_size=16, sliding_window_dedup_threshold=64)
        layer_name = "model.layers.2.self_attn.attn"
        store.register_layer_policy(
            layer_name,
            LayerCompressionPolicy(
                layer_idx=2,
                is_sliding_window=True,
                sliding_window_size=512,
                sliding_window_dedup_threshold=64,
            ),
        )

        tokens = tuple(range(200, 216))
        k_base = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v_base = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        hash_base = compute_full_prefix_block_hash("BASE_PREFIX", tokens)
        store.store_block(layer_name, hash_base, tokens, 32, k_base, v_base)

        # Near prefix edit (d_div = 16 < 64): uses Tier 2 (INT4 Delta)
        k_near = k_base + 0.20 * torch.randn_like(k_base)
        v_near = v_base + 0.15 * torch.randn_like(v_base)
        hash_near = compute_full_prefix_block_hash("NEAR_PREFIX", tokens)
        rec_near = store.store_block(
            layer_name, hash_near, tokens, 32, k_near, v_near, divergence_distance=16
        )
        self.assertEqual(rec_near.tier, CompressionTier.TIER2_INT4_DELTA)

        # Past sliding window decay boundary (d_div = 64 >= 64): uses Tier 0 (0-bit dedup)
        k_far = k_base + 0.03 * torch.randn_like(k_base)
        v_far = v_base + 0.02 * torch.randn_like(v_base)
        hash_far = compute_full_prefix_block_hash("FAR_PREFIX", tokens)
        rec_far = store.store_block(
            layer_name, hash_far, tokens, 32, k_far, v_far, divergence_distance=64
        )
        self.assertEqual(rec_far.tier, CompressionTier.TIER0_ZERO_BIT)
        self.assertEqual(rec_far.nbytes, 16)

    def test_regime_r2_sparse_bf16_plus_int4_delta(self) -> None:
        """Verifies Regime R2 intra-block token edit uses Tier 3 Sparse BF16 + INT4 Delta."""
        store = DeltaBlockStore(block_size=16)
        layer_name = "model.layers.5.self_attn.attn"
        store.register_layer_policy(layer_name, LayerCompressionPolicy(layer_idx=5))

        base_tokens = tuple(range(500, 516))
        k_base = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v_base = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        hash_base = compute_full_prefix_block_hash("ROOT", base_tokens)
        store.store_block(layer_name, hash_base, base_tokens, 0, k_base, v_base)

        # Edit 2 tokens inside the 16-token block (indices 3 and 11)
        edited_tokens = list(base_tokens)
        edited_tokens[3] = 9991
        edited_tokens[11] = 9992
        edited_tuple = tuple(edited_tokens)

        # Edited rows have completely new KV vectors; other 14 rows have small delta
        k_target = k_base + 0.12 * torch.randn_like(k_base)
        v_target = v_base + 0.10 * torch.randn_like(v_base)
        k_target[3] = torch.randn(4, 128, dtype=torch.bfloat16) * 3.0
        k_target[11] = torch.randn(4, 128, dtype=torch.bfloat16) * 3.0
        v_target[3] = torch.randn(4, 128, dtype=torch.bfloat16) * 3.0
        v_target[11] = torch.randn(4, 128, dtype=torch.bfloat16) * 3.0

        hash_edited = compute_full_prefix_block_hash("ROOT", edited_tuple)
        rec_edited = store.store_block(
            layer_name, hash_edited, edited_tuple, 0, k_target, v_target
        )

        self.assertEqual(rec_edited.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        assert rec_edited.sparse_token_indices is not None
        self.assertEqual(list(rec_edited.sparse_token_indices.tolist()), [3, 11])

        k_recon, v_recon = store.reconstruct_block(layer_name, hash_edited)
        assert v_recon is not None
        # Edited rows 3 and 11 must be reconstructed with 0.0 bitwise error!
        self.assertTrue(torch.equal(k_target[3], k_recon[3]))
        self.assertTrue(torch.equal(k_target[11], k_recon[11]))
        self.assertTrue(torch.equal(v_target[3], v_recon[3]))
        self.assertTrue(torch.equal(v_target[11], v_recon[11]))
        # Overall block SQNR > 38 dB
        self.assertGreater(_compute_sqnr_db(k_target, k_recon), 38.0)

    def test_single_hop_reference_rule_prevents_error_accumulation(self) -> None:
        """Verifies multi-turn deltas always anchor to Turn 1's UNCOMPRESSED_BASE block."""
        store = DeltaBlockStore(block_size=16)
        layer_name = "model.layers.3.self_attn.attn"
        store.register_layer_policy(layer_name, LayerCompressionPolicy(layer_idx=3))

        tokens = tuple(range(300, 316))
        k_turn1 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v_turn1 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        hash_turn1 = compute_full_prefix_block_hash("TURN_1", tokens)
        base_rec = store.store_block(
            layer_name, hash_turn1, tokens, 16, k_turn1, v_turn1
        )
        self.assertEqual(base_rec.tier, CompressionTier.UNCOMPRESSED_BASE)

        for turn_idx in range(2, 7):
            delta_p = 16 * (turn_idx - 1)
            k_turn_rotated = apply_relative_rope_rotation(k_turn1, delta_pos=delta_p)
            k_turn = k_turn_rotated + 0.18 * torch.randn_like(k_turn1)
            v_turn = v_turn1 + 0.15 * torch.randn_like(v_turn1)
            h_turn = compute_full_prefix_block_hash(f"TURN_{turn_idx}", tokens)
            rec = store.store_block(
                layer_name, h_turn, tokens, 16 * turn_idx, k_turn, v_turn
            )
            self.assertEqual(rec.tier, CompressionTier.TIER2_INT4_DELTA)
            # Single-hop invariant: always points directly to Turn 1's base hash
            self.assertEqual(rec.ref_base_hash, hash_turn1)

    def test_end_to_end_vllm_v1_connector_save_and_load_lifecycle(self) -> None:
        """Tests full Scheduler + Worker DeltaCompressionConnector via KVConnectorFactory."""
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="test_engine_e2e",
                kv_connector_extra_config={"sliding_window_dedup_threshold": 32},
            ),
        )

        sched_conn = KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=None  # type: ignore[arg-type]
        )
        worker_conn = KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config=None  # type: ignore[arg-type]
        )
        self.assertIsInstance(sched_conn, DeltaCompressionConnector)
        self.assertIsInstance(worker_conn, DeltaCompressionConnector)
        assert isinstance(sched_conn, DeltaCompressionConnector)
        assert isinstance(worker_conn, DeltaCompressionConnector)

        # Configure a 3-layer hybrid model: Layer 0, Layer 1 (Local SWA), Layer 2 (Global)
        layer_names = [
            "model.layers.0.self_attn.attn",
            "model.layers.1.self_attn.attn",
            "model.layers.2.self_attn.attn",
        ]
        worker_conn.store.register_layer_policy(
            layer_names[0], LayerCompressionPolicy(layer_idx=0)
        )
        worker_conn.store.register_layer_policy(
            layer_names[1],
            LayerCompressionPolicy(
                layer_idx=1,
                is_sliding_window=True,
                sliding_window_size=512,
                sliding_window_dedup_threshold=32,
            ),
        )
        worker_conn.store.register_layer_policy(
            layer_names[2], LayerCompressionPolicy(layer_idx=2, is_sliding_window=False)
        )

        # Create paged KV cache buffers [num_pages=16, 2, block_size=16, num_heads=4, head_dim=64]
        kv_buffers = {
            name: torch.randn(16, 2, 16, 4, 64, dtype=torch.bfloat16)
            for name in layer_names
        }

        # Request 1 (Base prompt: 4 blocks = 64 tokens, physical blocks [0, 1, 2, 3])
        req1_tokens = list(range(1000, 1064))
        sched_out_1 = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_1",
                    prompt_token_ids=req1_tokens,
                    block_ids=([0, 1, 2, 3],),
                )
            ]
        )
        meta_1 = sched_conn.build_connector_meta(sched_out_1)  # type: ignore[arg-type]
        worker_conn.bind_connector_metadata(meta_1)
        for name in layer_names:
            worker_conn.save_kv_layer(
                name, kv_buffers[name], attn_metadata=mock.MagicMock()
            )
        worker_conn.clear_connector_metadata()

        # Request 2: Divergent first block (tokens 0..15 edited), followed by the exact
        # same 3 suffix blocks (tokens 1016..1063) in physical blocks [4, 5, 6, 7].
        req2_tokens = list(range(2000, 2016)) + list(range(1016, 1064))
        for i, src_blk in enumerate([1, 2, 3], start=5):
            # Layer 0 has identical K, V on shared token blocks (delta_p = 0)
            kv_buffers[layer_names[0]][i] = kv_buffers[layer_names[0]][src_blk].clone()
            # Layer 1 & 2 have small cross-prefix residuals on shared token blocks
            kv_buffers[layer_names[1]][i] = kv_buffers[layer_names[1]][
                src_blk
            ] + 0.12 * torch.randn_like(kv_buffers[layer_names[1]][src_blk])
            kv_buffers[layer_names[2]][i] = kv_buffers[layer_names[2]][
                src_blk
            ] + 0.18 * torch.randn_like(kv_buffers[layer_names[2]][src_blk])

        sched_out_2 = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_2",
                    prompt_token_ids=req2_tokens,
                    block_ids=([4, 5, 6, 7],),
                )
            ]
        )
        meta_2 = sched_conn.build_connector_meta(sched_out_2)  # type: ignore[arg-type]
        worker_conn.bind_connector_metadata(meta_2)
        for name in layer_names:
            worker_conn.save_kv_layer(
                name, kv_buffers[name], attn_metadata=mock.MagicMock()
            )
        worker_conn.clear_connector_metadata()

        # Verify that Req 2's 3 shared suffix blocks were compressed across layers:
        # - Layer 0: 3 blocks in TIER0_ZERO_BIT
        # - Layer 1 (SWA): block at d_div=32 in TIER0_ZERO_BIT
        self.assertGreaterEqual(
            worker_conn.store.stats.tier_counts[CompressionTier.TIER0_ZERO_BIT], 4
        )
        self.assertGreater(
            worker_conn.store.stats.tier_counts[CompressionTier.TIER1_INT2_DELTA]
            + worker_conn.store.stats.tier_counts[CompressionTier.TIER2_INT4_DELTA],
            0,
        )

        # Now test loading Request 2 from DeltaBlockStore into fresh physical blocks [8, 9, 10, 11]
        req2_reload = SimpleNamespace(
            request_id="req_2_reload",
            prompt_token_ids=req2_tokens
            + [9999],  # +1 new token to trigger full 64-tok match
        )
        matched_tokens, _ = sched_conn.get_num_new_matched_tokens(req2_reload, 0)  # type: ignore[arg-type]
        self.assertEqual(matched_tokens, 64)

        sched_conn.update_state_after_alloc(req2_reload, mock.MagicMock(), matched_tokens)  # type: ignore[arg-type]
        sched_out_reload = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_2_reload",
                    prompt_token_ids=req2_tokens + [9999],
                    block_ids=([8, 9, 10, 11],),
                )
            ]
        )
        meta_reload = sched_conn.build_connector_meta(sched_out_reload)  # type: ignore[arg-type]

        # Zero out target physical blocks [8, 9, 10, 11] before start_load_kv
        for name in layer_names:
            kv_buffers[name][8:12].zero_()

        forward_ctx = SimpleNamespace(
            no_compile_layers={
                name: SimpleNamespace(kv_cache=kv_buffers[name])
                for name in layer_names
            },
            attn_metadata={name: mock.MagicMock() for name in layer_names},
        )
        worker_conn.bind_connector_metadata(meta_reload)
        worker_conn.start_load_kv(forward_ctx)  # type: ignore[arg-type]
        worker_conn.clear_connector_metadata()

        # Verify Layer 0 blocks [8, 9, 10, 11] match [4, 5, 6, 7] exactly!
        self.assertTrue(
            torch.equal(
                kv_buffers[layer_names[0]][4:8], kv_buffers[layer_names[0]][8:12]
            )
        )
        # Verify Layer 2 blocks [8, 9, 10, 11] reconstruct with > 35 dB SQNR!
        sqnr_l2 = _compute_sqnr_db(
            kv_buffers[layer_names[2]][4:8], kv_buffers[layer_names[2]][8:12]
        )
        self.assertGreater(sqnr_l2, 35.0)

    def test_standardized_4d_kv_layout_and_gpu_roundtrip(self) -> None:
        """Verifies 4D [num_blocks, num_kv_heads, block_size, 2*head_dim] extract/inject."""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        num_blocks, num_kv_heads, block_size, head_dim = 8, 1, 16, 256
        kv_4d = torch.randn(
            num_blocks, num_kv_heads, block_size, 2 * head_dim,
            dtype=torch.bfloat16, device=device
        )
        k_blk, v_blk = DeltaCompressionConnector._extract_block_kv(
            kv_4d, physical_block_id=3, is_mla=False
        )
        assert v_blk is not None
        self.assertEqual(tuple(k_blk.shape), (block_size, num_kv_heads, head_dim))
        self.assertEqual(tuple(v_blk.shape), (block_size, num_kv_heads, head_dim))

        kv_4d[5].zero_()
        DeltaCompressionConnector._inject_block_kv(
            kv_4d, physical_block_id=5, k_block=k_blk, v_block=v_blk, is_mla=False
        )
        self.assertTrue(torch.equal(kv_4d[3], kv_4d[5]))

        # Also test virtual kernel-block splitting when manager block_size=32, kernel_block_size=16
        k_blk32, v_blk32 = DeltaCompressionConnector._extract_block_kv(
            kv_4d, physical_block_id=1, is_mla=False, block_size=32
        )
        assert v_blk32 is not None
        self.assertEqual(tuple(k_blk32.shape), (32, num_kv_heads, head_dim))
        self.assertEqual(tuple(v_blk32.shape), (32, num_kv_heads, head_dim))
        kv_4d[4:6].zero_()
        DeltaCompressionConnector._inject_block_kv(
            kv_4d,
            physical_block_id=2,
            k_block=k_blk32,
            v_block=v_blk32,
            is_mla=False,
            block_size=32,
        )
        self.assertTrue(torch.equal(kv_4d[2:4], kv_4d[4:6]))

    def test_qwen3_rope_parameters_and_partial_rotary_auto_config(self) -> None:
        """Verifies rope_parameters['rope_theta'] (Qwen3) and partial_rotary_factor (Qwen3.5)."""
        DeltaCompressionConnector.reset_shared_stores()
        hf_cfg = SimpleNamespace(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            rope_theta=None,
            rope_local_base_freq=None,
            rotary_dim=None,
            head_dim=256,
            layer_types=None,
            sliding_window=None,
        )
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            model_config=SimpleNamespace(hf_text_config=hf_cfg),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="qwen3-cfg-test",
            ),
        )

        grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(sliding_window=None),
            layer_names=["model.layers.0.self_attn.attn"],
        )
        kv_cache_cfg = SimpleNamespace(kv_cache_groups=[grp])

        conn = DeltaCompressionConnector(
            vllm_config=vllm_config,  # type: ignore[arg-type]
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_cfg,  # type: ignore[arg-type]
        )
        pol = conn.store.get_layer_policy("model.layers.0.self_attn.attn")
        self.assertEqual(pol.rope_theta, 1000000.0)
        self.assertEqual(pol.rotary_dim, 64)

    def test_qwen3_5_hybrid_gdn_mamba_and_full_attention_roundtrip(self) -> None:
        """Verifies Qwen3.5 hybrid GatedDeltaNet (conv_state, ssm_state) + FullAttention compression."""
        DeltaCompressionConnector.reset_shared_stores()
        torch.manual_seed(42)

        hf_cfg = SimpleNamespace(
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
            rope_theta=None,
            rope_local_base_freq=None,
            rotary_dim=None,
            head_dim=64,
            layer_types=["linear_attention", "full_attention"],
            sliding_window=None,
        )
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            model_config=SimpleNamespace(hf_text_config=hf_cfg),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="qwen3_5-hybrid-test",
            ),
        )

        mamba_grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(sliding_window=None, mamba_type="gdn"),
            layer_names=["model.layers.0.linear_attn"],
        )
        attn_grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(sliding_window=None),
            layer_names=["model.layers.1.self_attn.attn"],
        )
        kv_cache_cfg = SimpleNamespace(kv_cache_groups=[mamba_grp, attn_grp])

        # Create worker first without kv_cache_config (mirroring vLLM V1 init order),
        # then scheduler with kv_cache_config, verifying shared layer_to_group_idx.
        worker_conn = DeltaCompressionConnector(
            vllm_config=vllm_config,  # type: ignore[arg-type]
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        sched_conn = DeltaCompressionConnector(
            vllm_config=vllm_config,  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=kv_cache_cfg,  # type: ignore[arg-type]
        )

        gdn_pol = worker_conn.store.get_layer_policy("model.layers.0.linear_attn")
        attn_pol = worker_conn.store.get_layer_policy("model.layers.1.self_attn.attn")
        self.assertTrue(gdn_pol.is_mamba)
        self.assertEqual(gdn_pol.rotary_dim, 0)
        self.assertEqual(gdn_pol.max_sparse_exceptions, 0)
        self.assertFalse(attn_pol.is_mamba)
        self.assertEqual(attn_pol.rope_theta, 10000000.0)
        self.assertEqual(attn_pol.rotary_dim, 16)

        # Allocate paged caches:
        # Group 0 (GDN): tuple(conv_state [8, 256, 3], ssm_state [8, 4, 32, 32])
        # Group 1 (Attn): 4D tensor [8, 2, 16, 128] (num_kv_heads=2, head_dim=64)
        conv_cache = torch.randn(8, 256, 3, dtype=torch.bfloat16)
        ssm_cache = torch.randn(8, 4, 32, 32, dtype=torch.bfloat16)
        attn_cache = torch.randn(8, 2, 16, 128, dtype=torch.bfloat16)

        gdn_layer = SimpleNamespace(kv_cache=(conv_cache, ssm_cache))
        attn_layer = SimpleNamespace(kv_cache=attn_cache)
        fwd_ctx = SimpleNamespace(
            no_compile_layers={
                "model.layers.0.linear_attn": gdn_layer,
                "model.layers.1.self_attn.attn": attn_layer,
            },
            attn_metadata={},
            virtual_engine=0,
        )

        # 1. Base request P_A: 2 full blocks (32 tokens) + 4 tail tokens = 36 tokens
        # Group 0 (Mamba align mode) has [0 (null_block), 2 (boundary block)]
        # Group 1 (FullAttention) has [1, 2]
        blk0 = tuple(range(100, 116))
        blk1 = tuple(range(200, 216))
        tail = (901, 902, 903, 904)
        tokens_a = list(blk0 + blk1 + tail)

        sched_out_a = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_a",
                    prompt_token_ids=tokens_a,
                    block_ids=([0, 2], [1, 2]),
                )
            ],
            scheduled_cached_reqs=None,
            finished_req_ids=set(),
        )
        meta_a = sched_conn.build_connector_meta(sched_out_a)  # type: ignore[arg-type]
        worker_conn.bind_connector_metadata(meta_a)
        worker_conn.start_load_kv(fwd_ctx)  # type: ignore[arg-type]
        worker_conn.save_kv_layer(
            "model.layers.1.self_attn.attn", attn_cache, SimpleNamespace()  # type: ignore[arg-type]
        )
        worker_conn.wait_for_save()
        worker_conn.clear_connector_metadata()

        # Verify that GDN skipped block 0 (null_block) and stored only block 2 (1 block),
        # while FullAttention stored both blocks 1 and 2 (2 blocks) -> 3 total blocks stored.
        self.assertEqual(worker_conn.store.stats.total_blocks_stored, 3)

        # 2. Target request P_B: edited Block 0 + shared Block 1 + 4 tail tokens
        # Populate physical blocks 4 (GDN boundary) and [3, 4] (FullAttention) with
        # perturbed states (eta ~ 0.08 -> TIER1_INT2_DELTA).
        conv_cache[4] = conv_cache[2] + 0.05 * torch.randn_like(conv_cache[2])
        ssm_cache[4] = ssm_cache[2] + 0.05 * torch.randn_like(ssm_cache[2])
        attn_cache[3] = attn_cache[1] + 0.05 * torch.randn_like(attn_cache[1])
        attn_cache[4] = attn_cache[2] + 0.05 * torch.randn_like(attn_cache[2])

        blk0_edited = (999,) + blk0[1:]
        tokens_b = list(blk0_edited + blk1 + tail)
        sched_out_b = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_b",
                    prompt_token_ids=tokens_b,
                    block_ids=([0, 4], [3, 4]),
                )
            ],
            scheduled_cached_reqs=None,
            finished_req_ids=set(),
        )
        meta_b = sched_conn.build_connector_meta(sched_out_b)  # type: ignore[arg-type]
        worker_conn.bind_connector_metadata(meta_b)
        worker_conn.start_load_kv(fwd_ctx)  # type: ignore[arg-type]
        worker_conn.save_kv_layer(
            "model.layers.1.self_attn.attn", attn_cache, SimpleNamespace()  # type: ignore[arg-type]
        )
        worker_conn.wait_for_save()
        worker_conn.clear_connector_metadata()

        # Verify that P_B's shared Block 1 in GDN was delta-compressed (TIER3_SPARSE_BF16_INT4_DELTA)
        # and FullAttention was delta-compressed (TIER1_INT2_DELTA)
        self.assertGreaterEqual(
            worker_conn.store.stats.tier_counts[CompressionTier.TIER1_INT2_DELTA]
            + worker_conn.store.stats.tier_counts[CompressionTier.TIER2_INT4_DELTA]
            + worker_conn.store.stats.tier_counts[
                CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA
            ],
            2,
        )

        # 3. Verify get_num_new_matched_tokens and start_load_kv reconstruction for P_B
        req_b_reload = SimpleNamespace(
            request_id="req_b_reload",
            prompt_token_ids=tokens_b,
        )
        matched_toks, _ = sched_conn.get_num_new_matched_tokens(
            req_b_reload,  # type: ignore[arg-type]
            num_computed_tokens=0,
        )
        self.assertEqual(matched_toks, 32)

        sched_conn.update_state_after_alloc(
            req_b_reload,  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            num_external_tokens=32,
        )
        sched_out_reload = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_b_reload",
                    prompt_token_ids=tokens_b,
                    block_ids=([0, 6, 7], [5, 6, 7]),
                )
            ],
            num_scheduled_tokens={"req_b_reload": 4},
            scheduled_cached_reqs=None,
            finished_req_ids=set(),
        )
        conv_cache[6].zero_()
        ssm_cache[6].zero_()
        conv_cache[7].zero_()
        ssm_cache[7].zero_()
        attn_cache[5].zero_()
        attn_cache[6].zero_()

        meta_reload = sched_conn.build_connector_meta(sched_out_reload)  # type: ignore[arg-type]
        worker_conn.bind_connector_metadata(meta_reload)
        worker_conn.start_load_kv(fwd_ctx)  # type: ignore[arg-type]
        worker_conn.clear_connector_metadata()

        # Verify both GDN boundary block 6 AND running block 7 (post-preprocess_mamba),
        # plus FullAttention (attn_cache[5, 6]) were reconstructed accurately!
        for gdn_blk in (6, 7):
            ssm_err = torch.linalg.norm(
                (ssm_cache[gdn_blk] - ssm_cache[4]).to(torch.float32)
            ) / torch.linalg.norm(ssm_cache[4].to(torch.float32))
            conv_err = torch.linalg.norm(
                (conv_cache[gdn_blk] - conv_cache[4]).to(torch.float32)
            ) / torch.linalg.norm(conv_cache[4].to(torch.float32))
            self.assertLess(ssm_err.item(), 0.05)
            self.assertLess(conv_err.item(), 0.05)

    def test_qwen3_5_moe_partial_edit_mamba_delta_compression(self) -> None:
        """Verifies Mamba/GDN partial-edit delta compression for Qwen3.5-MoE (e.g. 35B-A3B)."""
        store = DeltaBlockStore(max_sparse_exceptions=4)
        pol = store.get_layer_policy("model.language_model.layers.0.linear_attn")
        self.assertTrue(pol.is_mamba)
        self.assertEqual(pol.tau_zero_bit, 0.0)
        self.assertEqual(pol.tau_int2, 0.0)
        torch.manual_seed(123)
        base_tokens = tuple(range(1000, 1064))
        ssm_base = torch.randn(4096, 1, dtype=torch.bfloat16)
        conv_base = torch.randn(512, 1, dtype=torch.bfloat16)

        rec_base = store.store_block(
            layer_name="model.language_model.layers.0.linear_attn",
            full_block_hash="mamba_base_h",
            token_ids=base_tokens,
            start_position=0,
            k_block=ssm_base,
            v_block=conv_base,
        )
        self.assertEqual(rec_base.tier, CompressionTier.UNCOMPRESSED_BASE)

        # Edit 2 tokens inside the block (<= max_sparse_exceptions=4) with small state perturbation
        edited_tokens = (9991, 9992) + base_tokens[2:]
        ssm_target = ssm_base + 0.04 * torch.randn_like(ssm_base)
        conv_target = conv_base.clone()

        rec_target = store.store_block(
            layer_name="model.language_model.layers.0.linear_attn",
            full_block_hash="mamba_target_h",
            token_ids=edited_tokens,
            start_position=0,
            k_block=ssm_target,
            v_block=conv_target,
        )
        self.assertEqual(rec_target.tier, CompressionTier.TIER2_INT4_DELTA)
        self.assertEqual(rec_target.ref_base_hash, "mamba_base_h")

        ssm_recon, conv_recon = store.reconstruct_block(
            layer_name="model.language_model.layers.0.linear_attn",
            full_block_hash="mamba_target_h",
            token_ids=edited_tokens,
            target_start_position=0,
        )
        assert conv_recon is not None
        rel_err = (
            torch.linalg.norm((ssm_recon - ssm_target).to(torch.float32))
            / torch.linalg.norm(ssm_target.to(torch.float32))
        ).item()
        self.assertLess(rel_err, 0.015)
        self.assertTrue(torch.equal(conv_recon, conv_target))

    def test_gpt_oss_yarn_rope_and_sliding_window_null_block_handling(self) -> None:
        """Verifies openai/gpt-oss-20b YaRN RoPE scaling and 1:1 sliding+full attention."""
        DeltaCompressionConnector.reset_shared_stores()
        yarn_params = {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": False,
            "rope_theta": 150000,
        }
        hf_cfg = SimpleNamespace(
            model_type="gpt_oss",
            rope_parameters=yarn_params,
            rope_scaling=yarn_params,
            rope_theta=None,
            rope_local_base_freq=None,
            rotary_dim=None,
            head_dim=64,
            layer_types=["sliding_attention", "full_attention", "sliding_attention"],
            sliding_window=128,
        )
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            model_config=SimpleNamespace(hf_text_config=hf_cfg),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="gpt-oss-test",
            ),
        )
        sw_grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(sliding_window=128),
            layer_names=["model.layers.0.attn.attn", "model.layers.2.attn.attn"],
        )
        full_grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(sliding_window=None),
            layer_names=["model.layers.1.attn.attn"],
        )
        kv_cache_cfg = SimpleNamespace(kv_cache_groups=[sw_grp, full_grp])

        conn = DeltaCompressionConnector(
            vllm_config=vllm_config,  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=kv_cache_cfg,  # type: ignore[arg-type]
        )
        pol0 = conn.store.get_layer_policy("model.layers.0.attn.attn")
        pol1 = conn.store.get_layer_policy("model.layers.1.attn.attn")
        pol2 = conn.store.get_layer_policy("model.layers.2.attn.attn")

        # Both sliding_attention and full_attention layers in gpt-oss use YaRN theta=150000
        self.assertEqual(pol0.rope_theta, 150000.0)
        self.assertEqual(pol1.rope_theta, 150000.0)
        self.assertEqual(pol2.rope_theta, 150000.0)
        self.assertIsNotNone(pol0.rope_scaling)
        self.assertIsNotNone(pol1.rope_scaling)
        self.assertIsNotNone(pol2.rope_scaling)
        # Layer 2 follows a full_attention layer (Layer 1), so unconditional d_div 0-bit is disabled
        # and tau_zero_bit is tightened to 0.01
        self.assertGreaterEqual(pol2.sliding_window_dedup_threshold, 1_000_000)
        self.assertAlmostEqual(pol2.tau_zero_bit, 0.01)

        # Verify YaRN relative rotation composition: R_yarn(p + dp) == R_yarn(dp) * R_yarn(p)
        k_raw = torch.randn(16, 8, 64, dtype=torch.float32)
        pos_a = torch.arange(32, 48)
        pos_b = torch.arange(48, 64)  # delta_pos = +16
        k_a = apply_relative_rope_rotation(
            k_raw,
            delta_pos=pos_a,
            rope_theta=150000.0,
            rope_scaling=yarn_params,
        )
        k_b_direct = apply_relative_rope_rotation(
            k_raw,
            delta_pos=pos_b,
            rope_theta=150000.0,
            rope_scaling=yarn_params,
        )
        k_b_from_a = apply_relative_rope_rotation(
            k_a,
            delta_pos=16,
            rope_theta=150000.0,
            rope_scaling=yarn_params,
        )
        self.assertTrue(torch.allclose(k_b_direct, k_b_from_a, atol=1e-5))

        # Verify hybrid local prefix hit + external connector load offset:
        # Blocks 0 and 1 (32 tokens) hit in local GPU prefix cache;
        # Blocks 2 and 3 (32 tokens) hit in external DeltaBlockStore;
        # Block 4 (16 tokens) + tail (4 tokens) are newly computed.
        blk0 = tuple(range(10, 26))
        blk1 = tuple(range(26, 42))
        blk2 = tuple(range(42, 58))
        blk3 = tuple(range(58, 74))
        blk4 = tuple(range(74, 90))
        tail = (91, 92, 93, 94)
        all_tokens = list(blk0 + blk1 + blk2 + blk3 + blk4 + tail)

        # Populate external store with blocks 0..3
        parent_h = "ROOT"
        for idx, b_toks in enumerate((blk0, blk1, blk2, blk3)):
            full_h = compute_full_prefix_block_hash(parent_h, b_toks)
            for l_name in (
                "model.layers.0.attn.attn",
                "model.layers.1.attn.attn",
                "model.layers.2.attn.attn",
            ):
                conn.store.store_block(
                    layer_name=l_name,
                    full_block_hash=full_h,
                    token_ids=b_toks,
                    start_position=idx * 16,
                    k_block=torch.ones(16, 2, 64, dtype=torch.bfloat16) * (idx + 1),
                    v_block=torch.ones(16, 2, 64, dtype=torch.bfloat16) * (idx + 1),
                )
            parent_h = full_h

        req_hybrid = SimpleNamespace(
            request_id="req_hybrid_local_ext",
            prompt_token_ids=all_tokens,
        )
        # Local prefix cache matched 32 tokens (blocks 0 and 1); external store has up to 64 tokens (blocks 0..3)
        ext_toks, _ = conn.get_num_new_matched_tokens(
            req_hybrid,  # type: ignore[arg-type]
            num_computed_tokens=32,
        )
        self.assertEqual(ext_toks, 32)
        conn.update_state_after_alloc(
            req_hybrid,  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            num_external_tokens=32,
        )
        sched_out_hybrid = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="req_hybrid_local_ext",
                    prompt_token_ids=all_tokens,
                    block_ids=([10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]),
                )
            ],
            num_scheduled_tokens={"req_hybrid_local_ext": 20},
            scheduled_cached_reqs=None,
            finished_req_ids=set(),
        )
        meta_hybrid = conn.build_connector_meta(sched_out_hybrid)  # type: ignore[arg-type]
        self.assertEqual(len(meta_hybrid.requests), 2)
        load_req_meta = meta_hybrid.requests[0]
        store_req_meta = meta_hybrid.requests[1]
        self.assertFalse(load_req_meta.is_store)
        self.assertTrue(store_req_meta.is_store)
        # Load specs must be blocks [2, 3] (start_position 32 and 48), NOT blocks [0, 1]
        self.assertEqual([b.start_position for b in load_req_meta.blocks], [32, 48])
        self.assertEqual(
            [b.group_block_ids for b in load_req_meta.blocks],
            [(12, 22), (13, 23)],
        )
        # Store specs must be block [4] (start_position 64)
        self.assertEqual([b.start_position for b in store_req_meta.blocks], [64])
        self.assertEqual(
            [b.group_block_ids for b in store_req_meta.blocks],
            [(14, 24)],
        )

    def test_gemma4_proportional_rope_and_heterogeneous_group_block_sizes(
        self,
    ) -> None:
        """Verifies Gemma 4 proportional RoPE, nested rope_parameters, and 16 vs 32 group block_size."""
        DeltaCompressionConnector.reset_shared_stores()
        gemma4_rope_params = {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
        }

        class MockGemma4TextConfig:
            model_type = "gemma4_text"
            rope_parameters = gemma4_rope_params
            rope_scaling = None
            rope_theta = None
            rope_local_base_freq = None
            rotary_dim = None
            layer_types = ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"]
            sliding_window = 1024

            def __getattribute__(self, name: str):
                if name in ("head_dim", "num_key_value_heads"):
                    raise ValueError(
                        f"AmbiguousGlobalPerLayerAttributeError: {name} is per-layer"
                    )
                return super().__getattribute__(name)

        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            model_config=SimpleNamespace(hf_text_config=MockGemma4TextConfig()),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="gemma4-test",
            ),
        )
        sw_grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=16, sliding_window=1024),
            layer_names=[
                "model.language_model.layers.0.self_attn.attn",
                "model.language_model.layers.2.self_attn.attn",
                "model.language_model.layers.3.self_attn.attn",
            ],
        )
        full_grp = SimpleNamespace(
            kv_cache_spec=SimpleNamespace(block_size=32, sliding_window=None),
            layer_names=["model.language_model.layers.1.self_attn.attn"],
        )
        kv_cache_cfg = SimpleNamespace(kv_cache_groups=[sw_grp, full_grp])

        conn = DeltaCompressionConnector(
            vllm_config=vllm_config,  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=kv_cache_cfg,  # type: ignore[arg-type]
        )
        self.assertEqual(conn._scheduler_block_size, 32)
        self.assertEqual(conn._get_group_block_scale(0), 1)
        self.assertEqual(conn._get_group_block_scale(1), 2)

        pol_sw = conn.store.get_layer_policy(
            "model.language_model.layers.0.self_attn.attn"
        )
        pol_full = conn.store.get_layer_policy(
            "model.language_model.layers.1.self_attn.attn"
        )
        pol_sw3 = conn.store.get_layer_policy(
            "model.language_model.layers.3.self_attn.attn"
        )
        self.assertTrue(pol_sw.is_sliding_window)
        self.assertEqual(pol_sw.rope_theta, 10000.0)
        self.assertIsNone(pol_sw.rotary_dim)
        self.assertGreaterEqual(pol_sw.sliding_window_dedup_threshold, 1024)
        self.assertTrue(pol_sw3.is_sliding_window)
        self.assertGreaterEqual(pol_sw3.sliding_window_dedup_threshold, 1_000_000)
        self.assertAlmostEqual(pol_sw3.tau_zero_bit, 0.01)
        self.assertFalse(pol_full.is_sliding_window)
        self.assertEqual(pol_full.rope_theta, 1000000.0)
        self.assertIsNone(pol_full.rotary_dim)
        self.assertEqual(
            pol_full.rope_scaling,
            gemma4_rope_params["full_attention"],
        )

        # Verify proportional RoPE rotates only [0:64] paired with [256:320] for head_dim=512
        k_raw = torch.randn(16, 4, 512, dtype=torch.float32)
        k_rot = apply_relative_rope_rotation(
            k_raw,
            delta_pos=16,
            rope_theta=1000000.0,
            rotary_dim=None,
            is_neox_style=True,
            rope_scaling=gemma4_rope_params["full_attention"],
        )
        # Non-rotated regions [64:256] and [320:512] must be bitwise identical to k_raw
        self.assertTrue(torch.equal(k_rot[..., 64:256], k_raw[..., 64:256]))
        self.assertTrue(torch.equal(k_rot[..., 320:512], k_raw[..., 320:512]))
        # Rotated regions [0:64] and [256:320] must preserve norm and compose additively
        k_rot2 = apply_relative_rope_rotation(
            k_rot,
            delta_pos=-16,
            rope_theta=1000000.0,
            rotary_dim=None,
            is_neox_style=True,
            rope_scaling=gemma4_rope_params["full_attention"],
        )
        self.assertTrue(torch.allclose(k_rot2, k_raw, atol=1e-5))

        # Verify sub-block extraction & injection on Group 1 (kbs=32, block_size=16)
        kv_layer_32 = torch.zeros(2, 4, 32, 4, 512, dtype=torch.bfloat16)
        k_sub0 = torch.ones(16, 4, 512, dtype=torch.bfloat16) * 3.0
        v_sub0 = torch.ones(16, 4, 512, dtype=torch.bfloat16) * 4.0
        k_sub1 = torch.ones(16, 4, 512, dtype=torch.bfloat16) * 7.0
        v_sub1 = torch.ones(16, 4, 512, dtype=torch.bfloat16) * 8.0

        conn._inject_block_kv(
            kv_layer_32,
            physical_block_id=2,
            k_block=k_sub0,
            v_block=v_sub0,
            is_mla=False,
            block_size=16,
            start_position=0,
        )
        conn._inject_block_kv(
            kv_layer_32,
            physical_block_id=2,
            k_block=k_sub1,
            v_block=v_sub1,
            is_mla=False,
            block_size=16,
            start_position=16,
        )
        ext_k0, ext_v0 = conn._extract_block_kv(
            kv_layer_32,
            physical_block_id=2,
            is_mla=False,
            block_size=16,
            start_position=0,
        )
        ext_k1, ext_v1 = conn._extract_block_kv(
            kv_layer_32,
            physical_block_id=2,
            is_mla=False,
            block_size=16,
            start_position=16,
        )
        self.assertTrue(torch.equal(ext_k0, k_sub0))
        self.assertTrue(torch.equal(ext_v0, v_sub0))
        self.assertTrue(torch.equal(ext_k1, k_sub1))
        self.assertTrue(torch.equal(ext_v1, v_sub1))

        # Verify _build_block_specs_for_tokens maps b_idx 0,1 -> full_grp[0] and b_idx 2 -> full_grp[1]
        tokens_48 = list(range(100, 148))
        specs = conn._build_block_specs_for_tokens(
            tokens_48,
            block_ids=[10, 11, 12],
            all_group_block_ids=([10, 11, 12], [20, 21]),
        )
        self.assertEqual(len(specs), 3)
        self.assertEqual(
            [s.group_block_ids for s in specs],
            [(10, 20), (11, 20), (12, 21)],
        )
        # Populate all 3 blocks (48 tokens) into the store and check that get_num_new_matched_tokens
        # aligns external hits to _scheduler_block_size = 32 tokens
        for s in specs:
            conn.store.store_block(
                layer_name="model.language_model.layers.0.self_attn.attn",
                full_block_hash=s.full_block_hash,
                token_ids=s.token_ids,
                start_position=s.start_position,
                k_block=torch.ones(16, 8, 256, dtype=torch.bfloat16),
                v_block=torch.ones(16, 8, 256, dtype=torch.bfloat16),
            )
        req_50 = SimpleNamespace(
            request_id="req_gemma4_align",
            prompt_token_ids=tokens_48 + [998, 999],
        )
        matched_toks, _ = conn.get_num_new_matched_tokens(
            req_50,  # type: ignore[arg-type]
            num_computed_tokens=0,
        )
        self.assertEqual(matched_toks, 32)

    def test_large_head_dim_subhead_quant_and_tier3_threshold(self) -> None:
        """Verifies sub-head group_size=32 on head_dim=512 and TIER3 residual threshold fallback."""
        # 1. Simulate Gemma 4 full_attention head_dim=512 with learned k_norm channel outliers
        res_512 = 0.10 * torch.randn(16, 2, 512, dtype=torch.bfloat16)
        res_512[:, :, 0] *= 8.0  # Outlier channel in group 0 only
        packed = quantize_residual_packed(res_512, bits=4)
        # 16 * 2 * 512 / 32 = 512 scale groups
        self.assertEqual(packed.scales.shape[0], (16 * 2 * 512) // 32)
        recon = dequantize_residual_packed(packed)
        # Non-outlier channels (32..511) should have ~21 dB residual SQNR (vs ~4 dB without sub-head groups)
        sqnr_non_outlier = _compute_sqnr_db(res_512[:, :, 32:], recon[:, :, 32:])
        self.assertGreater(sqnr_non_outlier, 19.0)

        # 2. Verify TIER3 falls back to UNCOMPRESSED_BASE when non-edited rows exceed tau_int4
        store = DeltaBlockStore(block_size=16)
        layer_name = "model.language_model.layers.11.self_attn.attn"
        store.register_layer_policy(
            layer_name, LayerCompressionPolicy(layer_idx=11, tau_int4=0.70)
        )
        base_tokens = tuple(range(600, 616))
        k_base = torch.randn(16, 2, 512, dtype=torch.bfloat16)
        v_base = torch.randn(16, 2, 512, dtype=torch.bfloat16)
        hash_base = compute_full_prefix_block_hash("ROOT", base_tokens)
        store.store_block(layer_name, hash_base, base_tokens, 0, k_base, v_base)

        edited_tokens = list(base_tokens)
        edited_tokens[4] = 7777
        edited_tuple = tuple(edited_tokens)
        # Non-edited rows have large perturbation (eta ~ 0.83 > tau_int4=0.70)
        k_perturbed = k_base + 1.5 * torch.randn_like(k_base)
        v_perturbed = v_base + 1.5 * torch.randn_like(v_base)
        hash_edited = compute_full_prefix_block_hash("ROOT", edited_tuple)
        rec = store.store_block(
            layer_name, hash_edited, edited_tuple, 0, k_perturbed, v_perturbed
        )
        self.assertEqual(rec.tier, CompressionTier.UNCOMPRESSED_BASE)

    def test_gemma4_high_crest_zero_preserving_quant_and_moe_token_spike(self) -> None:
        """Verifies zero-preserving INT2/INT4 on high-crest V residuals and 1-token MoE spike promotion."""
        torch.manual_seed(7)
        # 1. High-crest-factor residual (1 outlier channel = 12.0, 255 near-zero channels ~ 0.02)
        res_sparse = 0.02 * torch.randn(16, 4, 256, dtype=torch.bfloat16)
        res_sparse[:, :, 0] = 12.0
        pre_norm = torch.linalg.norm(res_sparse.to(torch.float32)).item()

        for bits in (2, 4):
            q = quantize_residual_packed(res_sparse, bits=bits)
            recon = dequantize_residual_packed(q)
            err_norm = torch.linalg.norm(
                (res_sparse.to(torch.float32) - recon.to(torch.float32))
            ).item()
            # Zero-preserving quantization must sharply reduce residual norm (never amplify it!)
            self.assertLess(err_norm, 0.25 * pre_norm)

        # 2. Exact token match where 1 token out of 16 has a 120% MoE routing spike
        store = DeltaBlockStore(block_size=16)
        layer_name = "model.language_model.layers.22.self_attn.attn"
        store.register_layer_policy(layer_name, LayerCompressionPolicy(layer_idx=22))
        toks = tuple(range(800, 816))
        k_base = torch.randn(16, 4, 256, dtype=torch.bfloat16)
        v_base = torch.randn(16, 4, 256, dtype=torch.bfloat16)
        h_base = compute_full_prefix_block_hash("P1", toks)
        store.store_block(layer_name, h_base, toks, 32, k_base, v_base)

        k_target = k_base + 0.05 * torch.randn_like(k_base)
        v_target = v_base + 0.05 * torch.randn_like(v_base)
        # Token 9 flips MoE expert routing (eta_tok > 1.0) while block-wide eta ~ 0.30 <= tau_int4
        v_target[9] = -v_base[9] + 0.8 * torch.randn_like(v_base[9])
        h_target = compute_full_prefix_block_hash("P2", toks)
        rec_t = store.store_block(layer_name, h_target, toks, 32, k_target, v_target)
        self.assertEqual(rec_t.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        assert rec_t.sparse_token_indices is not None
        self.assertEqual(rec_t.sparse_token_indices.tolist(), [9])
        k_rec, v_rec = store.reconstruct_block(layer_name, h_target)
        assert v_rec is not None
        self.assertTrue(torch.equal(v_rec[9], v_target[9]))

        # 3. Intra-request exclude_base_hashes prevents same-request partial-match shadowing
        near_toks = list(toks)
        near_toks[0] = 99999
        h_same_req = compute_full_prefix_block_hash("P1_LATER", tuple(near_toks))
        rec_same = store.store_block(
            layer_name,
            h_same_req,
            tuple(near_toks),
            96,
            k_target,
            v_target,
            exclude_base_hashes={h_base},
        )
        self.assertEqual(rec_same.tier, CompressionTier.UNCOMPRESSED_BASE)

    def test_multiprocess_tp_scheduler_worker_store_separation(self) -> None:
        """Verifies TP>1 multi-process separation between Scheduler and Worker stores."""
        DeltaCompressionConnector.reset_shared_stores()
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="test_engine_tp8",
                kv_connector_extra_config={},
            ),
        )
        sched_conn = KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config=None  # type: ignore[arg-type]
        )
        worker_conn = KVConnectorFactory.create_connector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config=None  # type: ignore[arg-type]
        )
        assert isinstance(sched_conn, DeltaCompressionConnector)
        assert isinstance(worker_conn, DeltaCompressionConnector)
        # Simulate a separate worker subprocess with its own DeltaBlockStore instance
        worker_conn.store = DeltaBlockStore(block_size=16)

        toks = list(range(100, 136))
        req1 = SimpleNamespace(request_id="tp-req-1", prompt_token_ids=toks)
        m1, _ = sched_conn.get_num_new_matched_tokens(req1, 0)  # type: ignore[arg-type]
        self.assertEqual(m1, 0)
        sched_conn.update_state_after_alloc(req1, mock.MagicMock(), 0)  # type: ignore[arg-type]
        sched_out1 = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="tp-req-1",
                    prompt_token_ids=toks,
                    block_ids=([1, 2, 3],),
                )
            ]
        )
        meta1 = sched_conn.build_connector_meta(sched_out1)  # type: ignore[arg-type]
        # Scheduler store must already know the prefix hashes even before worker runs
        req2 = SimpleNamespace(request_id="tp-req-2", prompt_token_ids=toks)
        m2, _ = sched_conn.get_num_new_matched_tokens(req2, 0)  # type: ignore[arg-type]
        self.assertEqual(m2, 32)

        # Worker saves its TP KV shard into its own isolated store
        kv_shard = torch.randn(2, 8, 16, 1, 64, dtype=torch.bfloat16)
        worker_conn.bind_connector_metadata(meta1)
        worker_conn.save_kv_layer("model.layers.0.attn", kv_shard, mock.MagicMock())
        worker_conn.clear_connector_metadata()
        self.assertEqual(worker_conn.store.stats.total_blocks_stored, 2)
        self.assertEqual(sched_conn.store.stats.total_blocks_stored, 0)

        # Clear on Scheduler store propagates clear_epoch to Worker on next step
        sched_conn.store.clear()
        m3, _ = sched_conn.get_num_new_matched_tokens(req2, 0)  # type: ignore[arg-type]
        self.assertEqual(m3, 0)
        meta2 = sched_conn.build_connector_meta(SimpleNamespace(scheduled_new_reqs=[]))  # type: ignore[arg-type]
        worker_conn.bind_connector_metadata(meta2)
        self.assertEqual(worker_conn.store.stats.total_blocks_stored, 0)

    def test_batched_store_and_reconstruct_equivalence(self) -> None:
        """Verifies store_blocks_batch and reconstruct_blocks_batch match single-block execution."""
        torch.manual_seed(99)
        store_single = DeltaBlockStore(block_size=16)
        store_batch = DeltaBlockStore(block_size=16)
        layer_name = "model.layers.8.self_attn.attn"
        pol = LayerCompressionPolicy(
            layer_idx=8,
            rope_theta=1000000.0,
            tau_zero_bit=0.05,
            tau_int2=0.18,
            tau_int4=0.70,
        )
        store_single.register_layer_policy(layer_name, pol)
        store_batch.register_layer_policy(layer_name, pol)

        # Create 3 base blocks + 4 shifted/perturbed target blocks covering TIER0, TIER1, TIER2, TIER3
        tok0 = tuple(range(100, 116))
        tok1 = tuple(range(200, 216))
        tok2 = tuple(range(300, 316))
        k0 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v0 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        k1 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v1 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        k2 = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        v2 = torch.randn(16, 4, 128, dtype=torch.bfloat16)

        # Shifted K0 (+32 pos): TIER0_IDENTICAL
        k0_shift = apply_relative_rope_rotation(k0, 32, rope_theta=1000000.0)
        v0_shift = v0.clone()

        # Shifted K1 (+16 pos) + small perturbation (eta ~ 0.10): TIER1_INT2_DELTA
        k1_shift = apply_relative_rope_rotation(k1, 16, rope_theta=1000000.0) + 0.08 * torch.randn_like(k1)
        v1_shift = v1 + 0.08 * torch.randn_like(v1)

        # Shifted K2 (+48 pos) + moderate perturbation (eta ~ 0.30): TIER2_INT4_DELTA
        k2_shift = apply_relative_rope_rotation(k2, 48, rope_theta=1000000.0) + 0.25 * torch.randn_like(k2)
        v2_shift = v2 + 0.25 * torch.randn_like(v2)

        # Partial token edit (1 token changed in tok2): TIER3_SPARSE_BF16_INT4_DELTA
        tok2_edit = (9999,) + tok2[1:]
        k2_edit = k2_shift.clone()
        v2_edit = v2_shift.clone()
        k2_edit[0] = torch.randn(4, 128, dtype=torch.bfloat16)
        v2_edit[0] = torch.randn(4, 128, dtype=torch.bfloat16)

        base_items = [
            (
                DeltaBlockTransferSpec(
                    physical_block_id=0,
                    full_block_hash="h_base_0",
                    token_ids=tok0,
                    start_position=0,
                ),
                k0,
                v0,
            ),
            (
                DeltaBlockTransferSpec(
                    physical_block_id=1,
                    full_block_hash="h_base_1",
                    token_ids=tok1,
                    start_position=16,
                ),
                k1,
                v1,
            ),
            (
                DeltaBlockTransferSpec(
                    physical_block_id=2,
                    full_block_hash="h_base_2",
                    token_ids=tok2,
                    start_position=32,
                ),
                k2,
                v2,
            ),
        ]
        target_items = [
            (
                DeltaBlockTransferSpec(
                    physical_block_id=3,
                    full_block_hash="h_t0",
                    token_ids=tok0,
                    start_position=32,
                ),
                k0_shift,
                v0_shift,
            ),
            (
                DeltaBlockTransferSpec(
                    physical_block_id=4,
                    full_block_hash="h_t1",
                    token_ids=tok1,
                    start_position=32,
                ),
                k1_shift,
                v1_shift,
            ),
            (
                DeltaBlockTransferSpec(
                    physical_block_id=5,
                    full_block_hash="h_t2",
                    token_ids=tok2,
                    start_position=80,
                ),
                k2_shift,
                v2_shift,
            ),
            (
                DeltaBlockTransferSpec(
                    physical_block_id=6,
                    full_block_hash="h_t3",
                    token_ids=tok2_edit,
                    start_position=80,
                ),
                k2_edit,
                v2_edit,
            ),
        ]

        recs_single_base = [
            store_single.store_block(
                layer_name, spec.full_block_hash, spec.token_ids, spec.start_position, kb, vb
            )
            for (spec, kb, vb) in base_items
        ]
        recs_batch_base = store_batch.store_blocks_batch(layer_name, base_items)

        recs_single_target = [
            store_single.store_block(
                layer_name, spec.full_block_hash, spec.token_ids, spec.start_position, kb, vb
            )
            for (spec, kb, vb) in target_items
        ]
        recs_batch_target = store_batch.store_blocks_batch(layer_name, target_items)

        for r_s, r_b in zip(
            recs_single_base + recs_single_target, recs_batch_base + recs_batch_target
        ):
            self.assertEqual(r_s.tier, r_b.tier)
            self.assertEqual(r_s.nbytes, r_b.nbytes)

        all_specs = [spec for (spec, _, _) in base_items + target_items]
        recons_single = [
            store_single.reconstruct_block(
                layer_name, spec.full_block_hash, spec.token_ids, spec.start_position
            )
            for spec in all_specs
        ]
        recons_batch = store_batch.reconstruct_blocks_batch(layer_name, all_specs)
        for (ks, vs), (kb, vb) in zip(recons_single, recons_batch):
            self.assertTrue(torch.allclose(ks, kb, atol=1e-3, rtol=1e-3))
            assert vs is not None and vb is not None
            self.assertTrue(torch.allclose(vs, vb, atol=1e-3, rtol=1e-3))

    def test_approach1_cross_block_reference_stitching_unaligned_shift(self) -> None:
        """Verifies Approach 1 server-side cross-block reference stitching for delta_p != 0 (mod B)."""
        torch.manual_seed(2026)
        block_size = 16
        store_single = DeltaBlockStore(block_size=block_size)
        store_batch = DeltaBlockStore(block_size=block_size)

        l0 = "model.layers.0.self_attn.attn"
        l5 = "model.layers.5.self_attn.attn"
        for s in (store_single, store_batch):
            s.register_layer_policy(
                l0, LayerCompressionPolicy(layer_idx=0, rope_theta=10000.0)
            )
            s.register_layer_policy(
                l5,
                LayerCompressionPolicy(
                    layer_idx=5,
                    rope_theta=10000.0,
                    tau_zero_bit=0.05,
                    tau_int2=0.18,
                    tau_int4=0.70,
                ),
            )

        # Turn 1: two consecutive 16-token base blocks B0 (pos 16..31) and B1 (pos 32..47)
        tok_b0 = tuple(range(1000, 1016))
        tok_b1 = tuple(range(1016, 1032))
        h_b0 = compute_full_prefix_block_hash("T1_PREFIX", tok_b0)
        h_b1 = compute_full_prefix_block_hash(h_b0, tok_b1)
        t_hash_b0 = compute_token_block_hash(tok_b0)

        # Underlying unrotated K/V for the 32 contiguous tokens
        k_raw_32 = torch.randn(32, 4, 128, dtype=torch.bfloat16)
        v_raw_32 = torch.randn(32, 4, 128, dtype=torch.bfloat16)
        k_b0_l0 = apply_relative_rope_rotation(
            k_raw_32[:16], delta_pos=torch.arange(16, 32), rope_theta=10000.0
        )
        k_b1_l0 = apply_relative_rope_rotation(
            k_raw_32[16:], delta_pos=torch.arange(32, 48), rope_theta=10000.0
        )
        v_b0_l0 = v_raw_32[:16].clone()
        v_b1_l0 = v_raw_32[16:].clone()

        for s in (store_single, store_batch):
            s.store_block(l0, h_b0, tok_b0, 16, k_b0_l0, v_b0_l0)
            s.store_block(
                l0,
                h_b1,
                tok_b1,
                32,
                k_b1_l0,
                v_b1_l0,
                prev_token_block_hash=t_hash_b0,
            )
            s.store_block(l5, h_b0, tok_b0, 16, k_b0_l0, v_b0_l0)
            s.store_block(
                l5,
                h_b1,
                tok_b1,
                32,
                k_b1_l0,
                v_b1_l0,
                prev_token_block_hash=t_hash_b0,
            )

        # Turn 2: prefix grows by +5 tokens (unaligned shift!).
        # A block starting at pos=32 now contains tok_b0[11:16] + tok_b1[0:11] (stitch_offset s=11).
        # Its original tokens were at RoPE positions [16 + 11 .. 16 + 11 + 15] = [27 .. 42],
        # so delta_pos = 32 - (16 + 11) = +5!
        stitch_s = 11
        tok_stitched = tok_b0[stitch_s:] + tok_b1[:stitch_s]
        self.assertTrue(store_single.has_stitched_token_block(tok_stitched))
        h_stitched = compute_full_prefix_block_hash("T2_UNALIGNED_PREFIX", tok_stitched)
        k_stitched_l0 = apply_relative_rope_rotation(
            k_raw_32[stitch_s : stitch_s + 16],
            delta_pos=torch.arange(32, 48),
            rope_theta=10000.0,
        )
        v_stitched_l0 = v_raw_32[stitch_s : stitch_s + 16].clone()

        # 1. Layer 0 unaligned shift -> TIER0_ZERO_BIT via stitched (B0, B1)
        rec_l0 = store_single.store_block(
            l0, h_stitched, tok_stitched, 32, k_stitched_l0, v_stitched_l0
        )
        self.assertEqual(rec_l0.tier, CompressionTier.TIER0_ZERO_BIT)
        self.assertEqual(rec_l0.ref_base_hash, h_b0)
        self.assertEqual(rec_l0.ref_base_hash_2, h_b1)
        self.assertEqual(rec_l0.stitch_offset, stitch_s)
        self.assertEqual(rec_l0.delta_pos, 5)
        k_rec_l0, v_rec_l0 = store_single.reconstruct_block(l0, h_stitched)
        assert v_rec_l0 is not None
        self.assertTrue(torch.equal(v_rec_l0, v_stitched_l0))
        self.assertGreater(_compute_sqnr_db(k_stitched_l0, k_rec_l0), 50.0)

        # 2. Layer 5 unaligned shift + cross-prefix residual -> TIER2_INT4_DELTA via stitched (B0, B1)
        k_stitched_l5 = k_stitched_l0 + 0.16 * torch.randn_like(k_stitched_l0)
        v_stitched_l5 = v_stitched_l0 + 0.16 * torch.randn_like(v_stitched_l0)
        rec_l5 = store_single.store_block(
            l5, h_stitched, tok_stitched, 32, k_stitched_l5, v_stitched_l5
        )
        self.assertEqual(rec_l5.tier, CompressionTier.TIER2_INT4_DELTA)
        self.assertEqual(rec_l5.ref_base_hash, h_b0)
        self.assertEqual(rec_l5.ref_base_hash_2, h_b1)
        self.assertEqual(rec_l5.stitch_offset, stitch_s)
        self.assertEqual(rec_l5.delta_pos, 5)
        k_rec_l5, v_rec_l5 = store_single.reconstruct_block(l5, h_stitched)
        assert v_rec_l5 is not None
        self.assertGreater(_compute_sqnr_db(k_stitched_l5, k_rec_l5), 34.0)

        # 3. Layer 5 unaligned shift + 2 intra-block token edits -> TIER3_SPARSE_BF16_INT4_DELTA
        tok_stitched_edit = list(tok_stitched)
        tok_stitched_edit[2] = 88881
        tok_stitched_edit[13] = 88882
        tok_stitched_edit_t = tuple(tok_stitched_edit)
        h_stitched_edit = compute_full_prefix_block_hash(
            "T2_UNALIGNED_EDIT", tok_stitched_edit_t
        )
        k_stitched_edit = k_stitched_l5.clone()
        v_stitched_edit = v_stitched_l5.clone()
        k_stitched_edit[2] = torch.randn(4, 128, dtype=torch.bfloat16) * 2.5
        v_stitched_edit[2] = torch.randn(4, 128, dtype=torch.bfloat16) * 2.5
        k_stitched_edit[13] = torch.randn(4, 128, dtype=torch.bfloat16) * 2.5
        v_stitched_edit[13] = torch.randn(4, 128, dtype=torch.bfloat16) * 2.5
        rec_l5_edit = store_single.store_block(
            l5,
            h_stitched_edit,
            tok_stitched_edit_t,
            32,
            k_stitched_edit,
            v_stitched_edit,
        )
        self.assertEqual(
            rec_l5_edit.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA
        )
        self.assertEqual(rec_l5_edit.ref_base_hash, h_b0)
        self.assertEqual(rec_l5_edit.ref_base_hash_2, h_b1)
        self.assertEqual(rec_l5_edit.stitch_offset, stitch_s)
        assert rec_l5_edit.sparse_token_indices is not None
        self.assertEqual(rec_l5_edit.sparse_token_indices.tolist(), [2, 13])
        k_rec_edit, v_rec_edit = store_single.reconstruct_block(l5, h_stitched_edit)
        assert v_rec_edit is not None
        self.assertTrue(torch.equal(k_rec_edit[2], k_stitched_edit[2]))
        self.assertTrue(torch.equal(k_rec_edit[13], k_stitched_edit[13]))
        self.assertTrue(torch.equal(v_rec_edit[2], v_stitched_edit[2]))
        self.assertTrue(torch.equal(v_rec_edit[13], v_stitched_edit[13]))

        # 4. Verify batched store & reconstruct equivalence on stitched blocks
        batch_specs = [
            DeltaBlockTransferSpec(
                physical_block_id=10,
                full_block_hash=h_stitched,
                token_ids=tok_stitched,
                start_position=32,
            ),
            DeltaBlockTransferSpec(
                physical_block_id=11,
                full_block_hash=h_stitched_edit,
                token_ids=tok_stitched_edit_t,
                start_position=32,
            ),
        ]
        batch_recs = store_batch.store_blocks_batch(
            l5,
            [
                (batch_specs[0], k_stitched_l5, v_stitched_l5),
                (batch_specs[1], k_stitched_edit, v_stitched_edit),
            ],
        )
        self.assertEqual(batch_recs[0].tier, rec_l5.tier)
        self.assertEqual(batch_recs[0].stitch_offset, stitch_s)
        self.assertEqual(batch_recs[1].tier, rec_l5_edit.tier)
        self.assertEqual(batch_recs[1].stitch_offset, stitch_s)
        batch_recons = store_batch.reconstruct_blocks_batch(l5, batch_specs)
        self.assertTrue(
            torch.allclose(batch_recons[0][0], k_rec_l5, atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(batch_recons[1][0], k_rec_edit, atol=1e-3, rtol=1e-3)
        )

        # 5. Verify LRU dependency tracking & protection for BOTH ref_base_hash (h_b0) and ref_base_hash_2 (h_b1)
        self.assertIn(h_stitched, store_single._dependents_by_base[h_b0])
        self.assertIn(h_stitched, store_single._dependents_by_base[h_b1])
        self.assertIn(h_stitched_edit, store_single._dependents_by_base[h_b0])
        self.assertIn(h_stitched_edit, store_single._dependents_by_base[h_b1])
        # Protecting h_stitched during budget eviction transitively protects both h_b0 and h_b1!
        evicted_under_protect = store_single.evict_lru_to_budget(
            max_bytes=1, protected_hashes={h_stitched}
        )
        self.assertIn(h_stitched_edit, evicted_under_protect)
        self.assertNotIn(h_b0, evicted_under_protect)
        self.assertNotIn(h_b1, evicted_under_protect)
        self.assertTrue(store_single.has_full_block(h_b0))
        self.assertTrue(store_single.has_full_block(h_b1))
        # Cascading evict of h_b0 also evicts remaining dependent h_stitched
        cascaded = store_single.evict_block(h_b0)
        self.assertIn(h_stitched, cascaded)
        self.assertIn(h_b0, cascaded)

    def test_approach2_configurable_padding_token_and_server_alignment(self) -> None:
        """Verifies Approach 2 configurable padding token (default '\\n', map, ID) and server bucket alignment."""
        class MockCharTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return [ord(ch) for ch in text]

        tok = MockCharTokenizer()
        # 1. Default pad_token is "\n" (ord('\n') == 10)
        self.assertEqual(DEFAULT_PAD_TOKEN, "\n")
        self.assertEqual(resolve_pad_token_id(tok), ord("\n"))
        self.assertEqual(resolve_pad_token_id(tok, pad_token=" "), ord(" "))
        self.assertEqual(resolve_pad_token_id(tok, pad_token_id=777), 777)

        # 2. Content-dependent pad_token_map selection
        pad_map = {
            "[File:": "\n",
            r"^\s*\{": " ",
            "SELECT ": " ",
            "/*": "\t",
        }
        self.assertEqual(
            resolve_pad_token_id(
                tok, content_text="[File: app.py (100 lines total)]\n1:x=1\n", pad_token_map=pad_map
            ),
            ord("\n"),
        )
        self.assertEqual(
            resolve_pad_token_id(
                tok, content_text='  {"key": "value"}', pad_token_map=pad_map
            ),
            ord(" "),
        )
        self.assertEqual(
            resolve_pad_token_id(
                tok, content_text="SELECT * FROM users;", pad_token_map=pad_map
            ),
            ord(" "),
        )
        self.assertEqual(
            resolve_pad_token_id(
                tok, content_text="/* C block comment */", pad_token_map=pad_map
            ),
            ord("\t"),
        )

        # 3. Connector & Store initialization with custom pad_token and pad_token_map
        DeltaCompressionConnector.reset_shared_stores()
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                engine_id="pad-cfg-test",
                kv_connector_extra_config={
                    "pad_token": "\n",
                    "pad_token_map": pad_map,
                },
            ),
        )
        conn = DeltaCompressionConnector(
            vllm_config=vllm_config,  # type: ignore[arg-type]
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        self.assertEqual(conn.store.pad_token, "\n")
        self.assertEqual(conn.resolve_pad_token_id(tok), ord("\n"))
        self.assertEqual(
            conn.resolve_pad_token_id(tok, content_text='{"a": 1}'), ord(" ")
        )

        # Verify align_structural_prompt uses content-specific padding per segment
        seg_tokens = conn.align_structural_prompt(
            tok,
            segments=[
                ('{"a": 1}', True),
                ("def foo():\n", True),
                ("tail", False),
            ],
        )
        # First segment (8 chars) padded to 16 with ' ' (32)
        self.assertEqual(seg_tokens[:16], [ord(c) for c in '{"a": 1}'] + [ord(" ")] * 8)
        # Second segment (11 chars) padded to 16 with '\n' (10)
        self.assertEqual(
            seg_tokens[16:32], [ord(c) for c in "def foo():\n"] + [ord("\n")] * 5
        )
        # Final suffix segment unpadded (4 chars)
        self.assertEqual(seg_tokens[32:], [ord(c) for c in "tail"])

        # 4. Verify encode_file_viewer_bucketed_server isolates multi-token edits to their 10-line bucket
        lines_v1 = ["[File: foo.py (30 lines total)]\n"]
        lines_v2 = ["[File: foo.py (30 lines total)]\n"]
        for lno in range(1, 31):
            if lno == 5:
                lines_v1.append(f"{lno}:x = 1\n")
                lines_v2.append(f"{lno}:x = 1000 + 2000 + 3000\n")  # +16 chars in bucket 0 (lines 1..10)
            else:
                lines_v1.append(f"{lno}:y_{lno} = {lno}\n")
                lines_v2.append(f"{lno}:y_{lno} = {lno}\n")

        ids_v1 = conn.encode_aligned_segment(tok, "".join(lines_v1))
        ids_v2 = conn.encode_aligned_segment(tok, "".join(lines_v2))
        self.assertEqual(len(ids_v1), len(ids_v2))
        self.assertEqual(len(ids_v1) % 16, 0)
        # Banner is 1 block (0..16), Bucket 0 (lines 1..10) is 6 blocks (16..112),
        # Bucket 1 (lines 11..20) and Bucket 2 (lines 21..30) start at token 112 and must be identical!
        self.assertEqual(ids_v1[112:], ids_v2[112:])

    def test_large_block_size_windowed_norm_and_mamba_tier3_fidelity(self) -> None:
        """Verifies large block_size=2096 windowed L2 norm + sparse outliers and 3D Mamba TIER3."""
        torch.manual_seed(2026)
        b_size = 2096
        store_single = DeltaBlockStore(block_size=b_size, max_sparse_exceptions=4)
        store_batch = DeltaBlockStore(block_size=b_size, max_sparse_exceptions=4)

        attn_layer = "model.language_model.layers.3.self_attn.attn"
        mamba_layer = "model.language_model.layers.2.linear_attn"
        tokens = tuple(range(10000, 10000 + b_size))

        # 1. FullAttention block (2096 tokens): first 2020 tokens have tiny noise (~0.01),
        # final 76 tokens ([Risk Factors]) have 0.35 drift. Global L2 norm is diluted to ~0.068,
        # which previously fell into TIER1_INT2_DELTA without BF16 outliers.
        k_base = torch.randn(b_size, 2, 128, dtype=torch.bfloat16)
        v_base = torch.randn(b_size, 2, 128, dtype=torch.bfloat16)
        k_target = k_base + 0.01 * torch.randn_like(k_base)
        v_target = v_base + 0.01 * torch.randn_like(v_base)
        k_target[-76:] = k_base[-76:] + 0.35 * torch.randn_like(k_base[-76:])
        v_target[-76:] = v_base[-76:] + 0.35 * torch.randn_like(v_base[-76:])

        # 2. Mamba/GDN 3D ssm_state [8, 64, 64] + conv_state [1024, 3]
        ssm_base = torch.randn(8, 64, 64, dtype=torch.bfloat16)
        conv_base = torch.randn(1024, 3, dtype=torch.bfloat16)
        ssm_target = ssm_base + 0.04 * torch.randn_like(ssm_base)
        # 16 rows across heads have large state updates (0.45)
        ssm_target.view(-1, 64)[:16] += 0.45 * torch.randn(
            16, 64, dtype=torch.bfloat16
        )
        conv_target = conv_base + 0.25 * torch.randn_like(conv_base)

        for s in (store_single, store_batch):
            s.store_block(attn_layer, "h_attn_base", tokens, b_size, k_base, v_base)
            s.store_block(
                mamba_layer, "h_mamba_base", tokens, b_size, ssm_base, conv_base
            )

        rec_attn_s = store_single.store_block(
            attn_layer, "h_attn_tgt", tokens, b_size, k_target, v_target
        )
        rec_mamba_s = store_single.store_block(
            mamba_layer, "h_mamba_tgt", tokens, b_size, ssm_target, conv_target
        )

        spec_attn = DeltaBlockTransferSpec(
            full_block_hash="h_attn_tgt",
            token_ids=tokens,
            start_position=b_size,
            physical_block_id=2,
        )
        spec_mamba = DeltaBlockTransferSpec(
            full_block_hash="h_mamba_tgt",
            token_ids=tokens,
            start_position=b_size,
            physical_block_id=2,
        )
        rec_attn_b = store_batch.store_blocks_batch(
            attn_layer, [(spec_attn, k_target, v_target), (spec_attn, k_target, v_target)]
        )[0]
        rec_mamba_b = store_batch.store_blocks_batch(
            mamba_layer,
            [(spec_mamba, ssm_target, conv_target), (spec_mamba, ssm_target, conv_target)],
        )[0]

        self.assertEqual(rec_attn_s.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        self.assertEqual(rec_attn_b.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        self.assertEqual(rec_mamba_s.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        self.assertEqual(rec_mamba_b.tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)

        # Verify all 76 drifted tail tokens in FullAttention are preserved bitwise in BF16!
        k_rec_s, v_rec_s = store_single.reconstruct_block(attn_layer, "h_attn_tgt")
        k_rec_b, v_rec_b = store_batch.reconstruct_blocks_batch(
            attn_layer, [spec_attn, spec_attn]
        )[0]
        assert v_rec_s is not None and v_rec_b is not None
        self.assertTrue(torch.equal(k_rec_s[-76:], k_target[-76:]))
        self.assertTrue(torch.equal(v_rec_s[-76:], v_target[-76:]))
        self.assertTrue(torch.equal(k_rec_b[-76:], k_target[-76:]))
        self.assertTrue(torch.equal(v_rec_b[-76:], v_target[-76:]))

        # Verify Mamba conv_state is preserved bitwise and top-drift ssm_state rows are exact!
        ssm_rec_s, conv_rec_s = store_single.reconstruct_block(
            mamba_layer, "h_mamba_tgt"
        )
        ssm_rec_b, conv_rec_b = store_batch.reconstruct_blocks_batch(
            mamba_layer, [spec_mamba, spec_mamba]
        )[0]
        assert conv_rec_s is not None and conv_rec_b is not None
        self.assertTrue(torch.equal(conv_rec_s, conv_target))
        self.assertTrue(torch.equal(conv_rec_b, conv_target))
        self.assertTrue(
            torch.equal(ssm_rec_s.view(-1, 64)[:16], ssm_target.view(-1, 64)[:16])
        )
        self.assertTrue(
            torch.equal(ssm_rec_b.view(-1, 64)[:16], ssm_target.view(-1, 64)[:16])
        )

    def test_configurable_multi_cohort_threshold(self) -> None:
        """Verifies configurable multi_cohort_threshold (default=32 vs custom=4)."""
        torch.manual_seed(2026)
        b_size = 16
        store_default = DeltaBlockStore(block_size=b_size)
        store_custom = DeltaBlockStore(block_size=b_size, multi_cohort_threshold=4)
        self.assertEqual(store_default.multi_cohort_threshold, 32)
        self.assertEqual(store_custom.multi_cohort_threshold, 4)

        layer = "model.layers.3.self_attn.attn"
        for s in (store_default, store_custom):
            s.register_layer_policy(
                layer,
                LayerCompressionPolicy(
                    layer_idx=3,
                    rope_theta=10000.0,
                    tau_zero_bit=0.05,
                    tau_int2=0.12,
                    tau_int4=0.85,
                ),
            )

        # Seed 4 distinct pos-0 cohort headers + 1 shared pos-16 haystack block
        shared_tokens = tuple(range(500, 500 + b_size))
        k_shared_base = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
        v_shared_base = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)

        for c_idx in range(4):
            pos0_tokens = tuple(range(100 + c_idx * 20, 100 + c_idx * 20 + b_size))
            k_pos0 = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
            v_pos0 = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
            for s in (store_default, store_custom):
                s.store_block(layer, f"h_pos0_{c_idx}", pos0_tokens, 0, k_pos0, v_pos0)

        for s in (store_default, store_custom):
            s.store_block(
                layer, "h_shared_0", shared_tokens, b_size, k_shared_base, v_shared_base
            )

        # Store 2 sibling cases sharing `shared_tokens` at pos=16 with small context drift (~0.09)
        k_sib = k_shared_base + 0.08 * torch.randn_like(k_shared_base)
        v_sib = v_shared_base + 0.08 * torch.randn_like(v_shared_base)
        specs = [
            (
                DeltaBlockTransferSpec(
                    full_block_hash="h_shared_sib1",
                    token_ids=shared_tokens,
                    start_position=b_size,
                    physical_block_id=10,
                ),
                k_sib,
                v_sib,
            ),
            (
                DeltaBlockTransferSpec(
                    full_block_hash="h_shared_sib2",
                    token_ids=shared_tokens,
                    start_position=b_size,
                    physical_block_id=11,
                ),
                k_sib,
                v_sib,
            ),
        ]
        recs_default = store_default.store_blocks_batch(layer, specs)
        recs_custom = store_custom.store_blocks_batch(layer, specs)

        # With default threshold=32 (4 < 32), sibling blocks use quantized delta (INT2/INT4);
        # with custom threshold=4 (4 >= 4), is_multi_cohort promotes exact token matches to TIER0_ZERO_BIT!
        self.assertEqual(recs_custom[0].tier, CompressionTier.TIER0_ZERO_BIT)
        self.assertEqual(recs_custom[1].tier, CompressionTier.TIER0_ZERO_BIT)
        self.assertIn(
            recs_default[0].tier,
            (CompressionTier.TIER1_INT2_DELTA, CompressionTier.TIER2_INT4_DELTA),
        )
        self.assertGreater(
            store_custom.stats.compression_ratio,
            store_default.stats.compression_ratio,
        )

        # Now add 28 more pos-0 blocks that differ by only 2 tokens from pos0_0
        # (stored as TIER3_SPARSE_BF16_INT4_DELTA, like NIAH variant headers),
        # bringing total distinct pos-0 blocks in store_default to 32 (the default multi_cohort_threshold):
        base_pos0_tokens = list(range(100, 100 + b_size))
        for v_idx in range(1, 29):
            var_tokens = list(base_pos0_tokens)
            var_tokens[2] += v_idx * 1000
            var_tokens[5] += v_idx * 1000
            k_var = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
            v_var = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
            store_default.store_block(
                layer, f"h_pos0_var_{v_idx}", tuple(var_tokens), 0, k_var, v_var
            )

        self.assertEqual(len(store_default._pos0_blocks), 32)
        recs_default_after_32 = store_default.store_blocks_batch(
            layer,
            [
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_shared_sib3",
                        token_ids=shared_tokens,
                        start_position=b_size,
                        physical_block_id=12,
                    ),
                    k_sib,
                    v_sib,
                ),
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_shared_sib4",
                        token_ids=shared_tokens,
                        start_position=b_size,
                        physical_block_id=13,
                    ),
                    k_sib,
                    v_sib,
                ),
            ],
        )
        self.assertEqual(recs_default_after_32[0].tier, CompressionTier.TIER0_ZERO_BIT)
        self.assertEqual(recs_default_after_32[1].tier, CompressionTier.TIER0_ZERO_BIT)
        # Pre-threshold exact-token-match siblings are also retroactively promoted to TIER0_ZERO_BIT:
        self.assertEqual(
            store_default._records[(layer, "h_shared_sib1")].tier,
            CompressionTier.TIER0_ZERO_BIT,
        )
        self.assertEqual(
            store_default._records[(layer, "h_shared_sib2")].tier,
            CompressionTier.TIER0_ZERO_BIT,
        )

    def test_gqa_fp8_kv_delta_and_reconstruction(self) -> None:
        """Verifies GQA float8_e4m3fn K and V delta compression and reconstruction."""
        torch.manual_seed(2026)
        b_size = 16
        store = DeltaBlockStore(block_size=b_size, multi_cohort_threshold=32)
        layer = "model.layers.3.self_attn.attn"
        store.register_layer_policy(
            layer,
            LayerCompressionPolicy(
                layer_idx=3,
                rope_theta=10000.0,
                tau_zero_bit=0.01,
                tau_int2=0.15,
                tau_int4=0.85,
            ),
        )
        base_tokens = tuple(range(200, 200 + b_size))
        k_base_bf16 = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
        v_base_bf16 = torch.randn(b_size, 2, 64, dtype=torch.bfloat16)
        k_base_fp8 = k_base_bf16.to(torch.float8_e4m3fn)
        v_base_fp8 = v_base_bf16.to(torch.float8_e4m3fn)
        store.store_block(layer, "h_fp8_base", base_tokens, 0, k_base_fp8, v_base_fp8)

        # Shifted exact-token match (delta_pos=16) with small drift -> TIER1/TIER2 quantized delta
        k_rot = apply_relative_rope_rotation(
            k_base_fp8.to(torch.bfloat16),
            delta_pos=b_size,
            rope_theta=10000.0,
            rotary_dim=64,
        )
        k_sib_fp8 = (k_rot + 0.05 * torch.randn_like(k_rot)).clamp(-448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        v_sib_fp8 = (
            v_base_fp8.to(torch.bfloat16) + 0.05 * torch.randn_like(v_base_bf16)
        ).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # Partial token match (1 edited token at delta_pos=16) -> TIER3_SPARSE_BF16_INT4_DELTA
        part_tokens = list(base_tokens)
        part_tokens[4] = 9999
        k_part_fp8 = k_sib_fp8.clone()
        v_part_fp8 = v_sib_fp8.clone()

        recs = store.store_blocks_batch(
            layer,
            [
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_fp8_sib",
                        token_ids=base_tokens,
                        start_position=b_size,
                        physical_block_id=1,
                    ),
                    k_sib_fp8,
                    v_sib_fp8,
                ),
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_fp8_part",
                        token_ids=tuple(part_tokens),
                        start_position=b_size,
                        physical_block_id=2,
                    ),
                    k_part_fp8,
                    v_part_fp8,
                ),
            ],
        )
        self.assertIn(
            recs[0].tier,
            (CompressionTier.TIER1_INT2_DELTA, CompressionTier.TIER2_INT4_DELTA),
        )
        self.assertEqual(recs[1].tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)

        recon_batch = store.reconstruct_blocks_batch(
            layer,
            [
                DeltaBlockTransferSpec(
                    full_block_hash="h_fp8_sib",
                    token_ids=base_tokens,
                    start_position=b_size,
                    physical_block_id=1,
                ),
                DeltaBlockTransferSpec(
                    full_block_hash="h_fp8_part",
                    token_ids=tuple(part_tokens),
                    start_position=b_size,
                    physical_block_id=2,
                ),
            ],
        )
        for k_r, v_r in recon_batch:
            self.assertEqual(k_r.dtype, torch.float8_e4m3fn)
            self.assertIsNotNone(v_r)
            assert v_r is not None
            self.assertEqual(v_r.dtype, torch.float8_e4m3fn)
            self.assertFalse(torch.isnan(k_r.to(torch.float32)).any().item())
            self.assertFalse(torch.isnan(v_r.to(torch.float32)).any().item())

    def test_5d_hnd_layout_and_mixed_tier3_quant_delta(self) -> None:
        torch.manual_seed(77)
        b_size = 64
        num_heads = 2
        head_dim = 64

        # 1. Verify 5D LBHNC ([2, num_blocks, num_kv_heads=2, block_size=64, head_dim=64])
        # and BLHNC ([num_blocks, 2, num_kv_heads=2, block_size=64, head_dim=64]) round-trip
        kv_lbhnc = torch.randn(2, 4, num_heads, b_size, head_dim, dtype=torch.bfloat16)
        k_ext, v_ext = DeltaCompressionConnector._extract_block_kv(
            kv_lbhnc, 1, is_mla=False, block_size=b_size
        )
        self.assertEqual(tuple(k_ext.shape), (b_size, num_heads, head_dim))
        assert v_ext is not None
        self.assertEqual(tuple(v_ext.shape), (b_size, num_heads, head_dim))

        kv_lbhnc_out = torch.zeros_like(kv_lbhnc)
        DeltaCompressionConnector._inject_block_kv(
            kv_lbhnc_out, 1, k_ext, v_ext, is_mla=False, block_size=b_size
        )
        self.assertTrue(torch.equal(kv_lbhnc[:, 1], kv_lbhnc_out[:, 1]))

        # 2. Verify mixed TIER3_SPARSE_BF16_INT4_DELTA records (with and without quant_delta_k)
        # in the same reconstruct_blocks_batch call
        b_small = 16
        store = DeltaBlockStore(
            block_size=b_small,
            enable_cross_turn_reuse=True,
        )
        layer = "model.layers.3.self_attn.attn"
        store.register_layer_policy(
            layer,
            LayerCompressionPolicy(
                layer_idx=3,
                rotary_dim=0,
                tau_zero_bit=0.01,
                tau_int2=0.12,
                tau_int4=0.85,
                max_sparse_exceptions=4,
            ),
        )
        base_tokens = tuple(range(500, 500 + b_small))
        k_base = torch.randn(b_small, num_heads, head_dim, dtype=torch.bfloat16)
        v_base = torch.randn(b_small, num_heads, head_dim, dtype=torch.bfloat16)
        store.store_block(layer, "h_t3_base", base_tokens, 0, k_base, v_base)

        # Record A: partial token match (quant_delta_k is None)
        tok_a = list(base_tokens)
        tok_a[2] = 99999
        k_a = k_base.clone()
        k_a[2] += 5.0
        v_a = v_base.clone()
        v_a[2] -= 5.0

        # Record B: exact token match at b_size=16 with outlier row (quant_delta_k is not None)
        k_b = k_base + 0.02 * torch.randn_like(k_base)
        k_b[5] += 20.0
        v_b = v_base + 0.02 * torch.randn_like(v_base)
        v_b[5] -= 20.0

        recs = store.store_blocks_batch(
            layer,
            [
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_t3_part_a",
                        token_ids=tuple(tok_a),
                        start_position=0,
                        physical_block_id=1,
                    ),
                    k_a,
                    v_a,
                ),
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_t3_exact_b",
                        token_ids=base_tokens,
                        start_position=b_small,
                        physical_block_id=2,
                    ),
                    k_b,
                    v_b,
                ),
            ],
        )
        self.assertEqual(recs[0].tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        self.assertEqual(recs[1].tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA)
        self.assertIsNotNone(recs[1].quant_delta_k)
        # Simulate pos==0 multi-cohort sparse record (quant_delta_k=None) mixed with
        # full-residual TIER3_SPARSE_BF16_INT4_DELTA record (quant_delta_k!=None)
        recs[0].quant_delta_k = None
        recs[0].quant_delta_v = None

        recon_mixed = store.reconstruct_blocks_batch(
            layer,
            [
                DeltaBlockTransferSpec(
                    full_block_hash="h_t3_part_a",
                    token_ids=tuple(tok_a),
                    start_position=0,
                    physical_block_id=1,
                ),
                DeltaBlockTransferSpec(
                    full_block_hash="h_t3_exact_b",
                    token_ids=base_tokens,
                    start_position=b_small,
                    physical_block_id=2,
                ),
            ],
        )
        self.assertTrue(torch.allclose(recon_mixed[0][0], k_a, atol=1e-2))
        self.assertTrue(torch.allclose(recon_mixed[1][0], k_b, atol=0.1))

    def test_gqa_uint8_fp8_kv_extract_store_reconstruct_inject(self):
        """Verify 4D and 5D GQA `torch.uint8`-backed `float8_e4m3fn` KV caches
        (`--kv-cache-dtype fp8`) view as `float8_e4m3fn` on extraction and view back
        as `uint8` on injection without raw `uint8` (`0..255`) numeric corruption.
        """
        torch.manual_seed(2026)
        block_size = 16
        num_kv_heads = 2
        head_dim = 64

        for layout in ("5d_lbnhc", "5d_blhnc", "4d"):
            if layout == "5d_lbnhc":
                kv_layer = (
                    torch.randn(
                        2, 4, block_size, num_kv_heads, head_dim, dtype=torch.bfloat16
                    )
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                )
            elif layout == "5d_blhnc":
                kv_layer = (
                    torch.randn(
                        4, 2, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16
                    )
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                )
            else:
                kv_layer = (
                    torch.randn(
                        4, num_kv_heads, block_size, 2 * head_dim, dtype=torch.bfloat16
                    )
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                )

            self.assertFalse(
                DeltaCompressionConnector._is_mla_kv_tensor(kv_layer)
            )
            k_ext, v_ext = DeltaCompressionConnector._extract_block_kv(
                kv_layer,
                physical_block_id=1,
                is_mla=False,
                block_size=block_size,
            )
            self.assertEqual(k_ext.dtype, torch.float8_e4m3fn)
            self.assertIsNotNone(v_ext)
            assert v_ext is not None
            self.assertEqual(v_ext.dtype, torch.float8_e4m3fn)
            self.assertEqual(
                tuple(k_ext.shape), (block_size, num_kv_heads, head_dim)
            )

            store = DeltaBlockStore(
                block_size=block_size,
                enable_cross_turn_reuse=True,
            )
            layer_name = f"model.layers.0.self_attn.{layout}"
            store.register_layer_policy(
                layer_name,
                LayerCompressionPolicy(
                    layer_idx=0,
                    rotary_dim=32,
                    rope_theta=10000.0,
                    is_neox_style=True,
                ),
            )
            tokens = tuple(range(100, 100 + block_size))
            store.store_blocks_batch(
                layer_name,
                [
                    (
                        DeltaBlockTransferSpec(
                            full_block_hash="h_base",
                            token_ids=tokens,
                            start_position=0,
                            physical_block_id=1,
                        ),
                        k_ext,
                        v_ext,
                    )
                ],
            )
            k_shifted = apply_relative_rope_rotation(
                k_ext.to(torch.bfloat16),
                delta_pos=block_size,
                rope_theta=10000.0,
                rotary_dim=32,
                is_neox_style=True,
            ).to(torch.float8_e4m3fn)
            recs = store.store_blocks_batch(
                layer_name,
                [
                    (
                        DeltaBlockTransferSpec(
                            full_block_hash="h_shift",
                            token_ids=tokens,
                            start_position=block_size,
                            physical_block_id=2,
                        ),
                        k_shifted,
                        v_ext,
                    )
                ],
            )
            self.assertEqual(recs[0].tier, CompressionTier.TIER0_ZERO_BIT)
            recon = store.reconstruct_blocks_batch(
                layer_name,
                [
                    DeltaBlockTransferSpec(
                        full_block_hash="h_shift",
                        token_ids=tokens,
                        start_position=block_size,
                        physical_block_id=2,
                    )
                ],
            )
            assert recon[0] is not None
            k_rec, v_rec = recon[0]
            self.assertEqual(k_rec.dtype, torch.float8_e4m3fn)
            assert v_rec is not None
            self.assertEqual(v_rec.dtype, torch.float8_e4m3fn)

            DeltaCompressionConnector._inject_block_kv(
                kv_layer,
                physical_block_id=3,
                k_block=k_rec,
                v_block=v_rec,
                is_mla=False,
                block_size=block_size,
            )
            k_re_ext, v_re_ext = DeltaCompressionConnector._extract_block_kv(
                kv_layer,
                physical_block_id=3,
                is_mla=False,
                block_size=block_size,
            )
            self.assertTrue(
                torch.allclose(
                    k_re_ext.to(torch.float32),
                    k_shifted.to(torch.float32),
                    atol=1e-2,
                )
            )
            assert v_re_ext is not None
            self.assertTrue(
                torch.equal(
                    v_re_ext.view(torch.uint8), v_ext.view(torch.uint8)
                )
            )

    def test_dsv32_fp8_ds_mla_656b_dequant_rope_requant(self):
        """Verify 656-byte DeepSeek-V3.2 / GLM-5.3 `fp8_ds_mla` KV cache format
        (`512`B `float8_e4m3fn` NoPE + `16`B `4 x float32` scales + `128`B `64 x bfloat16` RoPE)
        round-trips through `DeltaBlockStore` with exact NoPE preservation under `TIER0_ZERO_BIT`
        and accurate dynamic scale re-quantization under `TIER1`/`TIER2`.
        """
        torch.manual_seed(2027)
        block_size = 16
        nope_fp32 = torch.randn(block_size, 4, 128, dtype=torch.float32) * 3.5
        amax = nope_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        q_fp8 = (nope_fp32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        rope_bf16 = torch.randn(block_size, 64, dtype=torch.bfloat16)
        raw_base = torch.cat(
            [
                q_fp8.view(torch.uint8).reshape(block_size, 512),
                scale.to(torch.float32).view(torch.uint8).reshape(block_size, 16),
                rope_bf16.view(torch.uint8).reshape(block_size, 128),
            ],
            dim=-1,
        )
        self.assertEqual(tuple(raw_base.shape), (block_size, 656))

        rope_rot = apply_relative_rope_rotation(
            rope_bf16,
            delta_pos=block_size,
            rope_theta=10000.0,
            rotary_dim=64,
            is_neox_style=False,
        )
        raw_shifted = torch.cat(
            [
                raw_base[:, :528],
                rope_rot.contiguous().view(torch.uint8).reshape(block_size, 128),
            ],
            dim=-1,
        )

        store = DeltaBlockStore(
            block_size=block_size,
            enable_cross_turn_reuse=True,
        )
        layer_name = "model.layers.1.self_attn.mla_attn"
        store.register_layer_policy(
            layer_name,
            LayerCompressionPolicy(
                layer_idx=1,
                rotary_dim=64,
                rope_theta=10000.0,
                is_neox_style=False,
                rope_at_tail=True,
            ),
        )
        tokens = tuple(range(200, 200 + block_size))
        store.store_blocks_batch(
            layer_name,
            [
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_ds_base",
                        token_ids=tokens,
                        start_position=0,
                        physical_block_id=1,
                    ),
                    raw_base,
                    None,
                )
            ],
        )
        recs = store.store_blocks_batch(
            layer_name,
            [
                (
                    DeltaBlockTransferSpec(
                        full_block_hash="h_ds_shift",
                        token_ids=tokens,
                        start_position=block_size,
                        physical_block_id=2,
                    ),
                    raw_shifted,
                    None,
                )
            ],
        )
        self.assertEqual(recs[0].tier, CompressionTier.TIER0_ZERO_BIT)
        recon = store.reconstruct_blocks_batch(
            layer_name,
            [
                DeltaBlockTransferSpec(
                    full_block_hash="h_ds_shift",
                    token_ids=tokens,
                    start_position=block_size,
                    physical_block_id=2,
                )
            ],
        )
        assert recon[0] is not None
        k_rec, _ = recon[0]
        self.assertTrue(torch.equal(k_rec[:, :528], raw_base[:, :528]))
        rec_rope = k_rec[:, 528:656].contiguous().view(torch.bfloat16)
        self.assertTrue(torch.allclose(rec_rope, rope_rot, atol=1e-2))

    def test_dsv41_planar_indexer_and_compress_ratio_shifted_stitched_hybrid_tiers(
        self,
    ):
        """Verify DeepSeek-V4.1-Flash planar indexer.k_cache ([128, 132]),
        compress_ratio=2 MLA ([64, 288]), compress_ratio=1 MLA ([128, 288]),
        and swa_cache ([128, 528]) support single-block shifted Block 0
        (TIER3_SPARSE_BF16_INT4_DELTA with quant_delta_k=None) and 2-block
        stitched Blocks 1..K (TIER0_ZERO_BIT).
        """
        torch.manual_seed(2028)
        block_size = 128
        shift_k = 6
        tok_b0 = tuple(range(1000, 1000 + block_size))
        tok_b1 = tuple(range(2000, 2000 + block_size))
        prefix_toks = tuple(range(9000, 9000 + shift_k))
        tok_t2_b0 = prefix_toks + tok_b0[: block_size - shift_k]
        tok_t2_b1 = (
            tok_b0[block_size - shift_k :] + tok_b1[: block_size - shift_k]
        )

        store = DeltaBlockStore(
            block_size=block_size,
            enable_cross_turn_reuse=True,
        )
        store.register_layer_policy(
            "layer.0.swa",
            LayerCompressionPolicy(
                layer_idx=0,
                rotary_dim=64,
                rope_theta=10000.0,
                is_neox_style=False,
                rope_at_tail=True,
                is_sliding_window=True,
            ),
        )
        store.register_layer_policy(
            "layer.2.mla_cr2",
            LayerCompressionPolicy(
                layer_idx=2,
                rotary_dim=64,
                rope_theta=160000.0,
                is_neox_style=False,
                rope_at_tail=True,
            ),
        )
        store.register_layer_policy(
            "layer.3.mla_cr1",
            LayerCompressionPolicy(
                layer_idx=3,
                rotary_dim=64,
                rope_theta=160000.0,
                is_neox_style=False,
                rope_at_tail=True,
            ),
        )
        store.register_layer_policy(
            "layer.3.indexer",
            LayerCompressionPolicy(
                layer_idx=3,
                rotary_dim=0,
            ),
        )

        def make_swa(n: int) -> torch.Tensor:
            bf = torch.randn(n, 1, 512, dtype=torch.bfloat16)
            return _requant_dsv41_kv(bf, torch.zeros(n, 528, dtype=torch.uint8))

        def make_mla(n: int) -> torch.Tensor:
            bf = torch.randn(n, 1, 512, dtype=torch.bfloat16)
            return _requant_dsv41_kv(bf, torch.zeros(n, 288, dtype=torch.uint8))

        def make_indexer(n: int) -> torch.Tensor:
            w = (
                torch.randn(n, 128, dtype=torch.bfloat16)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            s = torch.randint(120, 132, (n, 4), dtype=torch.uint8)
            return torch.cat([w.reshape(-1), s.reshape(-1)], dim=0).reshape(n, 132)

        shapes = {
            "layer.0.swa": (make_swa(128), make_swa(128), make_swa(128), make_swa(128)),
            "layer.2.mla_cr2": (make_mla(64), make_mla(64), make_mla(64), make_mla(64)),
            "layer.3.mla_cr1": (
                make_mla(128),
                make_mla(128),
                make_mla(128),
                make_mla(128),
            ),
            "layer.3.indexer": (
                make_indexer(128),
                make_indexer(128),
                make_indexer(128),
                make_indexer(128),
            ),
        }

        for lname, (kb0, kb1, kt2_0, kt2_1) in shapes.items():
            recs_t1 = store.store_blocks_batch(
                lname,
                [
                    (
                        DeltaBlockTransferSpec(
                            full_block_hash="h_t1_b0",
                            token_ids=tok_b0,
                            start_position=0,
                            physical_block_id=1,
                        ),
                        kb0,
                        None,
                    ),
                    (
                        DeltaBlockTransferSpec(
                            full_block_hash="h_t1_b1",
                            token_ids=tok_b1,
                            start_position=block_size,
                            physical_block_id=2,
                        ),
                        kb1,
                        None,
                    ),
                ],
            )
            self.assertEqual(recs_t1[0].tier, CompressionTier.UNCOMPRESSED_BASE)
            self.assertEqual(recs_t1[1].tier, CompressionTier.UNCOMPRESSED_BASE)

            recs_t2 = store.store_blocks_batch(
                lname,
                [
                    (
                        DeltaBlockTransferSpec(
                            full_block_hash="h_t2_b0",
                            token_ids=tok_t2_b0,
                            start_position=0,
                            physical_block_id=3,
                        ),
                        kt2_0,
                        None,
                    ),
                    (
                        DeltaBlockTransferSpec(
                            full_block_hash="h_t2_b1",
                            token_ids=tok_t2_b1,
                            start_position=block_size,
                            physical_block_id=4,
                        ),
                        kt2_1,
                        None,
                    ),
                ],
            )
            self.assertEqual(
                recs_t2[0].tier, CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA
            )
            self.assertIsNone(recs_t2[0].quant_delta_k)
            self.assertEqual(recs_t2[0].stitch_offset, -shift_k)
            self.assertEqual(recs_t2[1].tier, CompressionTier.TIER0_ZERO_BIT)
            self.assertEqual(recs_t2[1].stitch_offset, block_size - shift_k)

            recons = store.reconstruct_blocks_batch(
                lname,
                [
                    DeltaBlockTransferSpec(
                        full_block_hash="h_t2_b0",
                        token_ids=tok_t2_b0,
                        start_position=0,
                        physical_block_id=3,
                    ),
                    DeltaBlockTransferSpec(
                        full_block_hash="h_t2_b1",
                        token_ids=tok_t2_b1,
                        start_position=block_size,
                        physical_block_id=4,
                    ),
                ],
            )
            k_rec0, _ = recons[0]
            k_rec1, _ = recons[1]
            self.assertEqual(tuple(k_rec0.shape), tuple(kt2_0.shape))
            self.assertEqual(tuple(k_rec1.shape), tuple(kt2_1.shape))
            sparse_end = shift_k + 3
            if lname == "layer.3.indexer":
                # Planar indexer Block 0: first 128*128 bytes are FP8 weights, last 128*4 bytes are UE8M0 scales
                w_rec0 = k_rec0.reshape(-1)[: 128 * 128].reshape(128, 128)
                s_rec0 = k_rec0.reshape(-1)[128 * 128 :].reshape(128, 4)
                w_t2_0 = kt2_0.reshape(-1)[: 128 * 128].reshape(128, 128)
                s_t2_0 = kt2_0.reshape(-1)[128 * 128 :].reshape(128, 4)
                w_b0 = kb0.reshape(-1)[: 128 * 128].reshape(128, 128)
                s_b0 = kb0.reshape(-1)[128 * 128 :].reshape(128, 4)
                w_b1 = kb1.reshape(-1)[: 128 * 128].reshape(128, 128)
                s_b1 = kb1.reshape(-1)[128 * 128 :].reshape(128, 4)
                self.assertTrue(
                    torch.equal(w_rec0[:sparse_end], w_t2_0[:sparse_end])
                )
                self.assertTrue(
                    torch.equal(s_rec0[:sparse_end], s_t2_0[:sparse_end])
                )
                self.assertTrue(
                    torch.equal(
                        w_rec0[sparse_end:], w_b0[3 : 128 - shift_k]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        s_rec0[sparse_end:], s_b0[3 : 128 - shift_k]
                    )
                )
                # Planar indexer Block 1 stitched weights & scales must match kb0[s:] + kb1[:s]
                s_off = block_size - shift_k
                w_rec1 = k_rec1.reshape(-1)[: 128 * 128].reshape(128, 128)
                s_rec1 = k_rec1.reshape(-1)[128 * 128 :].reshape(128, 4)
                self.assertTrue(
                    torch.equal(
                        w_rec1,
                        torch.cat([w_b0[s_off:], w_b1[:s_off]], dim=0),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        s_rec1,
                        torch.cat([s_b0[s_off:], s_b1[:s_off]], dim=0),
                    )
                )
            elif lname == "layer.0.swa":
                # NoPE FP8 bytes (0..448) and NoPE UE8M0 scale bytes (512..526) on shifted/stitched
                # tokens must be 100% bit-exact with base!
                self.assertTrue(
                    torch.equal(
                        k_rec0[sparse_end:, :448], kb0[3 : 128 - shift_k, :448]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        k_rec0[sparse_end:, 512:526], kb0[3 : 128 - shift_k, 512:526]
                    )
                )


if __name__ == "__main__":
    unittest.main()

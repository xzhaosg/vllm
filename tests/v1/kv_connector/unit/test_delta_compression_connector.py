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
    CompressionTier,
    DeltaBlockStore,
    DeltaCompressionConnector,
    LayerCompressionPolicy,
    apply_relative_rope_rotation,
    compute_full_prefix_block_hash,
    dequantize_residual_packed,
    quantize_residual_packed,
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

        # Verify that P_B's shared Block 1 in GDN was delta-compressed (TIER2_INT4_DELTA)
        # and FullAttention was delta-compressed (TIER1_INT2_DELTA)
        self.assertGreaterEqual(
            worker_conn.store.stats.tier_counts[CompressionTier.TIER1_INT2_DELTA]
            + worker_conn.store.stats.tier_counts[CompressionTier.TIER2_INT4_DELTA],
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
            layer_types = ["sliding_attention", "full_attention"]
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
            layer_names=["model.language_model.layers.0.self_attn.attn"],
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
        self.assertTrue(pol_sw.is_sliding_window)
        self.assertEqual(pol_sw.rope_theta, 10000.0)
        self.assertIsNone(pol_sw.rotary_dim)
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
        self.assertEqual(matched_toks, 48)



if __name__ == "__main__":
    unittest.main()

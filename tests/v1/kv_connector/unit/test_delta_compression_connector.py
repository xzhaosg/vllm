# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vLLM V1 DeltaCompressionConnector (Path A Phase 1)."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

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


if __name__ == "__main__":
    unittest.main()

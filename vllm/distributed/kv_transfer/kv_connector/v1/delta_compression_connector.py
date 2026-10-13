# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeltaCompressionConnector: Layer-0 zero-bit reuse and Layer-n delta compression for vLLM V1."""

from dataclasses import dataclass, field
import enum
import math
import re
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class CompressionTier(enum.IntEnum):
    """Compression tiers for KV cache blocks across layers."""

    # Full-precision reference block (16 bits/elem) anchored in the base index.
    UNCOMPRESSED_BASE = 0
    # Tier 0: 0 bits/elem tensor payload (Layer 0 exact token match + R(delta_p),
    # or LOCAL_SLIDING layer with d_div >= sliding_window_dedup_threshold).
    TIER0_ZERO_BIT = 1
    # Tier 1: 2 bits/elem packed INT2 residual + R(delta_p) (8x compression).
    TIER1_INT2_DELTA = 2
    # Tier 2: 4 bits/elem packed INT4 residual + R(delta_p) (4x compression).
    TIER2_INT4_DELTA = 3
    # Tier 3: Sparse BF16 edited token rows + 4 bits/elem packed INT4 residual on
    # unchanged suffix rows for intra-block edits (Regime R2).
    TIER3_SPARSE_BF16_INT4_DELTA = 4


def compute_token_block_hash(token_ids: tuple[int, ...] | list[int]) -> str:
    """Compute a position-independent hash of a block's token IDs."""
    token_tuple = tuple(int(t) for t in token_ids)
    raw_bytes = torch.tensor(token_tuple, dtype=torch.int64).numpy().tobytes()
    return safe_hash(raw_bytes, usedforsecurity=False).hexdigest()


def compute_full_prefix_block_hash(
    parent_hash: str, token_ids: tuple[int, ...] | list[int]
) -> str:
    """Compute a prefix-dependent hash chaining parent_hash and block token IDs."""
    token_tuple = tuple(int(t) for t in token_ids)
    raw_bytes = (
        parent_hash.encode("utf-8")
        + torch.tensor(token_tuple, dtype=torch.int64).numpy().tobytes()
    )
    return safe_hash(raw_bytes, usedforsecurity=False).hexdigest()


def apply_relative_rope_rotation(
    k_tensor: torch.Tensor,
    delta_pos: int | torch.Tensor,
    rope_theta: float = 10000.0,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """Apply exact relative RoPE phase rotation R(delta_pos) to post-RoPE keys.

    Given K_ref = R(p_ref) * K_raw, computes:
        R(delta_pos) * K_ref = R(p_ref + delta_pos) * K_raw = R(p_target) * K_raw

    Args:
        k_tensor: Post-RoPE key tensor of shape [num_tokens, num_kv_heads, head_dim]
            or [num_tokens, head_dim] (for MLA).
        delta_pos: Scalar integer position shift (p_target - p_ref) or 1D tensor of
            shape [num_tokens].
        rope_theta: Base RoPE frequency theta (e.g., 10000.0 or 1000000.0).
        rotary_dim: Number of rotary dimensions (defaults to full head_dim; set to 0
            for NoPE layers or < head_dim for partial RoPE).
        is_neox_style: True for half-split NeoX RoPE (Gemma, Llama, Qwen); False for
            interleaved RoPE (DeepSeek, GLM).

    Returns:
        Rotated key tensor of the same shape and dtype as k_tensor.
    """
    if isinstance(delta_pos, int) and delta_pos == 0:
        return k_tensor.clone()
    if isinstance(delta_pos, torch.Tensor) and torch.all(delta_pos == 0):
        return k_tensor.clone()

    orig_dtype = k_tensor.dtype
    head_dim = k_tensor.shape[-1]
    r_dim = head_dim if rotary_dim is None else min(rotary_dim, head_dim)
    if r_dim <= 0:
        return k_tensor.clone()

    k_fp32 = k_tensor.to(torch.float32)
    k_rot = k_fp32[..., :r_dim]
    k_pass = k_fp32[..., r_dim:]

    half_dim = r_dim // 2
    freq_seq = torch.arange(0, half_dim, dtype=torch.float32, device=k_tensor.device)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / float(half_dim)))

    if isinstance(delta_pos, int):
        pos_tensor = torch.tensor(
            [float(delta_pos)], dtype=torch.float32, device=k_tensor.device
        )
    else:
        pos_tensor = delta_pos.to(dtype=torch.float32, device=k_tensor.device)

    # Broadcast angles to [num_tokens, 1, half_dim] or [num_tokens, half_dim]
    angles = torch.outer(pos_tensor.reshape(-1), inv_freq)
    if k_rot.ndim == 3:
        angles = angles.unsqueeze(1)

    cos_val = torch.cos(angles)
    sin_val = torch.sin(angles)

    if is_neox_style:
        x1 = k_rot[..., :half_dim]
        x2 = k_rot[..., half_dim : 2 * half_dim]
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val
        k_rot_out = torch.cat([out1, out2], dim=-1)
    else:
        x1 = k_rot[..., 0::2]
        x2 = k_rot[..., 1::2]
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val
        k_rot_out = torch.stack([out1, out2], dim=-1).flatten(-2)

    if r_dim < head_dim:
        k_out = torch.cat([k_rot_out, k_pass], dim=-1)
    else:
        k_out = k_rot_out
    return k_out.to(orig_dtype)


@dataclass
class PackedQuantizedTensor:
    """Bit-packed group-symmetric quantized residual tensor (INT4 or INT2)."""

    packed_data: torch.Tensor  # uint8 tensor of packed 4-bit or 2-bit codes
    scales: torch.Tensor  # float16 per-head/per-group scales
    orig_shape: tuple[int, ...]
    orig_dtype: torch.dtype
    bits: int  # 2 or 4

    @property
    def nbytes(self) -> int:
        return (
            self.packed_data.numel() * self.packed_data.element_size()
            + self.scales.numel() * self.scales.element_size()
        )


def quantize_residual_packed(
    residual: torch.Tensor, bits: int = 4
) -> PackedQuantizedTensor:
    """Quantize a residual tensor [block_size, num_heads, head_dim] into packed uint8.

    Uses per-head symmetric quantization:
      - INT4: range [-7, +7] mapped to unsigned [1, 15] (offset +8), 2 elems per byte.
      - INT2: range [-1, +1] or 4-level symmetric [-1.5, -0.5, +0.5, +1.5] mapped to
        unsigned [0, 3], 4 elems per byte.
    """
    if bits not in (2, 4):
        raise ValueError(f"Unsupported bit-width: {bits}. Expected 2 or 4.")

    orig_shape = tuple(residual.shape)
    orig_dtype = residual.dtype
    res_fp32 = residual.to(torch.float32)

    # Reshape to [num_groups, group_size] where group_size = head_dim
    head_dim = orig_shape[-1]
    flat_groups = res_fp32.reshape(-1, head_dim)
    max_abs = flat_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    if bits == 4:
        # 16-level symmetric mid-riser quantizer: levels {-7.5, -6.5, ..., +6.5, +7.5} * scale
        scales_fp32 = max_abs / 7.5
        normalized = flat_groups / scales_fp32
        q_unsigned = (
            torch.floor(normalized + 8.0).clamp(0.0, 15.0).to(torch.uint8).reshape(-1)
        )
        if q_unsigned.numel() % 2 != 0:
            q_unsigned = torch.cat(
                [q_unsigned, torch.zeros(1, dtype=torch.uint8, device=q_unsigned.device)]
            )
        low_nibble = q_unsigned[0::2] & 0x0F
        high_nibble = (q_unsigned[1::2] & 0x0F) << 4
        packed = low_nibble | high_nibble
    else:
        # 4-level Lloyd-Max calibrated symmetric mid-riser quantizer:
        # levels {-1.5, -0.5, +0.5, +1.5} * scale with optimal clipping for group_size=128
        scales_fp32 = max_abs / 2.15
        normalized = flat_groups / scales_fp32
        # Map to {0, 1, 2, 3} corresponding to {-1.5, -0.5, +0.5, +1.5}
        q_unsigned = (
            torch.floor(normalized + 2.0).clamp(0.0, 3.0).to(torch.uint8).reshape(-1)
        )
        rem = q_unsigned.numel() % 4
        if rem != 0:
            q_unsigned = torch.cat(
                [
                    q_unsigned,
                    torch.zeros(4 - rem, dtype=torch.uint8, device=q_unsigned.device),
                ]
            )
        packed = (
            (q_unsigned[0::4] & 0x03)
            | ((q_unsigned[1::4] & 0x03) << 2)
            | ((q_unsigned[2::4] & 0x03) << 4)
            | ((q_unsigned[3::4] & 0x03) << 6)
        )

    return PackedQuantizedTensor(
        packed_data=packed,
        scales=scales_fp32.to(torch.float16),
        orig_shape=orig_shape,
        orig_dtype=orig_dtype,
        bits=bits,
    )


def dequantize_residual_packed(packed_tensor: PackedQuantizedTensor) -> torch.Tensor:
    """Dequantize a PackedQuantizedTensor back to its original shape and dtype."""
    total_elems = math.prod(packed_tensor.orig_shape)
    head_dim = packed_tensor.orig_shape[-1]
    scales_fp32 = packed_tensor.scales.to(torch.float32)
    packed = packed_tensor.packed_data

    if packed_tensor.bits == 4:
        low_nibble = packed & 0x0F
        high_nibble = (packed >> 4) & 0x0F
        unpacked = torch.stack([low_nibble, high_nibble], dim=-1).reshape(-1)[
            :total_elems
        ]
        levels = unpacked.to(torch.float32) - 7.5
        flat_groups = levels.reshape(-1, head_dim)
        recon = flat_groups * scales_fp32
    elif packed_tensor.bits == 2:
        c0 = packed & 0x03
        c1 = (packed >> 2) & 0x03
        c2 = (packed >> 4) & 0x03
        c3 = (packed >> 6) & 0x03
        unpacked = torch.stack([c0, c1, c2, c3], dim=-1).reshape(-1)[:total_elems]
        # Map {0, 1, 2, 3} -> {-1.5, -0.5, +0.5, +1.5}
        levels = unpacked.to(torch.float32) - 1.5
        flat_groups = levels.reshape(-1, head_dim)
        recon = flat_groups * scales_fp32
    else:
        raise ValueError(f"Unsupported bit-width: {packed_tensor.bits}")

    return recon.reshape(packed_tensor.orig_shape).to(packed_tensor.orig_dtype)


@dataclass
class CompressedKVBlockRecord:
    """Stored representation of a single KV cache block in a specific layer."""

    tier: CompressionTier
    full_block_hash: str
    token_block_hash: str
    token_ids: tuple[int, ...]
    start_position: int
    layer_idx: int
    is_mla: bool = False

    # For UNCOMPRESSED_BASE:
    raw_k: torch.Tensor | None = None
    raw_v: torch.Tensor | None = None

    # For TIER0 / TIER1 / TIER2 / TIER3:
    # Single-hop reference pointer to an UNCOMPRESSED_BASE block's full_block_hash
    ref_base_hash: str | None = None
    delta_pos: int = 0

    # For TIER1 (INT2) and TIER2 (INT4) and TIER3 (Sparse + INT4):
    quant_delta_k: PackedQuantizedTensor | None = None
    quant_delta_v: PackedQuantizedTensor | None = None

    # For TIER3 (Sparse BF16 exception rows for intra-block token edits):
    sparse_token_indices: torch.Tensor | None = None  # uint8 [num_edited_rows]
    sparse_k_bf16: torch.Tensor | None = None  # [num_edited_rows, ...]
    sparse_v_bf16: torch.Tensor | None = None  # [num_edited_rows, ...]

    @property
    def nbytes(self) -> int:
        """Physical byte footprint of this block record's tensor payload."""
        total = 0
        if self.raw_k is not None:
            total += self.raw_k.numel() * self.raw_k.element_size()
        if self.raw_v is not None:
            total += self.raw_v.numel() * self.raw_v.element_size()
        if self.quant_delta_k is not None:
            total += self.quant_delta_k.nbytes
        if self.quant_delta_v is not None:
            total += self.quant_delta_v.nbytes
        if self.sparse_token_indices is not None:
            total += (
                self.sparse_token_indices.numel()
                * self.sparse_token_indices.element_size()
            )
        if self.sparse_k_bf16 is not None:
            total += self.sparse_k_bf16.numel() * self.sparse_k_bf16.element_size()
        if self.sparse_v_bf16 is not None:
            total += self.sparse_v_bf16.numel() * self.sparse_v_bf16.element_size()
        # Metadata header overhead (~16 bytes for tier, ref_id, delta_pos)
        if self.tier != CompressionTier.UNCOMPRESSED_BASE:
            total += 16
        return total


@dataclass
class DeltaCompressionStats:
    """Aggregated telemetry for delta compression across layers and tiers."""

    total_blocks_stored: int = 0
    total_blocks_loaded: int = 0
    raw_bytes_equivalent: int = 0
    compressed_bytes_stored: int = 0
    tier_counts: dict[CompressionTier, int] = field(
        default_factory=lambda: {tier: 0 for tier in CompressionTier}
    )

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes_stored <= 0:
            return 1.0
        return float(self.raw_bytes_equivalent) / float(self.compressed_bytes_stored)

    @property
    def memory_saving_pct(self) -> float:
        if self.raw_bytes_equivalent <= 0:
            return 0.0
        return 100.0 * (
            1.0 - float(self.compressed_bytes_stored) / float(self.raw_bytes_equivalent)
        )


@dataclass
class LayerCompressionPolicy:
    """Per-layer configuration for RoPE alignment and compression tier thresholds."""

    layer_idx: int
    is_sliding_window: bool = False
    sliding_window_size: int | None = None
    rope_theta: float = 10000.0
    rotary_dim: int | None = None
    is_neox_style: bool = True
    # Distance from prefix divergence (in tokens) after which LOCAL_SLIDING layers
    # use Tier 0 (0-bit deduplication).
    sliding_window_dedup_threshold: int = 64
    # Relative L2 residual norm thresholds for automatic tier selection:
    tau_zero_bit: float = 0.06
    tau_int2: float = 0.12
    tau_int4: float = 0.85
    max_sparse_exceptions: int = 4


class DeltaBlockStore:
    """In-memory / tiered store implementing 1-hop anchored 4-tier KV delta compression."""

    def __init__(
        self,
        block_size: int = 16,
        default_rope_theta: float = 10000.0,
        sliding_window_dedup_threshold: int = 64,
        tau_zero_bit: float = 0.06,
        tau_int2: float = 0.12,
        tau_int4: float = 0.85,
        max_sparse_exceptions: int = 4,
        device: str = "cpu",
    ):
        self.block_size = block_size
        self.default_rope_theta = default_rope_theta
        self.sliding_window_dedup_threshold = sliding_window_dedup_threshold
        self.tau_zero_bit = tau_zero_bit
        self.tau_int2 = tau_int2
        self.tau_int4 = tau_int4
        self.max_sparse_exceptions = max_sparse_exceptions
        self.device = device

        # Layer name -> LayerCompressionPolicy
        self.layer_policies: dict[str, LayerCompressionPolicy] = {}

        # Primary index: (layer_name, full_block_hash) -> CompressedKVBlockRecord
        self._records: dict[tuple[str, str], CompressedKVBlockRecord] = {}

        # Single-Hop Reference Index (ONLY stores UNCOMPRESSED_BASE blocks):
        # (layer_name, token_block_hash) -> CompressedKVBlockRecord
        self._base_by_token_hash: dict[tuple[str, str], CompressedKVBlockRecord] = {}

        # Secondary list of UNCOMPRESSED_BASE blocks per layer for R2 partial-edit matching:
        # layer_name -> list[CompressedKVBlockRecord]
        self._base_blocks_by_layer: dict[str, list[CompressedKVBlockRecord]] = {}
        # O(1) Pigeonhole sub-chunk index for R2 partial-edit matching (<= 4 edits in 16 tokens):
        # (layer_name, slice_idx, token_subtuple) -> list[CompressedKVBlockRecord]
        self._base_by_subchunk: dict[
            tuple[str, int, tuple[int, ...]], list[CompressedKVBlockRecord]
        ] = {}

        # Prefix hash registry at scheduler level: full_block_hash -> (token_ids, start_pos)
        self._known_prefix_blocks: dict[str, tuple[tuple[int, ...], int]] = {}
        # Token hash registry at scheduler level: token_block_hash -> full_block_hash
        self._known_token_blocks: dict[str, str] = {}

        self.stats = DeltaCompressionStats()

    def clear(self) -> None:
        """Clear all stored KV block records and reset telemetry while preserving layer policies."""
        self._records.clear()
        self._base_by_token_hash.clear()
        self._base_blocks_by_layer.clear()
        self._base_by_subchunk.clear()
        self._known_prefix_blocks.clear()
        self._known_token_blocks.clear()
        self.stats = DeltaCompressionStats()

    @staticmethod
    def _get_pigeonhole_slices(
        token_ids: tuple[int, ...], num_slices: int = 5
    ) -> list[tuple[int, tuple[int, ...]]]:
        """Partition token_ids into num_slices disjoint sub-tuples for O(1) candidate lookup."""
        n = len(token_ids)
        if n < num_slices:
            return [(0, token_ids)]
        out: list[tuple[int, tuple[int, ...]]] = []
        for s_idx in range(num_slices):
            start = (s_idx * n) // num_slices
            end = ((s_idx + 1) * n) // num_slices
            out.append((s_idx, token_ids[start:end]))
        return out

    def register_layer_policy(
        self, layer_name: str, policy: LayerCompressionPolicy
    ) -> None:
        self.layer_policies[layer_name] = policy

    def get_layer_policy(self, layer_name: str) -> LayerCompressionPolicy:
        if layer_name in self.layer_policies:
            return self.layer_policies[layer_name]
        # Infer layer_idx from layer_name (e.g. "model.layers.0.self_attn.attn")
        match = re.search(r"(?:layers|blocks|h)\.(\d+)", layer_name)
        layer_idx = int(match.group(1)) if match else 0
        policy = LayerCompressionPolicy(
            layer_idx=layer_idx,
            rope_theta=self.default_rope_theta,
            sliding_window_dedup_threshold=self.sliding_window_dedup_threshold,
            tau_zero_bit=self.tau_zero_bit,
            tau_int2=self.tau_int2,
            tau_int4=self.tau_int4,
            max_sparse_exceptions=self.max_sparse_exceptions,
        )
        self.layer_policies[layer_name] = policy
        return policy

    def has_full_block(self, full_block_hash: str) -> bool:
        return full_block_hash in self._known_prefix_blocks

    def has_token_block(self, token_block_hash: str) -> bool:
        return token_block_hash in self._known_token_blocks

    def _find_partial_match_base_block(
        self, layer_name: str, token_ids: tuple[int, ...]
    ) -> tuple[CompressedKVBlockRecord | None, list[int]]:
        """Find an UNCOMPRESSED_BASE block in the same layer differing in <= M tokens."""
        policy = self.get_layer_policy(layer_name)
        num_slices = policy.max_sparse_exceptions + 1
        seen_hashes: set[str] = set()
        candidates: list[CompressedKVBlockRecord] = []
        for s_idx, sub_tuple in self._get_pigeonhole_slices(token_ids, num_slices):
            bucket = self._base_by_subchunk.get((layer_name, s_idx, sub_tuple))
            if bucket:
                for rec in bucket[-8:]:
                    if rec.full_block_hash not in seen_hashes:
                        seen_hashes.add(rec.full_block_hash)
                        candidates.append(rec)

        best_base: CompressedKVBlockRecord | None = None
        best_diff_indices: list[int] = []

        for base_rec in candidates:
            if len(base_rec.token_ids) != len(token_ids):
                continue
            diff_idx = [
                i
                for i, (t_a, t_b) in enumerate(zip(token_ids, base_rec.token_ids))
                if t_a != t_b
            ]
            if 0 < len(diff_idx) <= policy.max_sparse_exceptions:
                if best_base is None or len(diff_idx) < len(best_diff_indices):
                    best_base = base_rec
                    best_diff_indices = diff_idx
                    if len(diff_idx) == 1:
                        break
        return best_base, best_diff_indices

    def store_block(
        self,
        layer_name: str,
        full_block_hash: str,
        token_ids: tuple[int, ...] | list[int],
        start_position: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor | None = None,
        divergence_distance: int | None = None,
    ) -> CompressedKVBlockRecord:
        """Compress and store a single KV cache block using the 4-tier policy.

        Args:
            layer_name: Name of the attention layer.
            full_block_hash: Prefix-dependent hash of the block.
            token_ids: Sequence of token IDs inside this block (length == block_size).
            start_position: Starting token position index of this block in the prompt.
            k_block: Key block tensor [block_size, num_kv_heads, head_dim] (or MLA
                latent [block_size, kv_lora_rank + qk_rope_head_dim]).
            v_block: Value block tensor [block_size, num_kv_heads, head_dim] (or None
                for MLA unified latent cache).
            divergence_distance: Number of contiguous matching tokens separating the
                start of this block from the most recent prefix divergence point.

        Returns:
            The stored CompressedKVBlockRecord.
        """
        key = (layer_name, full_block_hash)
        if key in self._records:
            return self._records[key]

        token_tuple = tuple(int(t) for t in token_ids)
        token_hash = compute_token_block_hash(token_tuple)
        policy = self.get_layer_policy(layer_name)
        is_mla = v_block is None

        k_cpu = k_block.detach().to(self.device)
        v_cpu = v_block.detach().to(self.device) if v_block is not None else None

        raw_bytes = k_cpu.numel() * k_cpu.element_size()
        if v_cpu is not None:
            raw_bytes += v_cpu.numel() * v_cpu.element_size()

        self._known_prefix_blocks[full_block_hash] = (token_tuple, start_position)
        self._known_token_blocks.setdefault(token_hash, full_block_hash)

        # 1. Check for an UNCOMPRESSED_BASE block with the exact same token_block_hash
        base_rec = self._base_by_token_hash.get((layer_name, token_hash))
        if base_rec is not None:
            delta_pos = start_position - base_rec.start_position

            # Case 1A: Layer 0 -> Always Tier 0 Zero-Bit Reuse!
            if policy.layer_idx == 0:
                record = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER0_ZERO_BIT,
                    full_block_hash=full_block_hash,
                    token_block_hash=token_hash,
                    token_ids=token_tuple,
                    start_position=start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=is_mla,
                    ref_base_hash=base_rec.full_block_hash,
                    delta_pos=delta_pos,
                )
                return self._commit_record(layer_name, record, raw_bytes)

            # Case 1B: Layer >= 1 LOCAL_SLIDING with divergence_distance >= threshold
            if (
                policy.is_sliding_window
                and divergence_distance is not None
                and divergence_distance >= policy.sliding_window_dedup_threshold
            ):
                record = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER0_ZERO_BIT,
                    full_block_hash=full_block_hash,
                    token_block_hash=token_hash,
                    token_ids=token_tuple,
                    start_position=start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=is_mla,
                    ref_base_hash=base_rec.full_block_hash,
                    delta_pos=delta_pos,
                )
                return self._commit_record(layer_name, record, raw_bytes)

            # Case 1C: Layer >= 1 exact token block match (Regimes R1a, R1b, R3)
            assert base_rec.raw_k is not None
            k_ref_aligned = apply_relative_rope_rotation(
                base_rec.raw_k,
                delta_pos=delta_pos,
                rope_theta=policy.rope_theta,
                rotary_dim=policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
            )
            delta_k = k_cpu.to(torch.float32) - k_ref_aligned.to(torch.float32)
            k_norm = torch.linalg.norm(k_cpu.to(torch.float32)).item()
            dk_norm = torch.linalg.norm(delta_k).item()
            eta_k = dk_norm / max(k_norm, 1e-8)

            if v_cpu is not None and base_rec.raw_v is not None:
                delta_v = v_cpu.to(torch.float32) - base_rec.raw_v.to(torch.float32)
                v_norm = torch.linalg.norm(v_cpu.to(torch.float32)).item()
                dv_norm = torch.linalg.norm(delta_v).item()
                eta_v = dv_norm / max(v_norm, 1e-8)
                eta_max = max(eta_k, eta_v)
            else:
                delta_v = None
                eta_max = eta_k

            # Select Tier based on residual magnitude eta_max:
            if policy.is_sliding_window and eta_max <= policy.tau_zero_bit:
                record = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER0_ZERO_BIT,
                    full_block_hash=full_block_hash,
                    token_block_hash=token_hash,
                    token_ids=token_tuple,
                    start_position=start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=is_mla,
                    ref_base_hash=base_rec.full_block_hash,
                    delta_pos=delta_pos,
                )
                return self._commit_record(layer_name, record, raw_bytes)

            if eta_max <= policy.tau_int2:
                q_dk = quantize_residual_packed(delta_k.to(k_cpu.dtype), bits=2)
                q_dv = (
                    quantize_residual_packed(delta_v.to(v_cpu.dtype), bits=2)
                    if (delta_v is not None and v_cpu is not None)
                    else None
                )
                record = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER1_INT2_DELTA,
                    full_block_hash=full_block_hash,
                    token_block_hash=token_hash,
                    token_ids=token_tuple,
                    start_position=start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=is_mla,
                    ref_base_hash=base_rec.full_block_hash,
                    delta_pos=delta_pos,
                    quant_delta_k=q_dk,
                    quant_delta_v=q_dv,
                )
                return self._commit_record(layer_name, record, raw_bytes)

            if eta_max <= policy.tau_int4:
                q_dk = quantize_residual_packed(delta_k.to(k_cpu.dtype), bits=4)
                q_dv = (
                    quantize_residual_packed(delta_v.to(v_cpu.dtype), bits=4)
                    if (delta_v is not None and v_cpu is not None)
                    else None
                )
                record = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER2_INT4_DELTA,
                    full_block_hash=full_block_hash,
                    token_block_hash=token_hash,
                    token_ids=token_tuple,
                    start_position=start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=is_mla,
                    ref_base_hash=base_rec.full_block_hash,
                    delta_pos=delta_pos,
                    quant_delta_k=q_dk,
                    quant_delta_v=q_dv,
                )
                return self._commit_record(layer_name, record, raw_bytes)

        # 2. Check for Regime R2: Intra-block partial token edit (<= M tokens changed)
        partial_base, diff_indices = self._find_partial_match_base_block(
            layer_name, token_tuple
        )
        if partial_base is not None and diff_indices:
            assert partial_base.raw_k is not None
            delta_pos = start_position - partial_base.start_position
            k_ref_aligned = apply_relative_rope_rotation(
                partial_base.raw_k,
                delta_pos=delta_pos,
                rope_theta=policy.rope_theta,
                rotary_dim=policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
            )
            idx_tensor = torch.tensor(diff_indices, dtype=torch.long, device=self.device)
            sparse_k = k_cpu[idx_tensor].clone()
            sparse_v = v_cpu[idx_tensor].clone() if v_cpu is not None else None

            # Zero out the edited rows in the residual so their large outliers do not
            # pollute the per-head INT4 quantization scale of the remaining rows!
            delta_k = k_cpu.to(torch.float32) - k_ref_aligned.to(torch.float32)
            delta_k[idx_tensor] = 0.0
            q_dk = (
                quantize_residual_packed(delta_k.to(k_cpu.dtype), bits=4)
                if policy.layer_idx > 0
                else None
            )

            if v_cpu is not None and partial_base.raw_v is not None:
                delta_v = v_cpu.to(torch.float32) - partial_base.raw_v.to(torch.float32)
                delta_v[idx_tensor] = 0.0
                q_dv = (
                    quantize_residual_packed(delta_v.to(v_cpu.dtype), bits=4)
                    if policy.layer_idx > 0
                    else None
                )
            else:
                q_dv = None

            record = CompressedKVBlockRecord(
                tier=CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA,
                full_block_hash=full_block_hash,
                token_block_hash=token_hash,
                token_ids=token_tuple,
                start_position=start_position,
                layer_idx=policy.layer_idx,
                is_mla=is_mla,
                ref_base_hash=partial_base.full_block_hash,
                delta_pos=delta_pos,
                quant_delta_k=q_dk,
                quant_delta_v=q_dv,
                sparse_token_indices=idx_tensor.to(torch.uint8),
                sparse_k_bf16=sparse_k,
                sparse_v_bf16=sparse_v,
            )
            return self._commit_record(layer_name, record, raw_bytes)

        # 3. Default: Store as UNCOMPRESSED_BASE and index for single-hop reference
        base_record = CompressedKVBlockRecord(
            tier=CompressionTier.UNCOMPRESSED_BASE,
            full_block_hash=full_block_hash,
            token_block_hash=token_hash,
            token_ids=token_tuple,
            start_position=start_position,
            layer_idx=policy.layer_idx,
            is_mla=is_mla,
            raw_k=k_cpu.clone(),
            raw_v=v_cpu.clone() if v_cpu is not None else None,
        )
        self._base_by_token_hash.setdefault((layer_name, token_hash), base_record)
        self._base_blocks_by_layer.setdefault(layer_name, []).append(base_record)
        num_slices = policy.max_sparse_exceptions + 1
        for s_idx, sub_tuple in self._get_pigeonhole_slices(token_tuple, num_slices):
            self._base_by_subchunk.setdefault(
                (layer_name, s_idx, sub_tuple), []
            ).append(base_record)
        return self._commit_record(layer_name, base_record, raw_bytes)

    def _commit_record(
        self,
        layer_name: str,
        record: CompressedKVBlockRecord,
        raw_bytes: int,
    ) -> CompressedKVBlockRecord:
        self._records[(layer_name, record.full_block_hash)] = record
        self.stats.total_blocks_stored += 1
        self.stats.raw_bytes_equivalent += raw_bytes
        self.stats.compressed_bytes_stored += record.nbytes
        self.stats.tier_counts[record.tier] += 1
        return record

    def reconstruct_block(
        self,
        layer_name: str,
        full_block_hash: str,
        token_ids: tuple[int, ...] | list[int] | None = None,
        target_start_position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reconstruct a KV cache block (K, V) from the store.

        Supports:
          1. Direct lookup by (layer_name, full_block_hash).
          2. Zero-bit cross-prefix synthesis for Layer 0 (or LOCAL_SLIDING layers)
             when only a token_block_hash match exists in _base_by_token_hash.
        """
        policy = self.get_layer_policy(layer_name)
        rec = self._records.get((layer_name, full_block_hash))

        # If exact full_block_hash wasn't stored, check if we can synthesize from
        # a token_block_hash match (e.g. Layer 0 or LOCAL_SLIDING zero-bit hit).
        if rec is None and token_ids is not None:
            token_hash = compute_token_block_hash(token_ids)
            base_rec = self._base_by_token_hash.get((layer_name, token_hash))
            if base_rec is not None:
                assert base_rec.raw_k is not None
                delta_pos = (
                    (target_start_position - base_rec.start_position)
                    if target_start_position is not None
                    else 0
                )
                k_recon = apply_relative_rope_rotation(
                    base_rec.raw_k,
                    delta_pos=delta_pos,
                    rope_theta=policy.rope_theta,
                    rotary_dim=policy.rotary_dim,
                    is_neox_style=policy.is_neox_style,
                )
                v_recon = base_rec.raw_v.clone() if base_rec.raw_v is not None else None
                self.stats.total_blocks_loaded += 1
                return k_recon, v_recon

        if rec is None:
            raise KeyError(
                f"Block {full_block_hash} not found in layer {layer_name} store."
            )

        self.stats.total_blocks_loaded += 1

        if rec.tier == CompressionTier.UNCOMPRESSED_BASE:
            assert rec.raw_k is not None
            return (
                rec.raw_k.clone(),
                rec.raw_v.clone() if rec.raw_v is not None else None,
            )

        # Single-hop base lookup
        assert rec.ref_base_hash is not None
        base_rec = self._records[(layer_name, rec.ref_base_hash)]
        assert base_rec.tier == CompressionTier.UNCOMPRESSED_BASE
        assert base_rec.raw_k is not None

        k_aligned = apply_relative_rope_rotation(
            base_rec.raw_k,
            delta_pos=rec.delta_pos,
            rope_theta=policy.rope_theta,
            rotary_dim=policy.rotary_dim,
            is_neox_style=policy.is_neox_style,
        )
        v_base = base_rec.raw_v.clone() if base_rec.raw_v is not None else None

        if rec.tier == CompressionTier.TIER0_ZERO_BIT:
            return k_aligned, v_base

        if rec.tier in (
            CompressionTier.TIER1_INT2_DELTA,
            CompressionTier.TIER2_INT4_DELTA,
        ):
            assert rec.quant_delta_k is not None
            dk = dequantize_residual_packed(rec.quant_delta_k)
            k_out = (k_aligned.to(torch.float32) + dk.to(torch.float32)).to(
                k_aligned.dtype
            )
            if v_base is not None and rec.quant_delta_v is not None:
                dv = dequantize_residual_packed(rec.quant_delta_v)
                v_out = (v_base.to(torch.float32) + dv.to(torch.float32)).to(
                    v_base.dtype
                )
            else:
                v_out = v_base
            return k_out, v_out

        if rec.tier == CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA:
            k_out = k_aligned.to(torch.float32)
            if rec.quant_delta_k is not None:
                dk = dequantize_residual_packed(rec.quant_delta_k)
                k_out = k_out + dk.to(torch.float32)
            k_out_typed = k_out.to(k_aligned.dtype)

            if v_base is not None:
                v_out = v_base.to(torch.float32)
                if rec.quant_delta_v is not None:
                    dv = dequantize_residual_packed(rec.quant_delta_v)
                    v_out = v_out + dv.to(torch.float32)
                v_out_typed: torch.Tensor | None = v_out.to(v_base.dtype)
            else:
                v_out_typed = None

            # Overwrite edited token positions with exact BF16 rows
            if rec.sparse_token_indices is not None and rec.sparse_k_bf16 is not None:
                idx = rec.sparse_token_indices.to(torch.long)
                k_out_typed[idx] = rec.sparse_k_bf16
                if v_out_typed is not None and rec.sparse_v_bf16 is not None:
                    v_out_typed[idx] = rec.sparse_v_bf16
            return k_out_typed, v_out_typed

        raise ValueError(f"Unsupported compression tier: {rec.tier}")


@dataclass
class DeltaBlockTransferSpec:
    """Metadata for a single block to be stored or loaded by the worker."""

    full_block_hash: str
    token_ids: tuple[int, ...]
    start_position: int
    physical_block_id: int
    divergence_distance: int | None = None
    group_block_ids: tuple[int | None, ...] = ()


@dataclass
class DeltaRequestMeta:
    """Per-request metadata passed from Scheduler connector to Worker connector."""

    req_id: str
    is_store: bool
    blocks: list[DeltaBlockTransferSpec] = field(default_factory=list)


@dataclass
class DeltaConnectorMetadata(KVConnectorMetadata):
    """Metadata container passed from Scheduler to Worker on each step."""

    requests: list[DeltaRequestMeta] = field(default_factory=list)


class DeltaCompressionConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM V1 KVConnector implementing Layer 0 reuse & Layer n delta compression.

    Supports both single-process (in-memory shared DeltaBlockStore) and
    multi-process setups, as well as Hybrid Memory Allocator (SupportsHMA).
    """

    # Process-local shared stores indexed by engine_id so Scheduler and Worker
    # connectors in the same process automatically share the DeltaBlockStore.
    _shared_stores: dict[str, DeltaBlockStore] = {}

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,  # type: ignore[arg-type]
        )
        self._block_size = vllm_config.cache_config.block_size
        extra_config = self._kv_transfer_config.kv_connector_extra_config or {}
        engine_id = self._kv_transfer_config.engine_id or "default"

        rope_theta = float(extra_config.get("rope_theta", 10000.0))
        sliding_window_dedup_threshold = int(
            extra_config.get("sliding_window_dedup_threshold", 64)
        )
        tau_zero_bit = float(extra_config.get("tau_zero_bit", 0.06))
        tau_int2 = float(extra_config.get("tau_int2", 0.12))
        tau_int4 = float(extra_config.get("tau_int4", 0.85))
        max_sparse_exceptions = int(extra_config.get("max_sparse_exceptions", 4))

        if engine_id not in self._shared_stores:
            self._shared_stores[engine_id] = DeltaBlockStore(
                block_size=self._block_size,
                default_rope_theta=rope_theta,
                sliding_window_dedup_threshold=sliding_window_dedup_threshold,
                tau_zero_bit=tau_zero_bit,
                tau_int2=tau_int2,
                tau_int4=tau_int4,
                max_sparse_exceptions=max_sparse_exceptions,
            )
        self.store: DeltaBlockStore = self._shared_stores[engine_id]
        self._requests_need_load: dict[str, tuple["Request", int]] = {}
        self._registered_kv_caches: dict[str, torch.Tensor] = {}
        self._layer_to_group_idx: dict[str, int] = {}
        self._auto_configure_layers(vllm_config, kv_cache_config)

    def _auto_configure_layers(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
    ) -> None:
        """Auto-populate layer-to-group mapping and RoPE/SWA policies from config."""
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = (
            getattr(model_config, "hf_text_config", None)
            if model_config is not None
            else None
        )
        global_theta = float(
            getattr(hf_config, "rope_theta", self.store.default_rope_theta)
            if hf_config is not None
            else self.store.default_rope_theta
        )
        local_theta = float(
            getattr(hf_config, "rope_local_base_freq", 10000.0)
            if hf_config is not None
            else 10000.0
        )
        layer_types = (
            getattr(hf_config, "layer_types", None) if hf_config is not None else None
        )
        default_sw = (
            getattr(hf_config, "sliding_window", None)
            if hf_config is not None
            else None
        )

        if kv_cache_config is not None and hasattr(kv_cache_config, "kv_cache_groups"):
            for group_idx, group_spec in enumerate(kv_cache_config.kv_cache_groups):
                spec = getattr(group_spec, "kv_cache_spec", None)
                spec_sw = getattr(spec, "sliding_window", None)
                for layer_name in getattr(group_spec, "layer_names", []):
                    self._layer_to_group_idx[layer_name] = group_idx
                    if layer_name in self.store.layer_policies:
                        continue
                    match = re.search(r"(?:layers|blocks|h)\.(\d+)", layer_name)
                    layer_idx = int(match.group(1)) if match else 0
                    is_sw = spec_sw is not None
                    if (
                        not is_sw
                        and isinstance(layer_types, (list, tuple))
                        and layer_idx < len(layer_types)
                    ):
                        is_sw = layer_types[layer_idx] == "sliding_attention"
                    sw_size = spec_sw if spec_sw is not None else (default_sw if is_sw else None)
                    theta = local_theta if is_sw else global_theta
                    self.store.register_layer_policy(
                        layer_name,
                        LayerCompressionPolicy(
                            layer_idx=layer_idx,
                            is_sliding_window=is_sw,
                            sliding_window_size=sw_size,
                            rope_theta=theta,
                            sliding_window_dedup_threshold=self.store.sliding_window_dedup_threshold,
                            tau_zero_bit=self.store.tau_zero_bit,
                            tau_int2=self.store.tau_int2,
                            tau_int4=self.store.tau_int4,
                            max_sparse_exceptions=self.store.max_sparse_exceptions,
                        ),
                    )

    @classmethod
    def reset_shared_stores(cls) -> None:
        cls._shared_stores.clear()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._registered_kv_caches = kv_caches

    # ==============================
    # Scheduler-side methods
    # ==============================

    def _build_block_specs_for_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        block_ids: list[int] | tuple[int, ...],
        all_group_block_ids: tuple[list[int], ...] | None = None,
    ) -> list[DeltaBlockTransferSpec]:
        num_full_blocks = min(len(token_ids) // self._block_size, len(block_ids))
        specs: list[DeltaBlockTransferSpec] = []
        parent_hash = "ROOT"
        divergence_distance: int | None = 0

        for b_idx in range(num_full_blocks):
            start_pos = b_idx * self._block_size
            blk_tokens = tuple(
                int(t) for t in token_ids[start_pos : start_pos + self._block_size]
            )
            full_hash = compute_full_prefix_block_hash(parent_hash, blk_tokens)
            token_hash = compute_token_block_hash(blk_tokens)

            if self.store.has_full_block(full_hash):
                # Prefix has not diverged yet
                divergence_distance = None
            elif self.store.has_token_block(token_hash):
                # Divergent prefix, but token block matches a reference block!
                if divergence_distance is None:
                    divergence_distance = 0
                else:
                    divergence_distance += self._block_size
            else:
                # Prefix diverged at this block
                divergence_distance = 0

            grp_ids: tuple[int | None, ...] = ()
            if all_group_block_ids:
                grp_ids = tuple(
                    int(g_blks[b_idx])
                    if (b_idx < len(g_blks) and g_blks[b_idx] is not None)
                    else None
                    for g_blks in all_group_block_ids
                )

            specs.append(
                DeltaBlockTransferSpec(
                    full_block_hash=full_hash,
                    token_ids=blk_tokens,
                    start_position=start_pos,
                    physical_block_id=int(block_ids[b_idx]),
                    divergence_distance=divergence_distance,
                    group_block_ids=grp_ids,
                )
            )
            parent_hash = full_hash
        return specs

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        token_ids = list(request.prompt_token_ids or [])
        num_tokens_to_check = (
            (len(token_ids) - 1) // self._block_size
        ) * self._block_size
        if num_tokens_to_check <= num_computed_tokens:
            return 0, False

        parent_hash = "ROOT"
        matched_tokens = 0
        for start_pos in range(0, num_tokens_to_check, self._block_size):
            blk_tokens = tuple(token_ids[start_pos : start_pos + self._block_size])
            full_hash = compute_full_prefix_block_hash(parent_hash, blk_tokens)
            if self.store.has_full_block(full_hash):
                matched_tokens = start_pos + self._block_size
                parent_hash = full_hash
            else:
                break

        new_matched = max(0, matched_tokens - num_computed_tokens)
        return new_matched, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                num_external_tokens,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = DeltaConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = list(new_req.block_ids[0]) if new_req.block_ids else []
            all_group_block_ids = tuple(new_req.block_ids) if new_req.block_ids else None
            specs = self._build_block_specs_for_tokens(
                token_ids, block_ids, all_group_block_ids=all_group_block_ids
            )
            if not specs:
                continue

            if new_req.req_id in self._requests_need_load:
                _, num_ext = self._requests_need_load.pop(new_req.req_id)
                num_ext_blocks = num_ext // self._block_size
                load_specs = specs[:num_ext_blocks]
                if load_specs:
                    meta.requests.append(
                        DeltaRequestMeta(
                            req_id=new_req.req_id,
                            is_store=False,
                            blocks=load_specs,
                        )
                    )
                # If there are remaining uncached blocks in the prompt, store them
                store_specs = specs[num_ext_blocks:]
                if store_specs:
                    meta.requests.append(
                        DeltaRequestMeta(
                            req_id=new_req.req_id,
                            is_store=True,
                            blocks=store_specs,
                        )
                    )
            else:
                meta.requests.append(
                    DeltaRequestMeta(
                        req_id=new_req.req_id,
                        is_store=True,
                        blocks=specs,
                    )
                )

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    # ==============================
    # Worker-side methods
    # ==============================

    def _resolve_physical_block_id(
        self, layer_name: str, blk_spec: DeltaBlockTransferSpec
    ) -> int | None:
        if not blk_spec.group_block_ids:
            return blk_spec.physical_block_id
        group_idx = self._layer_to_group_idx.get(layer_name, 0)
        if group_idx < len(blk_spec.group_block_ids):
            return blk_spec.group_block_ids[group_idx]
        return blk_spec.physical_block_id

    @staticmethod
    def _extract_block_kv(
        kv_layer: torch.Tensor,
        physical_block_id: int,
        is_mla: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract (K_block, V_block) of shape [block_size, num_kv_heads, head_dim]."""
        if is_mla:
            # MLA shape: [num_pages, 1, page_size, head_dim] or [num_pages, page_size, head_dim]
            if kv_layer.ndim == 4 and kv_layer.shape[1] == 1:
                return kv_layer[physical_block_id, 0], None
            return kv_layer[physical_block_id], None
        if kv_layer.ndim == 5 and kv_layer.shape[0] == 2:
            # Shape: [2, num_pages, page_size, num_kv_heads, head_dim]
            return kv_layer[0, physical_block_id], kv_layer[1, physical_block_id]
        if kv_layer.ndim == 5 and kv_layer.shape[1] == 2:
            # Shape: [num_pages, 2, page_size, num_kv_heads, head_dim]
            return kv_layer[physical_block_id, 0], kv_layer[physical_block_id, 1]
        if kv_layer.ndim == 4:
            # Standardized vLLM V1 4D layout: [num_blocks, num_kv_heads, block_size, 2 * head_dim]
            blk = kv_layer[physical_block_id].transpose(0, 1)
            k_blk, v_blk = blk.chunk(2, dim=-1)
            return k_blk.contiguous(), v_blk.contiguous()
        raise ValueError(f"Unsupported kv_layer shape: {tuple(kv_layer.shape)}")

    @staticmethod
    def _inject_block_kv(
        kv_layer: torch.Tensor,
        physical_block_id: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor | None,
        is_mla: bool,
    ) -> None:
        """Write reconstructed (K_block, V_block) into a paged KV cache tensor."""
        target_device = kv_layer.device
        if is_mla or v_block is None:
            if kv_layer.ndim == 4 and kv_layer.shape[1] == 1:
                kv_layer[physical_block_id, 0].copy_(
                    k_block.to(device=target_device, non_blocking=True)
                )
            else:
                kv_layer[physical_block_id].copy_(
                    k_block.to(device=target_device, non_blocking=True)
                )
            return
        if kv_layer.ndim == 5 and kv_layer.shape[0] == 2:
            kv_layer[0, physical_block_id].copy_(
                k_block.to(device=target_device, non_blocking=True)
            )
            kv_layer[1, physical_block_id].copy_(
                v_block.to(device=target_device, non_blocking=True)
            )
        elif kv_layer.ndim == 5 and kv_layer.shape[1] == 2:
            kv_layer[physical_block_id, 0].copy_(
                k_block.to(device=target_device, non_blocking=True)
            )
            kv_layer[physical_block_id, 1].copy_(
                v_block.to(device=target_device, non_blocking=True)
            )
        elif kv_layer.ndim == 4:
            # Standardized vLLM V1 4D layout: [num_blocks, num_kv_heads, block_size, 2 * head_dim]
            kv_combined = torch.cat([k_block, v_block], dim=-1).transpose(0, 1)
            kv_layer[physical_block_id].copy_(
                kv_combined.to(device=target_device, non_blocking=True)
            )
        else:
            raise ValueError(f"Unsupported kv_layer shape: {tuple(kv_layer.shape)}")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DeltaConnectorMetadata):
            return

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        no_compile_layers = getattr(forward_context, "no_compile_layers", {})

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue
            for layer_name, layer in no_compile_layers.items():
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                if isinstance(kv_cache_layer, list):
                    kv_tensor = kv_cache_layer[
                        getattr(forward_context, "virtual_engine", 0)
                    ]
                else:
                    kv_tensor = kv_cache_layer

                layer_attn_meta = (
                    attn_metadata.get(layer_name)
                    if isinstance(attn_metadata, dict)
                    else attn_metadata
                )
                is_mla = isinstance(layer_attn_meta, MLACommonMetadata)

                for blk_spec in req_meta.blocks:
                    phys_blk_id = self._resolve_physical_block_id(layer_name, blk_spec)
                    if phys_blk_id is None or phys_blk_id >= kv_tensor.shape[0]:
                        continue
                    try:
                        k_recon, v_recon = self.store.reconstruct_block(
                            layer_name=layer_name,
                            full_block_hash=blk_spec.full_block_hash,
                            token_ids=blk_spec.token_ids,
                            target_start_position=blk_spec.start_position,
                        )
                    except KeyError:
                        continue
                    self._inject_block_kv(
                        kv_tensor,
                        phys_blk_id,
                        k_recon,
                        v_recon,
                        is_mla=is_mla,
                    )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DeltaConnectorMetadata):
            return

        is_mla = isinstance(attn_metadata, MLACommonMetadata)
        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue
            for blk_spec in req_meta.blocks:
                phys_blk_id = self._resolve_physical_block_id(layer_name, blk_spec)
                if phys_blk_id is None or phys_blk_id >= kv_layer.shape[0]:
                    continue
                k_blk, v_blk = self._extract_block_kv(
                    kv_layer, phys_blk_id, is_mla=is_mla
                )
                self.store.store_block(
                    layer_name=layer_name,
                    full_block_hash=blk_spec.full_block_hash,
                    token_ids=blk_spec.token_ids,
                    start_position=blk_spec.start_position,
                    k_block=k_blk,
                    v_block=v_blk,
                    divergence_distance=blk_spec.divergence_distance,
                )

    def wait_for_save(self) -> None:
        return

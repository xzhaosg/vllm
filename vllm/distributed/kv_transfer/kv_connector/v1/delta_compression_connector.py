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


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely read an attribute even if __getattribute__ raises ValueError (e.g. Gemma4TextConfig)."""
    if obj is None:
        return default
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


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
    rope_scaling: dict[str, Any] | None = None,
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
        rope_scaling: Optional RoPE scaling dictionary (e.g. YaRN or linear scaling).

    Returns:
        Rotated key tensor of the same shape and dtype as k_tensor.
    """
    if isinstance(delta_pos, int) and delta_pos == 0:
        return k_tensor.clone()
    if isinstance(delta_pos, torch.Tensor) and torch.all(delta_pos == 0):
        return k_tensor.clone()

    orig_dtype = k_tensor.dtype
    head_dim = k_tensor.shape[-1]
    r_type = (
        (rope_scaling.get("rope_type") or rope_scaling.get("type"))
        if isinstance(rope_scaling, dict)
        else None
    )
    if r_type == "proportional":
        r_dim = (
            0
            if (rotary_dim is not None and rotary_dim <= 0)
            else head_dim
        )
    else:
        r_dim = head_dim if rotary_dim is None else min(rotary_dim, head_dim)
    if r_dim <= 0:
        return k_tensor.clone()

    k_fp32 = k_tensor.to(torch.float32)
    k_rot = k_fp32[..., :r_dim]
    k_pass = k_fp32[..., r_dim:]

    half_dim = r_dim // 2
    if r_type == "proportional" and isinstance(rope_scaling, dict):
        partial_factor = float(rope_scaling.get("partial_rotary_factor", 1.0))
        rope_angles = int(partial_factor * head_dim) // 2
        nope_angles = half_dim - rope_angles
        freq_seq = torch.arange(
            0, rope_angles, dtype=torch.float32, device=k_tensor.device
        )
        pos_freqs = rope_theta ** (freq_seq / float(half_dim))
        inv_freq_rot = 1.0 / pos_freqs
        inv_freq = (
            torch.cat(
                [
                    inv_freq_rot,
                    torch.zeros(
                        nope_angles, dtype=torch.float32, device=k_tensor.device
                    ),
                ]
            )
            if nope_angles > 0
            else inv_freq_rot
        )
    else:
        freq_seq = torch.arange(
            0, half_dim, dtype=torch.float32, device=k_tensor.device
        )
        pos_freqs = rope_theta ** (freq_seq / float(half_dim))
        inv_freq = 1.0 / pos_freqs

    if isinstance(rope_scaling, dict):
        factor = float(rope_scaling.get("factor", 1.0))
        if r_type in ("yarn", "deepseek_yarn") and factor > 1.0:
            orig_max_pos = int(
                rope_scaling.get("original_max_position_embeddings", 4096)
            )
            beta_fast = float(rope_scaling.get("beta_fast", 32.0))
            beta_slow = float(rope_scaling.get("beta_slow", 1.0))
            truncate = bool(rope_scaling.get("truncate", True))

            def _yarn_corr_dim(num_rot: float) -> float:
                return (
                    r_dim * math.log(orig_max_pos / (num_rot * 2.0 * math.pi))
                ) / (2.0 * math.log(rope_theta))

            low = _yarn_corr_dim(beta_fast)
            high = _yarn_corr_dim(beta_slow)
            if truncate:
                low = float(math.floor(low))
                high = float(math.ceil(high))
            low = max(low, 0.0)
            high = min(high, float(r_dim - 1))
            denom = (high - low) if high != low else 0.001
            ramp = torch.clamp((freq_seq - low) / denom, 0.0, 1.0)
            inv_freq_mask = 1.0 - ramp
            inv_freq_interp = 1.0 / (factor * pos_freqs)
            inv_freq = (
                inv_freq_interp * (1.0 - inv_freq_mask) + inv_freq * inv_freq_mask
            )
        elif r_type == "linear" and factor > 0.0:
            inv_freq = inv_freq / factor

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

    # Reshape to [num_groups, group_size]. Use orig_shape[-1] when >= 32 (per-head),
    # or a factor up to 128 when the trailing dimension is small (e.g. Mamba conv_state).
    total_elems = res_fp32.numel()
    group_size = orig_shape[-1]
    if group_size < 32 and total_elems >= 32:
        group_size = math.gcd(total_elems, 128)
        if group_size < 16:
            group_size = math.gcd(total_elems, 64) or orig_shape[-1]
    flat_groups = res_fp32.reshape(-1, group_size)
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
    scales_fp32 = packed_tensor.scales.to(torch.float32)
    group_size = total_elems // max(scales_fp32.shape[0], 1)
    packed = packed_tensor.packed_data

    if packed_tensor.bits == 4:
        low_nibble = packed & 0x0F
        high_nibble = (packed >> 4) & 0x0F
        unpacked = torch.stack([low_nibble, high_nibble], dim=-1).reshape(-1)[
            :total_elems
        ]
        levels = unpacked.to(torch.float32) - 7.5
        flat_groups = levels.reshape(-1, group_size)
        recon = flat_groups * scales_fp32
    elif packed_tensor.bits == 2:
        c0 = packed & 0x03
        c1 = (packed >> 2) & 0x03
        c2 = (packed >> 4) & 0x03
        c3 = (packed >> 6) & 0x03
        unpacked = torch.stack([c0, c1, c2, c3], dim=-1).reshape(-1)[:total_elems]
        # Map {0, 1, 2, 3} -> {-1.5, -0.5, +0.5, +1.5}
        levels = unpacked.to(torch.float32) - 1.5
        flat_groups = levels.reshape(-1, group_size)
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
    sparse_token_indices: torch.Tensor | None = None  # int16 [num_edited_rows]
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
    is_mamba: bool = False
    sliding_window_size: int | None = None
    rope_theta: float = 10000.0
    rotary_dim: int | None = None
    is_neox_style: bool = True
    rope_scaling: dict[str, Any] | None = None
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
        default_rotary_dim: int | None = None,
        default_rope_scaling: dict[str, Any] | None = None,
        sliding_window_dedup_threshold: int = 64,
        tau_zero_bit: float = 0.06,
        tau_int2: float = 0.12,
        tau_int4: float = 0.85,
        max_sparse_exceptions: int = 4,
        device: str = "cpu",
    ):
        self.block_size = block_size
        self.default_rope_theta = default_rope_theta
        self.default_rotary_dim = default_rotary_dim
        self.default_rope_scaling = default_rope_scaling
        self.sliding_window_dedup_threshold = sliding_window_dedup_threshold
        self.tau_zero_bit = tau_zero_bit
        self.tau_int2 = tau_int2
        self.tau_int4 = tau_int4
        self.max_sparse_exceptions = max_sparse_exceptions
        self.device = device

        # Layer name -> LayerCompressionPolicy
        self.layer_policies: dict[str, LayerCompressionPolicy] = {}
        # Layer name -> KV cache group index (shared across scheduler & worker roles)
        self.layer_to_group_idx: dict[str, int] = {}
        # Per-KV-cache-group physical block sizes and Mamba flags
        self.group_block_sizes: list[int] = []
        self.group_is_mamba: list[bool] = []

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
        # Set of full_block_hashes materialized in Mamba/GDN recurrent state layers
        self._known_mamba_blocks: set[str] = set()

        self.stats = DeltaCompressionStats()

    def clear(self) -> None:
        """Clear all stored KV block records and reset telemetry while preserving layer policies."""
        self._records.clear()
        self._base_by_token_hash.clear()
        self._base_blocks_by_layer.clear()
        self._base_by_subchunk.clear()
        self._known_prefix_blocks.clear()
        self._known_token_blocks.clear()
        self._known_mamba_blocks.clear()
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
        is_mamba = "linear_attn" in layer_name or ".mamba" in layer_name
        policy = LayerCompressionPolicy(
            layer_idx=layer_idx,
            is_mamba=is_mamba,
            rope_theta=0.0 if is_mamba else self.default_rope_theta,
            rotary_dim=0 if is_mamba else self.default_rotary_dim,
            rope_scaling=None if is_mamba else self.default_rope_scaling,
            sliding_window_dedup_threshold=self.sliding_window_dedup_threshold,
            tau_zero_bit=0.0 if is_mamba else self.tau_zero_bit,
            tau_int2=0.0 if is_mamba else self.tau_int2,
            tau_int4=self.tau_int4,
            max_sparse_exceptions=0 if is_mamba else self.max_sparse_exceptions,
        )
        self.layer_policies[layer_name] = policy
        return policy

    def has_full_block(self, full_block_hash: str) -> bool:
        return full_block_hash in self._known_prefix_blocks

    def has_token_block(self, token_block_hash: str) -> bool:
        return token_block_hash in self._known_token_blocks

    def has_mamba_block(self, full_block_hash: str) -> bool:
        return full_block_hash in self._known_mamba_blocks

    @property
    def has_mamba_layers(self) -> bool:
        return bool(self._known_mamba_blocks) or any(
            p.is_mamba for p in self.layer_policies.values()
        )

    def _find_partial_match_base_block(
        self, layer_name: str, token_ids: tuple[int, ...]
    ) -> tuple[CompressedKVBlockRecord | None, list[int]]:
        """Find an UNCOMPRESSED_BASE block in the same layer differing in <= M tokens."""
        policy = self.get_layer_policy(layer_name)
        max_exceptions = (
            self.max_sparse_exceptions
            if policy.is_mamba
            else policy.max_sparse_exceptions
        )
        if max_exceptions <= 0:
            return None, []
        num_slices = max_exceptions + 1
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
            if 0 < len(diff_idx) <= max_exceptions:
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
                latent [block_size, kv_lora_rank + qk_rope_head_dim], or Mamba/GDN
                ssm_state [num_v_heads, head_v_dim, head_k_dim]).
            v_block: Value block tensor [block_size, num_kv_heads, head_dim] (or None
                for MLA unified latent cache, or Mamba/GDN conv_state).
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
        is_mla = v_block is None and not policy.is_mamba

        k_cpu = k_block.detach().to(self.device)
        v_cpu = v_block.detach().to(self.device) if v_block is not None else None

        raw_bytes = k_cpu.numel() * k_cpu.element_size()
        if v_cpu is not None:
            raw_bytes += v_cpu.numel() * v_cpu.element_size()

        self._known_prefix_blocks[full_block_hash] = (token_tuple, start_position)
        self._known_token_blocks.setdefault(token_hash, full_block_hash)
        if policy.is_mamba:
            self._known_mamba_blocks.add(full_block_hash)

        # 1. Check for an UNCOMPRESSED_BASE block with the exact same token_block_hash
        base_rec = self._base_by_token_hash.get((layer_name, token_hash))
        if base_rec is not None:
            delta_pos = start_position - base_rec.start_position

            # Case 1A: Layer 0 non-recurrent attention -> Always Tier 0 Zero-Bit Reuse!
            if policy.layer_idx == 0 and not policy.is_mamba:
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

            # Case 1C: Exact token block match (Regimes R1a, R1b, R3, or Mamba/GDN)
            assert base_rec.raw_k is not None
            k_ref_aligned = apply_relative_rope_rotation(
                base_rec.raw_k,
                delta_pos=delta_pos,
                rope_theta=policy.rope_theta,
                rotary_dim=0 if policy.is_mamba else policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
                rope_scaling=policy.rope_scaling,
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
                eta_v = 0.0
                eta_max = eta_k

            # Select Tier based on residual magnitude:
            if (
                policy.is_sliding_window and eta_max <= policy.tau_zero_bit
            ) or (
                policy.is_mamba and eta_max <= min(policy.tau_zero_bit, 0.01)
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

            tier1_eligible = (
                (eta_k <= policy.tau_int2 and eta_v <= policy.tau_int4)
                if policy.is_mamba
                else (eta_max <= policy.tau_int2)
            )
            if tier1_eligible:
                q_dk = quantize_residual_packed(delta_k.to(k_cpu.dtype), bits=2)
                # For Mamba/GDN, conv_state (v_cpu) is only ~6.5% of the block bytes
                # and holds the 3-token boundary window, so use 4-bit quantization on it.
                v_bits = 4 if policy.is_mamba else 2
                q_dv = (
                    quantize_residual_packed(delta_v.to(v_cpu.dtype), bits=v_bits)
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
            if policy.is_mamba:
                delta_k = k_cpu.to(torch.float32) - partial_base.raw_k.to(torch.float32)
                k_norm = torch.linalg.norm(k_cpu.to(torch.float32)).item()
                dk_norm = torch.linalg.norm(delta_k).item()
                eta_k = dk_norm / max(k_norm, 1e-8)

                if v_cpu is not None and partial_base.raw_v is not None:
                    delta_v = v_cpu.to(torch.float32) - partial_base.raw_v.to(
                        torch.float32
                    )
                    v_norm = torch.linalg.norm(v_cpu.to(torch.float32)).item()
                    dv_norm = torch.linalg.norm(delta_v).item()
                    eta_v = dv_norm / max(v_norm, 1e-8)
                    eta_max = max(eta_k, eta_v)
                else:
                    delta_v = None
                    eta_v = 0.0
                    eta_max = eta_k

                if eta_max <= min(policy.tau_zero_bit, 0.01):
                    record = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER0_ZERO_BIT,
                        full_block_hash=full_block_hash,
                        token_block_hash=token_hash,
                        token_ids=token_tuple,
                        start_position=start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=is_mla,
                        ref_base_hash=partial_base.full_block_hash,
                        delta_pos=delta_pos,
                    )
                    return self._commit_record(layer_name, record, raw_bytes)

                if eta_k <= policy.tau_int2 and eta_v <= policy.tau_int4:
                    q_dk = quantize_residual_packed(delta_k.to(k_cpu.dtype), bits=2)
                    q_dv = (
                        quantize_residual_packed(delta_v.to(v_cpu.dtype), bits=4)
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
                        ref_base_hash=partial_base.full_block_hash,
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
                        ref_base_hash=partial_base.full_block_hash,
                        delta_pos=delta_pos,
                        quant_delta_k=q_dk,
                        quant_delta_v=q_dv,
                    )
                    return self._commit_record(layer_name, record, raw_bytes)
            else:
                k_ref_aligned = apply_relative_rope_rotation(
                    partial_base.raw_k,
                    delta_pos=delta_pos,
                    rope_theta=policy.rope_theta,
                    rotary_dim=policy.rotary_dim,
                    is_neox_style=policy.is_neox_style,
                    rope_scaling=policy.rope_scaling,
                )
                idx_tensor = torch.tensor(
                    diff_indices, dtype=torch.long, device=self.device
                )
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
                    delta_v = v_cpu.to(torch.float32) - partial_base.raw_v.to(
                        torch.float32
                    )
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
                    sparse_token_indices=idx_tensor.to(torch.int16),
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
        max_exceptions = (
            self.max_sparse_exceptions
            if policy.is_mamba
            else policy.max_sparse_exceptions
        )
        if max_exceptions > 0:
            num_slices = max_exceptions + 1
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
                    rope_scaling=policy.rope_scaling,
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
            rope_scaling=policy.rope_scaling,
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
    group_running_block_ids: tuple[int | None, ...] = ()


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
        self._requests_need_load: dict[str, tuple["Request", int, int]] = {}
        self._req_local_computed_tokens: dict[str, int] = {}
        self._inflight_req_state: dict[
            str, tuple[list[int], list[list[int]], int]
        ] = {}
        self._registered_kv_caches: dict[
            str, torch.Tensor | tuple[torch.Tensor, ...]
        ] = {}
        self._layer_to_group_idx: dict[str, int] = self.store.layer_to_group_idx
        self._group_block_sizes: list[int] = self.store.group_block_sizes
        self._group_is_mamba: list[bool] = self.store.group_is_mamba
        self._active_no_compile_layers: dict[str, Any] = {}
        self._active_virtual_engine: int = 0
        self._layers_saved_in_step: set[str] = set()
        self._auto_configure_layers(vllm_config, kv_cache_config)

    @property
    def _scheduler_block_size(self) -> int:
        attn_bs = [
            bs
            for bs, is_m in zip(self._group_block_sizes, self._group_is_mamba)
            if not is_m and bs > 0
        ]
        if not attn_bs:
            return self._block_size
        return math.lcm(self._block_size, *attn_bs)

    def _get_group_block_scale(self, g_idx: int) -> int:
        if (
            g_idx < len(self._group_is_mamba)
            and self._group_is_mamba[g_idx]
        ):
            return 1
        if g_idx < len(self._group_block_sizes):
            g_bs = self._group_block_sizes[g_idx]
            if g_bs > self._block_size and g_bs % self._block_size == 0:
                return g_bs // self._block_size
        return 1

    def _auto_configure_layers(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
    ) -> None:
        """Auto-populate layer-to-group mapping and RoPE/SWA/Mamba policies from config."""
        model_config = _safe_getattr(vllm_config, "model_config", None)
        hf_config = (
            _safe_getattr(model_config, "hf_text_config", None)
            if model_config is not None
            else None
        )
        rope_params = (
            _safe_getattr(hf_config, "rope_parameters", None)
            or _safe_getattr(hf_config, "rope_scaling", None)
            if hf_config is not None
            else None
        )
        if not isinstance(rope_params, dict):
            rope_params = None

        full_rope_params = (
            rope_params.get("full_attention")
            if rope_params is not None
            and isinstance(rope_params.get("full_attention"), dict)
            else rope_params
        )
        sliding_rope_params = (
            rope_params.get("sliding_attention")
            if rope_params is not None
            and isinstance(rope_params.get("sliding_attention"), dict)
            else None
        )

        raw_theta = (
            full_rope_params.get("rope_theta")
            if full_rope_params is not None
            and full_rope_params.get("rope_theta") is not None
            else (
                _safe_getattr(hf_config, "rope_theta", None)
                if hf_config is not None
                else None
            )
        )
        global_theta = (
            float(raw_theta) if raw_theta is not None else self.store.default_rope_theta
        )
        model_type = (
            _safe_getattr(hf_config, "model_type", None)
            if hf_config is not None
            else None
        )
        raw_local_theta = (
            _safe_getattr(hf_config, "rope_local_base_freq", None)
            if hf_config is not None
            else None
        )
        if raw_local_theta is None and rope_params is not None:
            raw_local_theta = rope_params.get("rope_local_base_freq")
        if raw_local_theta is None and sliding_rope_params is not None:
            raw_local_theta = sliding_rope_params.get("rope_theta")
        has_separate_local_rope = (
            raw_local_theta is not None
            or sliding_rope_params is not None
            or model_type
            in (
                "gemma3",
                "gemma3_text",
                "gemma4",
                "gemma4_text",
            )
        )
        local_theta = (
            float(raw_local_theta)
            if raw_local_theta is not None
            else (10000.0 if has_separate_local_rope else global_theta)
        )

        full_rope_type = (
            (full_rope_params.get("rope_type") or full_rope_params.get("type"))
            if isinstance(full_rope_params, dict)
            else None
        )
        # Resolve partial rotary factor / rotary_dim (e.g. Qwen3.5 / Qwen3-Next).
        # Note: for proportional RoPE (Gemma 4), rotary_dim stays None (full head_dim)
        # and partial_rotary_factor is handled via rope_scaling in apply_relative_rope_rotation.
        rotary_dim: int | None = (
            _safe_getattr(hf_config, "rotary_dim", None)
            if hf_config is not None
            else None
        )
        if (
            rotary_dim is None
            and hf_config is not None
            and full_rope_type != "proportional"
        ):
            partial_factor = (
                full_rope_params.get("partial_rotary_factor")
                if full_rope_params is not None
                and full_rope_params.get("partial_rotary_factor") is not None
                else _safe_getattr(hf_config, "partial_rotary_factor", None)
            )
            head_dim = _safe_getattr(hf_config, "head_dim", None)
            if head_dim is None:
                hidden_size = _safe_getattr(hf_config, "hidden_size", 0)
                num_heads = _safe_getattr(hf_config, "num_attention_heads", 0)
                if hidden_size and num_heads:
                    head_dim = hidden_size // num_heads
            if (
                partial_factor is not None
                and head_dim is not None
                and float(partial_factor) < 1.0
            ):
                rotary_dim = int(head_dim * float(partial_factor))

        if hf_config is not None:
            self.store.default_rope_theta = global_theta
            self.store.default_rotary_dim = rotary_dim
            self.store.default_rope_scaling = full_rope_params

        layer_types = (
            _safe_getattr(hf_config, "layer_types", None)
            if hf_config is not None
            else None
        )
        default_sw = (
            _safe_getattr(hf_config, "sliding_window", None)
            if hf_config is not None
            else None
        )

        if kv_cache_config is not None and hasattr(kv_cache_config, "kv_cache_groups"):
            self._group_block_sizes.clear()
            self._group_is_mamba.clear()
            for group_idx, group_spec in enumerate(kv_cache_config.kv_cache_groups):
                spec = getattr(group_spec, "kv_cache_spec", None)
                inner_specs = (
                    getattr(spec, "kv_cache_specs", None)
                    if spec is not None
                    else None
                )
                grp_bs = int(getattr(spec, "block_size", self._block_size) or self._block_size)
                grp_is_mamba = spec is not None and (
                    spec.__class__.__name__ == "MambaSpec"
                    or getattr(spec, "mamba_type", None) is not None
                )
                self._group_block_sizes.append(grp_bs)
                self._group_is_mamba.append(grp_is_mamba)
                for layer_name in getattr(group_spec, "layer_names", []):
                    self._layer_to_group_idx[layer_name] = group_idx
                    if layer_name in self.store.layer_policies:
                        continue
                    layer_spec = (
                        inner_specs.get(layer_name, spec)
                        if isinstance(inner_specs, dict)
                        else spec
                    )
                    spec_sw = getattr(layer_spec, "sliding_window", None)
                    match = re.search(r"(?:layers|blocks|h)\.(\d+)", layer_name)
                    layer_idx = int(match.group(1)) if match else 0
                    is_mamba = (
                        (
                            layer_spec is not None
                            and (
                                layer_spec.__class__.__name__ == "MambaSpec"
                                or getattr(layer_spec, "mamba_type", None) is not None
                            )
                        )
                        or (
                            isinstance(layer_types, (list, tuple))
                            and layer_idx < len(layer_types)
                            and layer_types[layer_idx] in ("linear_attention", "mamba")
                        )
                        or ("linear_attn" in layer_name or ".mamba" in layer_name)
                    )
                    if is_mamba and group_idx < len(self._group_is_mamba):
                        self._group_is_mamba[group_idx] = True
                    is_sw = False if is_mamba else (spec_sw is not None)
                    if (
                        not is_mamba
                        and not is_sw
                        and isinstance(layer_types, (list, tuple))
                        and layer_idx < len(layer_types)
                    ):
                        is_sw = layer_types[layer_idx] == "sliding_attention"
                    sw_size = (
                        spec_sw
                        if spec_sw is not None
                        else (default_sw if is_sw else None)
                    )
                    theta = local_theta if is_sw else global_theta
                    if is_mamba:
                        layer_rope_scaling = None
                        layer_rotary_dim: int | None = 0
                    elif is_sw and sliding_rope_params is not None:
                        layer_rope_scaling = sliding_rope_params
                        layer_rotary_dim = None
                    elif is_sw and has_separate_local_rope:
                        layer_rope_scaling = None
                        layer_rotary_dim = rotary_dim
                    else:
                        layer_rope_scaling = full_rope_params
                        layer_rotary_dim = rotary_dim
                    follows_global_layer = (
                        is_sw
                        and layer_idx > 0
                        and isinstance(layer_types, (list, tuple))
                        and layer_idx < len(layer_types)
                        and layer_types[layer_idx - 1] == "full_attention"
                    )
                    sw_dedup_thresh = (
                        max(self.store.sliding_window_dedup_threshold, 1_000_000)
                        if follows_global_layer
                        else self.store.sliding_window_dedup_threshold
                    )
                    sw_tau_zero = (
                        min(self.store.tau_zero_bit, 0.01)
                        if follows_global_layer
                        else self.store.tau_zero_bit
                    )
                    self.store.register_layer_policy(
                        layer_name,
                        LayerCompressionPolicy(
                            layer_idx=layer_idx,
                            is_sliding_window=is_sw,
                            is_mamba=is_mamba,
                            sliding_window_size=sw_size,
                            rope_theta=0.0 if is_mamba else theta,
                            rotary_dim=layer_rotary_dim,
                            rope_scaling=layer_rope_scaling,
                            sliding_window_dedup_threshold=sw_dedup_thresh,
                            tau_zero_bit=0.0
                            if is_mamba
                            else sw_tau_zero,
                            tau_int2=0.0 if is_mamba else self.store.tau_int2,
                            tau_int4=self.store.tau_int4,
                            max_sparse_exceptions=0
                            if is_mamba
                            else self.store.max_sparse_exceptions,
                        ),
                    )

    @classmethod
    def reset_shared_stores(cls) -> None:
        cls._shared_stores.clear()

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, ...]]
    ) -> None:
        self._registered_kv_caches = kv_caches
        for layer_name, kv_obj in kv_caches.items():
            if isinstance(kv_obj, tuple):
                policy = self.store.get_layer_policy(layer_name)
                policy.is_mamba = True
                policy.rotary_dim = 0
                policy.tau_zero_bit = 0.0
                policy.tau_int2 = 0.0
                policy.max_sparse_exceptions = 0

    # ==============================
    # Scheduler-side methods
    # ==============================

    def _build_block_specs_for_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        block_ids: list[int] | tuple[int, ...],
        all_group_block_ids: tuple[list[int], ...] | None = None,
    ) -> list[DeltaBlockTransferSpec]:
        max_avail_blocks = (
            max(
                (
                    len(g) * self._get_group_block_scale(g_idx)
                    for g_idx, g in enumerate(all_group_block_ids)
                ),
                default=len(block_ids) * self._get_group_block_scale(0),
            )
            if all_group_block_ids
            else len(block_ids) * self._get_group_block_scale(0)
        )
        num_full_blocks = min(len(token_ids) // self._block_size, max_avail_blocks)
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
                    int(g_blks[b_idx // self._get_group_block_scale(g_idx)])
                    if (
                        (b_idx // self._get_group_block_scale(g_idx)) < len(g_blks)
                        and g_blks[b_idx // self._get_group_block_scale(g_idx)]
                        is not None
                    )
                    else None
                    for g_idx, g_blks in enumerate(all_group_block_ids)
                )

            b0_idx = b_idx // self._get_group_block_scale(0)
            phys_id = (
                int(block_ids[b0_idx])
                if b0_idx < len(block_ids) and block_ids[b0_idx] is not None
                else 0
            )
            specs.append(
                DeltaBlockTransferSpec(
                    full_block_hash=full_hash,
                    token_ids=blk_tokens,
                    start_position=start_pos,
                    physical_block_id=phys_id,
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
        self._req_local_computed_tokens[request.request_id] = num_computed_tokens
        token_ids = list(request.prompt_token_ids or [])
        num_tokens_to_check = (
            (len(token_ids) - 1) // self._block_size
        ) * self._block_size
        if num_tokens_to_check <= num_computed_tokens:
            return 0, False

        parent_hash = "ROOT"
        matched_positions: list[tuple[int, str]] = []
        for start_pos in range(0, num_tokens_to_check, self._block_size):
            blk_tokens = tuple(token_ids[start_pos : start_pos + self._block_size])
            full_hash = compute_full_prefix_block_hash(parent_hash, blk_tokens)
            if self.store.has_full_block(full_hash):
                matched_positions.append((start_pos + self._block_size, full_hash))
                parent_hash = full_hash
            else:
                break

        if self.store.has_mamba_layers:
            while matched_positions and not self.store.has_mamba_block(
                matched_positions[-1][1]
            ):
                matched_positions.pop()

        matched_tokens = matched_positions[-1][0] if matched_positions else 0
        new_matched = max(0, matched_tokens - num_computed_tokens)
        return new_matched, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        local_tokens = self._req_local_computed_tokens.get(request.request_id, 0)
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                local_tokens,
                num_external_tokens,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = DeltaConnectorMetadata()

        for fin_id in getattr(scheduler_output, "finished_req_ids", ()) or ():
            self._inflight_req_state.pop(fin_id, None)
            self._req_local_computed_tokens.pop(fin_id, None)

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            block_ids = list(new_req.block_ids[0]) if new_req.block_ids else []
            all_group_block_ids = (
                tuple(list(g) for g in new_req.block_ids)
                if new_req.block_ids
                else None
            )
            specs = self._build_block_specs_for_tokens(
                token_ids, block_ids, all_group_block_ids=all_group_block_ids
            )
            total_prompt_full_blocks = len(token_ids) // self._block_size
            if all_group_block_ids and len(specs) < total_prompt_full_blocks:
                self._inflight_req_state[new_req.req_id] = (
                    token_ids,
                    [list(g) for g in all_group_block_ids],
                    len(specs),
                )
            if not specs:
                self._req_local_computed_tokens.pop(new_req.req_id, None)
                continue

            if new_req.req_id in self._requests_need_load:
                _, local_toks, num_ext = self._requests_need_load.pop(new_req.req_id)
                self._req_local_computed_tokens.pop(new_req.req_id, None)
                num_local_blocks = local_toks // self._block_size
                num_ext_blocks = num_ext // self._block_size
                end_load_block = num_local_blocks + num_ext_blocks
                load_specs = specs[num_local_blocks:end_load_block]
                if load_specs:
                    if all_group_block_ids:
                        num_sched_map = (
                            getattr(scheduler_output, "num_scheduled_tokens", None)
                            or {}
                        )
                        sched_toks = (
                            num_sched_map.get(new_req.req_id, 0)
                            if isinstance(num_sched_map, dict)
                            else 0
                        )
                        running_ids: list[int | None] = []
                        for g_blks in all_group_block_ids:
                            curr_idx = (
                                (local_toks + num_ext + sched_toks - 1)
                                // self._block_size
                                if sched_toks > 0
                                else len(g_blks) - 1
                            )
                            if (
                                curr_idx >= end_load_block
                                and curr_idx < len(g_blks)
                                and g_blks[curr_idx] is not None
                            ):
                                running_ids.append(int(g_blks[curr_idx]))
                            else:
                                running_ids.append(None)
                        load_specs[-1].group_running_block_ids = tuple(running_ids)
                    meta.requests.append(
                        DeltaRequestMeta(
                            req_id=new_req.req_id,
                            is_store=False,
                            blocks=load_specs,
                        )
                    )
                # If there are remaining uncached blocks in the prompt, store them
                store_specs = specs[end_load_block:]
                if store_specs:
                    meta.requests.append(
                        DeltaRequestMeta(
                            req_id=new_req.req_id,
                            is_store=True,
                            blocks=store_specs,
                        )
                    )
            else:
                local_toks = self._req_local_computed_tokens.pop(new_req.req_id, 0)
                num_local_blocks = local_toks // self._block_size
                store_specs = specs[num_local_blocks:]
                if store_specs:
                    meta.requests.append(
                        DeltaRequestMeta(
                            req_id=new_req.req_id,
                            is_store=True,
                            blocks=store_specs,
                        )
                    )

        # Handle multi-step prefill continuations (e.g. Mamba block-aligned splits)
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is not None and self._inflight_req_state:
            c_req_ids = getattr(cached_reqs, "req_ids", []) or []
            c_new_block_ids = getattr(cached_reqs, "new_block_ids", []) or []
            c_resumed = getattr(cached_reqs, "resumed_req_ids", set()) or set()
            for idx, req_id in enumerate(c_req_ids):
                if req_id not in self._inflight_req_state:
                    continue
                token_ids, group_lists, prev_stored = self._inflight_req_state[req_id]
                new_blk_tuple = (
                    c_new_block_ids[idx] if idx < len(c_new_block_ids) else None
                )
                if new_blk_tuple is not None:
                    if req_id in c_resumed:
                        group_lists = [list(g) for g in new_blk_tuple]
                    else:
                        for g_idx, g_new in enumerate(new_blk_tuple):
                            if g_idx < len(group_lists) and g_new:
                                group_lists[g_idx].extend(g_new)
                specs = self._build_block_specs_for_tokens(
                    token_ids,
                    group_lists[0] if group_lists else [],
                    all_group_block_ids=tuple(group_lists),
                )
                new_store_specs = specs[prev_stored:]
                if new_store_specs:
                    meta.requests.append(
                        DeltaRequestMeta(
                            req_id=req_id,
                            is_store=True,
                            blocks=new_store_specs,
                        )
                    )
                total_prompt_full_blocks = len(token_ids) // self._block_size
                if len(specs) >= total_prompt_full_blocks:
                    self._inflight_req_state.pop(req_id, None)
                else:
                    self._inflight_req_state[req_id] = (
                        token_ids,
                        group_lists,
                        len(specs),
                    )

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._inflight_req_state.pop(request.request_id, None)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._inflight_req_state.pop(request.request_id, None)
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

    def _resolve_running_block_id(
        self, layer_name: str, blk_spec: DeltaBlockTransferSpec
    ) -> int | None:
        if not blk_spec.group_running_block_ids:
            return None
        group_idx = self._layer_to_group_idx.get(layer_name, 0)
        if group_idx < len(blk_spec.group_running_block_ids):
            return blk_spec.group_running_block_ids[group_idx]
        return None

    @staticmethod
    def _get_kernel_block_ratio(
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        is_mla: bool = False,
        block_size: int | None = None,
    ) -> int:
        if isinstance(kv_layer, tuple) or block_size is None or block_size <= 0:
            return 1
        if is_mla:
            kbs = (
                kv_layer.shape[2]
                if (kv_layer.ndim == 4 and kv_layer.shape[1] == 1)
                else kv_layer.shape[1]
            )
        elif kv_layer.ndim in (4, 5):
            kbs = kv_layer.shape[2]
        else:
            return 1
        if kbs > 0 and block_size > kbs and block_size % kbs == 0:
            return block_size // kbs
        return 1

    @staticmethod
    def _get_kv_layer_num_blocks(
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        is_mla: bool = False,
        block_size: int | None = None,
    ) -> int:
        if isinstance(kv_layer, tuple):
            return kv_layer[0].shape[0] if kv_layer else 0
        raw_blocks = (
            kv_layer.shape[1]
            if (not is_mla and kv_layer.ndim == 5 and kv_layer.shape[0] == 2)
            else kv_layer.shape[0]
        )
        ratio = DeltaCompressionConnector._get_kernel_block_ratio(
            kv_layer, is_mla=is_mla, block_size=block_size
        )
        return raw_blocks // ratio

    def _get_layer_kv_cache(
        self,
        layer_name: str,
        layer: Any = None,
        virtual_engine: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        if layer is not None:
            kv_cache_layer = getattr(layer, "kv_cache", None)
            if isinstance(kv_cache_layer, list) and kv_cache_layer:
                kv_cache_layer = kv_cache_layer[
                    min(virtual_engine, len(kv_cache_layer) - 1)
                ]
            if isinstance(kv_cache_layer, tuple) and kv_cache_layer:
                if (
                    isinstance(kv_cache_layer[0], torch.Tensor)
                    and kv_cache_layer[0].numel() > 0
                ):
                    return kv_cache_layer
            elif (
                isinstance(kv_cache_layer, torch.Tensor)
                and kv_cache_layer.numel() > 0
            ):
                return kv_cache_layer
        reg = self._registered_kv_caches.get(layer_name)
        if isinstance(reg, tuple) and reg and reg[0].numel() > 0:
            return reg
        if isinstance(reg, torch.Tensor) and reg.numel() > 0:
            return reg
        return None

    @staticmethod
    def _extract_block_kv(
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        physical_block_id: int,
        is_mla: bool,
        block_size: int | None = None,
        start_position: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract (K_block, V_block) or Mamba/GDN (ssm_state, conv_state)."""
        if isinstance(kv_layer, tuple):
            if len(kv_layer) == 2:
                conv_cache, ssm_cache = kv_layer
                return (
                    ssm_cache[physical_block_id].contiguous(),
                    conv_cache[physical_block_id].contiguous(),
                )
            if len(kv_layer) == 1:
                return kv_layer[0][physical_block_id].contiguous(), None
            raise ValueError(f"Unsupported Mamba state tuple length: {len(kv_layer)}")

        ratio = DeltaCompressionConnector._get_kernel_block_ratio(
            kv_layer, is_mla=is_mla, block_size=block_size
        )
        start_blk = physical_block_id * ratio
        end_blk = start_blk + ratio

        if is_mla:
            # MLA shape: [num_pages, 1, page_size, head_dim] or [num_pages, page_size, head_dim]
            if ratio == 1:
                blk = (
                    kv_layer[physical_block_id, 0]
                    if (kv_layer.ndim == 4 and kv_layer.shape[1] == 1)
                    else kv_layer[physical_block_id]
                )
                if (
                    block_size is not None
                    and 0 < block_size < blk.shape[0]
                    and blk.shape[0] % block_size == 0
                ):
                    sub_start = start_position % blk.shape[0]
                    blk = blk[sub_start : sub_start + block_size]
                return blk, None
            if kv_layer.ndim == 4 and kv_layer.shape[1] == 1:
                sub = kv_layer[start_blk:end_blk, 0]
            else:
                sub = kv_layer[start_blk:end_blk]
            return sub.reshape(-1, *sub.shape[2:]).contiguous(), None
        if kv_layer.ndim == 5 and kv_layer.shape[0] == 2:
            # Shape: [2, num_pages, page_size, num_kv_heads, head_dim]
            if ratio == 1:
                k_blk = kv_layer[0, physical_block_id]
                v_blk = kv_layer[1, physical_block_id]
                if (
                    block_size is not None
                    and 0 < block_size < k_blk.shape[0]
                    and k_blk.shape[0] % block_size == 0
                ):
                    sub_start = start_position % k_blk.shape[0]
                    k_blk = k_blk[sub_start : sub_start + block_size]
                    v_blk = v_blk[sub_start : sub_start + block_size]
                return k_blk, v_blk
            k_sub = kv_layer[0, start_blk:end_blk]
            v_sub = kv_layer[1, start_blk:end_blk]
            return (
                k_sub.reshape(-1, *k_sub.shape[2:]).contiguous(),
                v_sub.reshape(-1, *v_sub.shape[2:]).contiguous(),
            )
        if kv_layer.ndim == 5 and kv_layer.shape[1] == 2:
            # Shape: [num_pages, 2, page_size, num_kv_heads, head_dim]
            if ratio == 1:
                k_blk = kv_layer[physical_block_id, 0]
                v_blk = kv_layer[physical_block_id, 1]
                if (
                    block_size is not None
                    and 0 < block_size < k_blk.shape[0]
                    and k_blk.shape[0] % block_size == 0
                ):
                    sub_start = start_position % k_blk.shape[0]
                    k_blk = k_blk[sub_start : sub_start + block_size]
                    v_blk = v_blk[sub_start : sub_start + block_size]
                return k_blk, v_blk
            k_sub = kv_layer[start_blk:end_blk, 0]
            v_sub = kv_layer[start_blk:end_blk, 1]
            return (
                k_sub.reshape(-1, *k_sub.shape[2:]).contiguous(),
                v_sub.reshape(-1, *v_sub.shape[2:]).contiguous(),
            )
        if kv_layer.ndim == 4:
            # Standardized vLLM V1 4D layout: [num_blocks, num_kv_heads, kernel_block_size, 2 * head_dim]
            if ratio == 1:
                blk = kv_layer[physical_block_id].transpose(0, 1)
                if (
                    block_size is not None
                    and 0 < block_size < blk.shape[0]
                    and blk.shape[0] % block_size == 0
                ):
                    sub_start = start_position % blk.shape[0]
                    blk = blk[sub_start : sub_start + block_size]
            else:
                sub = kv_layer[start_blk:end_blk]
                blk = sub.permute(0, 2, 1, 3).reshape(
                    ratio * sub.shape[2], sub.shape[1], sub.shape[3]
                )
            k_blk, v_blk = blk.chunk(2, dim=-1)
            return k_blk.contiguous(), v_blk.contiguous()
        raise ValueError(f"Unsupported kv_layer shape: {tuple(kv_layer.shape)}")

    @staticmethod
    def _inject_block_kv(
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        physical_block_id: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor | None,
        is_mla: bool,
        block_size: int | None = None,
        start_position: int = 0,
    ) -> None:
        """Write reconstructed (K_block, V_block) or Mamba/GDN (ssm_state, conv_state) into cache."""
        if isinstance(kv_layer, tuple):
            if len(kv_layer) == 2:
                conv_cache, ssm_cache = kv_layer
                ssm_cache[physical_block_id].copy_(
                    k_block.to(device=ssm_cache.device, non_blocking=True)
                )
                if v_block is not None:
                    conv_cache[physical_block_id].copy_(
                        v_block.to(device=conv_cache.device, non_blocking=True)
                    )
                return
            if len(kv_layer) == 1:
                kv_layer[0][physical_block_id].copy_(
                    k_block.to(device=kv_layer[0].device, non_blocking=True)
                )
                return
            raise ValueError(f"Unsupported Mamba state tuple length: {len(kv_layer)}")

        effective_block_size = (
            block_size if block_size is not None else int(k_block.shape[0])
        )
        ratio = DeltaCompressionConnector._get_kernel_block_ratio(
            kv_layer, is_mla=is_mla, block_size=effective_block_size
        )
        start_blk = physical_block_id * ratio
        end_blk = start_blk + ratio
        target_device = kv_layer.device
        if is_mla or v_block is None:
            k_dev = k_block.to(device=target_device, non_blocking=True)
            if ratio == 1:
                target = (
                    kv_layer[physical_block_id, 0]
                    if (kv_layer.ndim == 4 and kv_layer.shape[1] == 1)
                    else kv_layer[physical_block_id]
                )
                if (
                    0 < effective_block_size < target.shape[0]
                    and target.shape[0] % effective_block_size == 0
                ):
                    sub_start = start_position % target.shape[0]
                    target[sub_start : sub_start + effective_block_size].copy_(k_dev)
                else:
                    target.copy_(k_dev)
            else:
                if kv_layer.ndim == 4 and kv_layer.shape[1] == 1:
                    kbs = kv_layer.shape[2]
                    kv_layer[start_blk:end_blk, 0].copy_(
                        k_dev.reshape(ratio, kbs, *k_dev.shape[1:])
                    )
                else:
                    kbs = kv_layer.shape[1]
                    kv_layer[start_blk:end_blk].copy_(
                        k_dev.reshape(ratio, kbs, *k_dev.shape[1:])
                    )
            return
        if kv_layer.ndim == 5 and kv_layer.shape[0] == 2:
            k_dev = k_block.to(device=target_device, non_blocking=True)
            v_dev = v_block.to(device=target_device, non_blocking=True)
            if ratio == 1:
                k_target = kv_layer[0, physical_block_id]
                v_target = kv_layer[1, physical_block_id]
                if (
                    0 < effective_block_size < k_target.shape[0]
                    and k_target.shape[0] % effective_block_size == 0
                ):
                    sub_start = start_position % k_target.shape[0]
                    k_target[sub_start : sub_start + effective_block_size].copy_(k_dev)
                    v_target[sub_start : sub_start + effective_block_size].copy_(v_dev)
                else:
                    k_target.copy_(k_dev)
                    v_target.copy_(v_dev)
            else:
                kbs = kv_layer.shape[2]
                kv_layer[0, start_blk:end_blk].copy_(
                    k_dev.reshape(ratio, kbs, *k_dev.shape[1:])
                )
                kv_layer[1, start_blk:end_blk].copy_(
                    v_dev.reshape(ratio, kbs, *v_dev.shape[1:])
                )
        elif kv_layer.ndim == 5 and kv_layer.shape[1] == 2:
            k_dev = k_block.to(device=target_device, non_blocking=True)
            v_dev = v_block.to(device=target_device, non_blocking=True)
            if ratio == 1:
                k_target = kv_layer[physical_block_id, 0]
                v_target = kv_layer[physical_block_id, 1]
                if (
                    0 < effective_block_size < k_target.shape[0]
                    and k_target.shape[0] % effective_block_size == 0
                ):
                    sub_start = start_position % k_target.shape[0]
                    k_target[sub_start : sub_start + effective_block_size].copy_(k_dev)
                    v_target[sub_start : sub_start + effective_block_size].copy_(v_dev)
                else:
                    k_target.copy_(k_dev)
                    v_target.copy_(v_dev)
            else:
                kbs = kv_layer.shape[2]
                kv_layer[start_blk:end_blk, 0].copy_(
                    k_dev.reshape(ratio, kbs, *k_dev.shape[1:])
                )
                kv_layer[start_blk:end_blk, 1].copy_(
                    v_dev.reshape(ratio, kbs, *v_dev.shape[1:])
                )
        elif kv_layer.ndim == 4:
            # Standardized vLLM V1 4D layout: [num_blocks, num_kv_heads, kernel_block_size, 2 * head_dim]
            if ratio == 1:
                kv_combined = torch.cat([k_block, v_block], dim=-1).transpose(0, 1)
                target = kv_layer[physical_block_id]
                if (
                    0 < effective_block_size < target.shape[1]
                    and target.shape[1] % effective_block_size == 0
                ):
                    sub_start = start_position % target.shape[1]
                    target[
                        :, sub_start : sub_start + effective_block_size, :
                    ].copy_(kv_combined.to(device=target_device, non_blocking=True))
                else:
                    target.copy_(
                        kv_combined.to(device=target_device, non_blocking=True)
                    )
            else:
                kbs = kv_layer.shape[2]
                kv_combined = (
                    torch.cat([k_block, v_block], dim=-1)
                    .reshape(ratio, kbs, kv_layer.shape[1], kv_layer.shape[3])
                    .permute(0, 2, 1, 3)
                )
                kv_layer[start_blk:end_blk].copy_(
                    kv_combined.to(device=target_device, non_blocking=True)
                )
        else:
            raise ValueError(f"Unsupported kv_layer shape: {tuple(kv_layer.shape)}")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._active_no_compile_layers = (
            getattr(forward_context, "no_compile_layers", {})
            if forward_context is not None
            else {}
        ) or {}
        self._active_virtual_engine = (
            getattr(forward_context, "virtual_engine", 0)
            if forward_context is not None
            else 0
        )
        self._layers_saved_in_step.clear()

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DeltaConnectorMetadata):
            return

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        all_layer_names = list(self._active_no_compile_layers.keys())
        for reg_name in self._registered_kv_caches:
            if reg_name not in self._active_no_compile_layers:
                all_layer_names.append(reg_name)

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue
            for layer_name in all_layer_names:
                if (
                    self._layer_to_group_idx
                    and layer_name not in self._layer_to_group_idx
                ):
                    continue
                layer = self._active_no_compile_layers.get(layer_name)
                kv_tensor = self._get_layer_kv_cache(
                    layer_name, layer, self._active_virtual_engine
                )
                if kv_tensor is None:
                    continue

                policy = self.store.get_layer_policy(layer_name)
                if isinstance(kv_tensor, tuple) and not policy.is_mamba:
                    policy.is_mamba = True
                    policy.rotary_dim = 0
                    policy.tau_zero_bit = 0.0
                    policy.tau_int2 = 0.0
                    policy.max_sparse_exceptions = 0

                layer_attn_meta = (
                    attn_metadata.get(layer_name)
                    if isinstance(attn_metadata, dict)
                    else attn_metadata
                )
                is_mla = isinstance(layer_attn_meta, MLACommonMetadata)
                num_blocks = self._get_kv_layer_num_blocks(
                    kv_tensor, is_mla=is_mla, block_size=self._block_size
                )

                for blk_spec in req_meta.blocks:
                    phys_blk_id = self._resolve_physical_block_id(layer_name, blk_spec)
                    if phys_blk_id is None or phys_blk_id >= num_blocks:
                        continue
                    if (
                        isinstance(kv_tensor, tuple)
                        or policy.is_mamba
                        or len(blk_spec.group_block_ids) > 1
                    ) and phys_blk_id <= 0:
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
                        block_size=self._block_size,
                        start_position=blk_spec.start_position,
                    )
                    if isinstance(kv_tensor, tuple) or policy.is_mamba:
                        running_blk_id = self._resolve_running_block_id(
                            layer_name, blk_spec
                        )
                        if (
                            running_blk_id is not None
                            and 0 < running_blk_id < num_blocks
                            and running_blk_id != phys_blk_id
                        ):
                            self._inject_block_kv(
                                kv_tensor,
                                running_blk_id,
                                k_recon,
                                v_recon,
                                is_mla=is_mla,
                                block_size=self._block_size,
                                start_position=blk_spec.start_position,
                            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        self._layers_saved_in_step.add(layer_name)
        if self._layer_to_group_idx and layer_name not in self._layer_to_group_idx:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DeltaConnectorMetadata):
            return

        policy = self.store.get_layer_policy(layer_name)
        if isinstance(kv_layer, tuple) and not policy.is_mamba:
            policy.is_mamba = True
            policy.rotary_dim = 0
            policy.tau_zero_bit = 0.0
            policy.tau_int2 = 0.0
            policy.max_sparse_exceptions = 0

        is_mla = isinstance(attn_metadata, MLACommonMetadata)
        num_blocks = self._get_kv_layer_num_blocks(
            kv_layer, is_mla=is_mla, block_size=self._block_size
        )
        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue
            for blk_spec in req_meta.blocks:
                phys_blk_id = self._resolve_physical_block_id(layer_name, blk_spec)
                if phys_blk_id is None or phys_blk_id >= num_blocks:
                    continue
                if (
                    isinstance(kv_layer, tuple)
                    or policy.is_mamba
                    or len(blk_spec.group_block_ids) > 1
                ) and phys_blk_id <= 0:
                    continue
                k_blk, v_blk = self._extract_block_kv(
                    kv_layer,
                    phys_blk_id,
                    is_mla=is_mla,
                    block_size=self._block_size,
                    start_position=blk_spec.start_position,
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
        """Save any Mamba/GDN recurrent state layers not visited by @maybe_transfer_kv_layer."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DeltaConnectorMetadata):
            self._layers_saved_in_step.clear()
            return

        store_reqs = [r for r in metadata.requests if r.is_store]
        if not store_reqs:
            self._layers_saved_in_step.clear()
            return

        all_layer_names = list(self._active_no_compile_layers.keys())
        for reg_name in self._registered_kv_caches:
            if reg_name not in self._active_no_compile_layers:
                all_layer_names.append(reg_name)

        for layer_name in all_layer_names:
            if layer_name in self._layers_saved_in_step:
                continue
            layer = self._active_no_compile_layers.get(layer_name)
            kv_tensor = self._get_layer_kv_cache(
                layer_name, layer, self._active_virtual_engine
            )
            if kv_tensor is None:
                continue
            policy = self.store.get_layer_policy(layer_name)
            if not isinstance(kv_tensor, tuple) and not policy.is_mamba:
                continue
            if isinstance(kv_tensor, tuple) and not policy.is_mamba:
                policy.is_mamba = True
                policy.rotary_dim = 0
                policy.tau_zero_bit = 0.0
                policy.tau_int2 = 0.0
                policy.max_sparse_exceptions = 0

            num_blocks = self._get_kv_layer_num_blocks(
                kv_tensor, is_mla=False, block_size=self._block_size
            )
            for req_meta in store_reqs:
                for blk_spec in req_meta.blocks:
                    phys_blk_id = self._resolve_physical_block_id(layer_name, blk_spec)
                    if (
                        phys_blk_id is None
                        or phys_blk_id <= 0
                        or phys_blk_id >= num_blocks
                    ):
                        continue
                    k_blk, v_blk = self._extract_block_kv(
                        kv_tensor,
                        phys_blk_id,
                        is_mla=False,
                        block_size=self._block_size,
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

        self._layers_saved_in_step.clear()

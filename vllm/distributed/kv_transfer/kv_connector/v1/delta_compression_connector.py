# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeltaCompressionConnector: Layer-0 zero-bit reuse and Layer-n delta compression for vLLM V1."""

from dataclasses import dataclass, field
import enum
import math
import re
from typing import TYPE_CHECKING, Any

from prometheus_client import Gauge
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.utils.hashing import safe_hash
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.loggers import create_metric_per_engine
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
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


DEFAULT_PAD_TOKEN = "\n"


def _encode_text_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode `text` into token IDs without special tokens using a vLLM or HF tokenizer."""
    if hasattr(tokenizer, "encode"):
        try:
            res = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            res = tokenizer.encode(text)
    elif callable(tokenizer):
        try:
            out = tokenizer(text, add_special_tokens=False)
        except TypeError:
            out = tokenizer(text)
        res = out["input_ids"] if isinstance(out, dict) and "input_ids" in out else out
    else:
        raise TypeError(f"Unsupported tokenizer object: {type(tokenizer)!r}")
    if isinstance(res, torch.Tensor):
        return [int(x) for x in res.flatten().tolist()]
    return [int(x) for x in res]


def resolve_pad_token_id(
    tokenizer: Any,
    pad_token: str = DEFAULT_PAD_TOKEN,
    pad_token_id: int | None = None,
    content_text: str | None = None,
    pad_token_map: dict[str, str] | None = None,
) -> int:
    """Resolve the padding token ID for server-side block alignment (Approach 2).

    Supports:
      1. Explicit `pad_token_id` override if provided.
      2. Content-aware padding token selection via `pad_token_map` (mapping a substring
         or regex pattern in `content_text` to a padding string such as `" "` for JSON/SQL
         or `"\\n"` for source code / Markdown / SWE-agent observations).
      3. Configurable default `pad_token` string (defaults to `"\\n"`).
    """
    if pad_token_id is not None:
        return int(pad_token_id)

    effective_pad_str = pad_token if pad_token else DEFAULT_PAD_TOKEN
    if content_text and pad_token_map:
        for pattern, mapped_pad in pad_token_map.items():
            matched = pattern in content_text
            if not matched and (pattern.startswith("^") or pattern.startswith("re:")):
                rx = pattern[3:] if pattern.startswith("re:") else pattern
                try:
                    matched = bool(re.search(rx, content_text))
                except re.error:
                    matched = False
            if matched and mapped_pad:
                effective_pad_str = mapped_pad
                break

    if tokenizer is None:
        raise ValueError(
            "Either `tokenizer` or explicit `pad_token_id` must be provided to resolve_pad_token_id."
        )
    ids = _encode_text_without_special_tokens(tokenizer, effective_pad_str)
    if not ids:
        ids = _encode_text_without_special_tokens(tokenizer, DEFAULT_PAD_TOKEN)
    if not ids:
        raise ValueError(
            f"Tokenizer produced empty token IDs for pad_token={effective_pad_str!r}"
        )
    return int(ids[-1])


def align_tokens_to_block(
    token_ids: list[int] | tuple[int, ...],
    block_size: int,
    pad_token_id: int,
    target_blocks: int | None = None,
) -> list[int]:
    """Pad `token_ids` to an exact multiple of `block_size` using `pad_token_id`."""
    ids = [int(t) for t in token_ids]
    if block_size <= 0:
        return ids
    if target_blocks is not None:
        target_len = target_blocks * block_size
        if len(ids) < target_len:
            return ids + [int(pad_token_id)] * (target_len - len(ids))
        return ids[:target_len]
    rem = len(ids) % block_size
    if rem != 0:
        return ids + [int(pad_token_id)] * (block_size - rem)
    return ids


def encode_file_viewer_bucketed_server(
    tokenizer: Any,
    text: str,
    block_size: int,
    pad_token: str = DEFAULT_PAD_TOKEN,
    pad_token_id: int | None = None,
    pad_token_map: dict[str, str] | None = None,
    line_step: int = 10,
    blocks_per_bucket: int = 6,
    max_file_blocks: int | None = None,
    file_path_hint: str = "",
) -> list[int]:
    """Server-side SWE-agent `[File: ...]` viewer encoder with configurable padding token.

    Buckets numbered file viewer lines into `line_step`-line (default 10-line)
    block-aligned chunks (`blocks_per_bucket` blocks per 10-line bucket) so that:
      1. Multi-token edits inside one 10-line region do not shift token-chunk
         boundaries in downstream unchanged regions of the file.
      2. Scrolling the 100-line file window (`goto` / `scroll_down`) preserves exact
         `token_block_hash` matches on all overlapping 10-line buckets via RoPE shift.
    """
    resolved_pad_id = resolve_pad_token_id(
        tokenizer,
        pad_token=pad_token,
        pad_token_id=pad_token_id,
        content_text=text,
        pad_token_map=pad_token_map,
    )
    if line_step <= 0 or ("[File:" not in text and not file_path_hint):
        return align_tokens_to_block(
            _encode_text_without_special_tokens(tokenizer, text),
            block_size,
            resolved_pad_id,
            target_blocks=max_file_blocks,
        )

    if block_size >= 64:
        obs_lines = text.splitlines(keepends=True)
        buckets_50: dict[int, list[tuple[int, str]]] = {}
        line_re_50 = re.compile(r"^(\d+):")
        for ln in obs_lines:
            m50 = line_re_50.match(ln)
            if m50:
                lno = int(m50.group(1))
                buckets_50.setdefault((lno - 1) // 50, []).append((lno, ln))
        if not buckets_50:
            return align_tokens_to_block(
                _encode_text_without_special_tokens(tokenizer, text),
                block_size,
                resolved_pad_id,
                target_blocks=max_file_blocks,
            )
        out_ids_50: list[int] = []
        slot_len = block_size // 5
        for b_id in sorted(buckets_50.keys()):
            if (
                max_file_blocks is not None
                and (len(out_ids_50) // block_size) >= max_file_blocks
            ):
                break
            sub_buckets: dict[int, list[str]] = {}
            for lno, ln in buckets_50[b_id]:
                sub_id = ((lno - 1) % 50) // 10
                sub_buckets.setdefault(sub_id, []).append(ln)
            blk_ids: list[int] = []
            for s_id in range(5):
                stxt = "".join(sub_buckets.get(s_id, []))
                sids = (
                    _encode_text_without_special_tokens(tokenizer, stxt)
                    if stxt
                    else []
                )
                t_slot = slot_len if s_id < 4 else (block_size - 4 * slot_len)
                if len(sids) < t_slot:
                    sids = sids + [resolved_pad_id] * (t_slot - len(sids))
                else:
                    sids = sids[:t_slot]
                blk_ids.extend(sids)
            out_ids_50.extend(blk_ids)
        return align_tokens_to_block(
            out_ids_50,
            block_size,
            resolved_pad_id,
            target_blocks=max_file_blocks,
        )

    lines = text.splitlines(keepends=True)
    header_lines: list[str] = []
    numbered_lines: list[tuple[int, str]] = []
    footer_lines: list[str] = []
    in_body = False
    in_footer = False

    for line in lines:
        if not in_body and not in_footer:
            m = re.match(r"^(\d+):", line)
            if m:
                in_body = True
                numbered_lines.append((int(m.group(1)), line))
            else:
                header_lines.append(line)
        elif in_body and not in_footer:
            m = re.match(r"^(\d+):", line)
            if m:
                numbered_lines.append((int(m.group(1)), line))
            else:
                in_footer = True
                footer_lines.append(line)
        else:
            footer_lines.append(line)

    if not numbered_lines:
        return align_tokens_to_block(
            _encode_text_without_special_tokens(tokenizer, text),
            block_size,
            resolved_pad_id,
            target_blocks=max_file_blocks,
        )

    out_ids: list[int] = []
    hdr_str = (
        "".join(header_lines)
        if header_lines
        else (f"[File: {file_path_hint}]\n" if file_path_hint else "")
    )
    if hdr_str:
        hdr_pad_id = resolve_pad_token_id(
            tokenizer,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            content_text=hdr_str,
            pad_token_map=pad_token_map,
        )
        out_ids.extend(
            align_tokens_to_block(
                _encode_text_without_special_tokens(tokenizer, hdr_str),
                block_size,
                hdr_pad_id,
                target_blocks=1,
            )
        )

    buckets: dict[int, list[str]] = {}
    for line_no, line_str in numbered_lines:
        b_key = (line_no - 1) // line_step
        buckets.setdefault(b_key, []).append(line_str)

    eff_blocks_per_bucket = (
        max(1, (blocks_per_bucket * 16) // block_size)
        if block_size > 16 and blocks_per_bucket == 6
        else blocks_per_bucket
    )

    for b_key in sorted(buckets.keys()):
        if (
            max_file_blocks is not None
            and (len(out_ids) // block_size) + eff_blocks_per_bucket > max_file_blocks
        ):
            break
        if block_size == 64 and eff_blocks_per_bucket == 1 and line_step == 10:
            sub_buckets_64: dict[int, list[str]] = {}
            for ln_str in buckets[b_key]:
                m64 = re.match(r"^(\d+):", ln_str)
                lno64 = int(m64.group(1)) if m64 else 1
                sub_id64 = ((lno64 - 1) % 10) // 2
                sub_buckets_64.setdefault(sub_id64, []).append(ln_str)
            blk_ids_64: list[int] = []
            for s_id in range(5):
                stxt = "".join(sub_buckets_64.get(s_id, []))
                sids = (
                    _encode_text_without_special_tokens(tokenizer, stxt)
                    if stxt
                    else []
                )
                t_slot = 13 if s_id < 4 else 12
                if len(sids) < t_slot:
                    sids = sids + [resolved_pad_id] * (t_slot - len(sids))
                else:
                    sids = sids[:t_slot]
                blk_ids_64.extend(sids)
            out_ids.extend(blk_ids_64)
            continue
        b_str = "".join(buckets[b_key])
        b_pad_id = resolve_pad_token_id(
            tokenizer,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            content_text=b_str,
            pad_token_map=pad_token_map,
        )
        out_ids.extend(
            align_tokens_to_block(
                _encode_text_without_special_tokens(tokenizer, b_str),
                block_size,
                b_pad_id,
                target_blocks=eff_blocks_per_bucket,
            )
        )

    if footer_lines and (
        max_file_blocks is None
        or len(out_ids) < max_file_blocks * block_size
    ):
        max_ftr_blocks = 1 if block_size >= 64 else 2
        rem_blocks = (
            min(max_ftr_blocks, max_file_blocks - (len(out_ids) // block_size))
            if max_file_blocks is not None
            else None
        )
        ftr_str = "".join(footer_lines)
        ftr_pad_id = resolve_pad_token_id(
            tokenizer,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            content_text=ftr_str,
            pad_token_map=pad_token_map,
        )
        out_ids.extend(
            align_tokens_to_block(
                _encode_text_without_special_tokens(tokenizer, ftr_str),
                block_size,
                ftr_pad_id,
                target_blocks=rem_blocks,
            )
        )
    return out_ids


def align_structural_prompt_server(
    tokenizer: Any,
    segments: list[tuple[str, bool]] | list[str],
    block_size: int,
    pad_token: str = DEFAULT_PAD_TOKEN,
    pad_token_id: int | None = None,
    pad_token_map: dict[str, str] | None = None,
    bucket_file_viewer: bool = True,
) -> list[int]:
    """Server-side structural prompt tokenizer and block aligner (Approach 2).

    Each segment in `segments` can be either:
      - `(segment_text, align_this_segment)` where `align_this_segment=False` is used
        for the final open generation prefix (e.g. `<|im_start|>assistant\\n`) so
        next-token decoding starts at the exact unpadded generation boundary, OR
      - `segment_text` (a plain `str`), in which case all segments except the last
        open generation prefix are block-aligned to `block_size`.
    """
    out_ids: list[int] = []
    num_segs = len(segments)
    for idx, item in enumerate(segments):
        if isinstance(item, tuple):
            seg_text, should_align = item
        else:
            seg_text = str(item)
            should_align = idx < (num_segs - 1)
        if not seg_text:
            continue
        if not should_align:
            out_ids.extend(_encode_text_without_special_tokens(tokenizer, seg_text))
        elif bucket_file_viewer and "[File: " in seg_text:
            out_ids.extend(
                encode_file_viewer_bucketed_server(
                    tokenizer=tokenizer,
                    text=seg_text,
                    block_size=block_size,
                    pad_token=pad_token,
                    pad_token_id=pad_token_id,
                    pad_token_map=pad_token_map,
                )
            )
        else:
            seg_pad_id = resolve_pad_token_id(
                tokenizer,
                pad_token=pad_token,
                pad_token_id=pad_token_id,
                content_text=seg_text,
                pad_token_map=pad_token_map,
            )
            out_ids.extend(
                align_tokens_to_block(
                    _encode_text_without_special_tokens(tokenizer, seg_text),
                    block_size,
                    seg_pad_id,
                )
            )
    return out_ids


_ROPE_COS_SIN_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_INT2_SPARSE_CODEBOOK_CACHE: dict[torch.device, torch.Tensor] = {}
_E2M1_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_int2_sparse_codebook(device: torch.device) -> torch.Tensor:
    cb = _INT2_SPARSE_CODEBOOK_CACHE.get(device)
    if cb is None:
        cb = torch.tensor(
            [-1.0, 0.0, 0.38, 1.0], dtype=torch.float32, device=device
        )
        _INT2_SPARSE_CODEBOOK_CACHE[device] = cb
    return cb


def _get_e2m1_lut(device: torch.device) -> torch.Tensor:
    lut = _E2M1_LUT_CACHE.get(device)
    if lut is None:
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        _E2M1_LUT_CACHE[device] = lut
    return lut


def _is_dsv41_nvfp4_mla(t: torch.Tensor | None) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.uint8
        and t.ndim >= 2
        and t.shape[-1] == 288
    )


def _is_dsv41_mxfp8_swa(t: torch.Tensor | None) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.uint8
        and t.ndim >= 2
        and t.shape[-1] == 528
    )


def _is_dsv41_fp8_indexer(t: torch.Tensor | None) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == torch.uint8
        and t.ndim >= 2
        and t.shape[-1] == 132
    )


def _is_fp8_e4m3_mla(t: torch.Tensor | None) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.ndim >= 2
        and (
            t.dtype == torch.float8_e4m3fn
            or (t.dtype == torch.uint8 and t.shape[-1] in (576, 512))
        )
    )


def _is_dsv41_quant_kv(t: torch.Tensor | None) -> bool:
    return (
        _is_dsv41_nvfp4_mla(t)
        or _is_dsv41_mxfp8_swa(t)
        or _is_dsv41_fp8_indexer(t)
        or _is_fp8_e4m3_mla(t)
    )


def _dequant_dsv41_kv(raw: torch.Tensor) -> torch.Tensor:
    """Dequantize DeepSeek-V4.1 `nvfp4_ds_mla` (`[..., N, 288] uint8`),
    `swa_cache` (`[..., N, 528] uint8`), `indexer.k_cache` (`[..., N, 132] uint8`),
    or `fp8_e4m3` MLA (`[..., N, 576] uint8`) into `bfloat16`.
    Returns `raw` unchanged for standard floating-point KV tensors.
    """
    if _is_fp8_e4m3_mla(raw):
        fp8_view = (
            raw
            if raw.dtype == torch.float8_e4m3fn
            else raw.view(torch.float8_e4m3fn)
        )
        return fp8_view.to(torch.bfloat16)
    if _is_dsv41_mxfp8_swa(raw):
        prefix_shape = raw.shape[:-1]
        q_fp8 = (
            raw[..., :512]
            .view(torch.float8_e4m3fn)
            .reshape(*prefix_shape, 16, 32)
            .to(torch.float32)
        )
        s_u8 = raw[..., 512:528].reshape(*prefix_shape, 16, 1)
        s_fp32 = torch.exp2(s_u8.to(torch.float32) - 127.0)
        vals = (q_fp8 * s_fp32).reshape(*prefix_shape, 1, 512)
        return vals.to(torch.bfloat16)
    if _is_dsv41_nvfp4_mla(raw):
        prefix_shape = raw.shape[:-1]
        q_bytes = raw[..., :256]
        s_bytes = raw[..., 256:288]
        s_fp32 = s_bytes.view(torch.float8_e4m3fn).to(torch.float32)
        low = (q_bytes & 0x0F).to(torch.long)
        high = ((q_bytes >> 4) & 0x0F).to(torch.long)
        nibbles = torch.stack([low, high], dim=-1).reshape(*prefix_shape, 32, 16)
        lut = _get_e2m1_lut(raw.device)
        vals = (lut[nibbles] * s_fp32.unsqueeze(-1)).reshape(
            *prefix_shape, 1, 512
        )
        return vals.to(torch.bfloat16)
    if _is_dsv41_fp8_indexer(raw):
        prefix_shape = raw.shape[:-2]
        n_states = raw.shape[-2]
        flat = raw.reshape(*prefix_shape, n_states * 132)
        q_fp8 = (
            flat[..., : n_states * 128]
            .view(torch.float8_e4m3fn)
            .reshape(*prefix_shape, n_states, 128)
            .to(torch.float32)
        )
        s_raw = flat[..., n_states * 128 :].reshape(*prefix_shape, n_states, 4)
        # On SM100 (mqa_logits_use_mx_scales=True), scale is 4 equal packed UE8M0 bytes;
        # otherwise IEEE-754 float32 (whose low 2 mantissa bytes are 0 for power-of-two scales).
        use_ue8m0 = (
            raw.is_cuda and torch.cuda.get_device_capability(raw.device)[0] >= 10
        ) or bool(
            torch.any(s_raw[..., 0] != 0)
            and torch.all(s_raw[..., 0] == s_raw[..., 1])
        )
        if use_ue8m0:
            s_fp32 = torch.exp2(s_raw[..., :1].to(torch.float32) - 127.0)
        else:
            s_fp32 = (
                s_raw.reshape(-1)
                .view(torch.float32)
                .reshape(*prefix_shape, n_states, 1)
            )
        return (q_fp8 * s_fp32).unsqueeze(-2).to(torch.bfloat16)
    return raw


def _requant_dsv41_kv(
    vals_bf16: torch.Tensor,
    template_raw: torch.Tensor | None,
    rope_only: bool = False,
) -> torch.Tensor:
    """Re-quantize `bfloat16` back into `nvfp4_ds_mla` (`[..., N, 288] uint8`),
    `swa_cache` (`[..., N, 528] uint8`), `indexer.k_cache` (`[..., N, 132] uint8`),
    or `fp8_e4m3` MLA (`[..., N, 576] uint8`) matching `template_raw`.
    """
    if template_raw is None or not _is_dsv41_quant_kv(template_raw):
        return vals_bf16
    if _is_fp8_e4m3_mla(template_raw):
        q_fp8 = vals_bf16.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return (
            q_fp8
            if template_raw.dtype == torch.float8_e4m3fn
            else q_fp8.view(torch.uint8)
        )
    if _is_dsv41_mxfp8_swa(template_raw):
        prefix_shape = vals_bf16.shape[:-2]
        x = vals_bf16.to(torch.float32).reshape(*prefix_shape, 16, 32)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        exponent = torch.ceil(torch.log2(amax / 448.0))
        scale = torch.exp2(exponent)
        q_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_bytes = q_fp8.view(torch.uint8).reshape(*prefix_shape, 512)
        u_bytes = (
            (exponent + 127.0)
            .clamp(0.0, 254.0)
            .to(torch.uint8)
            .reshape(*prefix_shape, 16)
        )
        return torch.cat([q_bytes, u_bytes], dim=-1)
    if _is_dsv41_nvfp4_mla(template_raw):
        prefix_shape = vals_bf16.shape[:-2]
        if rope_only:
            x_r = vals_bf16[..., 448:].to(torch.float32).reshape(*prefix_shape, 4, 16)
            amax_r = x_r.abs().amax(dim=-1, keepdim=True)
            scale_r = (amax_r / 6.0).clamp(0.001953125, 448.0).to(torch.float8_e4m3fn)
            s_div_r = scale_r.to(torch.float32)
            s_div_r = torch.where(
                s_div_r > 0.0, s_div_r, torch.ones_like(s_div_r)
            )
            norm_r = x_r / s_div_r
            abs_r = norm_r.abs()
            sign_r = torch.where(
                norm_r < 0.0,
                torch.full_like(abs_r, 8, dtype=torch.uint8),
                torch.zeros_like(abs_r, dtype=torch.uint8),
            )
            mag_r = torch.zeros_like(abs_r, dtype=torch.uint8)
            mag_r = torch.where(abs_r >= 0.25, torch.full_like(mag_r, 1), mag_r)
            mag_r = torch.where(abs_r >= 0.75, torch.full_like(mag_r, 2), mag_r)
            mag_r = torch.where(abs_r >= 1.25, torch.full_like(mag_r, 3), mag_r)
            mag_r = torch.where(abs_r >= 1.75, torch.full_like(mag_r, 4), mag_r)
            mag_r = torch.where(abs_r >= 2.50, torch.full_like(mag_r, 5), mag_r)
            mag_r = torch.where(abs_r >= 3.50, torch.full_like(mag_r, 6), mag_r)
            mag_r = torch.where(abs_r >= 5.00, torch.full_like(mag_r, 7), mag_r)
            q_r = (mag_r | sign_r).reshape(*prefix_shape, 32, 2)
            packed_r = q_r[..., 0] | (q_r[..., 1] << 4)
            scale_bytes_r = scale_r.view(torch.uint8).reshape(*prefix_shape, 4)
            return torch.cat(
                [
                    template_raw[..., :224],
                    packed_r,
                    template_raw[..., 256:284],
                    scale_bytes_r,
                ],
                dim=-1,
            )
        x = vals_bf16.to(torch.float32).reshape(*prefix_shape, 32, 16)
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale_fp8 = (amax / 6.0).clamp(0.001953125, 448.0).to(torch.float8_e4m3fn)
        scale_div = scale_fp8.to(torch.float32)
        scale_div = torch.where(
            scale_div > 0.0, scale_div, torch.ones_like(scale_div)
        )
        norm = x / scale_div
        abs_n = norm.abs()
        sign_bit = torch.where(
            norm < 0.0,
            torch.full_like(abs_n, 8, dtype=torch.uint8),
            torch.zeros_like(abs_n, dtype=torch.uint8),
        )
        mag = torch.zeros_like(abs_n, dtype=torch.uint8)
        mag = torch.where(abs_n >= 0.25, torch.full_like(mag, 1), mag)
        mag = torch.where(abs_n >= 0.75, torch.full_like(mag, 2), mag)
        mag = torch.where(abs_n >= 1.25, torch.full_like(mag, 3), mag)
        mag = torch.where(abs_n >= 1.75, torch.full_like(mag, 4), mag)
        mag = torch.where(abs_n >= 2.50, torch.full_like(mag, 5), mag)
        mag = torch.where(abs_n >= 3.50, torch.full_like(mag, 6), mag)
        mag = torch.where(abs_n >= 5.00, torch.full_like(mag, 7), mag)
        q = (mag | sign_bit).reshape(*prefix_shape, 256, 2)
        packed_bytes = q[..., 0] | (q[..., 1] << 4)
        scale_bytes = scale_fp8.view(torch.uint8).reshape(*prefix_shape, 32)
        return torch.cat([packed_bytes, scale_bytes], dim=-1)
    if _is_dsv41_fp8_indexer(template_raw):
        prefix_shape = vals_bf16.shape[:-3]
        n_states = vals_bf16.shape[-3]
        x = vals_bf16.to(torch.float32).reshape(*prefix_shape, n_states, 128)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        exponent = torch.ceil(torch.log2(amax / 448.0))
        scale = torch.exp2(exponent)
        q_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_bytes = q_fp8.view(torch.uint8).reshape(*prefix_shape, n_states * 128)
        use_ue8m0 = (
            vals_bf16.is_cuda
            and torch.cuda.get_device_capability(vals_bf16.device)[0] >= 10
        )
        if use_ue8m0:
            u = (exponent + 127.0).clamp(0.0, 254.0).to(torch.uint8)
            s_bytes = u.expand(*prefix_shape, n_states, 4).reshape(
                *prefix_shape, n_states * 4
            )
        else:
            s_bytes = (
                scale.to(torch.float32)
                .view(torch.uint8)
                .reshape(*prefix_shape, n_states * 4)
            )
        return torch.cat([q_bytes, s_bytes], dim=-1).reshape(
            *prefix_shape, n_states, 132
        )
    return vals_bf16


def _get_cached_rope_cos_sin(
    device: torch.device,
    delta_pos: int,
    r_dim: int,
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict[str, Any] | None,
    k_rot_ndim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r_type = (
        (rope_scaling.get("rope_type") or rope_scaling.get("type"))
        if isinstance(rope_scaling, dict)
        else None
    )
    scale_key = (
        (
            r_type,
            float(rope_scaling.get("factor", 1.0)),
            float(rope_scaling.get("partial_rotary_factor", 1.0)),
            int(rope_scaling.get("original_max_position_embeddings", 4096)),
            float(rope_scaling.get("beta_fast", 32.0)),
            float(rope_scaling.get("beta_slow", 1.0)),
            bool(rope_scaling.get("truncate", True)),
        )
        if isinstance(rope_scaling, dict)
        else None
    )
    cache_key = (
        device,
        int(delta_pos),
        r_dim,
        head_dim,
        float(rope_theta),
        scale_key,
        k_rot_ndim,
    )
    cached = _ROPE_COS_SIN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    half_dim = r_dim // 2
    if delta_pos == 0:
        shape = (1, 1, half_dim) if k_rot_ndim == 3 else (1, half_dim)
        cos_val = torch.ones(shape, dtype=torch.float32, device=device)
        sin_val = torch.zeros(shape, dtype=torch.float32, device=device)
        _ROPE_COS_SIN_CACHE[cache_key] = (cos_val, sin_val)
        return cos_val, sin_val

    if r_type == "proportional" and isinstance(rope_scaling, dict):
        partial_factor = float(rope_scaling.get("partial_rotary_factor", 1.0))
        rope_angles = int(partial_factor * head_dim) // 2
        nope_angles = half_dim - rope_angles
        freq_seq = torch.arange(0, rope_angles, dtype=torch.float32, device=device)
        pos_freqs = rope_theta ** (freq_seq / float(half_dim))
        inv_freq_rot = 1.0 / pos_freqs
        inv_freq = (
            torch.cat(
                [
                    inv_freq_rot,
                    torch.zeros(nope_angles, dtype=torch.float32, device=device),
                ]
            )
            if nope_angles > 0
            else inv_freq_rot
        )
    else:
        freq_seq = torch.arange(0, half_dim, dtype=torch.float32, device=device)
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

    angles = (float(delta_pos) * inv_freq).unsqueeze(0)
    if k_rot_ndim == 3:
        angles = angles.unsqueeze(1)
    cos_val = torch.cos(angles)
    sin_val = torch.sin(angles)
    if len(_ROPE_COS_SIN_CACHE) > 4096:
        _ROPE_COS_SIN_CACHE.clear()
    _ROPE_COS_SIN_CACHE[cache_key] = (cos_val, sin_val)
    return cos_val, sin_val


def apply_relative_rope_rotation(
    k_tensor: torch.Tensor,
    delta_pos: int | torch.Tensor,
    rope_theta: float = 10000.0,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    rope_at_tail: bool = False,
    rope_pos_scale: float = 1.0,
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
        rope_at_tail: True when RoPE is applied to the last `rotary_dim` dimensions
            (e.g. DeepSeek MLA `NoPE | RoPE`).
        rope_pos_scale: Position scaling factor (e.g. 0.5 for 2x token-compressed MLA).

    Returns:
        Rotated key tensor of the same shape and dtype as k_tensor.
    """
    if rope_pos_scale != 1.0:
        if isinstance(delta_pos, int):
            delta_pos = int(round(delta_pos * rope_pos_scale))
        else:
            delta_pos = delta_pos.to(torch.float32) * float(rope_pos_scale)
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
    if rope_at_tail and r_dim < head_dim:
        k_pass = k_fp32[..., : head_dim - r_dim]
        k_rot = k_fp32[..., head_dim - r_dim :]
    else:
        k_rot = k_fp32[..., :r_dim]
        k_pass = k_fp32[..., r_dim:]
    half_dim = r_dim // 2

    if isinstance(delta_pos, int):
        cos_val, sin_val = _get_cached_rope_cos_sin(
            device=k_tensor.device,
            delta_pos=delta_pos,
            r_dim=r_dim,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            k_rot_ndim=k_rot.ndim,
        )
    else:
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

        pos_tensor = delta_pos.to(dtype=torch.float32, device=k_tensor.device)
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
        if rope_at_tail:
            k_out = torch.cat([k_pass, k_rot_out], dim=-1)
        else:
            k_out = torch.cat([k_rot_out, k_pass], dim=-1)
    else:
        k_out = k_rot_out
    return k_out.to(orig_dtype)


def apply_relative_rope_rotation_batch(
    k_batch: torch.Tensor,
    delta_pos_list: list[int],
    rope_theta: float = 10000.0,
    rotary_dim: int | None = None,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    rope_at_tail: bool = False,
    rope_pos_scale: float = 1.0,
) -> torch.Tensor:
    """Apply relative RoPE rotation across a batch of N blocks `[N, *block_shape]`."""
    if rope_pos_scale != 1.0:
        delta_pos_list = [int(round(d * rope_pos_scale)) for d in delta_pos_list]
    n = len(delta_pos_list)
    if n == 0 or all(d == 0 for d in delta_pos_list):
        return k_batch.clone()
    if n == 1:
        return apply_relative_rope_rotation(
            k_batch[0],
            delta_pos=delta_pos_list[0],
            rope_theta=rope_theta,
            rotary_dim=rotary_dim,
            is_neox_style=is_neox_style,
            rope_scaling=rope_scaling,
            rope_at_tail=rope_at_tail,
            rope_pos_scale=1.0,
        ).unsqueeze(0)

    orig_dtype = k_batch.dtype
    head_dim = k_batch.shape[-1]
    r_type = (
        (rope_scaling.get("rope_type") or rope_scaling.get("type"))
        if isinstance(rope_scaling, dict)
        else None
    )
    if r_type == "proportional":
        r_dim = 0 if (rotary_dim is not None and rotary_dim <= 0) else head_dim
    else:
        r_dim = head_dim if rotary_dim is None else min(rotary_dim, head_dim)
    if r_dim <= 0:
        return k_batch.clone()

    k_fp32 = k_batch.to(torch.float32)
    if rope_at_tail and r_dim < head_dim:
        k_pass = k_fp32[..., : head_dim - r_dim]
        k_rot = k_fp32[..., head_dim - r_dim :]
    else:
        k_rot = k_fp32[..., :r_dim]
        k_pass = k_fp32[..., r_dim:]
    half_dim = r_dim // 2
    single_ndim = k_rot.ndim - 1

    cos_list: list[torch.Tensor] = []
    sin_list: list[torch.Tensor] = []
    for d in delta_pos_list:
        c, s = _get_cached_rope_cos_sin(
            device=k_batch.device,
            delta_pos=d,
            r_dim=r_dim,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            k_rot_ndim=single_ndim,
        )
        cos_list.append(c)
        sin_list.append(s)
    cos_batch = torch.stack(cos_list, dim=0)
    sin_batch = torch.stack(sin_list, dim=0)

    if is_neox_style:
        x1 = k_rot[..., :half_dim]
        x2 = k_rot[..., half_dim : 2 * half_dim]
        out1 = x1 * cos_batch - x2 * sin_batch
        out2 = x1 * sin_batch + x2 * cos_batch
        k_rot_out = torch.cat([out1, out2], dim=-1)
    else:
        x1 = k_rot[..., 0::2]
        x2 = k_rot[..., 1::2]
        out1 = x1 * cos_batch - x2 * sin_batch
        out2 = x1 * sin_batch + x2 * cos_batch
        k_rot_out = torch.stack([out1, out2], dim=-1).flatten(-2)

    if r_dim < head_dim:
        if rope_at_tail:
            k_out = torch.cat([k_pass, k_rot_out], dim=-1)
        else:
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

    Uses sub-head group quantization:
      - INT4: zero-preserving 16-level mid-tread [-8, +7] mapped to unsigned [0, 15]
        (exact 0.0 at code 8, exact ±max_abs at codes 1 and 15), 2 elems per byte.
      - INT2: crest-adaptive 4-level quantizer (4 elems per byte) using the sign bit
        of the fp16 group scale to select between:
          * Low-crest dense mode (scale > 0): symmetric {-1.5, -0.5, +0.5, +1.5} * scale
          * High-crest outlier mode (scale < 0): zero-preserving {-1.0, 0.0, +0.38, +1.0} * |scale|
            (exact 0.0 at code 1, exact ±max_abs at codes 0 and 3).
    """
    if bits not in (2, 4):
        raise ValueError(f"Unsupported bit-width: {bits}. Expected 2 or 4.")

    orig_shape = tuple(residual.shape)
    orig_dtype = residual.dtype
    res_fp32 = residual.to(torch.float32)

    # Reshape to [num_groups, group_size]. Cap group_size at 32 for INT4 (64 for INT2)
    # so large-head_dim architectures (e.g. Gemma 4 head_dim=256/512 with learned k_norm
    # or unscaled v_norm) do not suffer from per-head outlier channel scale inflation.
    total_elems = res_fp32.numel()
    group_size = orig_shape[-1]
    max_group_size = 32 if bits == 4 else 64
    if group_size > max_group_size:
        group_size = math.gcd(total_elems, max_group_size) or group_size
    elif group_size < 32 and total_elems >= 32:
        group_size = math.gcd(total_elems, 128)
        if group_size < 16:
            group_size = math.gcd(total_elems, 64) or orig_shape[-1]
    flat_groups = res_fp32.reshape(-1, group_size)
    max_abs = flat_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    if bits == 4:
        # 16-level zero-preserving mid-tread quantizer:
        # levels {-8.0, -7.0, ..., 0.0, ..., +7.0} * scale (exact 0.0 at q=8)
        scales_fp32 = max_abs / 7.0
        normalized = flat_groups / scales_fp32
        q_unsigned = (
            torch.round(normalized + 8.0).clamp(0.0, 15.0).to(torch.uint8).reshape(-1)
        )
        if q_unsigned.numel() % 2 != 0:
            q_unsigned = torch.cat(
                [q_unsigned, torch.zeros(1, dtype=torch.uint8, device=q_unsigned.device)]
            )
        low_nibble = q_unsigned[0::2] & 0x0F
        high_nibble = (q_unsigned[1::2] & 0x0F) << 4
        packed = low_nibble | high_nibble
    else:
        # Crest-adaptive 4-level INT2 quantizer:
        # High-crest groups (crest > 3.2, e.g. Gemma 4 outlier V/K channels) use negative
        # scale to select zero-preserving codebook [-1.0, 0.0, +0.38, +1.0] * |scale|.
        # Low-crest groups (crest <= 3.2) use positive scale with Lloyd-Max [-1.5, -0.5, +0.5, +1.5].
        rms = torch.sqrt(torch.mean(flat_groups * flat_groups, dim=-1, keepdim=True)).clamp(
            min=1e-8
        )
        is_sparse = (max_abs / rms) > 3.2

        dense_scale = max_abs / 2.15
        norm_dense = flat_groups / dense_scale
        q_dense = torch.floor(norm_dense + 2.0).clamp(0.0, 3.0).to(torch.uint8)

        norm_sparse = flat_groups / max_abs
        q_sparse = torch.ones_like(q_dense)  # code 1 -> 0.0
        q_sparse = torch.where(norm_sparse < -0.50, torch.zeros_like(q_sparse), q_sparse)
        q_sparse = torch.where(
            norm_sparse >= 0.19, torch.full_like(q_sparse, 2), q_sparse
        )
        q_sparse = torch.where(
            norm_sparse >= 0.69, torch.full_like(q_sparse, 3), q_sparse
        )

        scales_fp32 = torch.where(is_sparse, -max_abs, dense_scale)
        q_unsigned = torch.where(is_sparse, q_sparse, q_dense).reshape(-1)
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
        levels = unpacked.to(torch.float32) - 8.0
        flat_groups = levels.reshape(-1, group_size)
        recon = flat_groups * scales_fp32
    elif packed_tensor.bits == 2:
        c0 = packed & 0x03
        c1 = (packed >> 2) & 0x03
        c2 = (packed >> 4) & 0x03
        c3 = (packed >> 6) & 0x03
        unpacked = torch.stack([c0, c1, c2, c3], dim=-1).reshape(-1)[:total_elems]
        u_groups = unpacked.reshape(-1, group_size)
        dense_levels = u_groups.to(torch.float32) - 1.5
        sparse_codebook = _get_int2_sparse_codebook(packed.device)
        sparse_levels = sparse_codebook[u_groups.to(torch.long)]
        is_sparse = scales_fp32 < 0.0
        levels = torch.where(is_sparse, sparse_levels, dense_levels)
        recon = levels * scales_fp32.abs()
    else:
        raise ValueError(f"Unsupported bit-width: {packed_tensor.bits}")

    return recon.reshape(packed_tensor.orig_shape).to(packed_tensor.orig_dtype)


def quantize_residual_packed_batch(
    residual_batch: torch.Tensor, bits: int = 4
) -> list[PackedQuantizedTensor]:
    """Quantize a batch of N residual tensors `[N, *orig_shape]` in a single fused pass."""
    n = int(residual_batch.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [quantize_residual_packed(residual_batch[0], bits=bits)]
    if bits not in (2, 4):
        raise ValueError(f"Unsupported bit-width: {bits}. Expected 2 or 4.")

    orig_shape = tuple(residual_batch.shape[1:])
    orig_dtype = residual_batch.dtype
    res_fp32 = residual_batch.to(torch.float32)

    total_elems = math.prod(orig_shape)
    group_size = orig_shape[-1]
    max_group_size = 32 if bits == 4 else 64
    if group_size > max_group_size:
        group_size = math.gcd(total_elems, max_group_size) or group_size
    elif group_size < 32 and total_elems >= 32:
        group_size = math.gcd(total_elems, 128)
        if group_size < 16:
            group_size = math.gcd(total_elems, 64) or orig_shape[-1]
    num_groups = total_elems // group_size
    flat_groups = res_fp32.reshape(n * num_groups, group_size)
    max_abs = flat_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

    if bits == 4:
        scales_fp32 = max_abs / 7.0
        normalized = flat_groups / scales_fp32
        q_unsigned = (
            torch.round(normalized + 8.0)
            .clamp(0.0, 15.0)
            .to(torch.uint8)
            .reshape(n, total_elems)
        )
        if total_elems % 2 != 0:
            q_unsigned = torch.cat(
                [
                    q_unsigned,
                    torch.zeros(
                        (n, 1), dtype=torch.uint8, device=q_unsigned.device
                    ),
                ],
                dim=-1,
            )
        low_nibble = q_unsigned[:, 0::2] & 0x0F
        high_nibble = (q_unsigned[:, 1::2] & 0x0F) << 4
        packed_batch = low_nibble | high_nibble
    else:
        rms = torch.sqrt(
            torch.mean(flat_groups * flat_groups, dim=-1, keepdim=True)
        ).clamp(min=1e-8)
        is_sparse = (max_abs / rms) > 3.2

        dense_scale = max_abs / 2.15
        norm_dense = flat_groups / dense_scale
        q_dense = torch.floor(norm_dense + 2.0).clamp(0.0, 3.0).to(torch.uint8)

        norm_sparse = flat_groups / max_abs
        q_sparse = torch.ones_like(q_dense)
        q_sparse = torch.where(
            norm_sparse < -0.50, torch.zeros_like(q_sparse), q_sparse
        )
        q_sparse = torch.where(
            norm_sparse >= 0.19, torch.full_like(q_sparse, 2), q_sparse
        )
        q_sparse = torch.where(
            norm_sparse >= 0.69, torch.full_like(q_sparse, 3), q_sparse
        )

        scales_fp32 = torch.where(is_sparse, -max_abs, dense_scale)
        q_unsigned = torch.where(is_sparse, q_sparse, q_dense).reshape(
            n, total_elems
        )
        rem = total_elems % 4
        if rem != 0:
            q_unsigned = torch.cat(
                [
                    q_unsigned,
                    torch.zeros(
                        (n, 4 - rem), dtype=torch.uint8, device=q_unsigned.device
                    ),
                ],
                dim=-1,
            )
        packed_batch = (
            (q_unsigned[:, 0::4] & 0x03)
            | ((q_unsigned[:, 1::4] & 0x03) << 2)
            | ((q_unsigned[:, 2::4] & 0x03) << 4)
            | ((q_unsigned[:, 3::4] & 0x03) << 6)
        )

    scales_f16 = scales_fp32.to(torch.float16).reshape(n, num_groups, 1)
    return [
        PackedQuantizedTensor(
            packed_data=packed_batch[i].clone(),
            scales=scales_f16[i].clone(),
            orig_shape=orig_shape,
            orig_dtype=orig_dtype,
            bits=bits,
        )
        for i in range(n)
    ]


def dequantize_residual_packed_batch(
    packed_list: list[PackedQuantizedTensor],
) -> torch.Tensor:
    """Dequantize a batch of N PackedQuantizedTensors to `[N, *orig_shape]`."""
    n = len(packed_list)
    if n == 1:
        return dequantize_residual_packed(packed_list[0]).unsqueeze(0)
    first = packed_list[0]
    orig_shape = first.orig_shape
    orig_dtype = first.orig_dtype
    bits = first.bits
    total_elems = math.prod(orig_shape)

    packed = torch.stack([p.packed_data for p in packed_list], dim=0)
    scales_fp32 = torch.stack([p.scales for p in packed_list], dim=0).to(
        torch.float32
    )
    num_groups = scales_fp32.shape[1]
    group_size = total_elems // max(num_groups, 1)

    if bits == 4:
        low_nibble = packed & 0x0F
        high_nibble = (packed >> 4) & 0x0F
        unpacked = torch.stack([low_nibble, high_nibble], dim=-1).reshape(
            n, -1
        )[:, :total_elems]
        levels = unpacked.to(torch.float32) - 8.0
        flat_groups = levels.reshape(n, num_groups, group_size)
        recon = flat_groups * scales_fp32
    elif bits == 2:
        c0 = packed & 0x03
        c1 = (packed >> 2) & 0x03
        c2 = (packed >> 4) & 0x03
        c3 = (packed >> 6) & 0x03
        unpacked = torch.stack([c0, c1, c2, c3], dim=-1).reshape(n, -1)[
            :, :total_elems
        ]
        u_groups = unpacked.reshape(n, num_groups, group_size)
        dense_levels = u_groups.to(torch.float32) - 1.5
        sparse_codebook = _get_int2_sparse_codebook(packed.device)
        sparse_levels = sparse_codebook[u_groups.to(torch.long)]
        is_sparse = scales_fp32 < 0.0
        levels = torch.where(is_sparse, sparse_levels, dense_levels)
        recon = levels * scales_fp32.abs()
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")

    return recon.reshape(n, *orig_shape).to(orig_dtype)


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
    prev_token_block_hash: str = ""

    # For UNCOMPRESSED_BASE:
    raw_k: torch.Tensor | None = None
    raw_v: torch.Tensor | None = None

    # For TIER0 / TIER1 / TIER2 / TIER3:
    # Single-hop reference pointer(s) to UNCOMPRESSED_BASE block(s).
    # When `ref_base_hash_2` is set and `0 < stitch_offset < block_size`,
    # the reference KV is stitched across `ref_base_hash[stitch_offset:]`
    # and `ref_base_hash_2[:stitch_offset]` (Approach 1 cross-block stitching).
    ref_base_hash: str | None = None
    delta_pos: int = 0
    ref_base_hash_2: str | None = None
    stitch_offset: int = 0

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
    evicted_blocks: int = 0
    evicted_unique_blocks: int = 0
    evicted_bytes: int = 0
    tier_counts: dict[CompressionTier, int] = field(
        default_factory=lambda: {tier: 0 for tier in CompressionTier}
    )
    cumulative_tier_counts: dict[CompressionTier, int] = field(
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
    rope_at_tail: bool = False
    rope_pos_scale: float = 1.0
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
        force_uncompressed: bool = False,
        pad_token: str = DEFAULT_PAD_TOKEN,
        pad_token_id: int | None = None,
        pad_token_map: dict[str, str] | None = None,
        enable_server_block_align: bool = False,
        max_store_bytes: int = 0,
        enable_cross_turn_reuse: bool = False,
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
        self.force_uncompressed = force_uncompressed
        self.max_store_bytes = max(0, int(max_store_bytes))
        self.enable_cross_turn_reuse = bool(enable_cross_turn_reuse)
        # Approach 2: Configurable server-side block alignment padding parameters
        self.pad_token = pad_token if pad_token else DEFAULT_PAD_TOKEN
        self.pad_token_id = int(pad_token_id) if pad_token_id is not None else None
        self.pad_token_map = dict(pad_token_map) if pad_token_map else None
        self.enable_server_block_align = bool(enable_server_block_align)

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
        self._base_by_token_hash_list: dict[
            tuple[str, str], list[CompressedKVBlockRecord]
        ] = {}

        # Secondary list of UNCOMPRESSED_BASE blocks per layer for R2 partial-edit matching:
        # layer_name -> list[CompressedKVBlockRecord]
        self._base_blocks_by_layer: dict[str, list[CompressedKVBlockRecord]] = {}
        # O(1) Pigeonhole sub-chunk index for R2 partial-edit matching (<= 4 edits in 16 tokens):
        # (layer_name, slice_idx, token_subtuple) -> list[CompressedKVBlockRecord]
        self._base_by_subchunk: dict[
            tuple[str, int, tuple[int, ...]], list[CompressedKVBlockRecord]
        ] = {}
        # Approach 1: O(1) boundary w-gram and position indices for cross-block reference
        # stitching across unaligned token shifts (delta_p != 0 mod block_size):
        self._base_by_layer_pos: dict[
            tuple[str, int], list[CompressedKVBlockRecord]
        ] = {}
        self._base_by_suffix_wgram: dict[
            tuple[str, tuple[int, ...]], list[CompressedKVBlockRecord]
        ] = {}
        self._base_by_prefix_wgram: dict[
            tuple[str, tuple[int, ...]], list[CompressedKVBlockRecord]
        ] = {}
        self._layers_with_unaligned_bases: set[str] = set()

        # Prefix hash registry at scheduler level: full_block_hash -> (token_ids, start_pos)
        self._known_prefix_blocks: dict[str, tuple[tuple[int, ...], int]] = {}
        # Token hash registry at scheduler level: token_block_hash -> full_block_hash
        self._known_token_blocks: dict[str, str] = {}
        # Scheduler-level pigeonhole sub-chunk index: (slice_idx, sub_tuple) -> [(full_hash, token_ids, start_pos)]
        self._known_subchunks: dict[
            tuple[int, tuple[int, ...]], list[tuple[str, tuple[int, ...], int]]
        ] = {}
        # Worker-level (layer_name, token_block_hash, start_position) -> anchor CompressedKVBlockRecord
        self._record_by_tok_pos: dict[
            tuple[str, str, int], CompressedKVBlockRecord
        ] = {}
        self._tok_pos_anchor_blocks: set[str] = set()
        # Set of full_block_hashes materialized in Mamba/GDN recurrent state layers
        self._known_mamba_blocks: set[str] = set()
        self._has_mamba_config: bool = False
        self._clear_epoch: int = 0
        # Compact GPU HBM byte-slab sub-allocator to avoid PyTorch CUDACachingAllocator's
        # 512-byte minimum allocation rounding on small quantized scale/index tensors:
        self._gpu_slabs: dict[torch.device, list[torch.Tensor]] = {}
        self._gpu_slab_offsets: dict[torch.device, int] = {}
        self._gpu_slabs_by_block: dict[tuple[torch.device, str], list[torch.Tensor]] = {}
        self._gpu_slab_offsets_by_block: dict[tuple[torch.device, str], int] = {}
        # LRU ordering (oldest first) and base->dependent reference tracking for eviction:
        self._block_lru: dict[str, None] = {}
        self._demoted_blocks: set[str] = set()
        self._retired_blocks: set[str] = set()
        self._attention_base_blocks: set[str] = set()
        self._pos0_base_blocks: set[str] = set()
        self._dependent_blocks: set[str] = set()
        self._active_by_root_pos: dict[tuple[str, int], str] = {}
        self._dependents_by_base: dict[str, set[str]] = {}
        self._record_raw_bytes: dict[tuple[str, str], int] = {}
        self._known_layers: set[str] = set()
        self._known_slab_devices: set[torch.device] = set()
        self._partial_match_cache: dict[
            tuple[Any, ...], tuple[str | None, list[int]]
        ] = {}
        self._stitched_match_cache: dict[
            tuple[Any, ...], tuple[str | None, str | None, int, list[int]]
        ] = {}
        self._pigeonhole_slice_cache: dict[
            tuple[str, int], list[tuple[int, tuple[int, ...]]]
        ] = {}

        self.stats = DeltaCompressionStats()

    def clear(self) -> None:
        """Clear all stored KV block records and reset telemetry while preserving layer policies."""
        self._records.clear()
        self._base_by_token_hash.clear()
        self._base_by_token_hash_list.clear()
        self._base_blocks_by_layer.clear()
        self._base_by_subchunk.clear()
        self._base_by_layer_pos.clear()
        self._base_by_suffix_wgram.clear()
        self._base_by_prefix_wgram.clear()
        self._layers_with_unaligned_bases.clear()
        self._known_prefix_blocks.clear()
        self._known_token_blocks.clear()
        self._known_subchunks.clear()
        self._record_by_tok_pos.clear()
        self._tok_pos_anchor_blocks.clear()
        self._known_mamba_blocks.clear()
        self._gpu_slabs.clear()
        self._gpu_slab_offsets.clear()
        self._gpu_slabs_by_block.clear()
        self._gpu_slab_offsets_by_block.clear()
        self._block_lru.clear()
        self._demoted_blocks.clear()
        self._retired_blocks.clear()
        self._attention_base_blocks.clear()
        self._pos0_base_blocks.clear()
        self._dependent_blocks.clear()
        self._active_by_root_pos.clear()
        self._dependents_by_base.clear()
        self._record_raw_bytes.clear()
        self._known_layers.clear()
        self._known_slab_devices.clear()
        self._partial_match_cache.clear()
        self._stitched_match_cache.clear()
        self._pigeonhole_slice_cache.clear()
        self._clear_epoch += 1
        self.stats = DeltaCompressionStats()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _touch_lru(
        self,
        full_block_hash: str,
        ref_base_hash: str | None = None,
        ref_base_hash_2: str | None = None,
    ) -> None:
        """Promote `full_block_hash` (and its `ref_base_hash` / `ref_base_hash_2` after it) to most-recently-used."""
        self._demoted_blocks.discard(full_block_hash)
        self._retired_blocks.discard(full_block_hash)
        self._block_lru.pop(full_block_hash, None)
        self._block_lru[full_block_hash] = None
        if ref_base_hash and ref_base_hash != full_block_hash:
            self._demoted_blocks.discard(ref_base_hash)
            self._retired_blocks.discard(ref_base_hash)
            self._block_lru.pop(ref_base_hash, None)
            self._block_lru[ref_base_hash] = None
        if (
            ref_base_hash_2
            and ref_base_hash_2 != full_block_hash
            and ref_base_hash_2 != ref_base_hash
        ):
            self._demoted_blocks.discard(ref_base_hash_2)
            self._retired_blocks.discard(ref_base_hash_2)
            self._block_lru.pop(ref_base_hash_2, None)
            self._block_lru[ref_base_hash_2] = None

    def resolve_pad_token_id(
        self,
        tokenizer: Any = None,
        content_text: str | None = None,
        pad_token: str | None = None,
        pad_token_id: int | None = None,
    ) -> int:
        """Resolve the padding token ID using this store's configured `pad_token` / `pad_token_map`."""
        return resolve_pad_token_id(
            tokenizer=tokenizer,
            pad_token=pad_token if pad_token is not None else self.pad_token,
            pad_token_id=(
                pad_token_id if pad_token_id is not None else self.pad_token_id
            ),
            content_text=content_text,
            pad_token_map=self.pad_token_map,
        )

    def align_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        pad_token_id: int | None = None,
        tokenizer: Any = None,
        content_text: str | None = None,
        target_blocks: int | None = None,
    ) -> list[int]:
        """Pad `token_ids` to an exact multiple of `self.block_size` (Approach 2)."""
        resolved_pad_id = (
            int(pad_token_id)
            if pad_token_id is not None
            else self.resolve_pad_token_id(
                tokenizer=tokenizer, content_text=content_text
            )
        )
        return align_tokens_to_block(
            token_ids=token_ids,
            block_size=self.block_size,
            pad_token_id=resolved_pad_id,
            target_blocks=target_blocks,
        )

    def encode_aligned_segment(
        self,
        tokenizer: Any,
        text: str,
        bucket_file_viewer: bool = True,
        target_blocks: int | None = None,
        pad_token: str | None = None,
        pad_token_id: int | None = None,
        line_step: int = 10,
    ) -> list[int]:
        """Encode and block-align a structural prompt segment on the server side (Approach 2)."""
        eff_pad_token = pad_token if pad_token is not None else self.pad_token
        eff_pad_id = (
            pad_token_id if pad_token_id is not None else self.pad_token_id
        )
        if bucket_file_viewer and "[File: " in text and target_blocks is None:
            return encode_file_viewer_bucketed_server(
                tokenizer=tokenizer,
                text=text,
                block_size=self.block_size,
                pad_token=eff_pad_token,
                pad_token_id=eff_pad_id,
                pad_token_map=self.pad_token_map,
                line_step=line_step,
            )
        resolved_pad_id = self.resolve_pad_token_id(
            tokenizer=tokenizer,
            content_text=text,
            pad_token=eff_pad_token,
            pad_token_id=eff_pad_id,
        )
        return align_tokens_to_block(
            token_ids=_encode_text_without_special_tokens(tokenizer, text),
            block_size=self.block_size,
            pad_token_id=resolved_pad_id,
            target_blocks=target_blocks,
        )

    def align_structural_prompt(
        self,
        tokenizer: Any,
        segments: list[tuple[str, bool]] | list[str],
        bucket_file_viewer: bool = True,
        pad_token: str | None = None,
        pad_token_id: int | None = None,
    ) -> list[int]:
        """Tokenize and block-align a sequence of structural prompt segments (Approach 2)."""
        return align_structural_prompt_server(
            tokenizer=tokenizer,
            segments=segments,
            block_size=self.block_size,
            pad_token=pad_token if pad_token is not None else self.pad_token,
            pad_token_id=(
                pad_token_id if pad_token_id is not None else self.pad_token_id
            ),
            pad_token_map=self.pad_token_map,
            bucket_file_viewer=bucket_file_viewer,
        )

    def demote_lru(self, full_block_hash: str) -> None:
        """Mark `full_block_hash` as demoted while preserving chronological LRU order."""
        if full_block_hash in self._block_lru:
            self._demoted_blocks.add(full_block_hash)

    def retire_lru(self, full_block_hash: str) -> None:
        """Mark `full_block_hash` as belonging to a completed trajectory's superseded history."""
        if full_block_hash in self._block_lru:
            self._demoted_blocks.add(full_block_hash)
            self._retired_blocks.add(full_block_hash)

    def _pack_cuda_tensor(
        self, t: torch.Tensor, block_hash: str = ""
    ) -> torch.Tensor:
        """Pack a CUDA tensor into a contiguous uint8 slab arena when not 512B-aligned.

        Eliminates PyTorch CUDACachingAllocator's 512-byte minimum block rounding
        for small quantized scale and sparse index tensors stored in GPU HBM while
        allowing per-block reclamation during LRU eviction.
        """
        if not t.is_cuda or t.numel() == 0:
            return t
        dev = t.device
        elem_size = t.element_size()
        nbytes = t.numel() * elem_size
        if nbytes >= 512 and nbytes % 512 == 0:
            return t.contiguous()
        if block_hash:
            slab_size = 4096  # 4 KB per-block slab (exact multiple of 512B)
            if nbytes >= slab_size:
                return t.contiguous()
            self._known_slab_devices.add(dev)
            bkey = (dev, block_hash)
            slabs = self._gpu_slabs_by_block.setdefault(bkey, [])
            off = self._gpu_slab_offsets_by_block.get(bkey, slab_size)
            aligned_off = (off + elem_size - 1) & ~(elem_size - 1)
            if not slabs or aligned_off + nbytes > slab_size:
                slabs.append(torch.empty(slab_size, dtype=torch.uint8, device=dev))
                aligned_off = 0
            slab = slabs[-1]
            view = (
                slab[aligned_off : aligned_off + nbytes]
                .view(t.dtype)
                .reshape(t.shape)
            )
            view.copy_(t)
            self._gpu_slab_offsets_by_block[bkey] = aligned_off + nbytes
            return view
        slab_size = 262144  # 256 KB (exact multiple of 512B allocator block)
        if nbytes >= slab_size:
            return t.contiguous()
        slabs = self._gpu_slabs.setdefault(dev, [])
        off = self._gpu_slab_offsets.get(dev, slab_size)
        aligned_off = (off + elem_size - 1) & ~(elem_size - 1)
        if not slabs or aligned_off + nbytes > slab_size:
            slabs.append(torch.empty(slab_size, dtype=torch.uint8, device=dev))
            aligned_off = 0
        slab = slabs[-1]
        view = (
            slab[aligned_off : aligned_off + nbytes]
            .view(t.dtype)
            .reshape(t.shape)
        )
        view.copy_(t)
        self._gpu_slab_offsets[dev] = aligned_off + nbytes
        return view

    def register_scheduled_store_block(
        self,
        full_block_hash: str,
        token_ids: tuple[int, ...] | list[int],
        start_position: int,
        has_mamba_state: bool = True,
        index_subchunks: bool = True,
    ) -> None:
        """Record scheduled store block hashes on the Scheduler side for multi-process TP/PP."""
        token_tuple = tuple(int(t) for t in token_ids)
        token_hash = compute_token_block_hash(token_tuple)
        self._known_prefix_blocks[full_block_hash] = (token_tuple, start_position)
        self._known_token_blocks.setdefault(token_hash, full_block_hash)
        if (
            self.enable_cross_turn_reuse
            and index_subchunks
            and self.max_sparse_exceptions > 0
        ):
            num_slices = self.max_sparse_exceptions + 1
            if not self._has_scheduler_partial_match(
                token_tuple, start_position, self.max_sparse_exceptions
            ):
                for s_idx, sub_tup in self._get_pigeonhole_slices(
                    token_tuple, num_slices=num_slices
                ):
                    bucket = self._known_subchunks.setdefault(
                        (s_idx, sub_tup), []
                    )
                    bucket.append((full_block_hash, token_tuple, start_position))
        if self.has_mamba_layers and has_mamba_state:
            self._known_mamba_blocks.add(full_block_hash)
        self._touch_lru(full_block_hash)

    def _has_scheduler_partial_match(
        self,
        token_tuple: tuple[int, ...],
        start_position: int,
        max_edits: int,
    ) -> bool:
        if max_edits <= 0 or not self._known_subchunks:
            return False
        n = len(token_tuple)
        num_slices = max_edits + 1
        seen_hashes: set[str] = set()
        for s_idx, sub_tup in self._get_pigeonhole_slices(
            token_tuple, num_slices=num_slices
        ):
            for cand_hash, cand_tokens, cand_pos in self._known_subchunks.get(
                (s_idx, sub_tup), ()
            ):
                if cand_hash in seen_hashes or len(cand_tokens) != n:
                    continue
                seen_hashes.add(cand_hash)
                if cand_pos != start_position:
                    continue
                diffs = self._diff_indices_bounded(
                    token_tuple, cand_tokens, max_edits
                )
                if diffs is not None:
                    return True
        return False

    def has_reconstructible_block(
        self,
        token_ids: tuple[int, ...] | list[int],
        start_position: int,
        max_edits: int = 4,
    ) -> bool:
        """Return True if `token_ids` matches an indexed exact or partial base block."""
        token_tuple = tuple(int(t) for t in token_ids)
        token_hash = compute_token_block_hash(token_tuple)
        if token_hash in self._known_token_blocks:
            return True
        return self._has_scheduler_partial_match(
            token_tuple, start_position, max_edits
        )

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
        return (
            self._has_mamba_config
            or bool(self._known_mamba_blocks)
            or any(p.is_mamba for p in self.layer_policies.values())
        )

    def _find_exact_base_block(
        self,
        layer_name: str,
        token_hash: str,
        start_position: int | None = None,
        exclude_hashes: set[str] | None = None,
        prev_token_block_hash: str = "",
    ) -> CompressedKVBlockRecord | None:
        """Find the closest-position UNCOMPRESSED_BASE block with the same token_hash."""
        candidates = self._base_by_token_hash_list.get((layer_name, token_hash), [])
        if not candidates:
            single = self._base_by_token_hash.get((layer_name, token_hash))
            if single is not None:
                candidates = [single]
        valid = [
            rec
            for rec in candidates
            if not exclude_hashes or rec.full_block_hash not in exclude_hashes
        ]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        return min(
            valid,
            key=lambda r: (
                0
                if (
                    prev_token_block_hash
                    and r.prev_token_block_hash == prev_token_block_hash
                )
                else 1,
                abs(start_position - r.start_position)
                if start_position is not None
                else 0,
            ),
        )

    @staticmethod
    def _diff_indices_bounded(
        tok_a: tuple[int, ...],
        tok_b: tuple[int, ...],
        max_diffs: int,
    ) -> list[int] | None:
        """Return indices where `tok_a != tok_b` if `0 < diffs <= max_diffs`, else None.

        Uses C-level tuple slice equality to skip identical regions and abort early
        when more than `max_diffs` chunks differ.
        """
        n = len(tok_a)
        if n != len(tok_b) or max_diffs <= 0 or tok_a == tok_b:
            return None
        if n >= 32:
            num_chunks = min(8, max(max_diffs + 2, 4))
            chunk_spans: list[tuple[int, int]] = []
            for c_idx in range(num_chunks):
                c_start = (c_idx * n) // num_chunks
                c_end = ((c_idx + 1) * n) // num_chunks
                if tok_a[c_start:c_end] != tok_b[c_start:c_end]:
                    chunk_spans.append((c_start, c_end))
                    if len(chunk_spans) > max_diffs:
                        return None
            diffs: list[int] = []
            for c_start, c_end in chunk_spans:
                for i in range(c_start, c_end):
                    if tok_a[i] != tok_b[i]:
                        diffs.append(i)
                        if len(diffs) > max_diffs:
                            return None
            return diffs if diffs else None

        diffs = []
        for i, (ta, tb) in enumerate(zip(tok_a, tok_b)):
            if ta != tb:
                diffs.append(i)
                if len(diffs) > max_diffs:
                    return None
        return diffs if diffs else None

    def _find_partial_match_base_block(
        self,
        layer_name: str,
        token_ids: tuple[int, ...],
        start_position: int | None = None,
        exclude_hashes: set[str] | None = None,
        prev_token_block_hash: str = "",
        full_block_hash: str = "",
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
        layer_bases = self._base_blocks_by_layer.get(layer_name)
        if not layer_bases:
            return None, []

        fast_cache_key = (
            (
                full_block_hash,
                start_position,
                prev_token_block_hash,
                max_exceptions,
                policy.is_mamba,
                len(layer_bases),
                layer_bases[0].full_block_hash,
                layer_bases[-1].full_block_hash,
                self.stats.evicted_blocks,
            )
            if full_block_hash and not exclude_hashes
            else None
        )
        if (
            fast_cache_key is not None
            and fast_cache_key in self._partial_match_cache
        ):
            cached_base_hash, cached_diff_idx = self._partial_match_cache[
                fast_cache_key
            ]
            if cached_base_hash is None:
                return None, []
            cached_rec = self._records.get((layer_name, cached_base_hash))
            if (
                cached_rec is not None
                and cached_rec.tier == CompressionTier.UNCOMPRESSED_BASE
            ):
                return cached_rec, cached_diff_idx

        num_slices = max_exceptions + 1
        slice_key = (full_block_hash, num_slices) if full_block_hash else None
        if slice_key is not None and slice_key in self._pigeonhole_slice_cache:
            slices = self._pigeonhole_slice_cache[slice_key]
        else:
            slices = self._get_pigeonhole_slices(token_ids, num_slices)
            if slice_key is not None:
                self._pigeonhole_slice_cache[slice_key] = slices

        seen_hashes: set[str] = set()
        candidates: list[CompressedKVBlockRecord] = []
        matched_buckets: list[list[CompressedKVBlockRecord]] = []
        for s_idx, sub_tuple in slices:
            bucket = self._base_by_subchunk.get((layer_name, s_idx, sub_tuple))
            if bucket:
                matched_buckets.append(bucket)
        if not matched_buckets:
            if fast_cache_key is not None:
                self._partial_match_cache[fast_cache_key] = (None, [])
            return None, []
        matched_buckets.sort(key=len)
        for bucket in matched_buckets:
            cand_slice = (
                bucket
                if len(bucket) <= 512
                else (bucket[:128] + bucket[-256:])
            )
            for rec in cand_slice:
                if exclude_hashes and rec.full_block_hash in exclude_hashes:
                    continue
                if rec.full_block_hash not in seen_hashes:
                    seen_hashes.add(rec.full_block_hash)
                    candidates.append(rec)
        if not candidates:
            if fast_cache_key is not None:
                self._partial_match_cache[fast_cache_key] = (None, [])
            return None, []

        cand_sig = tuple(c.full_block_hash for c in candidates)
        cache_key = (
            (
                full_block_hash,
                start_position,
                prev_token_block_hash,
                max_exceptions,
                policy.is_mamba,
                cand_sig,
            )
            if full_block_hash
            else None
        )
        if cache_key is not None and cache_key in self._partial_match_cache:
            cached_base_hash, cached_diff_idx = self._partial_match_cache[cache_key]
            if cached_base_hash is None:
                return None, []
            cached_rec = self._records.get((layer_name, cached_base_hash))
            if (
                cached_rec is not None
                and cached_rec.tier == CompressionTier.UNCOMPRESSED_BASE
                and (not exclude_hashes or cached_base_hash not in exclude_hashes)
            ):
                if fast_cache_key is not None:
                    self._partial_match_cache[fast_cache_key] = (
                        cached_base_hash,
                        cached_diff_idx,
                    )
                return cached_rec, cached_diff_idx

        best_base: CompressedKVBlockRecord | None = None
        best_diff_indices: list[int] = []
        best_score: tuple[int, int, int] = (999, 999, 999999)

        for base_rec in candidates:
            if len(base_rec.token_ids) != len(token_ids):
                continue
            if (
                policy.is_mamba
                and start_position is not None
                and base_rec.start_position != start_position
                and not (
                    prev_token_block_hash
                    and base_rec.prev_token_block_hash == prev_token_block_hash
                )
            ):
                continue
            diff_idx = self._diff_indices_bounded(
                token_ids, base_rec.token_ids, max_exceptions
            )
            if diff_idx is not None:
                prev_match = (
                    0
                    if (
                        prev_token_block_hash
                        and base_rec.prev_token_block_hash == prev_token_block_hash
                    )
                    else 1
                )
                pos_dist = (
                    abs(start_position - base_rec.start_position)
                    if start_position is not None
                    else 0
                )
                score = (prev_match, len(diff_idx), pos_dist)
                if best_base is None or score < best_score:
                    best_base = base_rec
                    best_diff_indices = diff_idx
                    best_score = score
                    if prev_match == 0 and len(diff_idx) <= 2 and pos_dist == 0:
                        break
        res_pair = (
            best_base.full_block_hash if best_base is not None else None,
            best_diff_indices,
        )
        if cache_key is not None:
            self._partial_match_cache[cache_key] = res_pair
        if fast_cache_key is not None:
            self._partial_match_cache[fast_cache_key] = res_pair
        return best_base, best_diff_indices

    @staticmethod
    def _get_boundary_wgram_size(block_len: int) -> int:
        return min(4, max(1, block_len // 4))

    def _index_base_block(
        self,
        layer_name: str,
        base_record: CompressedKVBlockRecord,
        max_exceptions: int,
    ) -> None:
        """Index an UNCOMPRESSED_BASE block for exact, partial, and stitched lookups."""
        token_hash = base_record.token_block_hash
        token_tuple = base_record.token_ids
        self._base_by_token_hash.setdefault((layer_name, token_hash), base_record)
        self._base_by_token_hash_list.setdefault((layer_name, token_hash), []).append(
            base_record
        )
        self._base_blocks_by_layer.setdefault(layer_name, []).append(base_record)
        if max_exceptions > 0:
            num_slices = max_exceptions + 1
            for s_idx, sub_tuple in self._get_pigeonhole_slices(
                token_tuple, num_slices
            ):
                self._base_by_subchunk.setdefault(
                    (layer_name, s_idx, sub_tuple), []
                ).append(base_record)
        b_len = len(token_tuple)
        if b_len >= 2:
            w = self._get_boundary_wgram_size(b_len)
            is_subbucket_padded = False
            if b_len >= 128:
                slot_len = b_len // 5
                for s_i in range(1, 5):
                    sb_end = s_i * slot_len
                    if sb_end >= w and len(set(token_tuple[sb_end - w : sb_end])) <= 1:
                        is_subbucket_padded = True
                        break
            if not is_subbucket_padded:
                self._base_by_layer_pos.setdefault(
                    (layer_name, base_record.start_position), []
                ).append(base_record)
                suf_w = token_tuple[-w:]
                if b_len < 16 or len(set(suf_w)) > 1:
                    self._base_by_suffix_wgram.setdefault(
                        (layer_name, suf_w), []
                    ).append(base_record)
                    self._layers_with_unaligned_bases.add(layer_name)
                pre_w = token_tuple[:w]
                if b_len < 16 or len(set(pre_w)) > 1:
                    self._base_by_prefix_wgram.setdefault(
                        (layer_name, pre_w), []
                    ).append(base_record)

    @staticmethod
    def _get_record_base_kv(
        base_rec_1: CompressedKVBlockRecord,
        base_rec_2: CompressedKVBlockRecord | None = None,
        stitch_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (raw_k, raw_v) from a single base block or stitched across two consecutive base blocks."""
        assert base_rec_1.raw_k is not None
        if base_rec_2 is None or stitch_offset <= 0:
            return base_rec_1.raw_k, base_rec_1.raw_v
        assert base_rec_2.raw_k is not None
        s = stitch_offset
        if (
            len(base_rec_1.token_ids) > base_rec_1.raw_k.shape[0]
            and base_rec_1.raw_k.shape[0] > 0
        ):
            s = max(
                1,
                (s * base_rec_1.raw_k.shape[0]) // len(base_rec_1.token_ids),
            )
        k_stitched = torch.cat([base_rec_1.raw_k[s:], base_rec_2.raw_k[:s]], dim=0)
        if base_rec_1.raw_v is not None and base_rec_2.raw_v is not None:
            v_stitched: torch.Tensor | None = torch.cat(
                [base_rec_1.raw_v[s:], base_rec_2.raw_v[:s]], dim=0
            )
        else:
            v_stitched = None
        return k_stitched, v_stitched

    def _find_stitched_base_blocks(
        self,
        layer_name: str,
        token_ids: tuple[int, ...],
        start_position: int | None = None,
        exclude_hashes: set[str] | None = None,
        max_edits: int = 0,
        full_block_hash: str = "",
    ) -> tuple[
        CompressedKVBlockRecord | None,
        CompressedKVBlockRecord | None,
        int,
        list[int],
    ]:
        """Find two consecutive UNCOMPRESSED_BASE blocks (B0, B1) whose stitched
        tokens `B0.token_ids[s:] + B1.token_ids[:s]` match `token_ids` within `max_edits`.

        Used by Approach 1 (Cross-Block Reference Stitching) when an unaligned
        token shift `delta_p != 0 (mod B)` causes blocks to straddle two adjacent
        reference blocks from an earlier turn.
        """
        b_len = len(token_ids)
        if b_len < 2 or layer_name not in self._layers_with_unaligned_bases:
            return None, None, 0, []
        layer_bases = self._base_blocks_by_layer.get(layer_name)
        if not layer_bases or len(layer_bases) < 2:
            return None, None, 0, []

        w = self._get_boundary_wgram_size(b_len)
        first_k = layer_bases[0].raw_k
        state_step = (
            max(1, b_len // int(first_k.shape[0]))
            if (
                first_k is not None
                and first_k.ndim >= 1
                and 0 < int(first_k.shape[0]) < b_len
                and b_len % int(first_k.shape[0]) == 0
            )
            else 1
        )
        cache_key = (
            (
                state_step,
                full_block_hash,
                start_position,
                max_edits,
                len(layer_bases),
                layer_bases[0].full_block_hash,
                layer_bases[-1].full_block_hash,
                self.stats.evicted_blocks,
            )
            if full_block_hash
            else None
        )
        if cache_key is not None and cache_key in self._stitched_match_cache:
            c_h1, c_h2, c_s, c_diffs = self._stitched_match_cache[cache_key]
            if c_h1 is None or c_h2 is None or c_s <= 0:
                return None, None, 0, []
            r1 = self._records.get((layer_name, c_h1))
            r2 = self._records.get((layer_name, c_h2))
            if (
                r1 is not None
                and r2 is not None
                and r1.tier == CompressionTier.UNCOMPRESSED_BASE
                and r2.tier == CompressionTier.UNCOMPRESSED_BASE
                and (
                    not exclude_hashes
                    or (c_h1 not in exclude_hashes and c_h2 not in exclude_hashes)
                )
            ):
                return r1, r2, c_s, c_diffs
        seen_pairs: set[tuple[str, str, int]] = set()
        candidates: list[
            tuple[CompressedKVBlockRecord, CompressedKVBlockRecord, int]
        ] = []
        require_both_seam_sides = max_edits == 0 or b_len > 64

        # O(1) seam lookup using boundary w-grams on either side of seam k = B - s:
        for k in range(1, b_len):
            if len(candidates) >= 16:
                break
            s = b_len - k
            if state_step > 1 and s % state_step != 0:
                continue
            if k >= w:
                suf_wgram = token_ids[k - w : k]
                if not (b_len >= 16 and len(set(suf_wgram)) <= 1):
                    b0_bucket = self._base_by_suffix_wgram.get((layer_name, suf_wgram))
                    if b0_bucket:
                        for b0 in b0_bucket[-16:]:
                            if exclude_hashes and b0.full_block_hash in exclude_hashes:
                                continue
                            if len(b0.token_ids) != b_len:
                                continue
                            if max_edits == 0 and token_ids[0] != b0.token_ids[s]:
                                continue
                            b1_list = self._base_by_layer_pos.get(
                                (layer_name, b0.start_position + b_len)
                            )
                            if not b1_list:
                                continue
                            for b1 in b1_list[-8:]:
                                if (
                                    exclude_hashes
                                    and b1.full_block_hash in exclude_hashes
                                ):
                                    continue
                                if len(b1.token_ids) != b_len:
                                    continue
                                if (
                                    b1.prev_token_block_hash
                                    and b0.token_block_hash
                                    and b1.prev_token_block_hash != b0.token_block_hash
                                ):
                                    continue
                                if (
                                    require_both_seam_sides
                                    and s >= w
                                    and token_ids[k : k + w] != b1.token_ids[:w]
                                ):
                                    continue
                                if (
                                    max_edits == 0
                                    and token_ids[-1] != b1.token_ids[s - 1]
                                ):
                                    continue
                                pkey = (b0.full_block_hash, b1.full_block_hash, s)
                                if pkey not in seen_pairs:
                                    seen_pairs.add(pkey)
                                    candidates.append((b0, b1, s))
                                    if len(candidates) >= 16:
                                        break
            if len(candidates) >= 16:
                break
            if s >= w and (not require_both_seam_sides or k < w):
                pre_wgram = token_ids[k : k + w]
                if not (b_len >= 16 and len(set(pre_wgram)) <= 1):
                    b1_bucket = self._base_by_prefix_wgram.get((layer_name, pre_wgram))
                    if b1_bucket:
                        for b1 in b1_bucket[-16:]:
                            if exclude_hashes and b1.full_block_hash in exclude_hashes:
                                continue
                            if len(b1.token_ids) != b_len:
                                continue
                            if max_edits == 0 and token_ids[-1] != b1.token_ids[s - 1]:
                                continue
                            b0_list = self._base_by_layer_pos.get(
                                (layer_name, b1.start_position - b_len)
                            )
                            if not b0_list:
                                continue
                            for b0 in b0_list[-8:]:
                                if (
                                    exclude_hashes
                                    and b0.full_block_hash in exclude_hashes
                                ):
                                    continue
                                if len(b0.token_ids) != b_len:
                                    continue
                                if b_len >= 16 and len(set(b0.token_ids[-w:])) <= 1:
                                    continue
                                if (
                                    b1.prev_token_block_hash
                                    and b0.token_block_hash
                                    and b1.prev_token_block_hash != b0.token_block_hash
                                ):
                                    continue
                                if max_edits == 0 and token_ids[0] != b0.token_ids[s]:
                                    continue
                                pkey = (b0.full_block_hash, b1.full_block_hash, s)
                                if pkey not in seen_pairs:
                                    seen_pairs.add(pkey)
                                    candidates.append((b0, b1, s))
                                    if len(candidates) >= 16:
                                        break

        # Fallback scan over recent adjacent base blocks when intra-block edits hit both
        # sides of the seam w-gram in a small base store:
        if not candidates and max_edits > 0 and b_len <= 64 and len(layer_bases) <= 64:
            for b0 in layer_bases[-16:]:
                if exclude_hashes and b0.full_block_hash in exclude_hashes:
                    continue
                if len(b0.token_ids) != b_len:
                    continue
                if b_len >= 16 and len(set(b0.token_ids[-w:])) <= 1:
                    continue
                b1_list = self._base_by_layer_pos.get(
                    (layer_name, b0.start_position + b_len)
                )
                if not b1_list:
                    continue
                for b1 in b1_list[-4:]:
                    if exclude_hashes and b1.full_block_hash in exclude_hashes:
                        continue
                    if len(b1.token_ids) != b_len:
                        continue
                    if (
                        b1.prev_token_block_hash
                        and b0.token_block_hash
                        and b1.prev_token_block_hash != b0.token_block_hash
                    ):
                        continue
                    for s in range(1, b_len):
                        if state_step > 1 and s % state_step != 0:
                            continue
                        pkey = (b0.full_block_hash, b1.full_block_hash, s)
                        if pkey not in seen_pairs:
                            seen_pairs.add(pkey)
                            candidates.append((b0, b1, s))

        best_b0: CompressedKVBlockRecord | None = None
        best_b1: CompressedKVBlockRecord | None = None
        best_s: int = 0
        best_diffs: list[int] = []
        best_score: tuple[int, int, int] = (999, 999, 999999)

        for b0, b1, s in candidates:
            k = b_len - s
            if max_edits == 0:
                if (
                    token_ids[:k] != b0.token_ids[s:]
                    or token_ids[k:] != b1.token_ids[:s]
                ):
                    continue
                diff_idx: list[int] = []
            else:
                if not any(ta == tb for ta, tb in zip(token_ids[:k], b0.token_ids[s:])):
                    continue
                if not any(ta == tb for ta, tb in zip(token_ids[k:], b1.token_ids[:s])):
                    continue
                stitched_toks = b0.token_ids[s:] + b1.token_ids[:s]
                bounded_diffs = self._diff_indices_bounded(
                    token_ids, stitched_toks, max_edits
                )
                if bounded_diffs is None:
                    continue
                diff_idx = bounded_diffs

            chain_ok = (
                0
                if (
                    b1.prev_token_block_hash
                    and b1.prev_token_block_hash == b0.token_block_hash
                )
                else 1
            )
            pos_dist = (
                abs(start_position - (b0.start_position + s))
                if start_position is not None
                else 0
            )
            score = (chain_ok, len(diff_idx), pos_dist)
            if best_b0 is None or score < best_score:
                best_b0 = b0
                best_b1 = b1
                best_s = s
                best_diffs = diff_idx
                best_score = score
                if chain_ok == 0 and len(diff_idx) == 0 and pos_dist == 0:
                    break

        if cache_key is not None:
            self._stitched_match_cache[cache_key] = (
                best_b0.full_block_hash if best_b0 is not None else None,
                best_b1.full_block_hash if best_b1 is not None else None,
                best_s,
                best_diffs,
            )
        return best_b0, best_b1, best_s, best_diffs

    def has_stitched_token_block(self, token_ids: tuple[int, ...]) -> bool:
        """Return True if `token_ids` matches a stitched pair of consecutive base blocks."""
        for layer_name in self._known_layers:
            policy = self.get_layer_policy(layer_name)
            if not policy.is_mamba and policy.rope_pos_scale >= 1.0:
                b0, b1, s, _ = self._find_stitched_base_blocks(
                    layer_name, token_ids, max_edits=0
                )
                return b0 is not None and b1 is not None and s > 0
        for layer_name in self._known_layers:
            policy = self.get_layer_policy(layer_name)
            if not policy.is_mamba:
                b0, b1, s, _ = self._find_stitched_base_blocks(
                    layer_name, token_ids, max_edits=0
                )
                return b0 is not None and b1 is not None and s > 0
        return False

    def store_block(
        self,
        layer_name: str,
        full_block_hash: str,
        token_ids: tuple[int, ...] | list[int],
        start_position: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor | None = None,
        divergence_distance: int | None = None,
        exclude_base_hashes: set[str] | None = None,
        prev_token_block_hash: str = "",
        root_block_hash: str = "",
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
            exclude_base_hashes: Optional set of UNCOMPRESSED_BASE hashes created earlier
                within the same request, preventing intra-request self-shadowing.

        Returns:
            The stored CompressedKVBlockRecord.
        """
        key = (layer_name, full_block_hash)
        if key in self._records:
            rec = self._records[key]
            self._touch_lru(
                full_block_hash, rec.ref_base_hash, rec.ref_base_hash_2
            )
            return rec

        token_tuple = tuple(int(t) for t in token_ids)
        token_hash = compute_token_block_hash(token_tuple)
        policy = self.get_layer_policy(layer_name)
        is_mla = v_block is None and not policy.is_mamba

        target_dev = (
            k_block.device
            if str(self.device).startswith("cuda")
            else self.device
        )
        k_cpu = k_block.detach().to(target_dev)
        v_cpu = v_block.detach().to(target_dev) if v_block is not None else None

        raw_bytes = k_cpu.numel() * k_cpu.element_size()
        if v_cpu is not None:
            raw_bytes += v_cpu.numel() * v_cpu.element_size()

        self._known_prefix_blocks[full_block_hash] = (token_tuple, start_position)
        self._known_token_blocks.setdefault(token_hash, full_block_hash)
        if policy.is_mamba:
            self._known_mamba_blocks.add(full_block_hash)

        # 1. Check for an UNCOMPRESSED_BASE block with the exact same token_block_hash,
        # or (Approach 1) a 0-edit cross-block stitched match across two consecutive base blocks:
        base_rec = (
            None
            if self.force_uncompressed
            else self._find_exact_base_block(
                layer_name,
                token_hash,
                start_position=start_position,
                exclude_hashes=exclude_base_hashes,
                prev_token_block_hash=prev_token_block_hash,
            )
        )
        base_rec_2: CompressedKVBlockRecord | None = None
        stitch_offset: int = 0
        if (
            base_rec is None
            and not self.force_uncompressed
            and not policy.is_mamba
        ):
            b0, b1, s_off, _ = self._find_stitched_base_blocks(
                layer_name,
                token_tuple,
                start_position=start_position,
                exclude_hashes=exclude_base_hashes,
                max_edits=0,
                full_block_hash=full_block_hash,
            )
            if b0 is not None and b1 is not None and s_off > 0:
                base_rec = b0
                base_rec_2 = b1
                stitch_offset = s_off

        if base_rec is not None:
            ref_hash_2 = (
                base_rec_2.full_block_hash if base_rec_2 is not None else None
            )
            delta_pos = start_position - (
                base_rec.start_position + stitch_offset
            )

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
                    ref_base_hash_2=ref_hash_2,
                    stitch_offset=stitch_offset,
                )
                return self._commit_record(
                    layer_name, record, raw_bytes, root_block_hash=root_block_hash
                )

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
                    ref_base_hash_2=ref_hash_2,
                    stitch_offset=stitch_offset,
                )
                return self._commit_record(
                    layer_name, record, raw_bytes, root_block_hash=root_block_hash
                )

            # Case 1C: Exact (or 0-edit stitched) token block match (Regimes R1a, R1b, R3, or Mamba/GDN)
            base_raw_k, base_raw_v = self._get_record_base_kv(
                base_rec, base_rec_2, stitch_offset
            )
            is_dsv41_q = _is_dsv41_quant_kv(k_cpu)
            is_dsv41_nvfp4 = (
                _is_dsv41_nvfp4_mla(k_cpu)
                or _is_dsv41_mxfp8_swa(k_cpu)
                or _is_dsv41_fp8_indexer(k_cpu)
            )
            is_fp8_mla = _is_fp8_e4m3_mla(k_cpu)
            delta_bits_4 = 2 if is_dsv41_nvfp4 else 4
            k_work = _dequant_dsv41_kv(k_cpu) if is_dsv41_q else k_cpu
            base_k_work = (
                _dequant_dsv41_kv(base_raw_k) if is_dsv41_q else base_raw_k
            )
            k_ref_aligned = apply_relative_rope_rotation(
                base_k_work,
                delta_pos=delta_pos,
                rope_theta=policy.rope_theta,
                rotary_dim=0 if policy.is_mamba else policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
                rope_scaling=policy.rope_scaling,
                rope_at_tail=policy.rope_at_tail,
                rope_pos_scale=policy.rope_pos_scale,
            )
            k_fp32 = k_work.to(torch.float32)
            delta_k = k_fp32 - k_ref_aligned.to(torch.float32)
            eta_k_t = torch.linalg.norm(delta_k) / torch.linalg.norm(k_fp32).clamp(
                min=1e-8
            )
            check_outliers = (
                not policy.is_mamba
                and policy.max_sparse_exceptions > 0
                and delta_k.ndim >= 2
                and delta_k.shape[0] >= 4
            )
            outlier_mask: torch.Tensor | None = None

            if v_cpu is not None and base_raw_v is not None:
                v_fp32 = v_cpu.to(torch.float32)
                delta_v = v_fp32 - base_raw_v.to(torch.float32)
                eta_v_t = torch.linalg.norm(delta_v) / torch.linalg.norm(v_fp32).clamp(
                    min=1e-8
                )
                if check_outliers:
                    b_tok = delta_k.shape[0]
                    dk_tok = torch.linalg.norm(delta_k.reshape(b_tok, -1), dim=-1)
                    k_tok = torch.linalg.norm(k_fp32.reshape(b_tok, -1), dim=-1).clamp(
                        min=1e-6
                    )
                    dv_tok = torch.linalg.norm(delta_v.reshape(b_tok, -1), dim=-1)
                    v_tok = torch.linalg.norm(v_fp32.reshape(b_tok, -1), dim=-1).clamp(
                        min=1e-6
                    )
                    tok_eta = torch.maximum(dk_tok / k_tok, dv_tok / v_tok)
                    outlier_mask = tok_eta > 0.65
                    eta_k, eta_v, num_outliers_f = torch.stack(
                        [eta_k_t, eta_v_t, outlier_mask.sum().to(torch.float32)]
                    ).tolist()
                    num_outliers = int(num_outliers_f)
                else:
                    eta_k, eta_v = torch.stack([eta_k_t, eta_v_t]).tolist()
                    num_outliers = 0
                eta_max = max(eta_k, eta_v)
            else:
                delta_v = None
                v_fp32 = None
                if check_outliers:
                    b_tok = delta_k.shape[0]
                    dk_tok = torch.linalg.norm(delta_k.reshape(b_tok, -1), dim=-1)
                    k_tok = torch.linalg.norm(k_fp32.reshape(b_tok, -1), dim=-1).clamp(
                        min=1e-6
                    )
                    outlier_mask = (dk_tok / k_tok) > 0.65
                    eta_k, num_outliers_f = torch.stack(
                        [eta_k_t, outlier_mask.sum().to(torch.float32)]
                    ).tolist()
                    num_outliers = int(num_outliers_f)
                else:
                    eta_k = float(eta_k_t.item())
                    num_outliers = 0
                eta_v = 0.0
                eta_max = eta_k

            # Select Tier based on residual magnitude:
            if (
                (
                    policy.is_sliding_window
                    or (is_dsv41_q and not is_fp8_mla)
                )
                and eta_max <= policy.tau_zero_bit
            ) or (
                is_dsv41_nvfp4
            ) or (
                (policy.is_mamba or is_fp8_mla)
                and eta_max <= min(policy.tau_zero_bit, 0.01)
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
                    ref_base_hash_2=ref_hash_2,
                    stitch_offset=stitch_offset,
                )
                return self._commit_record(
                    layer_name, record, raw_bytes, root_block_hash=root_block_hash
                )

            # Check for localized 1-2 token MoE/attention-sink spikes inside an
            # exact-token-match block before falling through to dense INT2/INT4:
            if check_outliers and outlier_mask is not None:
                max_sparse = min(2, policy.max_sparse_exceptions)
                if 0 < num_outliers <= max_sparse:
                    outlier_idx = torch.nonzero(outlier_mask, as_tuple=False).reshape(-1)
                    delta_k_rem = delta_k.clone()
                    delta_k_rem[outlier_idx] = 0.0
                    k_rem = k_fp32.clone()
                    k_rem[outlier_idx] = 0.0
                    rem_eta_k_t = torch.linalg.norm(delta_k_rem) / torch.linalg.norm(
                        k_rem
                    ).clamp(min=1e-8)
                    if delta_v is not None and v_fp32 is not None:
                        delta_v_rem = delta_v.clone()
                        delta_v_rem[outlier_idx] = 0.0
                        v_rem = v_fp32.clone()
                        v_rem[outlier_idx] = 0.0
                        rem_eta_v_t = torch.linalg.norm(
                            delta_v_rem
                        ) / torch.linalg.norm(v_rem).clamp(min=1e-8)
                        rem_eta_k, rem_eta_v = torch.stack(
                            [rem_eta_k_t, rem_eta_v_t]
                        ).tolist()
                        rem_eta_max = max(rem_eta_k, rem_eta_v)
                    else:
                        delta_v_rem = None
                        rem_eta_max = float(rem_eta_k_t.item())

                    if rem_eta_max <= policy.tau_int4:
                        sparse_k = k_cpu[outlier_idx].clone()
                        sparse_v = (
                            v_cpu[outlier_idx].clone() if v_cpu is not None else None
                        )
                        q_dk = quantize_residual_packed(
                            delta_k_rem.to(k_work.dtype), bits=delta_bits_4
                        )
                        q_dv = (
                            quantize_residual_packed(
                                delta_v_rem.to(v_cpu.dtype), bits=4
                            )
                            if (delta_v_rem is not None and v_cpu is not None)
                            else None
                        )
                        record = CompressedKVBlockRecord(
                            tier=CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA,
                            full_block_hash=full_block_hash,
                            token_block_hash=token_hash,
                            token_ids=token_tuple,
                            start_position=start_position,
                            layer_idx=policy.layer_idx,
                            is_mla=is_mla,
                            ref_base_hash=base_rec.full_block_hash,
                            delta_pos=delta_pos,
                            ref_base_hash_2=ref_hash_2,
                            stitch_offset=stitch_offset,
                            quant_delta_k=q_dk,
                            quant_delta_v=q_dv,
                            sparse_token_indices=outlier_idx.to(torch.int16),
                            sparse_k_bf16=sparse_k,
                            sparse_v_bf16=sparse_v,
                        )
                        return self._commit_record(
                            layer_name,
                            record,
                            raw_bytes,
                            root_block_hash=root_block_hash,
                        )

            eff_tau_int2 = (
                policy.tau_int2 if is_fp8_mla else min(policy.tau_int2, 0.08)
            )
            tier1_eligible = (
                (eta_k <= policy.tau_int2 and eta_v <= policy.tau_int4)
                if policy.is_mamba
                else (
                    eta_max <= policy.tau_int4
                    if is_dsv41_nvfp4
                    else eta_max <= eff_tau_int2
                )
            )
            if tier1_eligible:
                q_dk = quantize_residual_packed(delta_k.to(k_work.dtype), bits=2)
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
                    ref_base_hash_2=ref_hash_2,
                    stitch_offset=stitch_offset,
                    quant_delta_k=q_dk,
                    quant_delta_v=q_dv,
                )
                return self._commit_record(
                    layer_name, record, raw_bytes, root_block_hash=root_block_hash
                )

            if not policy.is_mamba or eta_max <= policy.tau_int4:
                q_dk = quantize_residual_packed(
                    delta_k.to(k_work.dtype), bits=delta_bits_4
                )
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
                    ref_base_hash_2=ref_hash_2,
                    stitch_offset=stitch_offset,
                    quant_delta_k=q_dk,
                    quant_delta_v=q_dv,
                )
                return self._commit_record(
                    layer_name, record, raw_bytes, root_block_hash=root_block_hash
                )

        # 2. Check for Regime R2: Intra-block partial token edit (<= M tokens changed)
        # or (Approach 1) stitched cross-block match with <= M intra-block token edits:
        partial_base, diff_indices = (
            (None, [])
            if self.force_uncompressed
            else self._find_partial_match_base_block(
                layer_name,
                token_tuple,
                start_position=start_position,
                exclude_hashes=exclude_base_hashes,
                prev_token_block_hash=prev_token_block_hash,
                full_block_hash=full_block_hash,
            )
        )
        partial_base_2: CompressedKVBlockRecord | None = None
        partial_stitch_offset: int = 0
        if (
            partial_base is None
            and not self.force_uncompressed
            and not policy.is_mamba
            and policy.max_sparse_exceptions > 0
        ):
            sb0, sb1, s_off, s_diffs = self._find_stitched_base_blocks(
                layer_name,
                token_tuple,
                start_position=start_position,
                exclude_hashes=exclude_base_hashes,
                max_edits=policy.max_sparse_exceptions,
                full_block_hash=full_block_hash,
            )
            if sb0 is not None and sb1 is not None and s_off > 0 and s_diffs:
                partial_base = sb0
                partial_base_2 = sb1
                partial_stitch_offset = s_off
                diff_indices = s_diffs

        if partial_base is not None and diff_indices:
            assert partial_base.raw_k is not None
            partial_ref_hash_2 = (
                partial_base_2.full_block_hash
                if partial_base_2 is not None
                else None
            )
            delta_pos = start_position - (
                partial_base.start_position + partial_stitch_offset
            )
            if policy.is_mamba:
                k_fp32 = k_cpu.to(torch.float32)
                delta_k = k_fp32 - partial_base.raw_k.to(torch.float32)
                eta_k_t = torch.linalg.norm(delta_k) / torch.linalg.norm(k_fp32).clamp(
                    min=1e-8
                )

                if v_cpu is not None and partial_base.raw_v is not None:
                    v_fp32 = v_cpu.to(torch.float32)
                    delta_v = v_fp32 - partial_base.raw_v.to(torch.float32)
                    eta_v_t = torch.linalg.norm(delta_v) / torch.linalg.norm(
                        v_fp32
                    ).clamp(min=1e-8)
                    eta_k, eta_v = torch.stack([eta_k_t, eta_v_t]).tolist()
                    eta_max = max(eta_k, eta_v)
                else:
                    delta_v = None
                    eta_k = float(eta_k_t.item())
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
                    return self._commit_record(
                        layer_name,
                        record,
                        raw_bytes,
                        root_block_hash=root_block_hash,
                    )

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
                    return self._commit_record(
                        layer_name,
                        record,
                        raw_bytes,
                        root_block_hash=root_block_hash,
                    )

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
                    return self._commit_record(
                        layer_name,
                        record,
                        raw_bytes,
                        root_block_hash=root_block_hash,
                    )
            else:
                partial_raw_k, partial_raw_v = self._get_record_base_kv(
                    partial_base, partial_base_2, partial_stitch_offset
                )
                is_dsv41_q = _is_dsv41_quant_kv(k_cpu)
                is_dsv41_nvfp4 = (
                    _is_dsv41_nvfp4_mla(k_cpu)
                    or _is_dsv41_mxfp8_swa(k_cpu)
                    or _is_dsv41_fp8_indexer(k_cpu)
                )
                delta_bits_4 = 2 if is_dsv41_nvfp4 else 4
                k_work = _dequant_dsv41_kv(k_cpu) if is_dsv41_q else k_cpu
                partial_k_work = (
                    _dequant_dsv41_kv(partial_raw_k)
                    if is_dsv41_q
                    else partial_raw_k
                )
                k_ref_aligned = apply_relative_rope_rotation(
                    partial_k_work,
                    delta_pos=delta_pos,
                    rope_theta=policy.rope_theta,
                    rotary_dim=policy.rotary_dim,
                    is_neox_style=policy.is_neox_style,
                    rope_scaling=policy.rope_scaling,
                    rope_at_tail=policy.rope_at_tail,
                    rope_pos_scale=policy.rope_pos_scale,
                )
                k_work_fp32 = k_work.to(torch.float32)
                delta_k = k_work_fp32 - k_ref_aligned.to(torch.float32)
                delta_v = (
                    (v_cpu.to(torch.float32) - partial_raw_v.to(torch.float32))
                    if (v_cpu is not None and partial_raw_v is not None)
                    else None
                )

                # If there are unused sparse exception slots and an adjacent token
                # inside the edited block experienced a large MoE/attention shift,
                # promote that token to BF16 as well.
                b_tok = delta_k.shape[0]
                if 0 < b_tok < len(token_tuple):
                    all_sparse_indices = sorted(
                        {(idx * b_tok) // len(token_tuple) for idx in diff_indices}
                    )
                else:
                    all_sparse_indices = list(diff_indices)
                rem_slots = policy.max_sparse_exceptions - len(all_sparse_indices)
                if rem_slots > 0 and delta_k.ndim >= 2 and b_tok >= 4:
                    dk_tok = torch.linalg.norm(delta_k.reshape(b_tok, -1), dim=-1)
                    k_tok = torch.linalg.norm(
                        k_work_fp32.reshape(b_tok, -1), dim=-1
                    ).clamp(min=1e-6)
                    tok_eta = dk_tok / k_tok
                    if delta_v is not None and v_cpu is not None:
                        dv_tok = torch.linalg.norm(delta_v.reshape(b_tok, -1), dim=-1)
                        v_tok = torch.linalg.norm(
                            v_cpu.to(torch.float32).reshape(b_tok, -1), dim=-1
                        ).clamp(min=1e-6)
                        tok_eta = torch.maximum(tok_eta, dv_tok / v_tok)
                    for idx_i in all_sparse_indices:
                        tok_eta[idx_i] = 0.0
                    top_vals, top_idxs = torch.topk(tok_eta, k=min(rem_slots, b_tok))
                    for val, idx_i in torch.stack(
                        [top_vals, top_idxs.to(torch.float32)], dim=-1
                    ).tolist():
                        if val > 0.65:
                            all_sparse_indices.append(int(idx_i))
                    all_sparse_indices.sort()

                idx_tensor = torch.tensor(
                    all_sparse_indices, dtype=torch.long, device=k_cpu.device
                )

                # Zero out the sparse BF16 rows in the residual so their large outliers do not
                # pollute the per-head INT4 quantization scale of the remaining rows!
                delta_k[idx_tensor] = 0.0
                k_rem = k_work_fp32.clone()
                k_rem[idx_tensor] = 0.0
                eta_k_t = torch.linalg.norm(delta_k) / torch.linalg.norm(k_rem).clamp(
                    min=1e-8
                )

                if delta_v is not None and v_cpu is not None:
                    delta_v[idx_tensor] = 0.0
                    v_rem = v_cpu.to(torch.float32).clone()
                    v_rem[idx_tensor] = 0.0
                    eta_v_t = torch.linalg.norm(delta_v) / torch.linalg.norm(
                        v_rem
                    ).clamp(min=1e-8)
                    eta_k, eta_v = torch.stack([eta_k_t, eta_v_t]).tolist()
                    eta_max = max(eta_k, eta_v)
                else:
                    eta_k = float(eta_k_t.item())
                    eta_max = eta_k

                if eta_max <= policy.tau_int4:
                    sparse_k = k_cpu[idx_tensor].clone()
                    sparse_v = v_cpu[idx_tensor].clone() if v_cpu is not None else None
                    q_dk = (
                        quantize_residual_packed(
                            delta_k.to(k_work.dtype), bits=delta_bits_4
                        )
                        if (policy.layer_idx > 0 and not is_dsv41_nvfp4)
                        else None
                    )
                    q_dv = (
                        quantize_residual_packed(delta_v.to(v_cpu.dtype), bits=4)
                        if (
                            policy.layer_idx > 0
                            and delta_v is not None
                            and v_cpu is not None
                        )
                        else None
                    )

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
                        ref_base_hash_2=partial_ref_hash_2,
                        stitch_offset=partial_stitch_offset,
                        quant_delta_k=q_dk,
                        quant_delta_v=q_dv,
                        sparse_token_indices=idx_tensor.to(torch.int16),
                        sparse_k_bf16=sparse_k,
                        sparse_v_bf16=sparse_v,
                    )
                    return self._commit_record(
                        layer_name,
                        record,
                        raw_bytes,
                        root_block_hash=root_block_hash,
                    )

        # 3. Default: Store as UNCOMPRESSED_BASE and index for single-hop reference
        base_record = CompressedKVBlockRecord(
            tier=CompressionTier.UNCOMPRESSED_BASE,
            full_block_hash=full_block_hash,
            token_block_hash=token_hash,
            token_ids=token_tuple,
            start_position=start_position,
            layer_idx=policy.layer_idx,
            is_mla=is_mla,
            prev_token_block_hash=prev_token_block_hash,
            raw_k=k_cpu.clone(),
            raw_v=v_cpu.clone() if v_cpu is not None else None,
        )
        max_exceptions = (
            self.max_sparse_exceptions
            if policy.is_mamba
            else policy.max_sparse_exceptions
        )
        if (
            (self.enable_cross_turn_reuse and _is_dsv41_quant_kv(k_cpu))
            or full_block_hash not in self._dependent_blocks
        ):
            self._index_base_block(layer_name, base_record, max_exceptions)
        return self._commit_record(
            layer_name, base_record, raw_bytes, root_block_hash=root_block_hash
        )

    def _commit_record(
        self,
        layer_name: str,
        record: CompressedKVBlockRecord,
        raw_bytes: int,
        batch_hashes: set[str] | None = None,
        root_block_hash: str = "",
    ) -> CompressedKVBlockRecord:
        bhash = record.full_block_hash
        if str(self.device).startswith("cuda"):
            if record.raw_k is not None:
                record.raw_k = self._pack_cuda_tensor(record.raw_k, bhash)
            if record.raw_v is not None:
                record.raw_v = self._pack_cuda_tensor(record.raw_v, bhash)
            if record.quant_delta_k is not None:
                record.quant_delta_k.packed_data = self._pack_cuda_tensor(
                    record.quant_delta_k.packed_data, bhash
                )
                record.quant_delta_k.scales = self._pack_cuda_tensor(
                    record.quant_delta_k.scales, bhash
                )
            if record.quant_delta_v is not None:
                record.quant_delta_v.packed_data = self._pack_cuda_tensor(
                    record.quant_delta_v.packed_data, bhash
                )
                record.quant_delta_v.scales = self._pack_cuda_tensor(
                    record.quant_delta_v.scales, bhash
                )
            if record.sparse_token_indices is not None:
                record.sparse_token_indices = self._pack_cuda_tensor(
                    record.sparse_token_indices, bhash
                )
            if record.sparse_k_bf16 is not None:
                record.sparse_k_bf16 = self._pack_cuda_tensor(
                    record.sparse_k_bf16, bhash
                )
            if record.sparse_v_bf16 is not None:
                record.sparse_v_bf16 = self._pack_cuda_tensor(
                    record.sparse_v_bf16, bhash
                )
        self._known_layers.add(layer_name)
        self._records[(layer_name, bhash)] = record
        self._record_raw_bytes[(layer_name, bhash)] = raw_bytes
        if self.enable_cross_turn_reuse:
            tok_pos_key = (
                layer_name,
                record.token_block_hash,
                record.start_position,
            )
            existing_tp = self._record_by_tok_pos.get(tok_pos_key)
            if (
                existing_tp is None
                or (layer_name, existing_tp.full_block_hash)
                not in self._records
                or (
                    existing_tp.tier == CompressionTier.TIER0_ZERO_BIT
                    and (
                        record.tier == CompressionTier.UNCOMPRESSED_BASE
                        or record.quant_delta_k is not None
                        or record.sparse_k_bf16 is not None
                    )
                )
            ):
                self._record_by_tok_pos[tok_pos_key] = record
                if (
                    record.tier == CompressionTier.UNCOMPRESSED_BASE
                    or record.quant_delta_k is not None
                    or record.sparse_k_bf16 is not None
                ):
                    self._tok_pos_anchor_blocks.add(bhash)
        if record.layer_idx == 0 or bhash not in self._block_lru:
            if record.ref_base_hash or record.ref_base_hash_2:
                self._dependent_blocks.add(bhash)
                self._attention_base_blocks.discard(bhash)
                self._pos0_base_blocks.discard(bhash)
            elif (
                record.tier == CompressionTier.UNCOMPRESSED_BASE
                and bhash not in self._dependent_blocks
                and not self.get_layer_policy(layer_name).is_mamba
            ):
                self._attention_base_blocks.add(bhash)
                if record.start_position == 0:
                    self._pos0_base_blocks.add(bhash)
            if record.ref_base_hash:
                deps_1 = self._dependents_by_base.setdefault(
                    record.ref_base_hash, set()
                )
                if not root_block_hash:
                    for old_dep in tuple(deps_1):
                        if old_dep != bhash and (
                            batch_hashes is None or old_dep not in batch_hashes
                        ):
                            self.demote_lru(old_dep)
                deps_1.add(bhash)
            if record.ref_base_hash_2:
                deps_2 = self._dependents_by_base.setdefault(
                    record.ref_base_hash_2, set()
                )
                if not root_block_hash:
                    for old_dep in tuple(deps_2):
                        if old_dep != bhash and (
                            batch_hashes is None or old_dep not in batch_hashes
                        ):
                            self.demote_lru(old_dep)
                deps_2.add(bhash)
            if (
                root_block_hash
                and bhash in self._dependent_blocks
                and record.start_position > 0
            ):
                rkey = (root_block_hash, record.start_position)
                old_dep = self._active_by_root_pos.get(rkey)
                if (
                    old_dep
                    and old_dep != bhash
                    and (batch_hashes is None or old_dep not in batch_hashes)
                ):
                    if (
                        (
                            len(self._known_prefix_blocks) >= 100
                            or len(self._block_lru) >= 100
                        )
                        and record.start_position >= 2 * max(1, len(record.token_ids))
                        and old_dep in self._dependent_blocks
                        and not self._dependents_by_base.get(old_dep)
                        and not self._is_attention_base_block(old_dep)
                        and old_dep not in self._tok_pos_anchor_blocks
                    ):
                        self.evict_block(old_dep, count_as_eviction=False)
                    else:
                        self.demote_lru(old_dep)
                self._active_by_root_pos[rkey] = bhash
            self._touch_lru(bhash, record.ref_base_hash, record.ref_base_hash_2)
        self.stats.total_blocks_stored += 1
        self.stats.raw_bytes_equivalent += raw_bytes
        self.stats.compressed_bytes_stored += record.nbytes
        self.stats.tier_counts[record.tier] += 1
        self.stats.cumulative_tier_counts[record.tier] += 1
        return record

    def store_blocks_batch(
        self,
        layer_name: str,
        items: list[tuple["DeltaBlockTransferSpec", torch.Tensor, torch.Tensor | None]],
    ) -> list[CompressedKVBlockRecord]:
        """Batch-compress and store multiple KV blocks for `layer_name` in a single step.

        Eliminates per-block GPU-to-CPU synchronization stalls (`.tolist()` / `.item()`)
        and batches RoPE phase alignment, L2 norm evaluation, and INT2/INT4 quantization
        across all blocks in the layer while preserving exact sequential commit order.
        """
        if len(items) <= 1 or self.force_uncompressed:
            step_new_base_hashes: set[str] = set()
            out_single: list[CompressedKVBlockRecord] = []
            for blk_spec, k_blk, v_blk in items:
                rec = self.store_block(
                    layer_name=layer_name,
                    full_block_hash=blk_spec.full_block_hash,
                    token_ids=blk_spec.token_ids,
                    start_position=blk_spec.start_position,
                    k_block=k_blk,
                    v_block=v_blk,
                    divergence_distance=blk_spec.divergence_distance,
                    exclude_base_hashes=step_new_base_hashes,
                    prev_token_block_hash=blk_spec.prev_token_block_hash,
                    root_block_hash=blk_spec.root_block_hash,
                )
                if rec.tier == CompressionTier.UNCOMPRESSED_BASE:
                    step_new_base_hashes.add(blk_spec.full_block_hash)
                out_single.append(rec)
            return out_single

        policy = self.get_layer_policy(layer_name)
        n = len(items)
        k_cpus: list[torch.Tensor] = []
        v_cpus: list[torch.Tensor | None] = []
        raw_bytes_list: list[int] = []
        token_tuples: list[tuple[int, ...]] = []
        token_hashes: list[str] = []
        records: list[CompressedKVBlockRecord | None] = [None] * n
        already_committed: list[bool] = [False] * n
        seen_in_batch: dict[str, int] = {}

        # (item_idx, base_rec_1, base_rec_2, stitch_offset, delta_pos)
        exact_group: list[
            tuple[
                int,
                CompressedKVBlockRecord,
                CompressedKVBlockRecord | None,
                int,
                int,
            ]
        ] = []
        # (item_idx, partial_base_1, partial_base_2, stitch_offset, delta_pos, diff_indices)
        attn_partial_group: list[
            tuple[
                int,
                CompressedKVBlockRecord,
                CompressedKVBlockRecord | None,
                int,
                int,
                list[int],
            ]
        ] = []

        for i, (blk_spec, k_blk, v_blk) in enumerate(items):
            token_tuple = tuple(int(t) for t in blk_spec.token_ids)
            token_hash = compute_token_block_hash(token_tuple)
            target_dev = (
                k_blk.device
                if str(self.device).startswith("cuda")
                else self.device
            )
            k_cpu = k_blk.detach().to(target_dev)
            v_cpu = v_blk.detach().to(target_dev) if v_blk is not None else None
            is_mla = v_cpu is None and not policy.is_mamba
            raw_b = k_cpu.numel() * k_cpu.element_size() + (
                v_cpu.numel() * v_cpu.element_size() if v_cpu is not None else 0
            )
            k_cpus.append(k_cpu)
            v_cpus.append(v_cpu)
            raw_bytes_list.append(raw_b)
            token_tuples.append(token_tuple)
            token_hashes.append(token_hash)

            existing_rec = self._records.get(
                (layer_name, blk_spec.full_block_hash)
            )
            if existing_rec is not None:
                self._touch_lru(
                    blk_spec.full_block_hash,
                    existing_rec.ref_base_hash,
                    existing_rec.ref_base_hash_2,
                )
                records[i] = existing_rec
                already_committed[i] = True
                continue
            if blk_spec.full_block_hash in seen_in_batch:
                already_committed[i] = True
                continue
            seen_in_batch[blk_spec.full_block_hash] = i

            self._known_prefix_blocks[blk_spec.full_block_hash] = (
                token_tuple,
                blk_spec.start_position,
            )
            self._known_token_blocks.setdefault(
                token_hash, blk_spec.full_block_hash
            )
            if policy.is_mamba:
                self._known_mamba_blocks.add(blk_spec.full_block_hash)

            base_rec = self._find_exact_base_block(
                layer_name,
                token_hash,
                start_position=blk_spec.start_position,
                prev_token_block_hash=blk_spec.prev_token_block_hash,
            )
            base_rec_2: CompressedKVBlockRecord | None = None
            stitch_offset: int = 0
            if (
                base_rec is None
                and not policy.is_mamba
                and not self.enable_cross_turn_reuse
            ):
                b0, b1, s_off, _ = self._find_stitched_base_blocks(
                    layer_name,
                    token_tuple,
                    start_position=blk_spec.start_position,
                    max_edits=0,
                    full_block_hash=blk_spec.full_block_hash,
                )
                if b0 is not None and b1 is not None and s_off > 0:
                    base_rec = b0
                    base_rec_2 = b1
                    stitch_offset = s_off

            if base_rec is not None:
                delta_pos = blk_spec.start_position - (
                    base_rec.start_position + stitch_offset
                )
                if (policy.layer_idx == 0 and not policy.is_mamba) or (
                    policy.is_sliding_window
                    and blk_spec.divergence_distance is not None
                    and blk_spec.divergence_distance
                    >= policy.sliding_window_dedup_threshold
                ):
                    records[i] = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER0_ZERO_BIT,
                        full_block_hash=blk_spec.full_block_hash,
                        token_block_hash=token_hashes[i],
                        token_ids=token_tuple,
                        start_position=blk_spec.start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=is_mla,
                        ref_base_hash=base_rec.full_block_hash,
                        delta_pos=delta_pos,
                        ref_base_hash_2=(
                            base_rec_2.full_block_hash
                            if base_rec_2 is not None
                            else None
                        ),
                        stitch_offset=stitch_offset,
                    )
                else:
                    exact_group.append(
                        (i, base_rec, base_rec_2, stitch_offset, delta_pos)
                    )
                continue

            partial_base, diff_indices = self._find_partial_match_base_block(
                layer_name,
                token_tuple,
                start_position=blk_spec.start_position,
                prev_token_block_hash=blk_spec.prev_token_block_hash,
                full_block_hash=blk_spec.full_block_hash,
            )
            partial_base_2: CompressedKVBlockRecord | None = None
            partial_stitch_offset: int = 0
            if (
                partial_base is None
                and not policy.is_mamba
                and not self.enable_cross_turn_reuse
                and policy.max_sparse_exceptions > 0
            ):
                sb0, sb1, s_off, s_diffs = self._find_stitched_base_blocks(
                    layer_name,
                    token_tuple,
                    start_position=blk_spec.start_position,
                    max_edits=policy.max_sparse_exceptions,
                    full_block_hash=blk_spec.full_block_hash,
                )
                if sb0 is not None and sb1 is not None and s_off > 0 and s_diffs:
                    partial_base = sb0
                    partial_base_2 = sb1
                    partial_stitch_offset = s_off
                    diff_indices = s_diffs

            if partial_base is not None and diff_indices:
                delta_pos = blk_spec.start_position - (
                    partial_base.start_position + partial_stitch_offset
                )
                if policy.is_mamba:
                    exact_group.append(
                        (
                            i,
                            partial_base,
                            partial_base_2,
                            partial_stitch_offset,
                            delta_pos,
                        )
                    )
                else:
                    attn_partial_group.append(
                        (
                            i,
                            partial_base,
                            partial_base_2,
                            partial_stitch_offset,
                            delta_pos,
                            diff_indices,
                        )
                    )

        # Pass 2A: Batched GPU evaluation for exact (or 0-edit stitched) token matches + Mamba partial matches
        is_multi_cohort = len(self._pos0_base_blocks) >= 16
        if (
            exact_group
            and not policy.is_mamba
            and (
                _is_dsv41_quant_kv(k_cpus[exact_group[0][0]])
                or is_multi_cohort
            )
        ):
            for (
                item_idx,
                base_rec,
                base_rec_2,
                stitch_offset,
                delta_pos,
            ) in exact_group:
                if (
                    not is_multi_cohort
                    and (delta_pos != 0 or stitch_offset > 0)
                    and policy.layer_idx > 0
                ):
                    continue
                blk_spec = items[item_idx][0]
                records[item_idx] = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER0_ZERO_BIT,
                    full_block_hash=blk_spec.full_block_hash,
                    token_block_hash=token_hashes[item_idx],
                    token_ids=token_tuples[item_idx],
                    start_position=blk_spec.start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=v_cpus[item_idx] is None,
                    ref_base_hash=base_rec.full_block_hash,
                    delta_pos=delta_pos,
                    ref_base_hash_2=(
                        base_rec_2.full_block_hash
                        if base_rec_2 is not None
                        else None
                    ),
                    stitch_offset=stitch_offset,
                )
            exact_group = [it for it in exact_group if records[it[0]] is None]

        if exact_group:
            m = len(exact_group)
            k_stack_raw = torch.stack(
                [k_cpus[idx] for idx, _, _, _, _ in exact_group], dim=0
            )
            base_kv_pairs = [
                self._get_record_base_kv(b1, b2, s_off)
                for _, b1, b2, s_off, _ in exact_group
            ]
            base_k_stack_raw = torch.stack(
                [kv[0] for kv in base_kv_pairs], dim=0
            )
            is_dsv41_q = _is_dsv41_quant_kv(k_stack_raw)
            is_dsv41_nvfp4 = (
                _is_dsv41_nvfp4_mla(k_stack_raw)
                or _is_dsv41_mxfp8_swa(k_stack_raw)
                or _is_dsv41_fp8_indexer(k_stack_raw)
            )
            is_fp8_mla = _is_fp8_e4m3_mla(k_stack_raw)
            delta_bits_4 = 2 if is_dsv41_nvfp4 else 4
            k_stack = (
                _dequant_dsv41_kv(k_stack_raw)
                if is_dsv41_q
                else k_stack_raw
            )
            base_k_stack = (
                _dequant_dsv41_kv(base_k_stack_raw)
                if is_dsv41_q
                else base_k_stack_raw
            )
            dpos_list = [dpos for _, _, _, _, dpos in exact_group]
            k_ref_aligned = apply_relative_rope_rotation_batch(
                base_k_stack,
                dpos_list,
                rope_theta=policy.rope_theta,
                rotary_dim=0 if policy.is_mamba else policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
                rope_scaling=policy.rope_scaling,
                rope_at_tail=policy.rope_at_tail,
                rope_pos_scale=policy.rope_pos_scale,
            )
            k_fp32 = k_stack.to(torch.float32)
            delta_k = k_fp32 - k_ref_aligned.to(torch.float32)
            eta_k_t = torch.linalg.norm(delta_k.reshape(m, -1), dim=-1) / (
                torch.linalg.norm(k_fp32.reshape(m, -1), dim=-1).clamp(min=1e-8)
            )

            has_v = (
                v_cpus[exact_group[0][0]] is not None
                and base_kv_pairs[0][1] is not None
            )
            if has_v:
                v_stack = torch.stack(
                    [v_cpus[idx] for idx, _, _, _, _ in exact_group], dim=0
                )
                base_v_stack = torch.stack(
                    [kv[1] for kv in base_kv_pairs], dim=0  # type: ignore[misc]
                )
                v_fp32 = v_stack.to(torch.float32)
                delta_v = v_fp32 - base_v_stack.to(torch.float32)
                eta_v_t = torch.linalg.norm(delta_v.reshape(m, -1), dim=-1) / (
                    torch.linalg.norm(v_fp32.reshape(m, -1), dim=-1).clamp(
                        min=1e-8
                    )
                )
            else:
                v_stack = None
                v_fp32 = None
                delta_v = None
                eta_v_t = torch.zeros_like(eta_k_t)

            check_outliers = (
                not policy.is_mamba
                and policy.max_sparse_exceptions > 0
                and delta_k.ndim >= 3
                and delta_k.shape[1] >= 4
            )
            if check_outliers:
                b_tok = delta_k.shape[1]
                dk_tok = torch.linalg.norm(
                    delta_k.reshape(m, b_tok, -1), dim=-1
                )
                k_tok = torch.linalg.norm(
                    k_fp32.reshape(m, b_tok, -1), dim=-1
                ).clamp(min=1e-6)
                tok_eta = dk_tok / k_tok
                if has_v and delta_v is not None and v_fp32 is not None:
                    dv_tok = torch.linalg.norm(
                        delta_v.reshape(m, b_tok, -1), dim=-1
                    )
                    v_tok = torch.linalg.norm(
                        v_fp32.reshape(m, b_tok, -1), dim=-1
                    ).clamp(min=1e-6)
                    tok_eta = torch.maximum(tok_eta, dv_tok / v_tok)
                outlier_mask = tok_eta > 0.65
                num_outliers_t = outlier_mask.sum(dim=-1).to(torch.float32)
                metrics_rows = torch.stack(
                    [eta_k_t, eta_v_t, num_outliers_t], dim=-1
                ).tolist()
            else:
                outlier_mask = None
                metrics_rows = torch.stack([eta_k_t, eta_v_t], dim=-1).tolist()

            eff_tau_int2 = (
                policy.tau_int2 if is_fp8_mla else min(policy.tau_int2, 0.08)
            )
            tier1_g_idxs: list[int] = []
            tier2_g_idxs: list[int] = []

            for g_idx, (
                item_idx,
                base_rec,
                base_rec_2,
                stitch_offset,
                delta_pos,
            ) in enumerate(exact_group):
                row = metrics_rows[g_idx]
                eta_k = float(row[0])
                eta_v = float(row[1])
                num_outliers = int(row[2]) if check_outliers else 0
                eta_max = max(eta_k, eta_v)
                blk_spec = items[item_idx][0]
                is_mla = v_cpus[item_idx] is None
                ref_hash_2 = (
                    base_rec_2.full_block_hash
                    if base_rec_2 is not None
                    else None
                )

                if (
                    (
                        (
                            policy.is_sliding_window
                            or (is_dsv41_q and not is_fp8_mla)
                        )
                        and eta_max <= policy.tau_zero_bit
                    )
                    or is_dsv41_nvfp4
                    or (
                        (policy.is_mamba or is_fp8_mla)
                        and eta_max <= min(policy.tau_zero_bit, 0.01)
                    )
                ):
                    records[item_idx] = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER0_ZERO_BIT,
                        full_block_hash=blk_spec.full_block_hash,
                        token_block_hash=token_hashes[item_idx],
                        token_ids=token_tuples[item_idx],
                        start_position=blk_spec.start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=is_mla,
                        ref_base_hash=base_rec.full_block_hash,
                        delta_pos=delta_pos,
                        ref_base_hash_2=ref_hash_2,
                        stitch_offset=stitch_offset,
                    )
                    continue

                if (
                    check_outliers
                    and outlier_mask is not None
                    and 0 < num_outliers <= min(2, policy.max_sparse_exceptions)
                ):
                    outlier_idx = torch.nonzero(
                        outlier_mask[g_idx], as_tuple=False
                    ).reshape(-1)
                    dk_single = delta_k[g_idx].clone()
                    dk_single[outlier_idx] = 0.0
                    k_rem = k_fp32[g_idx].clone()
                    k_rem[outlier_idx] = 0.0
                    rem_eta_k_t = torch.linalg.norm(dk_single) / (
                        torch.linalg.norm(k_rem).clamp(min=1e-8)
                    )
                    if delta_v is not None and v_fp32 is not None:
                        dv_single = delta_v[g_idx].clone()
                        dv_single[outlier_idx] = 0.0
                        v_rem = v_fp32[g_idx].clone()
                        v_rem[outlier_idx] = 0.0
                        rem_eta_v_t = torch.linalg.norm(dv_single) / (
                            torch.linalg.norm(v_rem).clamp(min=1e-8)
                        )
                        rem_eta_k, rem_eta_v = torch.stack(
                            [rem_eta_k_t, rem_eta_v_t]
                        ).tolist()
                        rem_eta_max = max(rem_eta_k, rem_eta_v)
                    else:
                        dv_single = None
                        rem_eta_max = float(rem_eta_k_t.item())

                    if rem_eta_max <= policy.tau_int4:
                        k_c = k_cpus[item_idx]
                        v_c = v_cpus[item_idx]
                        sparse_k = k_c[outlier_idx].clone()
                        sparse_v = (
                            v_c[outlier_idx].clone() if v_c is not None else None
                        )
                        q_dk = quantize_residual_packed(
                            dk_single.to(k_stack.dtype), bits=delta_bits_4
                        )
                        q_dv = (
                            quantize_residual_packed(
                                dv_single.to(v_c.dtype), bits=4
                            )
                            if (dv_single is not None and v_c is not None)
                            else None
                        )
                        records[item_idx] = CompressedKVBlockRecord(
                            tier=CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA,
                            full_block_hash=blk_spec.full_block_hash,
                            token_block_hash=token_hashes[item_idx],
                            token_ids=token_tuples[item_idx],
                            start_position=blk_spec.start_position,
                            layer_idx=policy.layer_idx,
                            is_mla=is_mla,
                            ref_base_hash=base_rec.full_block_hash,
                            delta_pos=delta_pos,
                            ref_base_hash_2=ref_hash_2,
                            stitch_offset=stitch_offset,
                            quant_delta_k=q_dk,
                            quant_delta_v=q_dv,
                            sparse_token_indices=outlier_idx.to(torch.int16),
                            sparse_k_bf16=sparse_k,
                            sparse_v_bf16=sparse_v,
                        )
                        continue

                tier1_eligible = (
                    (eta_k <= policy.tau_int2 and eta_v <= policy.tau_int4)
                    if policy.is_mamba
                    else (
                        eta_max <= policy.tau_int4
                        if is_dsv41_nvfp4
                        else eta_max <= eff_tau_int2
                    )
                )
                if tier1_eligible:
                    tier1_g_idxs.append(g_idx)
                elif eta_max <= policy.tau_int4:
                    tier2_g_idxs.append(g_idx)

            if tier1_g_idxs:
                q_dk_list = quantize_residual_packed_batch(
                    delta_k[tier1_g_idxs].to(k_stack.dtype), bits=2
                )
                v_bits = 4 if policy.is_mamba else 2
                q_dv_list = (
                    quantize_residual_packed_batch(
                        delta_v[tier1_g_idxs].to(v_stack.dtype), bits=v_bits
                    )
                    if (delta_v is not None and v_stack is not None)
                    else [None] * len(tier1_g_idxs)
                )
                for sub_i, g_idx in enumerate(tier1_g_idxs):
                    (
                        item_idx,
                        base_rec,
                        base_rec_2,
                        stitch_offset,
                        delta_pos,
                    ) = exact_group[g_idx]
                    blk_spec = items[item_idx][0]
                    records[item_idx] = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER1_INT2_DELTA,
                        full_block_hash=blk_spec.full_block_hash,
                        token_block_hash=token_hashes[item_idx],
                        token_ids=token_tuples[item_idx],
                        start_position=blk_spec.start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=v_cpus[item_idx] is None,
                        ref_base_hash=base_rec.full_block_hash,
                        delta_pos=delta_pos,
                        ref_base_hash_2=(
                            base_rec_2.full_block_hash
                            if base_rec_2 is not None
                            else None
                        ),
                        stitch_offset=stitch_offset,
                        quant_delta_k=q_dk_list[sub_i],
                        quant_delta_v=q_dv_list[sub_i],
                    )

            if tier2_g_idxs:
                q_dk_list2 = quantize_residual_packed_batch(
                    delta_k[tier2_g_idxs].to(k_stack.dtype), bits=delta_bits_4
                )
                q_dv_list2 = (
                    quantize_residual_packed_batch(
                        delta_v[tier2_g_idxs].to(v_stack.dtype), bits=4
                    )
                    if (delta_v is not None and v_stack is not None)
                    else [None] * len(tier2_g_idxs)
                )
                for sub_i, g_idx in enumerate(tier2_g_idxs):
                    (
                        item_idx,
                        base_rec,
                        base_rec_2,
                        stitch_offset,
                        delta_pos,
                    ) = exact_group[g_idx]
                    blk_spec = items[item_idx][0]
                    records[item_idx] = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER2_INT4_DELTA,
                        full_block_hash=blk_spec.full_block_hash,
                        token_block_hash=token_hashes[item_idx],
                        token_ids=token_tuples[item_idx],
                        start_position=blk_spec.start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=v_cpus[item_idx] is None,
                        ref_base_hash=base_rec.full_block_hash,
                        delta_pos=delta_pos,
                        ref_base_hash_2=(
                            base_rec_2.full_block_hash
                            if base_rec_2 is not None
                            else None
                        ),
                        stitch_offset=stitch_offset,
                        quant_delta_k=q_dk_list2[sub_i],
                        quant_delta_v=q_dv_list2[sub_i],
                    )

        # Pass 2B: Batched GPU evaluation for full-attention partial (and stitched partial) matches
        if attn_partial_group and (
            _is_dsv41_quant_kv(k_cpus[attn_partial_group[0][0]])
            or is_multi_cohort
        ):
            for (
                item_idx,
                partial_base,
                partial_base_2,
                partial_stitch_offset,
                delta_pos,
                d_idx,
            ) in attn_partial_group:
                if (
                    not is_multi_cohort
                    and (delta_pos != 0 or partial_stitch_offset > 0)
                    and policy.layer_idx > 0
                ):
                    continue
                blk_spec = items[item_idx][0]
                k_c = k_cpus[item_idx]
                v_c = v_cpus[item_idx]
                b_tok = int(k_c.shape[0])
                t_len = len(token_tuples[item_idx])
                s_indices = (
                    sorted({(idx * b_tok) // t_len for idx in d_idx})
                    if (0 < b_tok < t_len)
                    else list(d_idx)
                )
                idx_tensor = torch.tensor(
                    s_indices, dtype=torch.long, device=k_c.device
                )
                records[item_idx] = CompressedKVBlockRecord(
                    tier=CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA,
                    full_block_hash=blk_spec.full_block_hash,
                    token_block_hash=token_hashes[item_idx],
                    token_ids=token_tuples[item_idx],
                    start_position=blk_spec.start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=v_c is None,
                    ref_base_hash=partial_base.full_block_hash,
                    delta_pos=delta_pos,
                    ref_base_hash_2=(
                        partial_base_2.full_block_hash
                        if partial_base_2 is not None
                        else None
                    ),
                    stitch_offset=partial_stitch_offset,
                    quant_delta_k=None,
                    quant_delta_v=None,
                    sparse_token_indices=idx_tensor.to(torch.int16),
                    sparse_k_bf16=k_c[idx_tensor].clone(),
                    sparse_v_bf16=(
                        v_c[idx_tensor].clone() if v_c is not None else None
                    ),
                )
            attn_partial_group = [
                it for it in attn_partial_group if records[it[0]] is None
            ]

        if attn_partial_group:
            p = len(attn_partial_group)
            k_stack_raw = torch.stack(
                [k_cpus[idx] for idx, _, _, _, _, _ in attn_partial_group],
                dim=0,
            )
            partial_kv_pairs = [
                self._get_record_base_kv(b1, b2, s_off)
                for _, b1, b2, s_off, _, _ in attn_partial_group
            ]
            base_k_stack_raw = torch.stack(
                [kv[0] for kv in partial_kv_pairs], dim=0
            )
            is_dsv41_q = _is_dsv41_quant_kv(k_stack_raw)
            is_dsv41_nvfp4 = (
                _is_dsv41_nvfp4_mla(k_stack_raw)
                or _is_dsv41_mxfp8_swa(k_stack_raw)
                or _is_dsv41_fp8_indexer(k_stack_raw)
            )
            is_fp8_mla = _is_fp8_e4m3_mla(k_stack_raw)
            delta_bits_4 = 2 if is_dsv41_nvfp4 else 4
            k_stack = (
                _dequant_dsv41_kv(k_stack_raw)
                if is_dsv41_q
                else k_stack_raw
            )
            base_k_stack = (
                _dequant_dsv41_kv(base_k_stack_raw)
                if is_dsv41_q
                else base_k_stack_raw
            )
            dpos_list = [dpos for _, _, _, _, dpos, _ in attn_partial_group]
            k_ref_aligned = apply_relative_rope_rotation_batch(
                base_k_stack,
                dpos_list,
                rope_theta=policy.rope_theta,
                rotary_dim=policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
                rope_scaling=policy.rope_scaling,
                rope_at_tail=policy.rope_at_tail,
                rope_pos_scale=policy.rope_pos_scale,
            )
            k_fp32 = k_stack.to(torch.float32)
            delta_k = k_fp32 - k_ref_aligned.to(torch.float32)
            has_v = (
                v_cpus[attn_partial_group[0][0]] is not None
                and partial_kv_pairs[0][1] is not None
            )
            if has_v:
                v_stack = torch.stack(
                    [v_cpus[idx] for idx, _, _, _, _, _ in attn_partial_group],
                    dim=0,
                )
                base_v_stack = torch.stack(
                    [kv[1] for kv in partial_kv_pairs], dim=0  # type: ignore[misc]
                )
                v_fp32 = v_stack.to(torch.float32)
                delta_v = v_fp32 - base_v_stack.to(torch.float32)
            else:
                v_stack = None
                v_fp32 = None
                delta_v = None

            b_tok = delta_k.shape[1]
            all_sparse_per_p: list[list[int]] = [
                sorted(
                    {
                        (idx * b_tok) // len(token_tuples[item_i])
                        for idx in d_idx
                    }
                )
                if (0 < b_tok < len(token_tuples[item_i]))
                else list(d_idx)
                for item_i, _, _, _, _, d_idx in attn_partial_group
            ]
            max_rem_slots = max(
                0,
                max(
                    policy.max_sparse_exceptions - len(s_idx)
                    for s_idx in all_sparse_per_p
                ),
            )
            if max_rem_slots > 0 and delta_k.ndim >= 3 and b_tok >= 4:
                dk_tok = torch.linalg.norm(
                    delta_k.reshape(p, b_tok, -1), dim=-1
                )
                k_tok = torch.linalg.norm(
                    k_fp32.reshape(p, b_tok, -1), dim=-1
                ).clamp(min=1e-6)
                tok_eta = dk_tok / k_tok
                if has_v and delta_v is not None and v_fp32 is not None:
                    dv_tok = torch.linalg.norm(
                        delta_v.reshape(p, b_tok, -1), dim=-1
                    )
                    v_tok = torch.linalg.norm(
                        v_fp32.reshape(p, b_tok, -1), dim=-1
                    ).clamp(min=1e-6)
                    tok_eta = torch.maximum(tok_eta, dv_tok / v_tok)
                diff_mask_cpu = torch.zeros((p, b_tok), dtype=torch.bool)
                for p_i, s_idx in enumerate(all_sparse_per_p):
                    if s_idx:
                        diff_mask_cpu[p_i, s_idx] = True
                tok_eta.masked_fill_(
                    diff_mask_cpu.to(device=tok_eta.device), 0.0
                )
                k_top = min(max_rem_slots, b_tok)
                top_vals, top_idxs = torch.topk(tok_eta, k=k_top, dim=-1)
                top_rows = torch.stack(
                    [top_vals, top_idxs.to(torch.float32)], dim=-1
                ).tolist()
                for p_i, row in enumerate(top_rows):
                    rem_s = policy.max_sparse_exceptions - len(
                        all_sparse_per_p[p_i]
                    )
                    if rem_s > 0:
                        for val, idx_f in row[:rem_s]:
                            if val > 0.65:
                                all_sparse_per_p[p_i].append(int(idx_f))
                        all_sparse_per_p[p_i].sort()

            sparse_mask_cpu = torch.zeros((p, b_tok), dtype=torch.bool)
            for p_i, s_idx in enumerate(all_sparse_per_p):
                if s_idx:
                    sparse_mask_cpu[p_i, s_idx] = True
            sparse_mask_gpu = sparse_mask_cpu.to(device=delta_k.device)
            mask_exp = sparse_mask_gpu.view(
                p, b_tok, *([1] * (delta_k.ndim - 2))
            )
            delta_k.masked_fill_(mask_exp, 0.0)
            k_rem = k_fp32.masked_fill(mask_exp, 0.0)
            eta_k_t = torch.linalg.norm(delta_k.reshape(p, -1), dim=-1) / (
                torch.linalg.norm(k_rem.reshape(p, -1), dim=-1).clamp(min=1e-8)
            )
            if has_v and delta_v is not None and v_fp32 is not None:
                delta_v.masked_fill_(mask_exp, 0.0)
                v_rem = v_fp32.masked_fill(mask_exp, 0.0)
                eta_v_t = torch.linalg.norm(delta_v.reshape(p, -1), dim=-1) / (
                    torch.linalg.norm(v_rem.reshape(p, -1), dim=-1).clamp(
                        min=1e-8
                    )
                )
                p_metrics = torch.stack([eta_k_t, eta_v_t], dim=-1).tolist()
            else:
                p_metrics = [[float(v), 0.0] for v in eta_k_t.tolist()]

            valid_p_idxs = [
                p_i
                for p_i, row in enumerate(p_metrics)
                if max(row[0], row[1]) <= policy.tau_int4
            ]
            if valid_p_idxs:
                if policy.layer_idx > 0 and not is_dsv41_nvfp4:
                    q_dk_p = quantize_residual_packed_batch(
                        delta_k[valid_p_idxs].to(k_stack.dtype),
                        bits=delta_bits_4,
                    )
                    q_dv_p = (
                        quantize_residual_packed_batch(
                            delta_v[valid_p_idxs].to(v_stack.dtype), bits=4
                        )
                        if (delta_v is not None and v_stack is not None)
                        else [None] * len(valid_p_idxs)
                    )
                else:
                    q_dk_p = [None] * len(valid_p_idxs)
                    q_dv_p = [None] * len(valid_p_idxs)

                for sub_i, p_i in enumerate(valid_p_idxs):
                    (
                        item_idx,
                        partial_base,
                        partial_base_2,
                        partial_stitch_offset,
                        delta_pos,
                        _,
                    ) = attn_partial_group[p_i]
                    blk_spec = items[item_idx][0]
                    k_c = k_cpus[item_idx]
                    v_c = v_cpus[item_idx]
                    idx_tensor = torch.tensor(
                        all_sparse_per_p[p_i],
                        dtype=torch.long,
                        device=k_c.device,
                    )
                    sparse_k = k_c[idx_tensor].clone()
                    sparse_v = (
                        v_c[idx_tensor].clone() if v_c is not None else None
                    )
                    records[item_idx] = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA,
                        full_block_hash=blk_spec.full_block_hash,
                        token_block_hash=token_hashes[item_idx],
                        token_ids=token_tuples[item_idx],
                        start_position=blk_spec.start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=v_c is None,
                        ref_base_hash=partial_base.full_block_hash,
                        delta_pos=delta_pos,
                        ref_base_hash_2=(
                            partial_base_2.full_block_hash
                            if partial_base_2 is not None
                            else None
                        ),
                        stitch_offset=partial_stitch_offset,
                        quant_delta_k=q_dk_p[sub_i],
                        quant_delta_v=q_dv_p[sub_i],
                        sparse_token_indices=idx_tensor.to(torch.int16),
                        sparse_k_bf16=sparse_k,
                        sparse_v_bf16=sparse_v,
                    )

        # Pass 3: Commit all records in exact sequential order
        max_exceptions = (
            self.max_sparse_exceptions
            if policy.is_mamba
            else policy.max_sparse_exceptions
        )
        batch_hash_set = set(seen_in_batch.keys())
        out_records: list[CompressedKVBlockRecord] = []
        for i in range(n):
            rec = records[i]
            if already_committed[i]:
                if rec is None:
                    first_i = seen_in_batch.get(items[i][0].full_block_hash)
                    if first_i is not None:
                        rec = records[first_i]
                if rec is not None:
                    out_records.append(rec)
                continue
            if rec is None:
                blk_spec = items[i][0]
                token_tuple = token_tuples[i]
                token_hash = token_hashes[i]
                k_c = k_cpus[i]
                v_c = v_cpus[i]
                rec = CompressedKVBlockRecord(
                    tier=CompressionTier.UNCOMPRESSED_BASE,
                    full_block_hash=blk_spec.full_block_hash,
                    token_block_hash=token_hash,
                    token_ids=token_tuple,
                    start_position=blk_spec.start_position,
                    layer_idx=policy.layer_idx,
                    is_mla=v_c is None,
                    prev_token_block_hash=blk_spec.prev_token_block_hash,
                    raw_k=k_c.clone(),
                    raw_v=v_c.clone() if v_c is not None else None,
                )
                if (
                    (self.enable_cross_turn_reuse and _is_dsv41_quant_kv(k_c))
                    or blk_spec.full_block_hash not in self._dependent_blocks
                ):
                    self._index_base_block(layer_name, rec, max_exceptions)
            out_records.append(
                self._commit_record(
                    layer_name,
                    rec,
                    raw_bytes_list[i],
                    batch_hashes=batch_hash_set,
                    root_block_hash=items[i][0].root_block_hash,
                )
            )
        return out_records

    def evict_block(
        self, full_block_hash: str, count_as_eviction: bool = True
    ) -> list[str]:
        """Evict `full_block_hash` (and any dependent delta blocks) across all layers."""
        evicted: list[str] = []
        was_known = (full_block_hash in self._block_lru) or (
            full_block_hash in self._known_prefix_blocks
        )
        deps = list(self._dependents_by_base.pop(full_block_hash, set()))
        for dep_hash in deps:
            if dep_hash != full_block_hash and dep_hash in self._block_lru:
                evicted.extend(
                    self.evict_block(
                        dep_hash, count_as_eviction=count_as_eviction
                    )
                )

        all_layers = set(self.layer_policies.keys()) | self._known_layers
        for layer_name in all_layers:
            rec = self._records.pop((layer_name, full_block_hash), None)
            raw_b = self._record_raw_bytes.pop((layer_name, full_block_hash), 0)
            if rec is None:
                continue
            self.stats.compressed_bytes_stored = max(
                0, self.stats.compressed_bytes_stored - rec.nbytes
            )
            self.stats.raw_bytes_equivalent = max(
                0, self.stats.raw_bytes_equivalent - raw_b
            )
            self.stats.tier_counts[rec.tier] = max(
                0, self.stats.tier_counts.get(rec.tier, 0) - 1
            )
            if count_as_eviction:
                self.stats.evicted_blocks += 1
                self.stats.evicted_bytes += rec.nbytes

            if rec.ref_base_hash and rec.ref_base_hash in self._dependents_by_base:
                self._dependents_by_base[rec.ref_base_hash].discard(full_block_hash)
            if (
                rec.ref_base_hash_2
                and rec.ref_base_hash_2 in self._dependents_by_base
            ):
                self._dependents_by_base[rec.ref_base_hash_2].discard(
                    full_block_hash
                )

            if rec.tier == CompressionTier.UNCOMPRESSED_BASE:
                t_key = (layer_name, rec.token_block_hash)
                t_list = self._base_by_token_hash_list.get(t_key)
                if t_list:
                    self._base_by_token_hash_list[t_key] = [
                        r for r in t_list if r.full_block_hash != full_block_hash
                    ]
                    if self._base_by_token_hash_list[t_key]:
                        self._base_by_token_hash[t_key] = (
                            self._base_by_token_hash_list[t_key][0]
                        )
                    else:
                        self._base_by_token_hash_list.pop(t_key, None)
                        self._base_by_token_hash.pop(t_key, None)
                else:
                    self._base_by_token_hash.pop(t_key, None)

                l_list = self._base_blocks_by_layer.get(layer_name)
                if l_list:
                    self._base_blocks_by_layer[layer_name] = [
                        r for r in l_list if r.full_block_hash != full_block_hash
                    ]

                b_len = len(rec.token_ids)
                if b_len >= 2:
                    w = self._get_boundary_wgram_size(b_len)
                    pos_key = (layer_name, rec.start_position)
                    pos_list = self._base_by_layer_pos.get(pos_key)
                    if pos_list:
                        new_pos = [
                            r
                            for r in pos_list
                            if r.full_block_hash != full_block_hash
                        ]
                        if new_pos:
                            self._base_by_layer_pos[pos_key] = new_pos
                        else:
                            self._base_by_layer_pos.pop(pos_key, None)

                    suf_key = (layer_name, rec.token_ids[-w:])
                    suf_list = self._base_by_suffix_wgram.get(suf_key)
                    if suf_list:
                        new_suf = [
                            r
                            for r in suf_list
                            if r.full_block_hash != full_block_hash
                        ]
                        if new_suf:
                            self._base_by_suffix_wgram[suf_key] = new_suf
                        else:
                            self._base_by_suffix_wgram.pop(suf_key, None)

                    pre_key = (layer_name, rec.token_ids[:w])
                    pre_list = self._base_by_prefix_wgram.get(pre_key)
                    if pre_list:
                        new_pre = [
                            r
                            for r in pre_list
                            if r.full_block_hash != full_block_hash
                        ]
                        if new_pre:
                            self._base_by_prefix_wgram[pre_key] = new_pre
                        else:
                            self._base_by_prefix_wgram.pop(pre_key, None)

                policy = self.get_layer_policy(layer_name)
                max_exceptions = (
                    self.max_sparse_exceptions
                    if policy.is_mamba
                    else policy.max_sparse_exceptions
                )
                if max_exceptions > 0:
                    num_slices = max_exceptions + 1
                    for s_idx, sub_tuple in self._get_pigeonhole_slices(
                        rec.token_ids, num_slices
                    ):
                        s_key = (layer_name, s_idx, sub_tuple)
                        s_bucket = self._base_by_subchunk.get(s_key)
                        if s_bucket:
                            new_b = [
                                r
                                for r in s_bucket
                                if r.full_block_hash != full_block_hash
                            ]
                            if new_b:
                                self._base_by_subchunk[s_key] = new_b
                            else:
                                self._base_by_subchunk.pop(s_key, None)

        tok_pos = self._known_prefix_blocks.pop(full_block_hash, None)
        self._known_mamba_blocks.discard(full_block_hash)
        if tok_pos is not None:
            tok_hash = compute_token_block_hash(tok_pos[0])
            if self._known_token_blocks.get(tok_hash) == full_block_hash:
                self._known_token_blocks.pop(tok_hash, None)

        self._block_lru.pop(full_block_hash, None)
        self._demoted_blocks.discard(full_block_hash)
        self._retired_blocks.discard(full_block_hash)
        self._attention_base_blocks.discard(full_block_hash)
        self._pos0_base_blocks.discard(full_block_hash)
        self._dependent_blocks.discard(full_block_hash)
        if was_known and count_as_eviction:
            self.stats.evicted_unique_blocks += 1
        self._pigeonhole_slice_cache.pop(
            (full_block_hash, self.max_sparse_exceptions + 1), None
        )
        for dev in self._known_slab_devices:
            bkey = (dev, full_block_hash)
            self._gpu_slabs_by_block.pop(bkey, None)
            self._gpu_slab_offsets_by_block.pop(bkey, None)

        evicted.append(full_block_hash)
        return evicted

    def _is_attention_base_block(self, full_block_hash: str) -> bool:
        """Return True if `full_block_hash` is an UNCOMPRESSED_BASE block in any attention layer."""
        if full_block_hash in self._dependent_blocks:
            return False
        if full_block_hash in self._attention_base_blocks:
            return True
        for layer_name in self._known_layers:
            policy = self.get_layer_policy(layer_name)
            if not policy.is_mamba:
                rec = self._records.get((layer_name, full_block_hash))
                if rec is not None and rec.tier == CompressionTier.UNCOMPRESSED_BASE:
                    self._attention_base_blocks.add(full_block_hash)
                    return True
        return False

    def _is_pos0_base_block(self, full_block_hash: str) -> bool:
        """Return True if `full_block_hash` is a `start_position == 0` UNCOMPRESSED_BASE block."""
        if full_block_hash in self._dependent_blocks:
            return False
        if full_block_hash in self._pos0_base_blocks:
            return True
        for layer_name in self._known_layers:
            policy = self.get_layer_policy(layer_name)
            if not policy.is_mamba:
                rec = self._records.get((layer_name, full_block_hash))
                if (
                    rec is not None
                    and rec.tier == CompressionTier.UNCOMPRESSED_BASE
                    and rec.start_position == 0
                ):
                    self._pos0_base_blocks.add(full_block_hash)
                    return True
        return False

    def evict_lru_to_budget(
        self, max_bytes: int, protected_hashes: set[str] | None = None
    ) -> list[str]:
        """Evict least-recently-used blocks until `compressed_bytes_stored <= max_bytes`."""
        if max_bytes <= 0 or self.stats.compressed_bytes_stored <= max_bytes:
            return []
        effective_protected: set[str] = set(protected_hashes or ())
        if effective_protected and self._dependents_by_base:
            changed = True
            while changed:
                changed = False
                for base_h, deps in self._dependents_by_base.items():
                    if base_h not in effective_protected and (deps & effective_protected):
                        effective_protected.add(base_h)
                        changed = True

        evicted_all: list[str] = []
        # Pass 1A: Prefer evicting demoted compressed delta leaf blocks (0 live dependents) first
        for cand in list(self._block_lru):
            if self.stats.compressed_bytes_stored <= max_bytes:
                break
            if (
                cand in self._block_lru
                and cand in self._demoted_blocks
                and cand not in effective_protected
                and not self._dependents_by_base.get(cand)
                and not self._is_attention_base_block(cand)
            ):
                evicted_all.extend(self.evict_block(cand))
        # Pass 1B: Evict demoted unreferenced start_position==0 UNCOMPRESSED_BASE blocks (superseded action headers)
        if self.stats.compressed_bytes_stored > max_bytes:
            for cand in list(self._block_lru):
                if self.stats.compressed_bytes_stored <= max_bytes:
                    break
                if (
                    cand in self._block_lru
                    and cand in self._demoted_blocks
                    and cand not in effective_protected
                    and not self._dependents_by_base.get(cand)
                    and self._is_pos0_base_block(cand)
                ):
                    evicted_all.extend(self.evict_block(cand))
        # Pass 1C: Evict completed trajectories' remaining unreferenced intermediate UNCOMPRESSED_BASE blocks
        if self.stats.compressed_bytes_stored > max_bytes:
            for cand in list(self._block_lru):
                if self.stats.compressed_bytes_stored <= max_bytes:
                    break
                if (
                    cand in self._block_lru
                    and cand in self._retired_blocks
                    and cand not in effective_protected
                    and not self._dependents_by_base.get(cand)
                ):
                    evicted_all.extend(self.evict_block(cand))
        # Pass 1D: Evict remaining demoted unreferenced UNCOMPRESSED_BASE blocks if still above budget
        if self.stats.compressed_bytes_stored > max_bytes:
            for cand in list(self._block_lru):
                if self.stats.compressed_bytes_stored <= max_bytes:
                    break
                if (
                    cand in self._block_lru
                    and cand in self._demoted_blocks
                    and cand not in effective_protected
                    and not self._dependents_by_base.get(cand)
                ):
                    evicted_all.extend(self.evict_block(cand))
        # Pass 1E: Evict any remaining unreferenced non-base delta leaf blocks in LRU order
        if self.stats.compressed_bytes_stored > max_bytes:
            for cand in list(self._block_lru):
                if self.stats.compressed_bytes_stored <= max_bytes:
                    break
                if (
                    cand in self._block_lru
                    and cand not in effective_protected
                    and not self._dependents_by_base.get(cand)
                    and not self._is_attention_base_block(cand)
                ):
                    evicted_all.extend(self.evict_block(cand))
        # Pass 1F: Evict any remaining unreferenced blocks (0 live dependents) in strict LRU order
        if self.stats.compressed_bytes_stored > max_bytes:
            for cand in list(self._block_lru):
                if self.stats.compressed_bytes_stored <= max_bytes:
                    break
                if (
                    cand in self._block_lru
                    and cand not in effective_protected
                    and not self._dependents_by_base.get(cand)
                ):
                    evicted_all.extend(self.evict_block(cand))
        # Pass 2: Fall back to oldest unprotected anchor block if still above budget
        if self.stats.compressed_bytes_stored > max_bytes:
            for cand in list(self._block_lru):
                if self.stats.compressed_bytes_stored <= max_bytes:
                    break
                if cand in self._block_lru and cand not in effective_protected:
                    evicted_all.extend(self.evict_block(cand))
        return evicted_all

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
             when only a token_block_hash match or stitched consecutive-base match exists.
        """
        policy = self.get_layer_policy(layer_name)
        rec = self._records.get((layer_name, full_block_hash))
        if rec is not None:
            self._touch_lru(
                full_block_hash, rec.ref_base_hash, rec.ref_base_hash_2
            )

        # If exact full_block_hash wasn't stored, check if we can synthesize from
        # a token_block_hash match or stitched base pair (e.g. Layer 0 or LOCAL_SLIDING zero-bit hit).
        if rec is None and token_ids is not None:
            token_tuple = tuple(int(t) for t in token_ids)
            token_hash = compute_token_block_hash(token_tuple)
            base_rec = self._find_exact_base_block(
                layer_name, token_hash, start_position=target_start_position
            )
            base_rec_2: CompressedKVBlockRecord | None = None
            stitch_offset: int = 0
            if base_rec is None and not policy.is_mamba:
                b0, b1, s_off, _ = self._find_stitched_base_blocks(
                    layer_name,
                    token_tuple,
                    start_position=target_start_position,
                    max_edits=0,
                )
                if b0 is not None and b1 is not None and s_off > 0:
                    base_rec = b0
                    base_rec_2 = b1
                    stitch_offset = s_off

            if base_rec is not None:
                raw_k_ref, raw_v_ref = self._get_record_base_kv(
                    base_rec, base_rec_2, stitch_offset
                )
                delta_pos = (
                    (
                        target_start_position
                        - (base_rec.start_position + stitch_offset)
                    )
                    if target_start_position is not None
                    else 0
                )
                is_dsv41_q = _is_dsv41_quant_kv(raw_k_ref)
                if is_dsv41_q and delta_pos == 0:
                    k_recon = raw_k_ref.clone()
                else:
                    k_ref_work = (
                        _dequant_dsv41_kv(raw_k_ref)
                        if is_dsv41_q
                        else raw_k_ref
                    )
                    k_recon = apply_relative_rope_rotation(
                        k_ref_work,
                        delta_pos=delta_pos,
                        rope_theta=policy.rope_theta,
                        rotary_dim=policy.rotary_dim,
                        is_neox_style=policy.is_neox_style,
                        rope_scaling=policy.rope_scaling,
                        rope_at_tail=policy.rope_at_tail,
                        rope_pos_scale=policy.rope_pos_scale,
                    )
                    if is_dsv41_q:
                        k_recon = _requant_dsv41_kv(
                            k_recon, raw_k_ref, rope_only=True
                        )
                v_recon = (
                    raw_v_ref.clone() if raw_v_ref is not None else None
                )
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

        # Single-hop base lookup (or stitched pair lookup for Approach 1)
        assert rec.ref_base_hash is not None
        base_rec = self._records[(layer_name, rec.ref_base_hash)]
        assert base_rec.tier == CompressionTier.UNCOMPRESSED_BASE
        assert base_rec.raw_k is not None
        base_rec_2 = (
            self._records[(layer_name, rec.ref_base_hash_2)]
            if rec.ref_base_hash_2
            else None
        )
        raw_k_ref, raw_v_ref = self._get_record_base_kv(
            base_rec, base_rec_2, rec.stitch_offset
        )
        is_dsv41_q = _is_dsv41_quant_kv(raw_k_ref)
        v_base = raw_v_ref.clone() if raw_v_ref is not None else None

        if rec.tier == CompressionTier.TIER0_ZERO_BIT:
            if is_dsv41_q and rec.delta_pos == 0:
                return raw_k_ref.clone(), v_base
            k_ref_work = (
                _dequant_dsv41_kv(raw_k_ref) if is_dsv41_q else raw_k_ref
            )
            k_aligned = apply_relative_rope_rotation(
                k_ref_work,
                delta_pos=rec.delta_pos,
                rope_theta=policy.rope_theta,
                rotary_dim=policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
                rope_scaling=policy.rope_scaling,
                rope_at_tail=policy.rope_at_tail,
                rope_pos_scale=policy.rope_pos_scale,
            )
            if is_dsv41_q:
                k_aligned = _requant_dsv41_kv(
                    k_aligned, raw_k_ref, rope_only=True
                )
            return k_aligned, v_base

        k_ref_work = _dequant_dsv41_kv(raw_k_ref) if is_dsv41_q else raw_k_ref

        k_aligned = apply_relative_rope_rotation(
            k_ref_work,
            delta_pos=rec.delta_pos,
            rope_theta=policy.rope_theta,
            rotary_dim=policy.rotary_dim,
            is_neox_style=policy.is_neox_style,
            rope_scaling=policy.rope_scaling,
            rope_at_tail=policy.rope_at_tail,
            rope_pos_scale=policy.rope_pos_scale,
        )

        if rec.tier in (
            CompressionTier.TIER1_INT2_DELTA,
            CompressionTier.TIER2_INT4_DELTA,
        ):
            assert rec.quant_delta_k is not None
            dk = dequantize_residual_packed(rec.quant_delta_k)
            k_out = (k_aligned.to(torch.float32) + dk.to(torch.float32)).to(
                k_aligned.dtype
            )
            if is_dsv41_q:
                k_out = _requant_dsv41_kv(k_out, raw_k_ref)
            if v_base is not None and rec.quant_delta_v is not None:
                dv = dequantize_residual_packed(rec.quant_delta_v)
                v_out = (v_base.to(torch.float32) + dv.to(torch.float32)).to(
                    v_base.dtype
                )
            else:
                v_out = v_base
            return k_out, v_out

        if rec.tier == CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA:
            if is_dsv41_q and rec.quant_delta_k is None and rec.delta_pos == 0:
                k_out_typed = raw_k_ref.clone()
            else:
                k_out = k_aligned.to(torch.float32)
                if rec.quant_delta_k is not None:
                    dk = dequantize_residual_packed(rec.quant_delta_k)
                    k_out = k_out + dk.to(torch.float32)
                k_out_typed = k_out.to(k_aligned.dtype)
                if is_dsv41_q:
                    k_out_typed = _requant_dsv41_kv(
                        k_out_typed,
                        raw_k_ref,
                        rope_only=(rec.quant_delta_k is None),
                    )

            if v_base is not None:
                v_out = v_base.to(torch.float32)
                if rec.quant_delta_v is not None:
                    dv = dequantize_residual_packed(rec.quant_delta_v)
                    v_out = v_out + dv.to(torch.float32)
                v_out_typed: torch.Tensor | None = v_out.to(v_base.dtype)
            else:
                v_out_typed = None

            # Overwrite edited token positions with exact BF16 (or exact uint8) rows
            if rec.sparse_token_indices is not None and rec.sparse_k_bf16 is not None:
                idx = rec.sparse_token_indices.to(torch.long)
                k_out_typed[idx] = rec.sparse_k_bf16
                if v_out_typed is not None and rec.sparse_v_bf16 is not None:
                    v_out_typed[idx] = rec.sparse_v_bf16
            return k_out_typed, v_out_typed

        raise ValueError(f"Unsupported compression tier: {rec.tier}")

    def reconstruct_blocks_batch(
        self,
        layer_name: str,
        specs: list["DeltaBlockTransferSpec"],
        clone_outputs: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor | None] | None]:
        """Batch-reconstruct multiple KV blocks for `layer_name` in a single pass."""
        n = len(specs)
        if n == 0:
            return []
        if n == 1 and not self.enable_cross_turn_reuse:
            try:
                return [
                    self.reconstruct_block(
                        layer_name=layer_name,
                        full_block_hash=specs[0].full_block_hash,
                        token_ids=specs[0].token_ids,
                        target_start_position=specs[0].start_position,
                    )
                ]
            except KeyError:
                return [None]

        policy = self.get_layer_policy(layer_name)
        results: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [
            None
        ] * n

        # Group indices by tier for batched RoPE rotation + dequantization
        tier0_items: list[
            tuple[list[int], torch.Tensor, torch.Tensor | None, int]
        ] = []
        tier0_dedup: dict[tuple[int, int, int], list[int]] = {}
        tier1_items: list[
            tuple[
                list[int],
                CompressedKVBlockRecord,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []
        tier1_dedup: dict[int, list[int]] = {}
        tier2_items: list[
            tuple[
                list[int],
                CompressedKVBlockRecord,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []
        tier2_dedup: dict[int, list[int]] = {}
        tier3_items: list[
            tuple[
                list[int],
                CompressedKVBlockRecord,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []
        tier3_dedup: dict[int, list[int]] = {}

        for i, blk_spec in enumerate(specs):
            rec = self._records.get((layer_name, blk_spec.full_block_hash))
            if rec is not None:
                if policy.layer_idx == 0:
                    self._touch_lru(
                        blk_spec.full_block_hash,
                        rec.ref_base_hash,
                        rec.ref_base_hash_2,
                    )
                if (
                    self.enable_cross_turn_reuse
                    and rec.tier == CompressionTier.TIER0_ZERO_BIT
                ):
                    pos_rec = self._record_by_tok_pos.get(
                        (layer_name, rec.token_block_hash, rec.start_position)
                    )
                    if pos_rec is not None and (
                        (
                            pos_rec.tier == CompressionTier.UNCOMPRESSED_BASE
                            and (layer_name, pos_rec.full_block_hash)
                            in self._records
                        )
                        or (
                            (
                                pos_rec.quant_delta_k is not None
                                or pos_rec.sparse_k_bf16 is not None
                            )
                            and pos_rec.ref_base_hash is not None
                            and (layer_name, pos_rec.ref_base_hash)
                            in self._records
                        )
                    ):
                        rec = pos_rec
            else:
                tok_hash_cache = getattr(self, "_tok_hash_by_full_hash", None)
                if tok_hash_cache is None:
                    tok_hash_cache = {}
                    self._tok_hash_by_full_hash = tok_hash_cache
                cached_th = (
                    tok_hash_cache.get(blk_spec.full_block_hash)
                    if blk_spec.full_block_hash
                    else None
                )
                if cached_th is not None:
                    token_tuple, token_hash = cached_th
                else:
                    token_tuple = (
                        blk_spec.token_ids
                        if isinstance(blk_spec.token_ids, tuple)
                        else tuple(int(t) for t in blk_spec.token_ids)
                    )
                    token_hash = compute_token_block_hash(token_tuple)
                    if blk_spec.full_block_hash:
                        if len(tok_hash_cache) > 262144:
                            tok_hash_cache.clear()
                        tok_hash_cache[blk_spec.full_block_hash] = (
                            token_tuple,
                            token_hash,
                        )
                pos_rec = (
                    self._record_by_tok_pos.get(
                        (layer_name, token_hash, blk_spec.start_position)
                    )
                    if self.enable_cross_turn_reuse
                    else None
                )
                if pos_rec is not None and (
                    (
                        pos_rec.tier == CompressionTier.UNCOMPRESSED_BASE
                        and (layer_name, pos_rec.full_block_hash)
                        in self._records
                    )
                    or (
                        pos_rec.ref_base_hash is not None
                        and (layer_name, pos_rec.ref_base_hash) in self._records
                    )
                ):
                    ref_h = (
                        pos_rec.full_block_hash
                        if pos_rec.tier == CompressionTier.UNCOMPRESSED_BASE
                        else pos_rec.ref_base_hash
                    )
                    raw_b = self._record_raw_bytes.get(
                        (layer_name, pos_rec.full_block_hash),
                        self._record_raw_bytes.get((layer_name, ref_h or ""), 0),
                    )
                    zero_rec = CompressedKVBlockRecord(
                        tier=CompressionTier.TIER0_ZERO_BIT,
                        full_block_hash=blk_spec.full_block_hash,
                        token_block_hash=token_hash,
                        token_ids=token_tuple,
                        start_position=blk_spec.start_position,
                        layer_idx=policy.layer_idx,
                        is_mla=pos_rec.is_mla,
                        ref_base_hash=ref_h,
                        delta_pos=(
                            0
                            if pos_rec.tier == CompressionTier.UNCOMPRESSED_BASE
                            else pos_rec.delta_pos
                        ),
                    )
                    self._commit_record(
                        layer_name,
                        zero_rec,
                        raw_b,
                        root_block_hash=blk_spec.root_block_hash,
                    )
                    rec = pos_rec
                else:
                    base_rec = self._find_exact_base_block(
                        layer_name,
                        token_hash,
                        start_position=blk_spec.start_position,
                    )
                    base_rec_2: CompressedKVBlockRecord | None = None
                    stitch_offset: int = 0
                    if (
                        base_rec is None
                        and self.enable_cross_turn_reuse
                        and not policy.is_mamba
                        and policy.max_sparse_exceptions > 0
                    ):
                        p_base, _ = self._find_partial_match_base_block(
                            layer_name,
                            token_tuple,
                            start_position=blk_spec.start_position,
                            prev_token_block_hash=blk_spec.prev_token_block_hash,
                            full_block_hash=blk_spec.full_block_hash,
                        )
                        if p_base is not None:
                            base_rec = p_base
                    if (
                        base_rec is None
                        and not policy.is_mamba
                        and not self.enable_cross_turn_reuse
                    ):
                        b0, b1, s_off, _ = self._find_stitched_base_blocks(
                            layer_name,
                            token_tuple,
                            start_position=blk_spec.start_position,
                            max_edits=0,
                            full_block_hash=blk_spec.full_block_hash,
                        )
                        if b0 is not None and b1 is not None and s_off > 0:
                            base_rec = b0
                            base_rec_2 = b1
                            stitch_offset = s_off

                    if base_rec is not None and base_rec.raw_k is not None:
                        delta_pos_val = blk_spec.start_position - (
                            base_rec.start_position + stitch_offset
                        )
                        if self.enable_cross_turn_reuse:
                            raw_b = self._record_raw_bytes.get(
                                (layer_name, base_rec.full_block_hash), 0
                            )
                            zero_rec = CompressedKVBlockRecord(
                                tier=CompressionTier.TIER0_ZERO_BIT,
                                full_block_hash=blk_spec.full_block_hash,
                                token_block_hash=token_hash,
                                token_ids=token_tuple,
                                start_position=blk_spec.start_position,
                                layer_idx=policy.layer_idx,
                                is_mla=base_rec.is_mla,
                                ref_base_hash=base_rec.full_block_hash,
                                delta_pos=delta_pos_val,
                                ref_base_hash_2=(
                                    base_rec_2.full_block_hash
                                    if base_rec_2 is not None
                                    else None
                                ),
                                stitch_offset=stitch_offset,
                            )
                            self._commit_record(
                                layer_name,
                                zero_rec,
                                raw_b,
                                root_block_hash=blk_spec.root_block_hash,
                            )
                        self.stats.total_blocks_loaded += 1
                        if (
                            delta_pos_val == 0
                            and stitch_offset == 0
                            and base_rec_2 is None
                        ):
                            if clone_outputs:
                                results[i] = (
                                    base_rec.raw_k.clone(),
                                    (
                                        base_rec.raw_v.clone()
                                        if base_rec.raw_v is not None
                                        else None
                                    ),
                                )
                            else:
                                results[i] = (base_rec.raw_k, base_rec.raw_v)
                            continue
                        rk, rv = self._get_record_base_kv(
                            base_rec, base_rec_2, stitch_offset
                        )
                        t0_key = (
                            id(base_rec),
                            id(base_rec_2) if base_rec_2 is not None else 0,
                            stitch_offset * 1000003 + delta_pos_val,
                        )
                        if t0_key in tier0_dedup:
                            tier0_dedup[t0_key].append(i)
                        else:
                            idx_list = [i]
                            tier0_dedup[t0_key] = idx_list
                            tier0_items.append(
                                (
                                    idx_list,
                                    rk,
                                    rv,
                                    delta_pos_val,
                                )
                            )
                    continue

            self.stats.total_blocks_loaded += 1
            if rec.tier == CompressionTier.UNCOMPRESSED_BASE:
                assert rec.raw_k is not None
                if clone_outputs:
                    results[i] = (
                        rec.raw_k.clone(),
                        rec.raw_v.clone() if rec.raw_v is not None else None,
                    )
                else:
                    results[i] = (rec.raw_k, rec.raw_v)
                continue

            assert rec.ref_base_hash is not None
            base_rec = self._records.get((layer_name, rec.ref_base_hash))
            if base_rec is None or base_rec.raw_k is None:
                continue
            base_rec_2 = (
                self._records.get((layer_name, rec.ref_base_hash_2))
                if rec.ref_base_hash_2
                else None
            )
            if rec.ref_base_hash_2 and (
                base_rec_2 is None or base_rec_2.raw_k is None
            ):
                continue

            rec_id = id(rec)
            if rec.tier == CompressionTier.TIER0_ZERO_BIT:
                if (
                    rec.delta_pos == 0
                    and rec.stitch_offset == 0
                    and base_rec_2 is None
                ):
                    if clone_outputs:
                        results[i] = (
                            base_rec.raw_k.clone(),
                            (
                                base_rec.raw_v.clone()
                                if base_rec.raw_v is not None
                                else None
                            ),
                        )
                    else:
                        results[i] = (base_rec.raw_k, base_rec.raw_v)
                    continue
                t0_key = (
                    id(base_rec),
                    id(base_rec_2) if base_rec_2 is not None else 0,
                    rec.stitch_offset * 1000003 + rec.delta_pos,
                )
                if t0_key in tier0_dedup:
                    tier0_dedup[t0_key].append(i)
                else:
                    rk, rv = self._get_record_base_kv(
                        base_rec, base_rec_2, rec.stitch_offset
                    )
                    idx_list = [i]
                    tier0_dedup[t0_key] = idx_list
                    tier0_items.append((idx_list, rk, rv, rec.delta_pos))
            elif rec.tier == CompressionTier.TIER1_INT2_DELTA:
                if rec_id in tier1_dedup:
                    tier1_dedup[rec_id].append(i)
                else:
                    rk, rv = self._get_record_base_kv(
                        base_rec, base_rec_2, rec.stitch_offset
                    )
                    idx_list = [i]
                    tier1_dedup[rec_id] = idx_list
                    tier1_items.append((idx_list, rec, rk, rv))
            elif rec.tier == CompressionTier.TIER2_INT4_DELTA:
                if rec_id in tier2_dedup:
                    tier2_dedup[rec_id].append(i)
                else:
                    rk, rv = self._get_record_base_kv(
                        base_rec, base_rec_2, rec.stitch_offset
                    )
                    idx_list = [i]
                    tier2_dedup[rec_id] = idx_list
                    tier2_items.append((idx_list, rec, rk, rv))
            elif rec.tier == CompressionTier.TIER3_SPARSE_BF16_INT4_DELTA:
                if rec_id in tier3_dedup:
                    tier3_dedup[rec_id].append(i)
                else:
                    rk, rv = self._get_record_base_kv(
                        base_rec, base_rec_2, rec.stitch_offset
                    )
                    idx_list = [i]
                    tier3_dedup[rec_id] = idx_list
                    tier3_items.append((idx_list, rec, rk, rv))

        if tier0_items:
            base_k_s_raw = torch.stack(
                [rk for _, rk, _, _ in tier0_items], dim=0
            )
            is_dsv41_q = _is_dsv41_quant_kv(base_k_s_raw)
            dpos_s = [d for _, _, _, d in tier0_items]
            if is_dsv41_q and all(d == 0 for d in dpos_s):
                k_aligned_s = (
                    base_k_s_raw.clone() if clone_outputs else base_k_s_raw
                )
            else:
                base_k_s = (
                    _dequant_dsv41_kv(base_k_s_raw)
                    if is_dsv41_q
                    else base_k_s_raw
                )
                k_aligned_s = apply_relative_rope_rotation_batch(
                    base_k_s,
                    dpos_s,
                    rope_theta=policy.rope_theta,
                    rotary_dim=policy.rotary_dim,
                    is_neox_style=policy.is_neox_style,
                    rope_scaling=policy.rope_scaling,
                    rope_at_tail=policy.rope_at_tail,
                    rope_pos_scale=policy.rope_pos_scale,
                )
                if is_dsv41_q:
                    k_aligned_s = _requant_dsv41_kv(
                        k_aligned_s, base_k_s_raw, rope_only=True
                    )
            for sub_i, (item_indices, _, rv, _) in enumerate(tier0_items):
                k_sub = k_aligned_s[sub_i]
                for item_idx in item_indices:
                    results[item_idx] = (
                        k_sub,
                        (
                            rv.clone()
                            if (rv is not None and clone_outputs)
                            else rv
                        ),
                    )

        for dense_group in (tier1_items, tier2_items):
            if not dense_group:
                continue
            base_k_s_raw = torch.stack(
                [rk for _, _, rk, _ in dense_group], dim=0
            )
            is_dsv41_q = _is_dsv41_quant_kv(base_k_s_raw)
            base_k_s = (
                _dequant_dsv41_kv(base_k_s_raw)
                if is_dsv41_q
                else base_k_s_raw
            )
            dpos_s = [r.delta_pos for _, r, _, _ in dense_group]
            k_aligned_s = apply_relative_rope_rotation_batch(
                base_k_s,
                dpos_s,
                rope_theta=policy.rope_theta,
                rotary_dim=policy.rotary_dim,
                is_neox_style=policy.is_neox_style,
                rope_scaling=policy.rope_scaling,
                rope_at_tail=policy.rope_at_tail,
                rope_pos_scale=policy.rope_pos_scale,
            )
            dk_batch = dequantize_residual_packed_batch(
                [r.quant_delta_k for _, r, _, _ in dense_group]  # type: ignore[misc]
            )
            k_out_s = (
                k_aligned_s.to(torch.float32) + dk_batch.to(torch.float32)
            ).to(k_aligned_s.dtype)
            if is_dsv41_q:
                k_out_s = _requant_dsv41_kv(k_out_s, base_k_s_raw)

            has_v = (
                dense_group[0][3] is not None
                and dense_group[0][1].quant_delta_v is not None
            )
            if has_v:
                base_v_s = torch.stack(
                    [rv for _, _, _, rv in dense_group], dim=0  # type: ignore[misc]
                )
                dv_batch = dequantize_residual_packed_batch(
                    [r.quant_delta_v for _, r, _, _ in dense_group]  # type: ignore[misc]
                )
                v_out_s = (
                    base_v_s.to(torch.float32) + dv_batch.to(torch.float32)
                ).to(base_v_s.dtype)
                for sub_i, (item_indices, _, _, _) in enumerate(dense_group):
                    k_sub = k_out_s[sub_i]
                    v_sub = v_out_s[sub_i]
                    for item_idx in item_indices:
                        results[item_idx] = (k_sub, v_sub)
            else:
                for sub_i, (item_indices, _, _, rv) in enumerate(dense_group):
                    k_sub = k_out_s[sub_i]
                    for item_idx in item_indices:
                        results[item_idx] = (
                            k_sub,
                            (
                                rv.clone()
                                if (rv is not None and clone_outputs)
                                else rv
                            ),
                        )

        if tier3_items:
            base_k_s_raw = torch.stack(
                [rk for _, _, rk, _ in tier3_items], dim=0
            )
            is_dsv41_q = _is_dsv41_quant_kv(base_k_s_raw)
            dpos_s = [r.delta_pos for _, r, _, _ in tier3_items]
            has_q_dk = tier3_items[0][1].quant_delta_k is not None
            if (
                is_dsv41_q
                and not has_q_dk
                and all(d == 0 for d in dpos_s)
            ):
                k_out_s = base_k_s_raw.clone()
            else:
                base_k_s = (
                    _dequant_dsv41_kv(base_k_s_raw)
                    if is_dsv41_q
                    else base_k_s_raw
                )
                k_aligned_s = apply_relative_rope_rotation_batch(
                    base_k_s,
                    dpos_s,
                    rope_theta=policy.rope_theta,
                    rotary_dim=policy.rotary_dim,
                    is_neox_style=policy.is_neox_style,
                    rope_scaling=policy.rope_scaling,
                    rope_at_tail=policy.rope_at_tail,
                    rope_pos_scale=policy.rope_pos_scale,
                )
                if has_q_dk and all(
                    r.quant_delta_k is not None for _, r, _, _ in tier3_items
                ):
                    dk_batch3 = dequantize_residual_packed_batch(
                        [r.quant_delta_k for _, r, _, _ in tier3_items]  # type: ignore[misc]
                    )
                    k_out_s = (
                        k_aligned_s.to(torch.float32)
                        + dk_batch3.to(torch.float32)
                    ).to(k_aligned_s.dtype)
                else:
                    k_out_s = k_aligned_s.clone()
                if is_dsv41_q:
                    k_out_s = _requant_dsv41_kv(
                        k_out_s,
                        base_k_s_raw,
                        rope_only=(not has_q_dk),
                    )

            has_v = tier3_items[0][3] is not None
            if has_v:
                base_v_s = torch.stack(
                    [rv for _, _, _, rv in tier3_items], dim=0  # type: ignore[misc]
                )
                if has_q_dk and all(
                    r.quant_delta_v is not None for _, r, _, _ in tier3_items
                ):
                    dv_batch3 = dequantize_residual_packed_batch(
                        [r.quant_delta_v for _, r, _, _ in tier3_items]  # type: ignore[misc]
                    )
                    v_out_s: torch.Tensor | None = (
                        base_v_s.to(torch.float32)
                        + dv_batch3.to(torch.float32)
                    ).to(base_v_s.dtype)
                else:
                    v_out_s = base_v_s.clone()
            else:
                v_out_s = None

            for sub_i, (item_indices, rec, _, _) in enumerate(tier3_items):
                k_item = k_out_s[sub_i]
                v_item = v_out_s[sub_i] if v_out_s is not None else None
                if (
                    rec.sparse_token_indices is not None
                    and rec.sparse_k_bf16 is not None
                ):
                    idx = rec.sparse_token_indices.to(torch.long)
                    k_item[idx] = rec.sparse_k_bf16
                    if v_item is not None and rec.sparse_v_bf16 is not None:
                        v_item[idx] = rec.sparse_v_bf16
                for item_idx in item_indices:
                    results[item_idx] = (
                        k_item.clone() if clone_outputs else k_item,
                        (
                            v_item.clone()
                            if (v_item is not None and clone_outputs)
                            else v_item
                        ),
                    )

        return results


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
    prev_token_block_hash: str = ""
    root_block_hash: str = ""


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
    clear_epoch: int = 0


@dataclass
class DeltaWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-Scheduler metadata syncing evicted block hashes across processes."""

    evicted_hashes: set[str] = field(default_factory=set)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "DeltaWorkerMetadata":
        if isinstance(other, DeltaWorkerMetadata):
            self.evicted_hashes |= other.evicted_hashes
        return self


@dataclass
class DeltaConnectorStats(KVConnectorStats):
    """Serializable stats container for DeltaCompressionConnector."""

    def reset(self) -> None:
        self.data.clear()

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: "KVConnectorStats") -> "DeltaConnectorStats":
        if not other or not other.data:
            return self
        if not self.data:
            return DeltaConnectorStats(data=dict(other.data))
        if bool(other.data.get("from_worker", False)):
            # Intra-step aggregation across TP worker ranks: sum byte footprints,
            # take max of logical block counts.
            merged = dict(self.data)
            for k in (
                "raw_bytes_equivalent",
                "compressed_bytes_stored",
                "evicted_bytes",
            ):
                merged[k] = int(self.data.get(k, 0)) + int(other.data.get(k, 0))
            for k in (
                "evicted_blocks",
                "evicted_unique_blocks",
                "active_blocks",
                "total_blocks_stored",
                "total_blocks_loaded",
            ):
                merged[k] = max(
                    int(self.data.get(k, 0)), int(other.data.get(k, 0))
                )
            merged["from_worker"] = True
            return DeltaConnectorStats(data=merged)
        # Inter-step accumulation in KVConnectorLogging: keep latest snapshot.
        return DeltaConnectorStats(data=dict(other.data))

    def reduce(self) -> dict[str, int | float]:
        if not self.data:
            return {}
        raw_b = float(self.data.get("raw_bytes_equivalent", 0))
        comp_b = float(self.data.get("compressed_bytes_stored", 0))
        ratio = raw_b / comp_b if comp_b > 0 else 1.0
        return {
            "active_blocks": int(self.data.get("active_blocks", 0)),
            "evicted_blocks": int(self.data.get("evicted_unique_blocks", 0)),
            "compressed_mb": round(comp_b / (1024**2), 2),
            "raw_equiv_mb": round(raw_b / (1024**2), 2),
            "compression_ratio": round(ratio, 3),
        }


class DeltaConnectorPromMetrics(KVConnectorPromMetrics):
    """Prometheus gauges exposed on GET /metrics for DeltaCompressionConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
        self.gauge_raw_bytes = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:delta_connector_raw_bytes_equivalent",
                documentation="Uncompressed equivalent bytes of blocks currently in DeltaBlockStore.",
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            ),
            per_engine_labelvalues,
        )
        self.gauge_compressed_bytes = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:delta_connector_compressed_bytes_stored",
                documentation="Compressed bytes currently stored in DeltaBlockStore.",
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            ),
            per_engine_labelvalues,
        )
        self.gauge_evicted_bytes = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:delta_connector_evicted_bytes_total",
                documentation="Cumulative compressed bytes evicted from DeltaBlockStore.",
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            ),
            per_engine_labelvalues,
        )
        self.gauge_evicted_blocks = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:delta_connector_evicted_blocks_total",
                documentation="Cumulative unique prefix blocks evicted from DeltaBlockStore.",
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            ),
            per_engine_labelvalues,
        )
        self.gauge_active_blocks = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:delta_connector_active_blocks",
                documentation="Number of active unique prefix blocks currently stored in DeltaBlockStore.",
                labelnames=labelnames,
                multiprocess_mode="mostrecent",
            ),
            per_engine_labelvalues,
        )
        for engine_idx in per_engine_labelvalues:
            self.gauge_raw_bytes[engine_idx].set(0)
            self.gauge_compressed_bytes[engine_idx].set(0)
            self.gauge_evicted_bytes[engine_idx].set(0)
            self.gauge_evicted_blocks[engine_idx].set(0)
            self.gauge_active_blocks[engine_idx].set(0)

    def observe(
        self, transfer_stats_data: dict[str, Any], engine_idx: int = 0
    ) -> None:
        if not transfer_stats_data or engine_idx not in self.gauge_raw_bytes:
            return
        self.gauge_raw_bytes[engine_idx].set(
            float(transfer_stats_data.get("raw_bytes_equivalent", 0))
        )
        self.gauge_compressed_bytes[engine_idx].set(
            float(transfer_stats_data.get("compressed_bytes_stored", 0))
        )
        self.gauge_evicted_bytes[engine_idx].set(
            float(transfer_stats_data.get("evicted_bytes", 0))
        )
        self.gauge_evicted_blocks[engine_idx].set(
            float(
                transfer_stats_data.get(
                    "evicted_unique_blocks",
                    transfer_stats_data.get("evicted_blocks", 0),
                )
            )
        )
        self.gauge_active_blocks[engine_idx].set(
            float(transfer_stats_data.get("active_blocks", 0))
        )


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
        store_device = str(extra_config.get("store_device", "cpu"))
        force_uncompressed = bool(extra_config.get("force_uncompressed", False))
        # Optional HBM capacity budget (bytes per worker store, or total MB across TP):
        tp_size = int(
            getattr(
                getattr(vllm_config, "parallel_config", None),
                "tensor_parallel_size",
                1,
            )
            or 1
        )
        if "max_store_bytes" in extra_config and extra_config["max_store_bytes"]:
            max_store_bytes = int(extra_config["max_store_bytes"])
        elif "max_hbm_mb" in extra_config and extra_config["max_hbm_mb"]:
            max_store_bytes = int(float(extra_config["max_hbm_mb"]) * (1024**2))
        else:
            max_store_bytes = 0
        self._exclusive_connector_cache = bool(
            extra_config.get("exclusive_connector_cache", False)
        )
        # Approach 2: Configurable server-side block alignment padding options
        pad_token = str(extra_config.get("pad_token", DEFAULT_PAD_TOKEN))
        raw_pad_token_id = extra_config.get("pad_token_id", None)
        pad_token_id = (
            int(raw_pad_token_id) if raw_pad_token_id is not None else None
        )
        raw_pad_token_map = extra_config.get("pad_token_map", None)
        pad_token_map = (
            dict(raw_pad_token_map)
            if isinstance(raw_pad_token_map, dict)
            else None
        )
        enable_server_block_align = bool(
            extra_config.get("enable_server_block_align", False)
        )
        enable_cross_turn_reuse = bool(
            extra_config.get("enable_cross_turn_reuse", False)
        )

        if engine_id not in self._shared_stores:
            self._shared_stores[engine_id] = DeltaBlockStore(
                block_size=self._block_size,
                default_rope_theta=rope_theta,
                sliding_window_dedup_threshold=sliding_window_dedup_threshold,
                tau_zero_bit=tau_zero_bit,
                tau_int2=tau_int2,
                tau_int4=tau_int4,
                max_sparse_exceptions=max_sparse_exceptions,
                device=store_device,
                force_uncompressed=force_uncompressed,
                pad_token=pad_token,
                pad_token_id=pad_token_id,
                pad_token_map=pad_token_map,
                enable_server_block_align=enable_server_block_align,
                max_store_bytes=max_store_bytes,
                enable_cross_turn_reuse=enable_cross_turn_reuse,
            )
        else:
            existing_store = self._shared_stores[engine_id]
            if max_store_bytes > 0:
                existing_store.max_store_bytes = max_store_bytes
            if "pad_token" in extra_config:
                existing_store.pad_token = pad_token
            if "pad_token_id" in extra_config:
                existing_store.pad_token_id = pad_token_id
            if "pad_token_map" in extra_config:
                existing_store.pad_token_map = pad_token_map
            if "enable_server_block_align" in extra_config:
                existing_store.enable_server_block_align = (
                    enable_server_block_align
                )
            if "enable_cross_turn_reuse" in extra_config:
                existing_store.enable_cross_turn_reuse = (
                    enable_cross_turn_reuse
                )
        self.store: DeltaBlockStore = self._shared_stores[engine_id]
        self._gpu_block_pool: "BlockPool | None" = None
        self._pending_evicted_hashes: set[str] = set()
        self._last_reported_stats_sig: tuple[int, ...] | None = None
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
        layer_rope_theta = (
            _safe_getattr(hf_config, "layer_rope_theta", None)
            if hf_config is not None
            else None
        )
        if isinstance(layer_types, (list, tuple)) and any(
            lt in ("linear_attention", "mamba") for lt in layer_types
        ):
            self.store._has_mamba_config = True
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
                grp_non_cacheable = (
                    not grp_is_mamba
                    and spec is not None
                    and getattr(spec, "prefix_cacheable", True) is False
                )
                self._group_block_sizes.append(0 if grp_non_cacheable else grp_bs)
                self._group_is_mamba.append(grp_is_mamba)
                if grp_non_cacheable:
                    continue
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
                    tokens_per_state = int(
                        getattr(layer_spec, "tokens_per_state", 1) or 1
                    )
                    is_dsv41_layer = model_type in (
                        "deepseek_v41",
                        "deepseek_v41_text",
                        "deepseek_v4",
                        "deepseek_v4_text",
                    ) or (
                        getattr(layer_spec, "head_size", 0) in (288, 528)
                        and getattr(layer_spec, "dtype", None) == torch.uint8
                    )
                    is_dsa_or_mla_model = model_type in (
                        "glm_moe_dsa",
                        "deepseek_v32",
                        "deepseek_v3",
                        "deepseek_v2",
                    ) or (
                        not is_dsv41_layer
                        and _safe_getattr(hf_config, "kv_lora_rank", None) is not None
                    )
                    layer_is_neox = True
                    layer_rope_at_tail = False
                    layer_rope_pos_scale = 1.0
                    if is_dsv41_layer:
                        compress_ratios = _safe_getattr(
                            hf_config, "compress_ratios", None
                        )
                        layer_compress_ratio = (
                            int(compress_ratios[layer_idx] or 0)
                            if (
                                isinstance(compress_ratios, (list, tuple))
                                and layer_idx < len(compress_ratios)
                            )
                            else 1
                        )
                        if layer_compress_ratio > 0:
                            theta = float(
                                _safe_getattr(
                                    hf_config, "compress_rope_theta", 160000.0
                                )
                                or 160000.0
                            )
                            comp_rope = (
                                rope_params.get("compress")
                                if (
                                    isinstance(rope_params, dict)
                                    and isinstance(
                                        rope_params.get("compress"), dict
                                    )
                                )
                                else full_rope_params
                            )
                            if isinstance(comp_rope, dict) and (
                                comp_rope.get("rope_type")
                                or comp_rope.get("type")
                            ):
                                layer_rope_scaling = dict(comp_rope)
                                if layer_rope_scaling.get("rope_type") == "yarn":
                                    layer_rope_scaling["rope_type"] = "deepseek_yarn"
                            else:
                                layer_rope_scaling = {
                                    "rope_type": "deepseek_yarn",
                                    "factor": 16.0,
                                    "beta_fast": 32.0,
                                    "beta_slow": 1.0,
                                    "original_max_position_embeddings": 65536,
                                }
                        else:
                            theta = global_theta
                            layer_rope_scaling = None
                        layer_rotary_dim = int(
                            _safe_getattr(hf_config, "qk_rope_head_dim", 64) or 64
                        )
                        layer_is_neox = False
                        layer_rope_at_tail = True
                        layer_rope_pos_scale = 1.0
                    elif is_dsa_or_mla_model:
                        theta = local_theta if is_sw else global_theta
                        layer_rope_scaling = full_rope_params
                        layer_rotary_dim = int(
                            _safe_getattr(hf_config, "qk_rope_head_dim", 64) or 64
                        )
                        is_indexer_layer = "indexer" in layer_name
                        if is_indexer_layer:
                            layer_is_neox = not bool(
                                _safe_getattr(
                                    hf_config, "indexer_rope_interleave", False
                                )
                            )
                            layer_rope_at_tail = False
                        else:
                            layer_is_neox = not bool(
                                _safe_getattr(hf_config, "rope_interleave", True)
                            )
                            layer_rope_at_tail = True
                    else:
                        theta = local_theta if is_sw else global_theta
                        if is_mamba:
                            layer_rope_scaling = None
                            layer_rotary_dim = 0
                        elif is_sw and sliding_rope_params is not None:
                            layer_rope_scaling = sliding_rope_params
                            layer_rotary_dim = None
                        elif is_sw and has_separate_local_rope:
                            layer_rope_scaling = None
                            layer_rotary_dim = rotary_dim
                        else:
                            layer_rope_scaling = full_rope_params
                            layer_rotary_dim = rotary_dim
                        if (
                            isinstance(layer_rope_theta, (list, tuple))
                            and layer_idx < len(layer_rope_theta)
                        ):
                            per_layer_theta = float(
                                layer_rope_theta[layer_idx] or 0.0
                            )
                            if per_layer_theta <= 0.0:
                                theta = 0.0
                                layer_rope_scaling = None
                                layer_rotary_dim = 0
                            else:
                                theta = per_layer_theta
                        if model_type in (
                            "muse_spark",
                            "muse_spark_text",
                            "musespark1x_text",
                        ):
                            layer_is_neox = False
                    follows_global_layer = (
                        is_sw
                        and layer_idx > 0
                        and (
                            (
                                isinstance(layer_types, (list, tuple))
                                and "full_attention"
                                in layer_types[: min(layer_idx, len(layer_types))]
                            )
                            or (is_dsv41_layer and layer_idx > 1)
                        )
                    )
                    min_exact_sw_dedup = (
                        max(
                            self.store.sliding_window_dedup_threshold,
                            int(sw_size) * max(1, layer_idx),
                        )
                        if sw_size is not None
                        else self.store.sliding_window_dedup_threshold
                    )
                    sw_dedup_thresh = (
                        max(self.store.sliding_window_dedup_threshold, 1_000_000)
                        if follows_global_layer
                        else min_exact_sw_dedup
                    )
                    sw_tau_zero = (
                        min(self.store.tau_zero_bit, 0.01)
                        if (follows_global_layer or sw_size is not None)
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
                            is_neox_style=layer_is_neox,
                            rope_scaling=layer_rope_scaling,
                            rope_at_tail=layer_rope_at_tail,
                            rope_pos_scale=layer_rope_pos_scale,
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

    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        super().bind_connector_metadata(connector_metadata)
        self._layers_saved_in_step.clear()
        if isinstance(connector_metadata, DeltaConnectorMetadata):
            if connector_metadata.clear_epoch > self.store._clear_epoch:
                self.store.clear()
                self.store._clear_epoch = connector_metadata.clear_epoch

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self._layers_saved_in_step.clear()

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
    # Server-Side Structural Alignment & Padding Helpers (Approach 2)
    # ==============================

    def resolve_pad_token_id(
        self,
        tokenizer: Any,
        content_text: str | None = None,
        pad_token: str | None = None,
        pad_token_id: int | None = None,
    ) -> int:
        """Resolve padding token ID on the connector using its configured default or map."""
        return self.store.resolve_pad_token_id(
            tokenizer=tokenizer,
            content_text=content_text,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
        )

    def align_tokens(
        self,
        token_ids: list[int],
        pad_token_id: int | None = None,
        tokenizer: Any = None,
        content_text: str | None = None,
        target_blocks: int | None = None,
    ) -> list[int]:
        """Pad or truncate `token_ids` to an exact multiple of `self._block_size`."""
        return self.store.align_tokens(
            token_ids=token_ids,
            pad_token_id=pad_token_id,
            tokenizer=tokenizer,
            content_text=content_text,
            target_blocks=target_blocks,
        )

    def encode_aligned_segment(
        self,
        tokenizer: Any,
        text: str,
        bucket_file_viewer: bool = True,
        target_blocks: int | None = None,
        pad_token: str | None = None,
        pad_token_id: int | None = None,
        line_step: int = 10,
    ) -> list[int]:
        """Encode and block-align a structural prompt segment on the server side (Approach 2)."""
        return self.store.encode_aligned_segment(
            tokenizer=tokenizer,
            text=text,
            bucket_file_viewer=bucket_file_viewer,
            target_blocks=target_blocks,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            line_step=line_step,
        )

    def align_structural_prompt(
        self,
        tokenizer: Any,
        segments: list[tuple[str, bool]] | list[str],
        bucket_file_viewer: bool = True,
        pad_token: str | None = None,
        pad_token_id: int | None = None,
    ) -> list[int]:
        """Tokenize and block-align a sequence of structural prompt segments (Approach 2)."""
        return self.store.align_structural_prompt(
            tokenizer=tokenizer,
            segments=segments,
            bucket_file_viewer=bucket_file_viewer,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def _build_block_specs_for_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        block_ids: list[int] | tuple[int, ...],
        all_group_block_ids: tuple[list[int], ...] | None = None,
    ) -> list[DeltaBlockTransferSpec]:
        active_g_indices = sorted(set(self._layer_to_group_idx.values()))
        primary_g = active_g_indices[0] if active_g_indices else 0
        max_avail_blocks = (
            max(
                (
                    len(g) * self._get_group_block_scale(g_idx)
                    for g_idx, g in enumerate(all_group_block_ids)
                    if (not active_g_indices or g_idx in active_g_indices)
                ),
                default=len(block_ids) * self._get_group_block_scale(primary_g),
            )
            if all_group_block_ids
            else len(block_ids) * self._get_group_block_scale(primary_g)
        )
        num_full_blocks = min(len(token_ids) // self._block_size, max_avail_blocks)
        specs: list[DeltaBlockTransferSpec] = []
        parent_hash = "ROOT"
        root_hash = ""
        prev_tok_hash = ""
        divergence_distance: int | None = 0

        for b_idx in range(num_full_blocks):
            start_pos = b_idx * self._block_size
            blk_tokens = tuple(
                int(t) for t in token_ids[start_pos : start_pos + self._block_size]
            )
            full_hash = compute_full_prefix_block_hash(parent_hash, blk_tokens)
            if b_idx == 0:
                root_hash = full_hash
            token_hash = compute_token_block_hash(blk_tokens)

            if self.store.has_full_block(full_hash):
                # Prefix has not diverged yet
                divergence_distance = None
            elif self.store.has_token_block(token_hash) or (
                not self.store.enable_cross_turn_reuse
                and self.store.has_stitched_token_block(blk_tokens)
            ):
                # Divergent prefix, but token block matches an exact or stitched reference block!
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

            primary_blks = (
                all_group_block_ids[primary_g]
                if (all_group_block_ids and primary_g < len(all_group_block_ids))
                else block_ids
            )
            b0_idx = b_idx // self._get_group_block_scale(primary_g)
            phys_id = (
                int(primary_blks[b0_idx])
                if b0_idx < len(primary_blks) and primary_blks[b0_idx] is not None
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
                    prev_token_block_hash=prev_tok_hash,
                    root_block_hash=root_hash,
                )
            )
            parent_hash = full_hash
            prev_tok_hash = token_hash
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
            elif (
                self.store.enable_cross_turn_reuse
                and not self.store.has_mamba_layers
                and self.store.has_reconstructible_block(
                    blk_tokens,
                    start_position=start_pos,
                    max_edits=self.store.max_sparse_exceptions,
                )
            ):
                self.store.register_scheduled_store_block(
                    full_hash,
                    blk_tokens,
                    start_pos,
                    has_mamba_state=False,
                    index_subchunks=False,
                )
                matched_positions.append((start_pos + self._block_size, full_hash))
                parent_hash = full_hash
            else:
                break

        longest_attn_matched = matched_positions[-1][0] if matched_positions else 0
        if self.store.has_mamba_layers:
            while matched_positions and not self.store.has_mamba_block(
                matched_positions[-1][1]
            ):
                matched_positions.pop()

        matched_tokens = matched_positions[-1][0] if matched_positions else 0
        if longest_attn_matched > matched_tokens and hasattr(
            request, "shared_prefix_boundary"
        ):
            request.shared_prefix_boundary = max(
                getattr(request, "shared_prefix_boundary", 0) or 0,
                longest_attn_matched,
            )
        sched_bs = self._scheduler_block_size
        if sched_bs > self._block_size:
            matched_tokens = (matched_tokens // sched_bs) * sched_bs
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
        meta = DeltaConnectorMetadata(clear_epoch=self.store._clear_epoch)

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

        mamba_g_indices = [
            g_idx for g_idx, is_m in enumerate(self._group_is_mamba) if is_m
        ]
        active_g_indices = sorted(set(self._layer_to_group_idx.values()))
        for req_meta in meta.requests:
            if req_meta.is_store:
                for blk_spec in req_meta.blocks:
                    if mamba_g_indices and blk_spec.group_block_ids:
                        has_mamba_state = all(
                            g_idx < len(blk_spec.group_block_ids)
                            and blk_spec.group_block_ids[g_idx] is not None
                            and blk_spec.group_block_ids[g_idx] > 0
                            for g_idx in mamba_g_indices
                        )
                    elif blk_spec.group_block_ids:
                        check_g = active_g_indices or list(
                            range(len(blk_spec.group_block_ids))
                        )
                        has_mamba_state = all(
                            g_idx < len(blk_spec.group_block_ids)
                            and blk_spec.group_block_ids[g_idx] is not None
                            and blk_spec.group_block_ids[g_idx] > 0
                            for g_idx in check_g
                        )
                    else:
                        has_mamba_state = blk_spec.physical_block_id > 0
                    self.store.register_scheduled_store_block(
                        blk_spec.full_block_hash,
                        blk_spec.token_ids,
                        blk_spec.start_position,
                        has_mamba_state=has_mamba_state,
                    )

        self._requests_need_load.clear()
        if self._exclusive_connector_cache:
            self._evict_idle_gpu_block_pool()
        return meta

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._gpu_block_pool = gpu_block_pool

    def _evict_idle_gpu_block_pool(self) -> None:
        """Evict unreferenced cached blocks from local BlockPool when exclusive_connector_cache is enabled."""
        if not self._exclusive_connector_cache or self._gpu_block_pool is None:
            return
        cache_map = getattr(
            getattr(self._gpu_block_pool, "cached_block_hash_to_block", None),
            "_cache",
            None,
        )
        if isinstance(cache_map, dict):
            idle_block_ids: set[int] = set()
            for entry in cache_map.values():
                if isinstance(entry, dict):
                    for blk in entry.values():
                        if not blk.is_null and blk.ref_cnt == 0:
                            idle_block_ids.add(blk.block_id)
                elif (
                    entry is not None
                    and not getattr(entry, "is_null", True)
                    and getattr(entry, "ref_cnt", 1) == 0
                ):
                    idle_block_ids.add(entry.block_id)
        else:
            idle_block_ids = {
                blk.block_id
                for blk in self._gpu_block_pool.blocks
                if not blk.is_null and blk.ref_cnt == 0 and blk.block_hash is not None
            }
        if idle_block_ids:
            self._gpu_block_pool.evict_blocks(idle_block_ids)

    def on_new_request(self, request: "Request") -> None:
        if self._exclusive_connector_cache:
            self._evict_idle_gpu_block_pool()

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(worker_meta, DeltaWorkerMetadata) and worker_meta.evicted_hashes:
            for ev_hash in worker_meta.evicted_hashes:
                self.store.evict_block(ev_hash)
        if self._exclusive_connector_cache:
            self._evict_idle_gpu_block_pool()

    def reset_cache(self) -> bool | None:
        self.store.clear()
        self._pending_evicted_hashes.clear()
        return True

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
    def _is_mla_kv_tensor(
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        attn_metadata: Any = None,
    ) -> bool:
        if isinstance(kv_layer, tuple):
            return False
        if isinstance(attn_metadata, MLACommonMetadata):
            return True
        if _is_dsv41_quant_kv(kv_layer):
            return True
        if kv_layer.ndim == 3 or (kv_layer.ndim == 4 and kv_layer.shape[1] == 1):
            return True
        return False

    @staticmethod
    def _get_kernel_block_ratio(
        kv_layer: torch.Tensor | tuple[torch.Tensor, ...],
        is_mla: bool = False,
        block_size: int | None = None,
    ) -> int:
        if isinstance(kv_layer, tuple) or block_size is None or block_size <= 0:
            return 1
        if _is_dsv41_quant_kv(kv_layer):
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

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DeltaConnectorMetadata):
            return
        if not any(not req_meta.is_store for req_meta in metadata.requests):
            return

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        all_layer_names = list(self._active_no_compile_layers.keys())
        for reg_name in self._registered_kv_caches:
            if reg_name not in self._active_no_compile_layers:
                all_layer_names.append(reg_name)

        load_reqs = [r for r in metadata.requests if not r.is_store]
        if not load_reqs:
            return

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
            is_mla = self._is_mla_kv_tensor(kv_tensor, layer_attn_meta)
            num_blocks = self._get_kv_layer_num_blocks(
                kv_tensor, is_mla=is_mla, block_size=self._block_size
            )

            valid_specs: list[DeltaBlockTransferSpec] = []
            valid_phys_ids: list[int] = []
            for req_meta in load_reqs:
                for blk_spec in req_meta.blocks:
                    phys_blk_id = self._resolve_physical_block_id(
                        layer_name, blk_spec
                    )
                    if phys_blk_id is None or phys_blk_id >= num_blocks:
                        continue
                    if (
                        isinstance(kv_tensor, tuple)
                        or policy.is_mamba
                        or len(blk_spec.group_block_ids) > 1
                    ) and phys_blk_id <= 0:
                        continue
                    valid_specs.append(blk_spec)
                    valid_phys_ids.append(phys_blk_id)

            if not valid_specs:
                continue

            reconstructed_list = self.store.reconstruct_blocks_batch(
                layer_name=layer_name,
                specs=valid_specs,
                clone_outputs=False,
            )
            if not (isinstance(kv_tensor, tuple) or policy.is_mamba):
                ratio = self._get_kernel_block_ratio(
                    kv_tensor, is_mla=is_mla, block_size=self._block_size
                )
                if ratio == 1:
                    b_phys: list[int] = []
                    b_k: list[torch.Tensor] = []
                    b_v: list[torch.Tensor] = []
                    has_all_v = True
                    for phys_blk_id, kv_pair in zip(
                        valid_phys_ids, reconstructed_list
                    ):
                        if kv_pair is None:
                            continue
                        b_phys.append(phys_blk_id)
                        b_k.append(kv_pair[0])
                        if kv_pair[1] is not None:
                            b_v.append(kv_pair[1])
                        else:
                            has_all_v = False
                    if b_phys:
                        eff_bs = (
                            self._block_size
                            if self._block_size is not None
                            else int(b_k[0].shape[0])
                        )
                        if is_mla or not has_all_v:
                            page_dim = (
                                int(kv_tensor.shape[2])
                                if (kv_tensor.ndim == 4 and kv_tensor.shape[1] == 1)
                                else int(kv_tensor.shape[1])
                            )
                            if eff_bs == page_dim:
                                for c_s in range(0, len(b_phys), 4096):
                                    c_e = c_s + 4096
                                    phys_t = torch.tensor(
                                        b_phys[c_s:c_e],
                                        dtype=torch.long,
                                        device=kv_tensor.device,
                                    )
                                    k_batch = torch.stack(b_k[c_s:c_e], dim=0).to(
                                        device=kv_tensor.device, non_blocking=True
                                    )
                                    if kv_tensor.ndim == 4 and kv_tensor.shape[1] == 1:
                                        kv_tensor[phys_t, 0] = k_batch
                                    else:
                                        kv_tensor[phys_t] = k_batch
                                continue
                        elif kv_tensor.ndim == 5 and kv_tensor.shape[0] == 2:
                            if eff_bs == int(kv_tensor.shape[2]):
                                for c_s in range(0, len(b_phys), 4096):
                                    c_e = c_s + 4096
                                    phys_t = torch.tensor(
                                        b_phys[c_s:c_e],
                                        dtype=torch.long,
                                        device=kv_tensor.device,
                                    )
                                    kv_tensor[0, phys_t] = torch.stack(
                                        b_k[c_s:c_e], dim=0
                                    ).to(device=kv_tensor.device, non_blocking=True)
                                    kv_tensor[1, phys_t] = torch.stack(
                                        b_v[c_s:c_e], dim=0
                                    ).to(device=kv_tensor.device, non_blocking=True)
                                continue
                        elif kv_tensor.ndim == 5 and kv_tensor.shape[1] == 2:
                            if eff_bs == int(kv_tensor.shape[2]):
                                for c_s in range(0, len(b_phys), 4096):
                                    c_e = c_s + 4096
                                    phys_t = torch.tensor(
                                        b_phys[c_s:c_e],
                                        dtype=torch.long,
                                        device=kv_tensor.device,
                                    )
                                    kv_tensor[phys_t, 0] = torch.stack(
                                        b_k[c_s:c_e], dim=0
                                    ).to(device=kv_tensor.device, non_blocking=True)
                                    kv_tensor[phys_t, 1] = torch.stack(
                                        b_v[c_s:c_e], dim=0
                                    ).to(device=kv_tensor.device, non_blocking=True)
                                continue
                        elif kv_tensor.ndim == 4:
                            if eff_bs == int(kv_tensor.shape[2]):
                                for c_s in range(0, len(b_phys), 4096):
                                    c_e = c_s + 4096
                                    phys_t = torch.tensor(
                                        b_phys[c_s:c_e],
                                        dtype=torch.long,
                                        device=kv_tensor.device,
                                    )
                                    k_batch = torch.stack(b_k[c_s:c_e], dim=0).to(
                                        device=kv_tensor.device, non_blocking=True
                                    )
                                    v_batch = torch.stack(b_v[c_s:c_e], dim=0).to(
                                        device=kv_tensor.device, non_blocking=True
                                    )
                                    kv_tensor[phys_t] = torch.cat(
                                        [k_batch, v_batch], dim=-1
                                    ).transpose(1, 2)
                                continue

            for blk_spec, phys_blk_id, kv_pair in zip(
                valid_specs, valid_phys_ids, reconstructed_list
            ):
                if kv_pair is None:
                    continue
                k_recon, v_recon = kv_pair
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

        is_mla = self._is_mla_kv_tensor(kv_layer, attn_metadata)
        num_blocks = self._get_kv_layer_num_blocks(
            kv_layer, is_mla=is_mla, block_size=self._block_size
        )
        batch_items: list[
            tuple[DeltaBlockTransferSpec, torch.Tensor, torch.Tensor | None]
        ] = []
        seen_in_layer: set[str] = set()
        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue
            for blk_spec in req_meta.blocks:
                if blk_spec.full_block_hash in seen_in_layer:
                    continue
                seen_in_layer.add(blk_spec.full_block_hash)
                existing_rec = self.store._records.get(
                    (layer_name, blk_spec.full_block_hash)
                )
                if existing_rec is not None:
                    self.store._touch_lru(
                        blk_spec.full_block_hash,
                        existing_rec.ref_base_hash,
                        existing_rec.ref_base_hash_2,
                    )
                    if policy.is_mamba:
                        self.store._known_mamba_blocks.add(
                            blk_spec.full_block_hash
                        )
                    continue
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
                batch_items.append((blk_spec, k_blk, v_blk))
        if batch_items:
            self.store.store_blocks_batch(
                layer_name=layer_name,
                items=batch_items,
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

            is_mla = self._is_mla_kv_tensor(kv_tensor)
            num_blocks = self._get_kv_layer_num_blocks(
                kv_tensor, is_mla=is_mla, block_size=self._block_size
            )
            batch_items: list[
                tuple[DeltaBlockTransferSpec, torch.Tensor, torch.Tensor | None]
            ] = []
            seen_in_layer: set[str] = set()
            for req_meta in store_reqs:
                for blk_spec in req_meta.blocks:
                    if blk_spec.full_block_hash in seen_in_layer:
                        continue
                    seen_in_layer.add(blk_spec.full_block_hash)
                    existing_rec = self.store._records.get(
                        (layer_name, blk_spec.full_block_hash)
                    )
                    if existing_rec is not None:
                        self.store._touch_lru(
                            blk_spec.full_block_hash,
                            existing_rec.ref_base_hash,
                            existing_rec.ref_base_hash_2,
                        )
                        if policy.is_mamba:
                            self.store._known_mamba_blocks.add(
                                blk_spec.full_block_hash
                            )
                        continue
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
                        is_mla=is_mla,
                        block_size=self._block_size,
                        start_position=blk_spec.start_position,
                    )
                    batch_items.append((blk_spec, k_blk, v_blk))
            if batch_items:
                self.store.store_blocks_batch(
                    layer_name=layer_name,
                    items=batch_items,
                )

        if (
            self.store.max_store_bytes > 0
            and self.store.stats.compressed_bytes_stored
            > self.store.max_store_bytes
        ):
            step_protected = {
                blk.full_block_hash
                for req_meta in metadata.requests
                for blk in req_meta.blocks
            }
            evicted = self.store.evict_lru_to_budget(
                self.store.max_store_bytes, protected_hashes=step_protected
            )
            if evicted:
                self._pending_evicted_hashes.update(evicted)

        self._layers_saved_in_step.clear()

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if not self._pending_evicted_hashes:
            return None
        evicted = set(self._pending_evicted_hashes)
        self._pending_evicted_hashes.clear()
        return DeltaWorkerMetadata(evicted_hashes=evicted)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.role != KVConnectorRole.WORKER:
            return None
        st = self.store.stats
        cur_sig = (
            st.total_blocks_stored,
            st.total_blocks_loaded,
            st.evicted_blocks,
            st.evicted_unique_blocks,
            len(self.store._block_lru),
        )
        if cur_sig == self._last_reported_stats_sig:
            return None
        self._last_reported_stats_sig = cur_sig
        return DeltaConnectorStats(
            data={
                "from_worker": True,
                "raw_bytes_equivalent": int(st.raw_bytes_equivalent),
                "compressed_bytes_stored": int(st.compressed_bytes_stored),
                "evicted_bytes": int(st.evicted_bytes),
                "evicted_blocks": int(st.evicted_blocks),
                "evicted_unique_blocks": int(st.evicted_unique_blocks),
                "active_blocks": int(len(self.store._block_lru)),
                "total_blocks_stored": int(st.total_blocks_stored),
                "total_blocks_loaded": int(st.total_blocks_loaded),
            }
        )

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        payload = dict(data) if data else {}
        payload["from_worker"] = False
        return DeltaConnectorStats(data=payload)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics | None:
        return DeltaConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

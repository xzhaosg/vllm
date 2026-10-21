#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU HBM Memory Savings & Ground-Truth Accuracy Benchmark on Real Multi-Turn
SWE-Agent Trajectories (`nebius/SWE-agent-trajectories`, `target == True`).

Replays 20 verified-passing (`target == True`) multi-turn SWE-agent trajectories
where the agent inspects, edits, tests, and submits changes to a real repository
source file across 4 turns, comparing:
  1. Uncompressed GPU HBM Prefix Caching (`store_device="cuda", force_uncompressed=True`)
  2. Delta-Compressed GPU HBM Caching (`store_device="cuda", force_uncompressed=False`)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import os
import re
import time
from typing import Any
import urllib.request

import torch
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    DEFAULT_PAD_TOKEN,
    CompressionTier,
    DeltaBlockStore,
    DeltaCompressionConnector,
    align_tokens_to_block,
    encode_file_viewer_bucketed_server,
    resolve_pad_token_id,
)
from vllm.v1.core.kv_cache_utils import get_group_id


def load_nebius_passing_trajectories(
    cache_path: str | None = None,
    num_trajectories: int = 20,
    min_turns: int = 8,
    max_turns: int = 12,
) -> list[dict[str, Any]]:
    """Load verified-passing (`target == True`) multi-turn SWE-agent trajectories."""
    if cache_path is None:
        cache_path = (
            f"/tmp/nebius_swe_agent_passing_n{num_trajectories}"
            f"_min{min_turns}_max{max_turns}.json"
        )
    fallback_paths = [
        cache_path,
        f"/tmp/nebius_swe_agent_passing_n157_min{min_turns}_max{max_turns}.json",
        f"/workspace/nebius_swe_agent_passing_n157_min{min_turns}_max{max_turns}.json",
    ]
    for cand_path in fallback_paths:
        if cand_path and os.path.exists(cand_path):
            with open(cand_path, encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) >= num_trajectories:
                return cached[:num_trajectories]

    found: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    # Prefer local Parquet shard if available (contains 6,670 trajectories / 1,268 passing)
    parquet_candidates = [
        "/mnt/ssd/hf_cache/hub/datasets--nebius--SWE-agent-trajectories/snapshots/68195a1450865274106246d0d0296a1d6807b88e/data/train-00000-of-00012.parquet",
        "/mnt/ssd/hf_cache/datasets--nebius--SWE-agent-trajectories/snapshots/68195a1450865274106246d0d0296a1d6807b88e/data/train-00000-of-00012.parquet",
    ]
    parquet_path = next((p for p in parquet_candidates if os.path.exists(p)), None)
    if parquet_path is None:
        try:
            from huggingface_hub import hf_hub_download

            parquet_path = hf_hub_download(
                repo_id="nebius/SWE-agent-trajectories",
                filename="data/train-00000-of-00012.parquet",
                repo_type="dataset",
            )
        except Exception:
            parquet_path = None

    if parquet_path is not None and os.path.exists(parquet_path):
        import pyarrow.parquet as pq

        table = pq.read_table(
            parquet_path,
            columns=[
                "instance_id",
                "target",
                "trajectory",
                "generated_patch",
            ],
        )
        for row in table.to_pylist():
            if row.get("target") is not True:
                continue
            inst = str(row.get("instance_id", ""))
            traj = row.get("trajectory", [])
            sys_txt = (
                str(traj[0].get("system_prompt") or "") if len(traj) > 0 else ""
            )
            issue_txt = str(traj[1].get("text") or "") if len(traj) > 1 else ""
            by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for idx in range(len(traj) - 1):
                u_txt = str(traj[idx].get("text") or "")
                a_txt = str(traj[idx + 1].get("text") or "")
                prev_a = str(traj[idx - 1].get("text") or "") if idx > 0 else ""
                if (
                    traj[idx].get("role") == "user"
                    and traj[idx + 1].get("role") == "ai"
                    and "[File:" in u_txt
                    and len(u_txt) >= 600
                ):
                    m = re.search(r"\[File:\s*([^\s\]]+)", u_txt)
                    if m:
                        by_file[m.group(1)].append(
                            {
                                "step_idx": idx,
                                "user_obs": u_txt,
                                "ai_resp": a_txt,
                                "prev_ai_resp": prev_a,
                            }
                        )
            for fpath, steps in by_file.items():
                if len(steps) >= min_turns and (inst, fpath) not in seen_keys:
                    seen_keys.add((inst, fpath))
                    found.append(
                        {
                            "instance_id": inst,
                            "file_path": fpath,
                            "system_prompt": sys_txt,
                            "issue_text": issue_txt,
                            "generated_patch": str(row.get("generated_patch") or ""),
                            "total_file_turns_in_trajectory": len(steps),
                            "turns": steps[:max_turns],
                        }
                    )
            if len(found) >= num_trajectories:
                break

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(found, f, indent=2)
    return found[:num_trajectories]


def _align_to_block(
    token_ids: list[int],
    block_size: int,
    newline_id: int,
    target_blocks: int | None = None,
) -> list[int]:
    """Pad or truncate `token_ids` to an exact multiple of `block_size` via server-side helper."""
    return align_tokens_to_block(
        token_ids=token_ids,
        block_size=block_size,
        pad_token_id=newline_id,
        target_blocks=target_blocks,
    )


def _extract_patch_symbols(patch: str, file_path: str) -> list[str]:
    """Extract ground-truth modified function/class/file symbols from `generated_patch`."""
    symbols: list[str] = []
    base_name = os.path.basename(file_path)
    if base_name:
        symbols.append(base_name)
    for m in re.finditer(r"@@.*?@@\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", patch):
        sym = m.group(1)
        if sym not in symbols:
            symbols.append(sym)
    for m in re.finditer(r"^[+-]\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", patch, re.MULTILINE):
        sym = m.group(1)
        if sym not in symbols:
            symbols.append(sym)
    return symbols


def _encode_file_viewer_bucketed(
    user_obs: str,
    fpath: str,
    tokenizer: Any,
    block_size: int,
    newline_id: int,
    max_file_blocks: int,
) -> list[int]:
    """Encode a SWE-agent `[File: ...]` observation using server-side 10-line block-aligned buckets."""
    return encode_file_viewer_bucketed_server(
        tokenizer=tokenizer,
        text=user_obs,
        block_size=block_size,
        pad_token_id=newline_id,
        max_file_blocks=max_file_blocks,
        file_path_hint=fpath,
    )


def _classify_swe_action(text: str) -> str:
    """Classify a SWE-agent turn output into an action category."""
    low = text.lower()
    if "edit " in low or "end_of_edit" in low or "diff" in low or "replace" in low:
        return "edit"
    if "submit" in low:
        return "submit"
    if "pytest" in low or "unittest" in low or "python " in low or "test" in low:
        return "test"
    return "inspect"


def build_trajectory_samples(
    trajectories: list[dict[str, Any]],
    tokenizer: Any,
    block_size: int,
    pad_token: str = DEFAULT_PAD_TOKEN,
) -> list[dict[str, Any]]:
    """Build all-turn token sequences from real `nebius/SWE-agent-trajectories` turns."""
    newline_id = resolve_pad_token_id(tokenizer=tokenizer, pad_token=pad_token)
    raw_edit_toks = [
        tokenizer.encode(str(d), add_special_tokens=False)[-1]
        for d in range(1, 250)
    ]
    edit_toks: list[int] = []
    for tok in raw_edit_toks:
        if tok != newline_id and tok not in edit_toks:
            edit_toks.append(tok)
    edit_blk = 1 if block_size >= 64 else 2

    is_harmony = (
        hasattr(tokenizer, "get_vocab") and "<|channel|>" in tokenizer.get_vocab()
    )
    is_qwen_thinking = (
        hasattr(tokenizer, "get_vocab")
        and "<think>" in tokenizer.get_vocab()
        and "<|im_start|>" in tokenizer.get_vocab()
    )
    is_deepseek_v41 = (
        hasattr(tokenizer, "get_vocab")
        and "<｜Assistant｜>" in tokenizer.get_vocab()
        and "</think>" in tokenizer.get_vocab()
    )
    is_glm = (
        hasattr(tokenizer, "get_vocab")
        and "<|assistant|>" in tokenizer.get_vocab()
        and "<|user|>" in tokenizer.get_vocab()
    )

    samples: list[dict[str, Any]] = []
    for sample_idx, item in enumerate(trajectories):
        inst = item["instance_id"]
        fpath = item["file_path"]
        sys_txt = item.get("system_prompt", "")
        issue_txt = item["issue_text"]
        patch = item["generated_patch"]
        gt_symbols = _extract_patch_symbols(patch, fpath)
        sid_h = 100000000 + sample_idx
        sid_b = 200000000 + sample_idx
        sid_p = 300000000 + sample_idx

        # Shared GitHub Issue & Reference Patch Context block(s)
        issue_blocks = 1 if block_size >= 64 else 8
        obs_blocks = 2 if block_size >= 64 else 32
        issue_section_text = (
            f"ID: {sid_h} | Active SWE-Agent Issue & Patch Context: {sid_b}\n\n"
            f"### GitHub Issue ({inst})\n"
            f"{issue_txt[:2000]}\n\n"
            f"### Reference Patch Context\n"
            f"{patch[:1200]}\n\n"
        )
        issue_ids_raw = tokenizer.encode(issue_section_text, add_special_tokens=False)
        while len(issue_ids_raw) < issue_blocks * block_size:
            issue_ids_raw = issue_ids_raw + issue_ids_raw
        issue_ids_base = _align_to_block(
            issue_ids_raw, block_size, newline_id, target_blocks=issue_blocks
        )
        turn0_obs = item["turns"][0]["user_obs"]

        # Suffix prompt asking for the next SWE-agent action & patch analysis
        prompt_question = (
            f"You are working on `{fpath}` for `{inst}`. "
            f"State the target file `{os.path.basename(fpath)}`, the function/lines to edit "
            f"or test, and the next SWE-agent command (`edit`, `python -m unittest`, or `submit`)."
        )
        if is_harmony:
            suffix_text = (
                f"{prompt_question}"
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            )
        elif is_qwen_thinking:
            suffix_text = (
                f"{prompt_question}"
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        elif is_deepseek_v41:
            suffix_text = f"<｜User｜>{prompt_question}<｜Assistant｜></think>"
        elif is_glm:
            suffix_text = f"<|user|>{prompt_question}<|assistant|><think></think>"
        else:
            suffix_text = f"{prompt_question}\nResponse:"

        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if len(suffix_ids) < 4:
            suffix_ids = [newline_id] * (4 - len(suffix_ids)) + suffix_ids
        if len(suffix_ids) % block_size == 0:
            suffix_ids = [newline_id] + suffix_ids

        # Static System Prompt & Workspace Header (Block 0..edit_blk) for real-world regime
        static_header_text = (
            f"ID: {sid_h} | SWE-Agent Trajectory: {inst} | Open File: {fpath}\n"
            f"SWE-Agent ACI: Inspect the open file window and apply the minimal edit.\n\n"
        )
        if block_size >= 64:
            static_header_text += (
                f"System & Workspace Context ({fpath}):\n"
                f"{sys_txt[:800]}\n{issue_txt[:1200]}\n\n"
            )
        h_static = _align_to_block(
            tokenizer.encode(static_header_text, add_special_tokens=False),
            block_size,
            newline_id,
            target_blocks=edit_blk,
        )

        # Build ALL turns directly from each turn's real `step_idx`, `prev_ai_resp`, and `user_obs`
        stateful_turns: list[list[int]] = []
        rolling_turns: list[list[int]] = []
        real_world_turns: list[list[int]] = []
        gt_ai_resps: list[str] = []

        for t_idx, turn_data in enumerate(item["turns"]):
            gt_ai_resps.append(turn_data["ai_resp"])
            cur_step_idx = turn_data["step_idx"]
            cur_user_obs = turn_data["user_obs"]
            cur_prev_ai = turn_data.get("prev_ai_resp", "")

            # 1. Real Turn `t_idx` SWE-Agent ACI Header (Block 0..edit_blk)
            header_text = (
                f"ID: {sid_h} | SWE-Agent Trajectory: {inst} | "
                f"Step: {cur_step_idx} | Open File: {fpath}\n"
                f"SWE-Agent ACI: Inspect the open file window and apply the minimal edit.\n\n"
            )
            if block_size >= 64:
                header_text += (
                    f"System & Workspace Context ({fpath}):\n"
                    f"{sys_txt[:800]}\n{issue_txt[:1200]}\n\n"
                )
            h_turn = _align_to_block(
                tokenizer.encode(header_text, add_special_tokens=False),
                block_size,
                newline_id,
                target_blocks=edit_blk,
            )

            # 2. Real Turn `t_idx` `[File: ...]` Window from `turn_data["user_obs"]`
            obs_turn_ids = _encode_file_viewer_bucketed(
                cur_user_obs,
                fpath,
                tokenizer,
                block_size,
                newline_id,
                max_file_blocks=obs_blocks,
            )
            b_turn = issue_ids_base + obs_turn_ids
            turn_prompt = h_turn + b_turn + suffix_ids
            stateful_turns.append(turn_prompt)

            if t_idx == 0:
                rolling_turns.append(list(turn_prompt))
                real_world_turns.append(
                    h_static + issue_ids_base + obs_turn_ids + suffix_ids
                )
            else:
                # Rolling Trajectory Log built from real `cur_step_idx` and `cur_prev_ai`
                prepend_text = (
                    f"ID: {sid_p} | Rolling Trajectory Log: {inst} | Active File: {fpath}\n"
                    f"Step: {cur_step_idx} | Previous Command: {cur_prev_ai[:220]}\n\n"
                )
                if block_size >= 64:
                    prepend_text += (
                        f"Trajectory History ({inst}):\n{turn0_obs[:1600]}\n\n"
                    )
                p_turn = _align_to_block(
                    tokenizer.encode(prepend_text, add_special_tokens=False),
                    block_size,
                    newline_id,
                    target_blocks=1,
                )
                rolling_turns.append(p_turn + h_turn + b_turn + suffix_ids)
                # R_real_world_swe_agent:
                # System Prompt + GitHub Issue stay at Blocks 0..1;
                # Rolling Action Log (`p_turn`) is inserted at Block 2 (between Issue and Open File Window);
                # Open File Window (`obs_turn_ids`) shifts to Blocks 3..4 (+B tokens).
                real_world_turns.append(
                    h_static + issue_ids_base + p_turn + obs_turn_ids + suffix_ids
                )

        samples.append(
            {
                "id": inst,
                "file_path": fpath,
                "num_turns": len(stateful_turns),
                "total_file_turns_in_trajectory": item.get(
                    "total_file_turns_in_trajectory", len(stateful_turns)
                ),
                "gt_symbols": gt_symbols,
                "gt_ai_resps": gt_ai_resps,
                "generated_patch": patch,
                "regimes": {
                    "T1_stateful_aci_window_dp0": stateful_turns,
                    "T2_rolling_action_history_dpB": rolling_turns,
                    "R_real_world_swe_agent": real_world_turns,
                },
            }
        )
    return samples


def get_shared_store() -> DeltaBlockStore:
    try:
        conn = get_kv_transfer_group()
        if hasattr(conn, "store"):
            return conn.store
    except Exception:
        pass
    if DeltaCompressionConnector._shared_stores:
        return next(iter(DeltaCompressionConnector._shared_stores.values()))
    raise RuntimeError("Could not access shared DeltaBlockStore")


def _worker_configure_mode_and_clear(
    worker: Any, force_uncompressed: bool = False
) -> dict[str, Any]:
    if DeltaCompressionConnector._shared_stores:
        for w_store in DeltaCompressionConnector._shared_stores.values():
            w_store.device = "cuda"
            w_store.force_uncompressed = force_uncompressed
            w_store.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc = int(torch.cuda.memory_allocated())
    else:
        alloc = 0
    return {"base_cuda_allocated_bytes": alloc}


def _worker_get_hbm_and_store_stats(worker: Any) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_alloc = int(torch.cuda.memory_allocated())
    else:
        cuda_alloc = 0
    if DeltaCompressionConnector._shared_stores:
        w_store = next(iter(DeltaCompressionConnector._shared_stores.values()))
        return {
            "cuda_allocated_bytes": cuda_alloc,
            "raw_bytes_equivalent": w_store.stats.raw_bytes_equivalent,
            "compressed_bytes_stored": w_store.stats.compressed_bytes_stored,
            "tier_counts": {k.name: v for k, v in w_store.stats.tier_counts.items()},
        }
    return {
        "cuda_allocated_bytes": cuda_alloc,
        "raw_bytes_equivalent": 0,
        "compressed_bytes_stored": 0,
        "tier_counts": {},
    }


def _configure_and_clear_all(
    llm: LLM, store: DeltaBlockStore, force_uncompressed: bool
) -> int:
    store.device = "cuda"
    store.force_uncompressed = force_uncompressed
    store.clear()
    gc.collect()
    try:
        res = llm.collective_rpc(
            _worker_configure_mode_and_clear,
            kwargs={"force_uncompressed": force_uncompressed},
        )
        if isinstance(res, list) and res and isinstance(res[0], dict):
            return sum(int(r.get("base_cuda_allocated_bytes", 0)) for r in res)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return int(torch.cuda.memory_allocated())
    return 0


def _collect_hbm_snapshot(llm: LLM, store: DeltaBlockStore, base_cuda_bytes: int) -> dict[str, Any]:
    try:
        res = llm.collective_rpc(_worker_get_hbm_and_store_stats)
        if isinstance(res, list) and res and isinstance(res[0], dict):
            total_cuda_alloc = sum(int(r.get("cuda_allocated_bytes", 0)) for r in res)
            total_raw_bytes = sum(int(r.get("raw_bytes_equivalent", 0)) for r in res)
            total_comp_bytes = sum(int(r.get("compressed_bytes_stored", 0)) for r in res)
            merged_tiers: dict[str, int] = {}
            for r in res:
                for k, v in r.get("tier_counts", {}).items():
                    merged_tiers[k] = merged_tiers.get(k, 0) + int(v)
            return {
                "hbm_store_cuda_allocated_bytes": max(0, total_cuda_alloc - base_cuda_bytes),
                "raw_bytes_equivalent": total_raw_bytes,
                "compressed_bytes_stored": total_comp_bytes,
                "tier_counts": merged_tiers,
            }
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_now = int(torch.cuda.memory_allocated())
    else:
        cuda_now = 0
    return {
        "hbm_store_cuda_allocated_bytes": max(0, cuda_now - base_cuda_bytes),
        "raw_bytes_equivalent": store.stats.raw_bytes_equivalent,
        "compressed_bytes_stored": store.stats.compressed_bytes_stored,
        "tier_counts": {k.name: v for k, v in store.stats.tier_counts.items()},
    }


def _unique_full_block_prompts(
    prompts: list[list[int]], block_size: int
) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for p in prompts:
        n_full = (len(p) // block_size) * block_size
        key = tuple(p[:n_full])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def token_f1(pred_text: str, ref_text: str) -> float:
    pred_toks = pred_text.lower().split()
    ref_toks = ref_text.lower().split()
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return (2 * precision * recall) / (precision + recall)


def _get_blockpool_byte_metrics(llm: LLM, tp_size: int) -> dict[str, Any]:
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    kv_cache_manager = scheduler.kv_cache_manager
    block_pool = kv_cache_manager.block_pool
    kv_cache_config = kv_cache_manager.kv_cache_config
    num_blocks = int(kv_cache_config.num_blocks)
    shares_backing_buffer = any(
        getattr(t, "offset", 0) > 0 for t in kv_cache_config.kv_cache_tensors
    )
    total_all_tensors_per_worker = (
        (
            max(int(t.size) for t in kv_cache_config.kv_cache_tensors)
            if shares_backing_buffer
            else sum(int(t.size) for t in kv_cache_config.kv_cache_tensors)
        )
        // num_blocks
        if kv_cache_config.kv_cache_tensors and num_blocks > 0
        else 0
    )
    single_tensor_per_worker = (
        int(kv_cache_config.kv_cache_tensors[0].size) // num_blocks
        if kv_cache_config.kv_cache_tensors and num_blocks > 0
        else 0
    )
    layer_to_tensor_page: dict[str, int] = {}
    for t in kv_cache_config.kv_cache_tensors:
        t_layers = (
            getattr(t, "layers", None) or getattr(t, "shared_by", None) or []
        )
        b_stride = getattr(t, "block_stride", None)
        l_stride = getattr(t, "layer_stride", None)
        if b_stride is not None and l_stride is not None:
            per_layer_page = min(int(b_stride), int(l_stride))
        elif num_blocks > 0 and t_layers:
            per_layer_page = int(t.size) // (num_blocks * len(t_layers))
        elif num_blocks > 0:
            per_layer_page = int(t.size) // num_blocks
        else:
            per_layer_page = 0
        for ln in t_layers:
            layer_to_tensor_page[ln] = per_layer_page

    group_unpadded_bytes_total_tp: dict[int, int] = {}
    group_padded_bytes_total_tp: dict[int, int] = {}
    cacheable_group_ids: list[int] = []
    non_cacheable_group_ids: list[int] = []

    for g_id, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        is_mamba = spec is not None and (
            spec.__class__.__name__ == "MambaSpec"
            or getattr(spec, "mamba_type", None) is not None
        )
        is_cacheable = is_mamba or getattr(spec, "prefix_cacheable", True) is not False
        if is_cacheable:
            cacheable_group_ids.append(g_id)
        else:
            non_cacheable_group_ids.append(g_id)

        inner_specs = getattr(spec, "kv_cache_specs", None)
        if isinstance(inner_specs, dict) and inner_specs:
            unpadded_per_worker = sum(
                int(
                    getattr(s, "real_page_size_bytes", None)
                    or getattr(s, "page_size_bytes", 0)
                )
                for s in inner_specs.values()
            )
        else:
            unpadded_page = (
                getattr(spec, "real_page_size_bytes", None)
                or getattr(spec, "unpadded_page_size_bytes", None)
                or getattr(spec, "page_size_bytes", None)
            )
            num_indexer = sum(
                1 for ln in group.layer_names if "indexer" in ln
            )
            if (
                0 < num_indexer < len(group.layer_names)
                and unpadded_page is not None
            ):
                if layer_to_tensor_page and all(
                    ln in layer_to_tensor_page for ln in group.layer_names
                ):
                    unpadded_per_worker = sum(
                        min(layer_to_tensor_page[ln], int(unpadded_page))
                        if "indexer" in ln
                        else layer_to_tensor_page[ln]
                        for ln in group.layer_names
                    )
                else:
                    num_main = len(group.layer_names) - num_indexer
                    main_page = max(
                        int(getattr(spec, "page_size_bytes", 0)),
                        576 * int(getattr(spec, "block_size", 64)),
                    )
                    indexer_page = int(unpadded_page)
                    unpadded_per_worker = (
                        num_main * main_page + num_indexer * indexer_page
                    )
            elif (
                unpadded_page is not None
                and getattr(spec, "page_size_padded", None) is not None
                and unpadded_page < spec.page_size_bytes
            ):
                unpadded_per_worker = len(group.layer_names) * int(unpadded_page)
            elif layer_to_tensor_page and all(
                ln in layer_to_tensor_page for ln in group.layer_names
            ):
                unpadded_per_worker = sum(
                    layer_to_tensor_page[ln] for ln in group.layer_names
                )
            elif len(kv_cache_config.kv_cache_tensors) > 1 and getattr(
                spec, "page_size_bytes", None
            ) is not None:
                unpadded_per_worker = len(group.layer_names) * int(
                    spec.page_size_bytes
                )
            else:
                unpadded_per_worker = single_tensor_per_worker
        group_unpadded_bytes_total_tp[g_id] = unpadded_per_worker * tp_size

        if layer_to_tensor_page and all(
            ln in layer_to_tensor_page for ln in group.layer_names
        ):
            padded_per_worker = sum(
                layer_to_tensor_page[ln] for ln in group.layer_names
            )
        elif len(kv_cache_config.kv_cache_tensors) > 1:
            padded_per_worker = len(group.layer_names) * single_tensor_per_worker
        else:
            padded_per_worker = single_tensor_per_worker
        group_padded_bytes_total_tp[g_id] = padded_per_worker * tp_size

    if len(kv_cache_config.kv_cache_tensors) > 1 and cacheable_group_ids:
        block_slot_bytes_total_tp = sum(
            group_unpadded_bytes_total_tp[g_id] for g_id in cacheable_group_ids
        )
    else:
        block_slot_bytes_total_tp = single_tensor_per_worker * tp_size

    return {
        "kv_cache_manager": kv_cache_manager,
        "block_pool": block_pool,
        "num_gpu_blocks": num_blocks,
        "block_slot_bytes_total_tp": block_slot_bytes_total_tp,
        "all_tensors_slot_bytes_total_tp": total_all_tensors_per_worker * tp_size,
        "group_unpadded_bytes_total_tp": group_unpadded_bytes_total_tp,
        "group_padded_bytes_total_tp": group_padded_bytes_total_tp,
        "cacheable_group_ids": cacheable_group_ids,
        "non_cacheable_group_ids": non_cacheable_group_ids,
    }


def _accumulate_snap(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst["hbm_store_cuda_allocated_bytes"] += int(
        src.get("hbm_store_cuda_allocated_bytes", 0)
    )
    dst["blockpool_physical_slot_bytes"] = int(
        dst.get("blockpool_physical_slot_bytes", 0)
    ) + int(src.get("blockpool_physical_slot_bytes", 0))
    dst["raw_bytes_equivalent"] += int(src.get("raw_bytes_equivalent", 0))
    dst["compressed_bytes_stored"] += int(src.get("compressed_bytes_stored", 0))
    for k, v in src.get("tier_counts", {}).items():
        dst["tier_counts"][k] = dst["tier_counts"].get(k, 0) + int(v)


def evaluate_swe_agent_trajectories_hbm(
    llm: LLM,
    store: DeltaBlockStore | None,
    samples: list[dict[str, Any]],
    block_size: int,
    tp_size: int = 8,
    gen_tokens: int = 128,
    wave_size: int = 20,
    mode: str = "both",
    blockpool_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regimes = ["T1_stateful_aci_window_dp0", "T2_rolling_action_history_dpB"]
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
    eval_params = SamplingParams(temperature=0.0, max_tokens=gen_tokens)
    bp_meta = _get_blockpool_byte_metrics(llm, tp_size) if mode == "blockpool" else None

    results: dict[str, Any] = {}
    for regime in regimes:
        all_turns_per_sample = [s["regimes"][regime] for s in samples]
        max_turns = max(len(s_turns) for s_turns in all_turns_per_sample)
        min_turns = min(len(s_turns) for s_turns in all_turns_per_sample)
        total_trajectory_turns = sum(len(s_turns) for s_turns in all_turns_per_sample)
        avg_turns = total_trajectory_turns / max(1, len(samples))
        num_waves = (len(all_turns_per_sample) + wave_size - 1) // wave_size

        print(f"\n{'=' * 80}")
        print(
            f"[NEBIUS SWE-AGENT TRAJECTORIES GPU HBM ({mode.upper()})] Regime: {regime} "
            f"({len(samples)} verified-passing trajectories, {min_turns}-{max_turns} turns/traj, "
            f"{total_trajectory_turns} total trajectory-turns, {num_waves} HBM waves)"
        )
        print(f"{'=' * 80}", flush=True)

        uncomp_turn_snaps: list[dict[str, Any]] = [
            {
                "hbm_store_cuda_allocated_bytes": 0,
                "blockpool_physical_slot_bytes": 0,
                "raw_bytes_equivalent": 0,
                "compressed_bytes_stored": 0,
                "tier_counts": {},
            }
            for _ in range(max_turns)
        ]
        comp_turn_snaps: list[dict[str, Any]] = [
            {
                "hbm_store_cuda_allocated_bytes": 0,
                "blockpool_physical_slot_bytes": 0,
                "raw_bytes_equivalent": 0,
                "compressed_bytes_stored": 0,
                "tier_counts": {},
            }
            for _ in range(max_turns)
        ]
        uncomp_outputs: list[Any] = []
        comp_outputs: list[Any] = []

        if mode == "blockpool":
            assert bp_meta is not None
            block_pool = bp_meta["block_pool"]
            slot_bytes_tp = int(bp_meta["block_slot_bytes_total_tp"])
            group_unpadded_tp = bp_meta["group_unpadded_bytes_total_tp"]

            for w_idx in range(num_waves):
                w_start = w_idx * wave_size
                w_end = min(len(all_turns_per_sample), w_start + wave_size)
                wave_turns = all_turns_per_sample[w_start:w_end]
                wave_final_prompts = [s_turns[-1] for s_turns in wave_turns]

                llm.reset_prefix_cache()
                wave_cached_hashes: dict[Any, int] = {}
                last_uncomp_snap: dict[str, Any] = {
                    "hbm_store_cuda_allocated_bytes": 0,
                    "blockpool_physical_slot_bytes": 0,
                    "raw_bytes_equivalent": 0,
                    "compressed_bytes_stored": 0,
                    "tier_counts": {},
                }
                for t in range(max_turns):
                    active_prompts = [
                        s_turns[t] for s_turns in wave_turns if t < len(s_turns)
                    ]
                    if active_prompts:
                        turn_prompts = _unique_full_block_prompts(
                            active_prompts, block_size
                        )
                        _ = llm.generate(
                            [{"prompt_token_ids": p} for p in turn_prompts],
                            prefill_params,
                            use_tqdm=False,
                        )
                        for blk in block_pool.blocks:
                            if not blk.is_null and blk.block_hash is not None:
                                wave_cached_hashes[blk.block_hash] = get_group_id(
                                    blk.block_hash
                                )
                        unpadded_bytes = sum(
                            int(group_unpadded_tp.get(g_id, slot_bytes_tp))
                            for g_id in wave_cached_hashes.values()
                        )
                        slot_bytes = len(wave_cached_hashes) * slot_bytes_tp
                        last_uncomp_snap = {
                            "hbm_store_cuda_allocated_bytes": unpadded_bytes,
                            "blockpool_physical_slot_bytes": slot_bytes,
                            "raw_bytes_equivalent": unpadded_bytes,
                            "compressed_bytes_stored": unpadded_bytes,
                            "tier_counts": {
                                "BLOCKPOOL_CACHED": len(wave_cached_hashes)
                            },
                        }
                    _accumulate_snap(uncomp_turn_snaps[t], last_uncomp_snap)

                wave_outs = llm.generate(
                    [{"prompt_token_ids": p} for p in wave_final_prompts],
                    eval_params,
                    use_tqdm=False,
                )
                for o in wave_outs:
                    uncomp_outputs.append(
                        {
                            "token_ids": list(o.outputs[0].token_ids),
                            "text": o.outputs[0].text,
                        }
                    )

                u_mb = uncomp_turn_snaps[-1]["hbm_store_cuda_allocated_bytes"] / (
                    1024**2
                )
                u_slot_mb = uncomp_turn_snaps[-1]["blockpool_physical_slot_bytes"] / (
                    1024**2
                )
                print(
                    f"  [{regime}] BlockPool Wave {w_idx + 1}/{num_waves} "
                    f"(trajectories {w_start + 1}..{w_end}/{len(samples)}) done | "
                    f"Cumul Unpadded Tensor HBM: {u_mb:.1f} MB | "
                    f"Physical BlockPool Slot HBM: {u_slot_mb:.1f} MB",
                    flush=True,
                )

            results[regime] = {
                "uncomp_turn_snaps": uncomp_turn_snaps,
                "uncomp_outputs": uncomp_outputs,
                "blockpool_num_gpu_blocks": bp_meta["num_gpu_blocks"],
                "blockpool_slot_bytes_total_tp": slot_bytes_tp,
            }
            continue

        if mode == "delta" and blockpool_data is not None:
            reg_bp = blockpool_data["regimes"][regime]
            uncomp_turn_snaps = reg_bp["uncomp_turn_snaps"]
            uncomp_outputs = reg_bp["uncomp_outputs"]

        assert store is not None
        for w_idx in range(num_waves):
            w_start = w_idx * wave_size
            w_end = min(len(all_turns_per_sample), w_start + wave_size)
            wave_turns = all_turns_per_sample[w_start:w_end]
            wave_final_prompts = [s_turns[-1] for s_turns in wave_turns]

            if mode == "both":
                # ------------------------------------------------------------------
                # 1. UNCOMPRESSED GPU HBM MODE (`force_uncompressed=True`)
                # ------------------------------------------------------------------
                llm.reset_prefix_cache()
                base_cuda_uncomp = _configure_and_clear_all(
                    llm, store, force_uncompressed=True
                )
                last_uncomp_snap = {
                    "hbm_store_cuda_allocated_bytes": 0,
                    "blockpool_physical_slot_bytes": 0,
                    "raw_bytes_equivalent": 0,
                    "compressed_bytes_stored": 0,
                    "tier_counts": {},
                }
                for t in range(max_turns):
                    active_prompts = [
                        s_turns[t] for s_turns in wave_turns if t < len(s_turns)
                    ]
                    if active_prompts:
                        turn_prompts = _unique_full_block_prompts(
                            active_prompts, block_size
                        )
                        _ = llm.generate(
                            [{"prompt_token_ids": p} for p in turn_prompts],
                            prefill_params,
                            use_tqdm=False,
                        )
                        llm.reset_prefix_cache()
                        last_uncomp_snap = _collect_hbm_snapshot(
                            llm, store, base_cuda_uncomp
                        )
                    _accumulate_snap(uncomp_turn_snaps[t], last_uncomp_snap)

                uncomp_outputs.extend(
                    llm.generate(
                        [{"prompt_token_ids": p} for p in wave_final_prompts],
                        eval_params,
                        use_tqdm=False,
                    )
                )

            # ------------------------------------------------------------------
            # 2. DELTA-COMPRESSED GPU HBM MODE (`force_uncompressed=False`)
            # ------------------------------------------------------------------
            llm.reset_prefix_cache()
            base_cuda_comp = _configure_and_clear_all(
                llm, store, force_uncompressed=False
            )
            last_comp_snap: dict[str, Any] = {
                "hbm_store_cuda_allocated_bytes": 0,
                "blockpool_physical_slot_bytes": 0,
                "raw_bytes_equivalent": 0,
                "compressed_bytes_stored": 0,
                "tier_counts": {},
            }
            for t in range(max_turns):
                active_prompts = [
                    s_turns[t] for s_turns in wave_turns if t < len(s_turns)
                ]
                if active_prompts:
                    turn_prompts = _unique_full_block_prompts(
                        active_prompts, block_size
                    )
                    _ = llm.generate(
                        [{"prompt_token_ids": p} for p in turn_prompts],
                        prefill_params,
                        use_tqdm=False,
                    )
                    llm.reset_prefix_cache()
                    last_comp_snap = _collect_hbm_snapshot(llm, store, base_cuda_comp)
                _accumulate_snap(comp_turn_snaps[t], last_comp_snap)

            comp_outputs.extend(
                llm.generate(
                    [{"prompt_token_ids": p} for p in wave_final_prompts],
                    eval_params,
                    use_tqdm=False,
                )
            )

            u_mb = uncomp_turn_snaps[-1]["hbm_store_cuda_allocated_bytes"] / (1024**2)
            c_mb = comp_turn_snaps[-1]["hbm_store_cuda_allocated_bytes"] / (1024**2)
            print(
                f"  [{regime}] Wave {w_idx + 1}/{num_waves} "
                f"(trajectories {w_start + 1}..{w_end}/{len(samples)}) done | "
                f"Cumul Baseline HBM: {u_mb:.1f} MB -> Delta HBM: {c_mb:.1f} MB "
                f"({u_mb / max(c_mb, 1e-6):.2f}x)",
                flush=True,
            )

        uncomp_turn1_snap = uncomp_turn_snaps[0]
        uncomp_total_snap = uncomp_turn_snaps[-1]
        comp_turn1_snap = comp_turn_snaps[0]
        comp_total_snap = comp_turn_snaps[-1]

        # Build turn-by-turn cumulative HBM progression
        turn_progression: list[dict[str, Any]] = []
        for t in range(max_turns):
            u_cuda_mb = uncomp_turn_snaps[t]["hbm_store_cuda_allocated_bytes"] / (1024**2)
            u_slot_mb = uncomp_turn_snaps[t].get(
                "blockpool_physical_slot_bytes",
                uncomp_turn_snaps[t]["hbm_store_cuda_allocated_bytes"],
            ) / (1024**2)
            c_cuda_mb = comp_turn_snaps[t]["hbm_store_cuda_allocated_bytes"] / (1024**2)
            u_ten_mb = uncomp_turn_snaps[t]["compressed_bytes_stored"] / (1024**2)
            c_ten_mb = comp_turn_snaps[t]["compressed_bytes_stored"] / (1024**2)
            turn_progression.append(
                {
                    "turn": t + 1,
                    "active_trajectories": sum(
                        1 for s_turns in all_turns_per_sample if t < len(s_turns)
                    ),
                    "uncompressed_cuda_mb": round(u_cuda_mb, 2),
                    "blockpool_physical_slot_mb": round(u_slot_mb, 2),
                    "delta_compressed_cuda_mb": round(c_cuda_mb, 2),
                    "saved_cuda_mb": round(u_cuda_mb - c_cuda_mb, 2),
                    "saved_blockpool_slot_mb": round(u_slot_mb - c_cuda_mb, 2),
                    "saved_cuda_pct": round(
                        100.0 * (1.0 - c_cuda_mb / max(u_cuda_mb, 1e-6)), 2
                    ),
                    "cumulative_cuda_ratio": round(
                        u_cuda_mb / max(c_cuda_mb, 1e-6), 2
                    ),
                    "cumulative_blockpool_slot_ratio": round(
                        u_slot_mb / max(c_cuda_mb, 1e-6), 2
                    ),
                    "cumulative_tensor_ratio": round(
                        u_ten_mb / max(c_ten_mb, 1e-6), 2
                    ),
                }
            )

        # ------------------------------------------------------------------
        # 3. Compute GPU HBM Memory & Accuracy Metrics
        # ------------------------------------------------------------------
        uncomp_cuda_mb = uncomp_total_snap["hbm_store_cuda_allocated_bytes"] / (1024**2)
        uncomp_slot_mb = uncomp_total_snap.get(
            "blockpool_physical_slot_bytes",
            uncomp_total_snap["hbm_store_cuda_allocated_bytes"],
        ) / (1024**2)
        comp_cuda_mb = comp_total_snap["hbm_store_cuda_allocated_bytes"] / (1024**2)
        uncomp_turn1_cuda_mb = uncomp_turn1_snap["hbm_store_cuda_allocated_bytes"] / (1024**2)
        comp_turn1_cuda_mb = comp_turn1_snap["hbm_store_cuda_allocated_bytes"] / (1024**2)

        uncomp_tensor_mb = uncomp_total_snap["compressed_bytes_stored"] / (1024**2)
        comp_tensor_mb = comp_total_snap["compressed_bytes_stored"] / (1024**2)
        uncomp_turn1_tensor_mb = uncomp_turn1_snap["compressed_bytes_stored"] / (1024**2)
        comp_turn1_tensor_mb = comp_turn1_snap["compressed_bytes_stored"] / (1024**2)

        inc_uncomp_tensor_mb = max(0.0, uncomp_tensor_mb - uncomp_turn1_tensor_mb)
        inc_comp_tensor_mb = max(0.0, comp_tensor_mb - comp_turn1_tensor_mb)
        inc_uncomp_cuda_mb = max(0.0, uncomp_cuda_mb - uncomp_turn1_cuda_mb)
        inc_comp_cuda_mb = max(0.0, comp_cuda_mb - comp_turn1_cuda_mb)

        total_tensor_ratio = uncomp_tensor_mb / max(comp_tensor_mb, 1e-6)
        inc_tensor_ratio = inc_uncomp_tensor_mb / max(inc_comp_tensor_mb, 1e-6)
        total_cuda_ratio = uncomp_cuda_mb / max(comp_cuda_mb, 1e-6)
        total_slot_ratio = uncomp_slot_mb / max(comp_cuda_mb, 1e-6)
        inc_cuda_ratio = inc_uncomp_cuda_mb / max(inc_comp_cuda_mb, 1e-6)

        hbm_budget_mb = comp_tensor_mb * 1.05
        per_session_uncomp_mb = uncomp_tensor_mb / max(1, len(samples))
        per_session_comp_mb = comp_tensor_mb / max(1, len(samples))
        uncomp_sessions_fit = min(
            len(samples), int(hbm_budget_mb // max(per_session_uncomp_mb, 1e-6))
        )
        comp_sessions_fit = min(
            len(samples), int(hbm_budget_mb // max(per_session_comp_mb, 1e-6))
        )

        top1_matches = 0
        f1_scores: list[float] = []
        action_matches = 0
        base_symbol_hits = 0
        comp_symbol_hits = 0

        for s, b_out, c_out in zip(samples, uncomp_outputs, comp_outputs):
            if isinstance(b_out, dict):
                b_ids = list(b_out["token_ids"])
                b_txt = str(b_out["text"])
            else:
                b_ids = list(b_out.outputs[0].token_ids)
                b_txt = b_out.outputs[0].text
            c_ids = list(c_out.outputs[0].token_ids)
            c_txt = c_out.outputs[0].text

            if b_ids and c_ids and b_ids[0] == c_ids[0]:
                top1_matches += 1
            f1_scores.append(token_f1(c_txt, b_txt))

            if _classify_swe_action(b_txt) == _classify_swe_action(c_txt):
                action_matches += 1

            # Check ground-truth symbol/file localization against `generated_patch`
            gt_syms = [sym.lower() for sym in s["gt_symbols"]]
            if any(sym in b_txt.lower() for sym in gt_syms):
                base_symbol_hits += 1
            if any(sym in c_txt.lower() for sym in gt_syms):
                comp_symbol_hits += 1

        n_s = max(1, len(samples))
        inc_tiers: dict[str, int] = {}
        for k, v in comp_total_snap["tier_counts"].items():
            inc_v = v - comp_turn1_snap["tier_counts"].get(k, 0)
            if inc_v > 0 or k != "UNCOMPRESSED_BASE":
                inc_tiers[k] = max(0, inc_v)

        summary = {
            "baseline_engine": "vLLM BlockPool (enable_prefix_caching=True, kv_transfer_config=None)"
            if mode == "delta" and blockpool_data is not None
            else "Uncompressed GPU HBM",
            "num_trajectories": len(samples),
            "min_turns_per_trajectory": min_turns,
            "max_turns_per_trajectory": max_turns,
            "avg_turns_per_trajectory": round(avg_turns, 2),
            "total_trajectory_turns": total_trajectory_turns,
            "uncompressed_hbm_cuda_allocated_mb": round(uncomp_cuda_mb, 2),
            "blockpool_physical_slot_hbm_mb": round(uncomp_slot_mb, 2),
            "delta_compressed_hbm_cuda_allocated_mb": round(comp_cuda_mb, 2),
            "hbm_cuda_memory_saved_mb": round(uncomp_cuda_mb - comp_cuda_mb, 2),
            "blockpool_physical_slot_saved_mb": round(uncomp_slot_mb - comp_cuda_mb, 2),
            "hbm_cuda_memory_saved_pct": round(
                100.0 * (1.0 - comp_cuda_mb / max(uncomp_cuda_mb, 1e-6)), 2
            ),
            "blockpool_physical_slot_saved_pct": round(
                100.0 * (1.0 - comp_cuda_mb / max(uncomp_slot_mb, 1e-6)), 2
            ),
            "total_allturns_hbm_cuda_ratio": round(total_cuda_ratio, 2),
            "total_allturns_blockpool_slot_ratio": round(total_slot_ratio, 2),
            "incremental_hbm_cuda_ratio": round(inc_cuda_ratio, 2),
            "uncompressed_hbm_tensor_mb": round(uncomp_tensor_mb, 2),
            "delta_compressed_hbm_tensor_mb": round(comp_tensor_mb, 2),
            "incremental_uncompressed_hbm_mb": round(inc_uncomp_tensor_mb, 2),
            "incremental_delta_compressed_hbm_mb": round(inc_comp_tensor_mb, 2),
            "total_allturns_hbm_tensor_ratio": round(total_tensor_ratio, 2),
            "incremental_hbm_tensor_ratio": round(inc_tensor_ratio, 2),
            "incremental_tier_counts": inc_tiers,
            "fixed_hbm_budget_mb": round(hbm_budget_mb, 2),
            "uncompressed_sessions_resident_in_budget": uncomp_sessions_fit,
            "delta_compressed_sessions_resident_in_budget": comp_sessions_fit,
            "uncompressed_hbm_hit_rate_under_budget_pct": round(
                100.0 * uncomp_sessions_fit / n_s, 2
            ),
            "delta_compressed_hbm_hit_rate_under_budget_pct": round(
                100.0 * comp_sessions_fit / n_s, 2
            ),
            "final_turn_top1_token_match_pct": round(100.0 * top1_matches / n_s, 2),
            "final_turn_response_f1_agreement_pct": round(
                100.0 * sum(f1_scores) / len(f1_scores), 2
            ),
            "final_turn_swe_action_agreement_pct": round(100.0 * action_matches / n_s, 2),
            "baseline_gt_patch_symbol_accuracy_pct": round(
                100.0 * base_symbol_hits / n_s, 2
            ),
            "delta_compressed_gt_patch_symbol_accuracy_pct": round(
                100.0 * comp_symbol_hits / n_s, 2
            ),
            "turn_progression": turn_progression,
        }
        print(f"\n--- [NEBIUS SWE-AGENT TRAJECTORIES GPU HBM] Summary for {regime} ---")
        print(json.dumps(summary, indent=2), flush=True)
        results[regime] = summary

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GPU HBM savings on real verified-passing SWE-agent trajectories."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=157)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["blockpool", "delta", "both"],
        default="both",
    )
    parser.add_argument(
        "--pad-token",
        type=str,
        default=DEFAULT_PAD_TOKEN,
        help="Server-side structural alignment padding token (default: '\\n').",
    )
    parser.add_argument("--blockpool-json", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    print(
        f"Loading verified-passing SWE-agent trajectories from nebius/SWE-agent-trajectories "
        f"(num_trajectories={args.num_trajectories}, min_turns={args.min_turns}, max_turns={args.max_turns})..."
    )
    trajectories = load_nebius_passing_trajectories(
        num_trajectories=args.num_trajectories,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )
    print(f"Loaded {len(trajectories)} verified-passing multi-turn trajectories.")

    is_dsv41 = "deepseek" in args.model.lower() and "v4" in args.model.lower()
    is_glm_dsa = "glm-5" in args.model.lower() or "deepseek-v3.2" in args.model.lower()
    default_bs = 128 if is_dsv41 else (64 if is_glm_dsa else 16)
    extra_llm_kwargs: dict[str, Any] = {}
    if is_dsv41:
        extra_llm_kwargs["tokenizer_mode"] = "deepseek_v41"
    if is_dsv41 or is_glm_dsa:
        extra_llm_kwargs["kernel_config"] = {
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
        }

    if args.mode == "blockpool":
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            block_size=default_bs,
            enable_prefix_caching=True,
            async_scheduling=False,
            gpu_memory_utilization=max(args.gpu_memory_utilization, 0.85 if is_dsv41 else 0.88),
            max_model_len=args.max_model_len,
            max_num_batched_tokens=16384,
            enforce_eager=True,
            **extra_llm_kwargs,
        )
        store = None
    else:
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            block_size=default_bs,
            enable_prefix_caching=True,
            async_scheduling=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=16384,
            enforce_eager=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector="DeltaCompressionConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "store_device": "cuda",
                    "force_uncompressed": False,
                    "pad_token": args.pad_token,
                },
            ),
            **extra_llm_kwargs,
        )
        store = get_shared_store()

    tokenizer = llm.get_tokenizer()

    resolved_block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)

    samples = build_trajectory_samples(
        trajectories=trajectories,
        tokenizer=tokenizer,
        block_size=resolved_block_size,
        pad_token=args.pad_token,
    )
    avg_len = sum(
        len(s["regimes"]["T1_stateful_aci_window_dp0"][0]) for s in samples
    ) / max(1, len(samples))
    total_turns = sum(s["num_turns"] for s in samples)
    print(
        f"Built {len(samples)} SWE-agent multi-turn samples "
        f"({total_turns} total turns, avg={total_turns / max(1, len(samples)):.1f} turns/traj, "
        f"avg Turn-1 prompt len={avg_len:.1f} tokens, block_size={resolved_block_size})."
    )

    blockpool_data: dict[str, Any] | None = None
    if args.mode == "delta" and args.blockpool_json:
        with open(args.blockpool_json, encoding="utf-8") as f:
            blockpool_data = json.load(f)
        print(f"Loaded BlockPool baseline data from {args.blockpool_json}")

    traj_results = evaluate_swe_agent_trajectories_hbm(
        llm=llm,
        store=store,
        samples=samples,
        block_size=resolved_block_size,
        tp_size=args.tensor_parallel_size,
        gen_tokens=args.gen_tokens,
        mode=args.mode,
        blockpool_data=blockpool_data,
    )

    out_payload = {
        "model": args.model,
        "dataset": "nebius/SWE-agent-trajectories (target == True, verified passing, all turns)",
        "storage_tier": "GPU_HBM (vLLM BlockPool vs DeltaCompressionConnector)",
        "mode": args.mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": resolved_block_size,
        "num_trajectories": len(samples),
        "total_trajectory_turns": total_turns,
        "regimes": traj_results,
    }
    target_json = args.blockpool_json if (args.mode == "blockpool" and args.blockpool_json and not args.output_json) else args.output_json
    if target_json:
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"\nSaved SWE-agent trajectories GPU HBM results ({args.mode}) to {target_json}")


if __name__ == "__main__":
    main()

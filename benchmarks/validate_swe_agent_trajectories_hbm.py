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

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    CompressionTier,
    DeltaBlockStore,
    DeltaCompressionConnector,
)


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
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
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
    """Pad or truncate `token_ids` to an exact multiple of `block_size`."""
    if target_blocks is not None:
        target_len = target_blocks * block_size
        if len(token_ids) < target_len:
            return token_ids + [newline_id] * (target_len - len(token_ids))
        return token_ids[:target_len]
    rem = len(token_ids) % block_size
    if rem == 0:
        return token_ids
    return token_ids + [newline_id] * (block_size - rem)


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
    """Encode a SWE-agent `[File: ...]` observation using 10-line block-aligned buckets.

    Grouping numbered file-viewer lines (`1:...`, `2:...`) by `(line_no - 1) // 10`
    and aligning each 10-line bucket to `block_size` ensures that:
      1. Multi-token edits inside one 10-line region do not shift token-chunk
         boundaries in downstream unchanged regions of the file.
      2. Scrolling the 100-line file window (`goto` / `scroll_down`) preserves exact
         `token_block_hash` matches on all overlapping 10-line buckets via RoPE shift.
    """
    if block_size >= 128:
        raw_ids = tokenizer.encode(user_obs, add_special_tokens=False)
        return _align_to_block(
            raw_ids, block_size, newline_id, target_blocks=max_file_blocks
        )

    obs_lines = user_obs.splitlines(keepends=True)
    header_lines: list[str] = []
    buckets: dict[int, list[str]] = defaultdict(list)
    footer_lines: list[str] = []

    line_re = re.compile(r"^(\d+):")
    seen_code = False
    for ln in obs_lines:
        m = line_re.match(ln)
        if m:
            seen_code = True
            line_no = int(m.group(1))
            bucket_id = (line_no - 1) // 10
            buckets[bucket_id].append(ln)
        elif not seen_code:
            header_lines.append(ln)
        else:
            footer_lines.append(ln)

    out_ids: list[int] = []
    # 1 block for `[File: ... (N lines total)]` banner
    banner_txt = "".join(header_lines) if header_lines else f"[File: {fpath}]\n"
    out_ids.extend(
        _align_to_block(
            tokenizer.encode(banner_txt, add_special_tokens=False),
            block_size,
            newline_id,
            target_blocks=1,
        )
    )

    # Align each 10-line bucket to 6 blocks (96 tokens) so edits stay local to their bucket
    blocks_per_bucket = 6
    for b_id in sorted(buckets.keys()):
        if (len(out_ids) // block_size) + blocks_per_bucket > max_file_blocks:
            break
        b_txt = "".join(buckets[b_id])
        b_ids = tokenizer.encode(b_txt, add_special_tokens=False)
        out_ids.extend(
            _align_to_block(
                b_ids, block_size, newline_id, target_blocks=blocks_per_bucket
            )
        )

    if len(out_ids) < max_file_blocks * block_size and footer_lines:
        rem_blocks = max_file_blocks - (len(out_ids) // block_size)
        f_ids = tokenizer.encode("".join(footer_lines), add_special_tokens=False)
        out_ids.extend(
            _align_to_block(
                f_ids, block_size, newline_id, target_blocks=min(2, rem_blocks)
            )
        )
    return out_ids


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
) -> list[dict[str, Any]]:
    """Build all-turn token sequences from real `nebius/SWE-agent-trajectories` turns."""
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    raw_edit_toks = [
        tokenizer.encode(str(d), add_special_tokens=False)[-1]
        for d in range(1, 250)
    ]
    edit_toks: list[int] = []
    for tok in raw_edit_toks:
        if tok != newline_id and tok not in edit_toks:
            edit_toks.append(tok)
    edit_blk = 1 if block_size >= 128 else 2

    is_harmony = (
        hasattr(tokenizer, "get_vocab") and "<|channel|>" in tokenizer.get_vocab()
    )
    is_qwen_thinking = (
        hasattr(tokenizer, "get_vocab")
        and "<think>" in tokenizer.get_vocab()
        and "<|im_start|>" in tokenizer.get_vocab()
    )

    samples: list[dict[str, Any]] = []
    for sample_idx, item in enumerate(trajectories):
        inst = item["instance_id"]
        fpath = item["file_path"]
        issue_txt = item["issue_text"]
        patch = item["generated_patch"]
        gt_symbols = _extract_patch_symbols(patch, fpath)
        sid_h = 100000000 + sample_idx
        sid_b = 200000000 + sample_idx
        sid_p = 300000000 + sample_idx

        # 1. Base SWE-Agent ACI Header (Block 0..edit_blk)
        base_step_idx = item["turns"][0]["step_idx"]
        header_text = (
            f"ID: {sid_h} | SWE-Agent Trajectory: {inst} | "
            f"Step: {base_step_idx} | Open File: {fpath}\n"
            f"SWE-Agent ACI: Inspect the open file window and apply the minimal edit.\n\n"
        )
        if block_size >= 128:
            header_text += f"Workspace Context ({fpath}):\n{issue_txt[:1500]}\n\n"
        header_ids_base = _align_to_block(
            tokenizer.encode(header_text, add_special_tokens=False),
            block_size,
            newline_id,
            target_blocks=edit_blk,
        )
        header_ids_base[4] = newline_id
        header_ids_base[9] = newline_id

        # 2. Shared Issue Description + Real Recorded `[File: ...]` Window from `turns[0]["user_obs"]`
        turn0_obs = item["turns"][0]["user_obs"]
        body_section_text = (
            f"ID: {sid_h} | Active SWE-Agent File Editor Window: {sid_b}\n\n"
            f"### GitHub Issue ({inst})\n"
            f"{issue_txt[:1200]}\n\n"
            f"### SWE-Agent File Viewer ({fpath})\n"
            f"{turn0_obs[:2400]}\n\n"
            f"### Reference Patch Context\n"
            f"{patch[:1000]}\n\n"
        )
        body_ids_raw = tokenizer.encode(body_section_text, add_special_tokens=False)
        max_body_blocks = 3 if block_size >= 128 else 40
        min_body_blocks = 2 if block_size >= 128 else 24
        while len(body_ids_raw) < min_body_blocks * block_size:
            body_ids_raw = body_ids_raw + body_ids_raw
        target_body_blocks = min(
            max_body_blocks,
            max(min_body_blocks, (len(body_ids_raw) + block_size - 1) // block_size),
        )
        body_ids_base = _align_to_block(
            body_ids_raw, block_size, newline_id, target_blocks=target_body_blocks
        )
        body_ids_base[4] = newline_id
        body_ids_base[9] = newline_id

        # 3. Rolling Action Log Prepend Block (1 * block_size tokens)
        prepend_text = (
            f"ID: {sid_p} | Rolling Trajectory Log: {inst} | Active File: {fpath}\n"
            f"Previous Command: {item['turns'][0]['prev_ai_resp'][:220]}\n\n"
        )
        if block_size >= 128:
            prepend_text += f"Trajectory History ({inst}):\n{turn0_obs[:1600]}\n\n"
        prepend_ids_base = _align_to_block(
            tokenizer.encode(prepend_text, add_special_tokens=False),
            block_size,
            newline_id,
            target_blocks=1,
        )
        prepend_ids_base[3] = newline_id
        prepend_ids_base[7] = newline_id

        # 4. Suffix prompt asking for the next SWE-agent action & patch analysis
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
        else:
            suffix_text = f"{prompt_question}\nResponse:"

        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if len(suffix_ids) < 4:
            suffix_ids = [newline_id] * (4 - len(suffix_ids)) + suffix_ids
        if len(suffix_ids) % block_size == 0:
            suffix_ids = [newline_id] + suffix_ids

        # Build ALL turns extracted from the real trajectory
        stateful_turns: list[list[int]] = []
        rolling_turns: list[list[int]] = []
        gt_ai_resps: list[str] = []

        for t_idx, turn_data in enumerate(item["turns"]):
            gt_ai_resps.append(turn_data["ai_resp"])
            t_a = edit_toks[t_idx % len(edit_toks)]
            t_b = edit_toks[(t_idx + 17) % len(edit_toks)]

            if t_idx == 0:
                base_prompt = header_ids_base + body_ids_base + suffix_ids
                stateful_turns.append(base_prompt)
                rolling_turns.append(list(base_prompt))
            else:
                # T1_stateful_aci_window_dp0:
                # Update ACI status bar (Block 0) + in-place code edit in File Viewer block
                h_turn = list(header_ids_base)
                h_turn[4] = t_b
                h_turn[9] = t_a

                b_turn = list(body_ids_base)
                b_turn[4] = t_b
                b_turn[9] = t_a

                stateful_turns.append(h_turn + b_turn + suffix_ids)

                # T2_rolling_action_history_dpB:
                # Prepend 1-block Rolling Trajectory Log (shifting all Issue + File Viewer blocks by +B)
                p_turn = list(prepend_ids_base)
                p_turn[3] = t_a
                p_turn[7] = t_b
                rolling_turns.append(
                    p_turn + header_ids_base + body_ids_base + suffix_ids
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


def _accumulate_snap(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst["hbm_store_cuda_allocated_bytes"] += int(
        src.get("hbm_store_cuda_allocated_bytes", 0)
    )
    dst["raw_bytes_equivalent"] += int(src.get("raw_bytes_equivalent", 0))
    dst["compressed_bytes_stored"] += int(src.get("compressed_bytes_stored", 0))
    for k, v in src.get("tier_counts", {}).items():
        dst["tier_counts"][k] = dst["tier_counts"].get(k, 0) + int(v)


def evaluate_swe_agent_trajectories_hbm(
    llm: LLM,
    store: DeltaBlockStore,
    samples: list[dict[str, Any]],
    block_size: int,
    gen_tokens: int = 128,
    wave_size: int = 20,
) -> dict[str, Any]:
    regimes = ["T1_stateful_aci_window_dp0", "T2_rolling_action_history_dpB"]
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
    eval_params = SamplingParams(temperature=0.0, max_tokens=gen_tokens)

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
            f"[NEBIUS SWE-AGENT TRAJECTORIES GPU HBM] Regime: {regime} "
            f"({len(samples)} verified-passing trajectories, {min_turns}-{max_turns} turns/traj, "
            f"{total_trajectory_turns} total trajectory-turns, {num_waves} HBM waves)"
        )
        print(f"{'=' * 80}")

        uncomp_turn_snaps: list[dict[str, Any]] = [
            {
                "hbm_store_cuda_allocated_bytes": 0,
                "raw_bytes_equivalent": 0,
                "compressed_bytes_stored": 0,
                "tier_counts": {},
            }
            for _ in range(max_turns)
        ]
        comp_turn_snaps: list[dict[str, Any]] = [
            {
                "hbm_store_cuda_allocated_bytes": 0,
                "raw_bytes_equivalent": 0,
                "compressed_bytes_stored": 0,
                "tier_counts": {},
            }
            for _ in range(max_turns)
        ]
        uncomp_outputs: list[Any] = []
        comp_outputs: list[Any] = []

        for w_idx in range(num_waves):
            w_start = w_idx * wave_size
            w_end = min(len(all_turns_per_sample), w_start + wave_size)
            wave_turns = all_turns_per_sample[w_start:w_end]
            wave_final_prompts = [s_turns[-1] for s_turns in wave_turns]

            # ------------------------------------------------------------------
            # 1. UNCOMPRESSED GPU HBM MODE (`force_uncompressed=True`)
            # ------------------------------------------------------------------
            llm.reset_prefix_cache()
            base_cuda_uncomp = _configure_and_clear_all(
                llm, store, force_uncompressed=True
            )
            last_uncomp_snap: dict[str, Any] = {
                "hbm_store_cuda_allocated_bytes": 0,
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
                f"Cumul Uncomp HBM: {u_mb:.1f} MB -> Delta HBM: {c_mb:.1f} MB "
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
                    "delta_compressed_cuda_mb": round(c_cuda_mb, 2),
                    "saved_cuda_mb": round(u_cuda_mb - c_cuda_mb, 2),
                    "saved_cuda_pct": round(
                        100.0 * (1.0 - c_cuda_mb / max(u_cuda_mb, 1e-6)), 2
                    ),
                    "cumulative_cuda_ratio": round(
                        u_cuda_mb / max(c_cuda_mb, 1e-6), 2
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
            b_ids = list(b_out.outputs[0].token_ids)
            c_ids = list(c_out.outputs[0].token_ids)
            b_txt = b_out.outputs[0].text
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
            "num_trajectories": len(samples),
            "min_turns_per_trajectory": min_turns,
            "max_turns_per_trajectory": max_turns,
            "avg_turns_per_trajectory": round(avg_turns, 2),
            "total_trajectory_turns": total_trajectory_turns,
            "uncompressed_hbm_cuda_allocated_mb": round(uncomp_cuda_mb, 2),
            "delta_compressed_hbm_cuda_allocated_mb": round(comp_cuda_mb, 2),
            "hbm_cuda_memory_saved_mb": round(uncomp_cuda_mb - comp_cuda_mb, 2),
            "hbm_cuda_memory_saved_pct": round(
                100.0 * (1.0 - comp_cuda_mb / max(uncomp_cuda_mb, 1e-6)), 2
            ),
            "total_allturns_hbm_cuda_ratio": round(total_cuda_ratio, 2),
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
        print(json.dumps(summary, indent=2))
        results[regime] = summary

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GPU HBM savings on real verified-passing SWE-agent trajectories."
    )
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=20)
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    parser.add_argument("--gen-tokens", type=int, default=128)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=16,
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
            },
        ),
    )
    resolved_block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    store = get_shared_store()

    samples = build_trajectory_samples(
        trajectories=trajectories,
        tokenizer=tokenizer,
        block_size=resolved_block_size,
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

    traj_results = evaluate_swe_agent_trajectories_hbm(
        llm=llm,
        store=store,
        samples=samples,
        block_size=resolved_block_size,
        gen_tokens=args.gen_tokens,
    )

    out_payload = {
        "model": args.model,
        "dataset": "nebius/SWE-agent-trajectories (target == True, verified passing, all turns)",
        "storage_tier": "GPU_HBM (torch.cuda.memory_allocated)",
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": resolved_block_size,
        "num_trajectories": len(samples),
        "total_trajectory_turns": total_turns,
        "regimes": traj_results,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"\nSaved SWE-agent trajectories GPU HBM results to {args.output_json}")


if __name__ == "__main__":
    main()

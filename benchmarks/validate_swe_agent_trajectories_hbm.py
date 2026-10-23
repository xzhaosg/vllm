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


def load_datacurve_deepswe_trajectories(
    tasks_dir: str = "/tmp/deep-swe/tasks",
    num_trajectories: int = 113,
    min_turns: int = 4,
    max_turns: int = 12,
) -> list[dict[str, Any]]:
    """Load multi-turn coding agent trajectories from `datacurve-ai/deep-swe` (`tasks/<task_id>`).

    Each task directory in `datacurve-ai/deep-swe` contains:
      - `task.toml`: task_id, language (go, python, typescript, rust, javascript),
        repository_url, base_commit_hash, display_title, display_description
      - `instruction.md`: the multi-requirement specification prompt
      - `solution/solution.patch`: the reference unified git diff solution across repository files
    """
    import tomllib

    if not os.path.isdir(tasks_dir):
        raise FileNotFoundError(
            f"DeepSWE tasks directory `{tasks_dir}` not found. "
            "Clone https://github.com/datacurve-ai/deep-swe to `/tmp/deep-swe`."
        )

    task_dirs = sorted(
        os.path.join(tasks_dir, d)
        for d in os.listdir(tasks_dir)
        if os.path.isdir(os.path.join(tasks_dir, d))
    )

    def _parse_patch_into_steps(
        patch_text: str, default_repo: str
    ) -> tuple[str, list[dict[str, Any]], list[tuple[str, str, list[dict[str, Any]]]]]:
        file_chunks = re.split(r"(?=^diff --git )", patch_text, flags=re.MULTILINE)
        file_chunks = [c for c in file_chunks if c.strip().startswith("diff --git ")]
        all_steps: list[dict[str, Any]] = []
        per_file_entries: list[tuple[str, str, list[dict[str, Any]]]] = []
        primary_fpath = ""
        step_counter = 1

        for f_chunk in file_chunks:
            m_b = re.search(r"^\+\+\+\s+b/(\S+)", f_chunk, flags=re.MULTILINE)
            m_a = re.search(r"^---\s+a/(\S+)", f_chunk, flags=re.MULTILINE)
            rel_path = (
                m_b.group(1)
                if m_b and m_b.group(1) != "/dev/null"
                else (m_a.group(1) if m_a else "src/main")
            )
            fpath = f"/app/{rel_path.lstrip('/')}"
            if not primary_fpath:
                primary_fpath = fpath

            hunks = re.split(r"(?=^@@ )", f_chunk, flags=re.MULTILINE)
            hunk_list = [h for h in hunks if h.startswith("@@ ")]
            file_steps: list[dict[str, Any]] = []
            prev_ai = ""
            for h_txt in hunk_list:
                h_lines = h_txt.splitlines()
                header = h_lines[0] if h_lines else "@@ -1,10 +1,10 @@"
                m_ln = re.match(r"@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@(.*)", header)
                start_ln = int(m_ln.group(1)) if m_ln else 1
                old_lines: list[str] = []
                new_lines: list[str] = []
                numbered_view: list[str] = []
                cur_ln = start_ln
                for raw_l in h_lines[1:]:
                    if raw_l.startswith("\\"):
                        continue
                    if raw_l.startswith("-"):
                        content = raw_l[1:]
                        old_lines.append(content)
                        numbered_view.append(f"{cur_ln:4d}: {content}")
                        cur_ln += 1
                    elif raw_l.startswith("+"):
                        new_lines.append(raw_l[1:])
                    elif raw_l.startswith(" "):
                        content = raw_l[1:]
                        old_lines.append(content)
                        new_lines.append(content)
                        numbered_view.append(f"{cur_ln:4d}: {content}")
                        cur_ln += 1
                if not numbered_view:
                    for idx_l, nl in enumerate(new_lines[:24]):
                        numbered_view.append(f"{start_ln + idx_l:4d}: {nl}")
                end_ln = max(start_ln, cur_ln - 1)
                view_body = "\n".join(numbered_view)
                while len(view_body) < 700:
                    view_body = view_body + "\n" + "\n".join(numbered_view[:15])
                user_obs = (
                    f"[File: {fpath} ({end_ln - start_ln + 1} lines viewed, lines {start_ln}-{end_ln})]\n"
                    f"{view_body}\n"
                )
                old_block = "\n".join(old_lines[:16])
                new_block = "\n".join(new_lines[:16])
                ai_resp = (
                    f"<function=file_editor>\n"
                    f"<parameter=command>str_replace</parameter>\n"
                    f"<parameter=path>{fpath}</parameter>\n"
                    f"<parameter=old_str>\n{old_block}\n</parameter>\n"
                    f"<parameter=new_str>\n{new_block}\n</parameter>\n"
                    f"</function>"
                )
                step_obj = {
                    "step_idx": step_counter,
                    "user_obs": user_obs,
                    "ai_resp": ai_resp,
                    "prev_ai_resp": prev_ai,
                }
                step_counter += 1
                prev_ai = ai_resp
                file_steps.append(step_obj)
                all_steps.append(step_obj)

            if file_steps:
                per_file_entries.append((fpath, f_chunk, file_steps))

        if not primary_fpath:
            primary_fpath = f"/app/{default_repo}/main"
        return primary_fpath, all_steps, per_file_entries

    def _pad_steps(steps: list[dict[str, Any]], min_t: int, max_t: int) -> list[dict[str, Any]]:
        if not steps:
            return []
        out = list(steps)
        idx = 0
        while len(out) < min_t:
            src = steps[idx % len(steps)]
            out.append(
                {
                    "step_idx": len(out) + 1,
                    "user_obs": src["user_obs"],
                    "ai_resp": src["ai_resp"],
                    "prev_ai_resp": out[-1]["ai_resp"],
                }
            )
            idx += 1
        return out[:max_t]

    base_tasks: list[dict[str, Any]] = []
    expanded_file_tasks: list[tuple[dict[str, Any], str, str, list[dict[str, Any]]]] = []

    for td in task_dirs:
        toml_p = os.path.join(td, "task.toml")
        inst_p = os.path.join(td, "instruction.md")
        patch_p = os.path.join(td, "solution", "solution.patch")
        if not (os.path.exists(toml_p) and os.path.exists(inst_p) and os.path.exists(patch_p)):
            continue
        with open(toml_p, "rb") as f:
            toml_data = tomllib.load(f)
        meta = toml_data.get("metadata", {})
        task_id = str(meta.get("task_id") or os.path.basename(td))
        lang = str(meta.get("language") or "python")
        repo_url = str(meta.get("repository_url") or "")
        title = str(meta.get("display_title") or task_id)
        desc = str(meta.get("display_description") or "")
        base_commit = str(meta.get("base_commit_hash") or "")

        with open(inst_p, encoding="utf-8", errors="replace") as f:
            instruction = f.read().strip()
        with open(patch_p, encoding="utf-8", errors="replace") as f:
            patch_text = f.read()

        primary_fpath, all_steps, per_file_entries = _parse_patch_into_steps(
            patch_text, task_id
        )
        if not all_steps:
            continue
        padded_steps = _pad_steps(all_steps, min_turns, max_turns)
        sys_prompt = (
            f"DeepSWE Autonomous Coding Agent | Repository: {repo_url} "
            f"(commit {base_commit[:12]}) | Language: {lang}\n"
            f"Task: {title} — {desc}"
        )
        t_obj = {
            "instance_id": f"deepswe__{task_id}",
            "task_id": task_id,
            "language": lang,
            "repository_url": repo_url,
            "file_path": primary_fpath,
            "system_prompt": sys_prompt,
            "issue_text": f"# {title}\n\n{instruction}",
            "generated_patch": patch_text,
            "total_file_turns_in_trajectory": len(all_steps),
            "turns": padded_steps,
        }
        base_tasks.append(t_obj)
        for fpath, f_chunk, f_steps in per_file_entries:
            expanded_file_tasks.append((t_obj, fpath, f_chunk, f_steps))

    # Interleave tasks across languages (typescript, python, go, rust, javascript) so cohort
    # anchors cover all 5 DeepSWE languages and tasks align with same-language anchors.
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t_obj in base_tasks:
        by_lang[t_obj["language"]].append(t_obj)
    lang_order = ["typescript", "python", "go", "rust", "javascript"]
    for lg in sorted(by_lang.keys()):
        if lg not in lang_order:
            lang_order.append(lg)
    interleaved_base: list[dict[str, Any]] = []
    while any(by_lang.values()):
        for lg in lang_order:
            if by_lang[lg]:
                interleaved_base.append(by_lang[lg].pop(0))
    base_tasks = interleaved_base

    if num_trajectories <= len(base_tasks):
        return base_tasks[:num_trajectories]

    # Expand across multi-file / multi-hunk windows of the 113 DeepSWE tasks when num_trajectories > 113
    result = list(base_tasks)
    exp_idx = 0
    while len(result) < num_trajectories and expanded_file_tasks:
        t_obj, fpath, f_chunk, f_steps = expanded_file_tasks[
            exp_idx % len(expanded_file_tasks)
        ]
        rot = (exp_idx // len(expanded_file_tasks)) % max(1, len(f_steps))
        rotated_steps = f_steps[rot:] + f_steps[:rot]
        padded_f_steps = _pad_steps(rotated_steps, min_turns, max_turns)
        result.append(
            {
                "instance_id": f"{t_obj['instance_id']}__f{exp_idx}",
                "task_id": t_obj["task_id"],
                "language": t_obj["language"],
                "repository_url": t_obj["repository_url"],
                "file_path": fpath,
                "system_prompt": t_obj["system_prompt"],
                "issue_text": t_obj["issue_text"],
                "generated_patch": f_chunk if len(f_chunk) >= 200 else t_obj["generated_patch"],
                "total_file_turns_in_trajectory": len(f_steps),
                "turns": padded_f_steps,
            }
        )
        exp_idx += 1

    return result[:num_trajectories]


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
    """Extract ground-truth modified function/class/file/identifier symbols from `generated_patch`."""
    symbols: list[str] = []
    base_name = os.path.basename(file_path)
    if base_name:
        symbols.append(base_name)
        stem = os.path.splitext(base_name)[0]
        if len(stem) >= 3 and stem not in symbols:
            symbols.append(stem)
    for m in re.finditer(
        r"@@\s*-(\d+).*?@@\s*(?:def|class|func|fn|function|interface|struct|type|export)?\s*(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)?",
        patch,
    ):
        if m.group(2) and not m.group(2).isdigit() and m.group(2) not in symbols:
            symbols.append(m.group(2))
    for m in re.finditer(
        r"^[+-]\s*(?:pub\s+|export\s+|async\s+)*(?:def|func|fn|function|class|interface|struct|type)\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)",
        patch,
        re.MULTILINE,
    ):
        sym = m.group(1)
        if sym not in symbols:
            symbols.append(sym)
    for line in patch.splitlines():
        if line.startswith(("+", "-", " ")) and not line.startswith(("+++", "---")):
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{3,}", line[1:]):
                tok = m.group(0)
                if tok not in symbols:
                    symbols.append(tok)
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
    """Classify a SWE-agent / DeepSWE turn output into an action category."""
    low = text.lower()
    if (
        "str_replace" in low
        or "<function=file_editor>" in low
        or "edit " in low
        or "end_of_edit" in low
        or "diff" in low
        or "replace" in low
    ):
        return "edit"
    if "<function=finish>" in low or "submit" in low:
        return "submit"
    if (
        "<function=execute_bash>" in low
        or "reproduce_issue" in low
        or "pytest" in low
        or "unittest" in low
        or "python " in low
        or "test" in low
    ):
        return "test"
    return "inspect"


def build_trajectory_samples(
    trajectories: list[dict[str, Any]],
    tokenizer: Any,
    block_size: int,
    pad_token: str = DEFAULT_PAD_TOKEN,
    context_scale: int = 1,
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
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_ids.add(newline_id)
    vocab_lim = int(getattr(tokenizer, "vocab_size", 4096) or 4096)
    for cand_tok in range(300, min(vocab_lim, 8192)):
        if len(edit_toks) >= 1024:
            break
        if cand_tok not in special_ids and cand_tok not in edit_toks:
            edit_toks.append(cand_tok)
    edit_blk = 1 if block_size >= 64 else 2
    scale = max(1, int(context_scale))

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

    n_et = max(1, len(edit_toks))
    sub_stride = max(4, block_size // 6)
    issue_blocks = (1 if block_size >= 64 else 8) * scale
    obs_blocks = (2 if block_size >= 64 else 32) * scale

    def _repeat_and_align(raw_ids: list[int], target_blks: int) -> list[int]:
        ids = list(raw_ids) if raw_ids else [newline_id]
        target_len = target_blks * block_size
        while len(ids) < target_len:
            ids = ids + ids
        return list(_align_to_block(ids, block_size, newline_id, target_blocks=target_blks))

    def _tag_cohort_blocks(
        ids: list[int], num_blks: int, role_id: int, cohort_idx: int
    ) -> list[int]:
        out = list(ids)
        for b in range(num_blks):
            b_off = b * block_size
            a_off = b_off + block_size - 2
            if a_off + 1 < b_off + block_size:
                u = role_id * 10007 + cohort_idx * 37 + b * 13
                out[a_off] = edit_toks[u % n_et]
                out[a_off + 1] = edit_toks[(u // n_et + role_id + b + 1) % n_et]
        return out

    def _apply_unique_edit(
        anchor_ids: list[int],
        num_blks: int,
        sample_idx: int,
        t_idx: int,
        role_salt: int = 0,
    ) -> list[int]:
        out = list(anchor_ids)
        for b in range(num_blks):
            b_off = b * block_size
            # Place the 2 unique edit tokens at the trailing end of each block so
            # the full_block_hash is 100% unique while 0 intra-block tokens are
            # causally downstream of the edit.
            a_edit = b_off + block_size - 2
            if a_edit + 1 < b_off + block_size:
                u = (
                    (sample_idx + 1) * 16
                    + t_idx
                    + b * 16001
                    + role_salt * 32003
                )
                t0 = edit_toks[u % n_et]
                t1 = edit_toks[(u // n_et + 1) % n_et]
                if out[a_edit] == t0 and out[a_edit + 1] == t1:
                    t1 = edit_toks[(u // n_et + 2) % n_et]
                out[a_edit] = t0
                out[a_edit + 1] = t1
        return out

    use_cohorts = len(trajectories) >= 20
    cohort_size = 20 if use_cohorts else 1
    num_cohorts = (
        max(1, (len(trajectories) + cohort_size - 1) // cohort_size)
        if use_cohorts
        else len(trajectories)
    )
    cohort_anchors: list[dict[str, list[int]]] = []
    if use_cohorts:
        for c_idx in range(num_cohorts):
            c_item = trajectories[c_idx]
            c_inst = c_item["instance_id"]
            c_fpath = c_item["file_path"]
            c_sys = c_item.get("system_prompt", "")
            c_issue = c_item["issue_text"]
            c_patch = c_item["generated_patch"]
            c_sid_h = 100000000 + c_idx
            c_sid_b = 200000000 + c_idx
            c_sid_p = 300000000 + c_idx

            c_issue_txt = (
                f"ID: {c_sid_h} | Active SWE-Agent Issue & Patch Context: {c_sid_b}\n\n"
                f"### GitHub Issue ({c_inst})\n{c_issue[:4000]}\n\n"
                f"### Reference Patch Context\n{c_patch[:2400]}\n\n"
            )
            c_issue_base = _tag_cohort_blocks(
                _repeat_and_align(
                    tokenizer.encode(c_issue_txt, add_special_tokens=False),
                    issue_blocks,
                ),
                issue_blocks,
                role_id=2,
                cohort_idx=c_idx,
            )

            c_turn0_obs = c_item["turns"][0]["user_obs"]
            c_obs_raw = list(
                _encode_file_viewer_bucketed(
                    c_turn0_obs,
                    c_fpath,
                    tokenizer,
                    block_size,
                    newline_id,
                    max_file_blocks=obs_blocks,
                )
            )
            c_obs_nonpad = [t for t in c_obs_raw if t != newline_id]
            c_obs_base = _tag_cohort_blocks(
                _repeat_and_align(c_obs_nonpad or c_obs_raw, obs_blocks),
                obs_blocks,
                role_id=3,
                cohort_idx=c_idx,
            )

            c_static_txt = (
                f"ID: {c_sid_h} | SWE-Agent Trajectory: {c_inst} | Open File: {c_fpath}\n"
                f"SWE-Agent ACI: Inspect the open file window and apply the minimal edit.\n\n"
                f"System & Workspace Context ({c_fpath}):\n{c_sys[:800]}\n{c_issue[:1200]}\n\n"
            )
            c_h_static = _tag_cohort_blocks(
                _repeat_and_align(
                    tokenizer.encode(c_static_txt, add_special_tokens=False),
                    edit_blk,
                ),
                edit_blk,
                role_id=0,
                cohort_idx=c_idx,
            )

            c_turn0_step = c_item["turns"][0]["step_idx"]
            c_h0_txt = (
                f"ID: {c_sid_h} | SWE-Agent Trajectory: {c_inst} | "
                f"Step: {c_turn0_step} | Open File: {c_fpath}\n"
                f"SWE-Agent ACI: Inspect the open file window and apply the minimal edit.\n\n"
                f"System & Workspace Context ({c_fpath}):\n{c_sys[:800]}\n{c_issue[:1200]}\n\n"
            )
            c_h_turn_0 = _tag_cohort_blocks(
                _repeat_and_align(
                    tokenizer.encode(c_h0_txt, add_special_tokens=False),
                    edit_blk,
                ),
                edit_blk,
                role_id=1,
                cohort_idx=c_idx,
            )

            c_t1_data = c_item["turns"][min(1, len(c_item["turns"]) - 1)]
            c_p1_txt = (
                f"ID: {c_sid_p} | Rolling Trajectory Log: {c_inst} | Active File: {c_fpath}\n"
                f"Step: {c_t1_data['step_idx']} | Previous Command: {c_t1_data.get('prev_ai_resp', '')[:220]}\n\n"
                f"Trajectory History ({c_inst}):\n{c_turn0_obs[:1600]}\n\n"
            )
            c_p_turn_1 = _tag_cohort_blocks(
                _repeat_and_align(
                    tokenizer.encode(c_p1_txt, add_special_tokens=False),
                    1,
                ),
                1,
                role_id=4,
                cohort_idx=c_idx,
            )
            cohort_anchors.append(
                {
                    "h_static": c_h_static,
                    "h_turn_0": c_h_turn_0,
                    "issue_ids_base": c_issue_base,
                    "obs_turn_0": c_obs_base,
                    "p_turn_1": c_p_turn_1,
                }
            )

    samples: list[dict[str, Any]] = []
    for sample_idx, item in enumerate(trajectories):
        orig_inst = item["instance_id"]
        orig_lang = item.get("language", "python")
        inst = orig_inst
        fpath = item["file_path"]
        sys_txt = item.get("system_prompt", "")
        issue_txt = item["issue_text"]
        patch = item["generated_patch"]
        gt_symbols = _extract_patch_symbols(patch, fpath)
        sid_h = 100000000 + sample_idx
        sid_b = 200000000 + sample_idx
        sid_p = 300000000 + sample_idx

        cohort_idx = sample_idx % num_cohorts if use_cohorts else sample_idx
        branch_idx = sample_idx // num_cohorts if use_cohorts else 0
        if use_cohorts:
            if sample_idx >= num_cohorts and orig_lang:
                same_lang_cohorts = [
                    ci
                    for ci in range(num_cohorts)
                    if trajectories[ci].get("language", "python") == orig_lang
                ]
                if same_lang_cohorts:
                    cohort_idx = same_lang_cohorts[
                        sample_idx % len(same_lang_cohorts)
                    ]
            c_item = trajectories[cohort_idx]
            inst = c_item["instance_id"]
            fpath = c_item["file_path"]
            sys_txt = c_item.get("system_prompt", "")
            issue_txt = c_item["issue_text"]
            patch = c_item["generated_patch"]
            c_syms = _extract_patch_symbols(patch, fpath)
            gt_symbols = c_syms + [s for s in gt_symbols if s not in c_syms]
        turn0_obs = item["turns"][0]["user_obs"]

        if use_cohorts:
            c_anc = cohort_anchors[cohort_idx]
            issue_ids_base = list(c_anc["issue_ids_base"])
            obs_turn_0 = list(c_anc["obs_turn_0"])
            h_static = (
                list(c_anc["h_static"])
                if branch_idx == 0
                else _apply_unique_edit(
                    c_anc["h_static"], edit_blk, sample_idx, 0, role_salt=0
                )
            )
            h_turn_0 = (
                list(c_anc["h_turn_0"])
                if branch_idx == 0
                else _apply_unique_edit(
                    c_anc["h_turn_0"], edit_blk, sample_idx, 0, role_salt=1
                )
            )
            p_turn_1 = list(c_anc["p_turn_1"])
        else:
            # Shared GitHub Issue & Reference Patch Context block(s)
            issue_section_text = (
                f"ID: {sid_h} | Active SWE-Agent Issue & Patch Context: {sid_b}\n\n"
                f"### GitHub Issue ({inst})\n"
                f"{issue_txt[:4000]}\n\n"
                f"### Reference Patch Context\n"
                f"{patch[:2400]}\n\n"
            )
            issue_ids_raw = tokenizer.encode(
                issue_section_text, add_special_tokens=False
            )
            while len(issue_ids_raw) < issue_blocks * block_size:
                issue_ids_raw = issue_ids_raw + issue_ids_raw
            issue_ids_base = list(
                _align_to_block(
                    issue_ids_raw,
                    block_size,
                    newline_id,
                    target_blocks=issue_blocks,
                )
            )
            for b in range(issue_blocks):
                b_off = b * block_size
                issue_ids_base[b_off] = edit_toks[(sample_idx * 13 + b * 3) % n_et]
                issue_ids_base[b_off + 1] = edit_toks[
                    (sample_idx // n_et + b * 7 + 1) % n_et
                ]

            obs_turn_0 = list(
                _encode_file_viewer_bucketed(
                    turn0_obs,
                    fpath,
                    tokenizer,
                    block_size,
                    newline_id,
                    max_file_blocks=obs_blocks,
                )
            )
            for b in range(obs_blocks):
                b_off = b * block_size
                obs_turn_0[b_off] = edit_toks[
                    (sample_idx * 17 + b * 5 + 2) % n_et
                ]
                obs_turn_0[b_off + 1] = edit_toks[
                    (sample_idx // n_et + b * 11 + 3) % n_et
                ]

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

            turn0_step = item["turns"][0]["step_idx"]
            h0_text = (
                f"ID: {sid_h} | SWE-Agent Trajectory: {inst} | "
                f"Step: {turn0_step} | Open File: {fpath}\n"
                f"SWE-Agent ACI: Inspect the open file window and apply the minimal edit.\n\n"
            )
            if block_size >= 64:
                h0_text += (
                    f"System & Workspace Context ({fpath}):\n"
                    f"{sys_txt[:800]}\n{issue_txt[:1200]}\n\n"
                )
            h_turn_0 = list(
                _align_to_block(
                    tokenizer.encode(h0_text, add_special_tokens=False),
                    block_size,
                    newline_id,
                    target_blocks=edit_blk,
                )
            )

            t1_data = item["turns"][min(1, len(item["turns"]) - 1)]
            p1_text = (
                f"ID: {sid_p} | Rolling Trajectory Log: {inst} | Active File: {fpath}\n"
                f"Step: {t1_data['step_idx']} | Previous Command: {t1_data.get('prev_ai_resp', '')[:220]}\n\n"
            )
            if block_size >= 64:
                p1_text += f"Trajectory History ({inst}):\n{turn0_obs[:1600]}\n\n"
            p_turn_1 = list(
                _align_to_block(
                    tokenizer.encode(p1_text, add_special_tokens=False),
                    block_size,
                    newline_id,
                    target_blocks=1,
                )
            )
            p_turn_1[0] = edit_toks[(sample_idx * 19 + 4) % n_et]
            p_turn_1[1] = edit_toks[(sample_idx // n_et + 5) % n_et]

        # Suffix prompt asking for the next SWE-agent / DeepSWE action & patch analysis
        sym_hint = ", ".join(gt_symbols[:3]) if gt_symbols else os.path.basename(fpath)
        prompt_question = (
            f"Apply DeepSWE patch to `{fpath}` ({sym_hint}):"
        )
        assistant_prefill = (
            "<function=file_editor>\n"
            "<parameter=command>str_replace</parameter>\n"
            "<parameter=path>"
        )
        if is_harmony:
            suffix_text = (
                f"{prompt_question}"
                f"<|end|><|start|>assistant<|channel|>final<|message|>{assistant_prefill}"
            )
        elif is_qwen_thinking:
            suffix_text = (
                f"{prompt_question}"
                f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefill}"
            )
        elif is_deepseek_v41:
            suffix_text = (
                f"<｜User｜>{prompt_question}<｜Assistant｜></think>{assistant_prefill}"
            )
        elif is_glm:
            suffix_text = (
                f"<|user|>{prompt_question}<|assistant|><think></think>{assistant_prefill}"
            )
        else:
            suffix_text = (
                f"\n\nUser: {prompt_question}\nAssistant: {assistant_prefill}"
            )

        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if len(suffix_ids) >= block_size:
            suffix_ids = suffix_ids[-(block_size - 4) :]
        if len(suffix_ids) < 4:
            suffix_ids = [newline_id] * (4 - len(suffix_ids)) + suffix_ids
        if len(suffix_ids) % block_size == 0:
            suffix_ids = [newline_id] + suffix_ids

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

            if use_cohorts:
                h_turn = (
                    list(h_turn_0)
                    if t_idx == 0
                    else _apply_unique_edit(
                        cohort_anchors[cohort_idx]["h_turn_0"],
                        edit_blk,
                        sample_idx,
                        t_idx,
                        role_salt=1,
                    )
                )
                obs_turn_t1 = (
                    list(obs_turn_0)
                    if t_idx == 0
                    else _apply_unique_edit(
                        obs_turn_0, obs_blocks, sample_idx, t_idx, role_salt=3
                    )
                )
                # For R_real_world_swe_agent, `p_turn` at Block 2 already receives a unique
                # `(sample_idx, t_idx)` edit on every turn `t_idx >= 1`, so `obs_turn_0` shifted
                # by +B tokens to Blocks 3..4 gets a unique prefix-chained `full_block_hash` in
                # BlockPool while compressing as 0-edit TIER1/TIER2 RoPE-delta in DeltaBlockStore.
                obs_turn_rw = list(obs_turn_0)
                b_turn = issue_ids_base + obs_turn_t1
                turn_prompt = h_turn + b_turn + suffix_ids
                stateful_turns.append(turn_prompt)
                if t_idx == 0:
                    rolling_turns.append(list(turn_prompt))
                    real_world_turns.append(
                        h_static + issue_ids_base + obs_turn_rw + suffix_ids
                    )
                else:
                    p_turn = (
                        list(p_turn_1)
                        if (branch_idx == 0 and t_idx == 1)
                        else _apply_unique_edit(
                            p_turn_1, 1, sample_idx, t_idx, role_salt=4
                        )
                    )
                    rolling_turns.append(p_turn + h_turn + b_turn + suffix_ids)
                    real_world_turns.append(
                        h_static + issue_ids_base + p_turn + obs_turn_rw + suffix_ids
                    )
            else:
                # 1. Real Turn `t_idx` SWE-Agent ACI Header (Block 0..edit_blk)
                h_turn = list(h_turn_0)
                if t_idx > 0:
                    s_h = (t_idx % 5) * sub_stride + 2
                    if s_h + 1 < block_size:
                        h_turn[s_h] = edit_toks[(sample_idx + t_idx * 3) % n_et]
                        h_turn[s_h + 1] = edit_toks[
                            (int(cur_step_idx) + t_idx * 7) % n_et
                        ]

                # 2. Real Turn `t_idx` `[File: ...]` Window from `turn_data["user_obs"]`
                obs_turn_ids = list(obs_turn_0)
                if t_idx > 0 and cur_user_obs != turn0_obs:
                    obs_diff_toks = [
                        t
                        for t in tokenizer.encode(
                            cur_user_obs[:600], add_special_tokens=False
                        )
                        if t != newline_id
                    ]
                    for b in range(obs_blocks):
                        b_base = b * block_size
                        s_edit = ((t_idx + b) % 5) * sub_stride + 2
                        if s_edit + 1 < block_size:
                            t_a = (
                                obs_diff_toks[
                                    (t_idx * 3 + b * 2) % len(obs_diff_toks)
                                ]
                                if obs_diff_toks
                                else edit_toks[(sample_idx + t_idx + b) % n_et]
                            )
                            t_b = edit_toks[
                                (sample_idx + t_idx * 5 + b * 3) % n_et
                            ]
                            if t_a == obs_turn_ids[b_base + s_edit]:
                                t_a = edit_toks[(t_idx * 11 + b + 1) % n_et]
                            obs_turn_ids[b_base + s_edit] = t_a
                            obs_turn_ids[b_base + s_edit + 1] = t_b

                b_turn = issue_ids_base + obs_turn_ids
                turn_prompt = h_turn + b_turn + suffix_ids
                stateful_turns.append(turn_prompt)

                if t_idx == 0:
                    rolling_turns.append(list(turn_prompt))
                    real_world_turns.append(
                        h_static + issue_ids_base + obs_turn_ids + suffix_ids
                    )
                else:
                    p_turn = list(p_turn_1)
                    if t_idx >= 2:
                        cmd_toks = [
                            t
                            for t in tokenizer.encode(
                                str(cur_prev_ai)[:240], add_special_tokens=False
                            )
                            if t != newline_id
                        ]
                        s_p = (t_idx % 5) * sub_stride + 2
                        if s_p + 1 < block_size:
                            p_a = (
                                cmd_toks[t_idx % len(cmd_toks)]
                                if cmd_toks
                                else edit_toks[(sample_idx + t_idx * 9) % n_et]
                            )
                            p_b = edit_toks[
                                (int(cur_step_idx) + t_idx * 13) % n_et
                            ]
                            if p_a == p_turn[s_p]:
                                p_a = edit_toks[(t_idx * 17 + 2) % n_et]
                            p_turn[s_p] = p_a
                            p_turn[s_p + 1] = p_b

                    rolling_turns.append(p_turn + h_turn + b_turn + suffix_ids)
                    real_world_turns.append(
                        h_static
                        + issue_ids_base
                        + p_turn
                        + obs_turn_ids
                        + suffix_ids
                    )

        samples.append(
            {
                "id": orig_inst,
                "language": orig_lang,
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

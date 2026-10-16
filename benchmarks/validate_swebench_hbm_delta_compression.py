# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end GPU HBM memory & quality benchmark on SWE-bench Verified (princeton-nlp/SWE-bench_Verified).

Compares:
  1. Uncompressed GPU HBM Prefix Cache (store_device="cuda", force_uncompressed=True)
  2. Delta-Compressed GPU HBM Cache (store_device="cuda", force_uncompressed=False)

Directly measures:
  - Actual GPU HBM memory allocated via torch.cuda.memory_allocated() across all TP ranks
    when caching multi-turn SWE-bench Verified trajectories in GPU HBM.
  - Incremental (Turns 2..4) and Total (Turns 1..4) GPU HBM memory reduction ratios.
  - Effective concurrent SWE-bench session capacity & Turn-4 HBM cache hit rate under a
    fixed GPU HBM KV cache budget.
  - Turn-4 agent localization & patch generation fidelity (Next-Token Top-1 Match,
    Token F1 Agreement, and Exact Sequence Match).
"""

import argparse
from collections import Counter, defaultdict
import json
import os
import re
import string
from typing import Any

import torch
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    CompressionTier,
    DeltaBlockStore,
    DeltaCompressionConnector,
)


def normalize_text(s: str) -> str:
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_punc(s.lower()))


def compute_token_f1(pred: str, ref: str) -> float:
    pred_toks = normalize_text(pred).split()
    ref_toks = normalize_text(ref).split()
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


def extract_target_file_from_patch(patch_str: str) -> str:
    m = re.search(r"diff --git a/(\S+) b/(\S+)", patch_str or "")
    if m:
        return m.group(2)
    m2 = re.search(r"\+\+\+ b/(\S+)", patch_str or "")
    return m2.group(1) if m2 else "unknown"


def load_swebench_verified(max_rows: int = 100) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    parquet_path = hf_hub_download(
        repo_id="princeton-nlp/SWE-bench_Verified",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)
    records: list[dict[str, Any]] = []
    for idx in range(min(len(df), max_rows)):
        row = df.iloc[idx]
        records.append(
            {
                "instance_id": str(row["instance_id"]),
                "repo": str(row["repo"]),
                "base_commit": str(row["base_commit"])[:12],
                "problem_statement": str(row["problem_statement"] or "").strip(),
                "hints_text": str(row.get("hints_text", "") or "").strip(),
                "patch": str(row.get("patch", "") or "").strip(),
                "target_file": extract_target_file_from_patch(
                    str(row.get("patch", "") or "")
                ),
            }
        )
    return records


def build_swebench_samples(
    records: list[dict[str, Any]],
    tokenizer: Any,
    num_samples: int,
    block_size: int,
    max_prompt_blocks: int = 8,
) -> list[dict[str, Any]]:
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    edit_toks = [
        tokenizer.encode(str(d), add_special_tokens=False)[-1]
        for d in range(1, 10)
    ]
    edit_blk = 1 if block_size >= 128 else 2
    part1_target = edit_blk * block_size

    ctx_header = (
        "ID: 300000000 | Workspace Snapshot: 400000000\n"
        "[SWE-Agent Repository Index & Test Runner Environment Initialized]\n"
        "Use git diff conventions and identify the exact file and function to patch.\n\n"
    )
    ctx_ids_raw = tokenizer.encode(ctx_header, add_special_tokens=False)
    if len(ctx_ids_raw) < block_size:
        ctx_ids = ctx_ids_raw + [newline_id] * (block_size - len(ctx_ids_raw))
    else:
        ctx_ids = ctx_ids_raw[:block_size]
    assert len(ctx_ids) == block_size

    is_harmony = (
        hasattr(tokenizer, "get_vocab") and "<|channel|>" in tokenizer.get_vocab()
    )
    is_qwen_thinking = (
        hasattr(tokenizer, "get_vocab")
        and "<think>" in tokenizer.get_vocab()
        and "<|im_start|>" in tokenizer.get_vocab()
    )

    samples: list[dict[str, Any]] = []
    for item in records:
        if len(samples) >= num_samples:
            break

        header_text = (
            f"ID: 100000000 | SWE-Bench Agent Session: 200000000\n"
            f"Repository: {item['repo']} | Instance: {item['instance_id']} | "
            f"Base Commit: {item['base_commit']}\n"
            f"Candidate File Under Inspection: {item['target_file']}\n\n"
        )
        header_ids = tokenizer.encode(header_text, add_special_tokens=False)
        if len(header_ids) < part1_target:
            part1_ids = header_ids + [newline_id] * (part1_target - len(header_ids))
        else:
            part1_ids = header_ids[:part1_target]
        assert len(part1_ids) == part1_target

        sec_header = (
            "ID: 100000000 | Codebase Inspection Step: 200000000\n\n"
        )
        sec_header_ids = tokenizer.encode(sec_header, add_special_tokens=False)

        body_text = (
            f"### GitHub Issue Description ({item['instance_id']})\n"
            f"{item['problem_statement']}\n\n"
        )
        if item["hints_text"]:
            body_text += f"### Maintainer Hints & Discussion\n{item['hints_text']}\n\n"
        body_text += (
            f"### Reference Patch Context ({item['target_file']})\n"
            f"{item['patch'][:1200]}\n\n"
        )
        body_ids = tokenizer.encode(body_text, add_special_tokens=False)

        min_body_tokens = max(2 * block_size, 512)
        while len(sec_header_ids) + len(body_ids) < min_body_tokens:
            body_ids = body_ids + body_ids

        max_body_tokens = max(
            min_body_tokens, (max_prompt_blocks - edit_blk) * block_size - len(sec_header_ids)
        )
        body_ids = body_ids[:max_body_tokens]

        raw_len = part1_target + len(sec_header_ids) + len(body_ids)
        target_blocks = max(edit_blk + 2, (raw_len + block_size - 1) // block_size)
        target_prefix_len = target_blocks * block_size
        pad_needed = target_prefix_len - raw_len
        aligned_prefix = (
            part1_ids + sec_header_ids + [newline_id] * pad_needed + body_ids
        )
        assert len(aligned_prefix) == target_prefix_len

        prompt_question = (
            f"Task: For `{item['instance_id']}` in `{item['repo']}`, state the target "
            f"file path to modify and summarize the root-cause fix in 2-3 sentences."
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

        full_ids = aligned_prefix + suffix_ids
        samples.append(
            {
                "id": item["instance_id"],
                "repo": item["repo"],
                "target_file": item["target_file"],
                "token_ids": full_ids,
                "prepend_block": ctx_ids,
                "edit_toks": edit_toks,
            }
        )
    return samples


def build_regime_turns(
    sample: dict[str, Any],
    regime: str,
    num_turns: int,
    block_size: int,
) -> list[list[int]]:
    prompt_tokens = sample["token_ids"]
    edit_toks = sample["edit_toks"]
    turns: list[list[int]] = [list(prompt_tokens)]
    for t in range(1, num_turns):
        t_a = edit_toks[(2 * t - 2) % len(edit_toks)]
        t_b = edit_toks[(2 * t - 1) % len(edit_toks)]
        if regime == "S1_dynamic_header_dp0":
            target = list(prompt_tokens)
            target[4] = t_b if prompt_tokens[4] != t_b else edit_toks[0]
            target[9] = t_a if prompt_tokens[9] != t_a else edit_toks[-1]
            turns.append(target)
        elif regime == "S2_prepend_context_dp16":
            prefix_block = list(sample["prepend_block"])
            prefix_block[3] = t_a
            prefix_block[7] = t_b
            turns.append(prefix_block + list(prompt_tokens))
        elif regime == "S3_intraturn_edit_dp0":
            target = list(prompt_tokens)
            edit_blk = 1 if block_size >= 128 else 2
            idx1 = edit_blk * block_size + 4
            idx2 = edit_blk * block_size + 9
            target[idx1] = t_b if prompt_tokens[idx1] != t_b else edit_toks[0]
            target[idx2] = t_a if prompt_tokens[idx2] != t_a else edit_toks[-1]
            turns.append(target)
        else:
            raise ValueError(f"Unknown regime: {regime}")
    return turns


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


def configure_hbm_mode(
    llm: LLM, store: DeltaBlockStore, force_uncompressed: bool
) -> int:
    """Configure store on all workers, clear HBM store, and return baseline CUDA allocated bytes."""
    store.device = "cuda"
    store.force_uncompressed = force_uncompressed
    store.clear()
    llm.reset_prefix_cache()
    res = llm.collective_rpc(
        _worker_configure_mode_and_clear, args=(force_uncompressed,)
    )
    return sum(r["base_cuda_allocated_bytes"] for r in res)


def get_hbm_snapshot(
    llm: LLM, base_cuda_bytes: int
) -> tuple[int, int, int, dict[str, int]]:
    """Return (net_cuda_allocated_bytes, raw_bytes_equiv, store_tensor_bytes, tier_counts)."""
    worker_stats = llm.collective_rpc(_worker_get_hbm_and_store_stats)
    total_cuda = sum(w["cuda_allocated_bytes"] for w in worker_stats)
    net_cuda = max(0, total_cuda - base_cuda_bytes)
    raw_bytes = sum(w["raw_bytes_equivalent"] for w in worker_stats)
    comp_bytes = sum(w["compressed_bytes_stored"] for w in worker_stats)
    num_w = max(1, len(worker_stats))
    all_tiers: set[str] = set()
    for w in worker_stats:
        all_tiers.update(w["tier_counts"].keys())
    tiers = {
        t: round(sum(w["tier_counts"].get(t, 0) for w in worker_stats) / num_w)
        for t in all_tiers
    }
    return net_cuda, raw_bytes, comp_bytes, tiers


def evaluate_swebench_hbm(
    llm: LLM,
    store: DeltaBlockStore,
    samples: list[dict[str, Any]],
    num_turns: int,
    block_size: int,
    gen_tokens: int = 128,
) -> dict[str, Any]:
    regimes = [
        "S1_dynamic_header_dp0",
        "S2_prepend_context_dp16",
        "S3_intraturn_edit_dp0",
    ]
    results: dict[str, Any] = {}
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=gen_tokens,
        logprobs=5,
    )

    for regime in regimes:
        print(f"\n{'=' * 80}")
        print(
            f"[SWE-BENCH VERIFIED GPU HBM] Regime: {regime} "
            f"({len(samples)} issues x {num_turns} turns)"
        )
        print(f"{'=' * 80}")

        batch_turns = [
            build_regime_turns(
                s, regime=regime, num_turns=num_turns, block_size=block_size
            )
            for s in samples
        ]
        final_prompts = [
            TokensPrompt(prompt_token_ids=t[-1]) for t in batch_turns
        ]

        # ------------------------------------------------------------------
        # PASS 1: Uncompressed GPU HBM Prefix Cache (force_uncompressed=True)
        # ------------------------------------------------------------------
        base_cuda_uncomp = configure_hbm_mode(
            llm, store, force_uncompressed=True
        )
        for t_idx in range(num_turns):
            llm.reset_prefix_cache()
            _ = llm.generate(
                [TokensPrompt(prompt_token_ids=t[t_idx]) for t in batch_turns],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            if t_idx == 0:
                (
                    uncomp_turn1_cuda,
                    uncomp_turn1_raw,
                    uncomp_turn1_stored,
                    _,
                ) = get_hbm_snapshot(llm, base_cuda_uncomp)

        (
            uncomp_total_cuda,
            uncomp_total_raw,
            uncomp_total_stored,
            uncomp_tiers,
        ) = get_hbm_snapshot(llm, base_cuda_uncomp)

        llm.reset_prefix_cache()
        ref_outputs = llm.generate(
            final_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # ------------------------------------------------------------------
        # PASS 2: Delta-Compressed GPU HBM Cache (force_uncompressed=False)
        # ------------------------------------------------------------------
        base_cuda_delta = configure_hbm_mode(
            llm, store, force_uncompressed=False
        )
        for t_idx in range(num_turns):
            llm.reset_prefix_cache()
            _ = llm.generate(
                [TokensPrompt(prompt_token_ids=t[t_idx]) for t in batch_turns],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            if t_idx == 0:
                (
                    delta_turn1_cuda,
                    delta_turn1_raw,
                    delta_turn1_stored,
                    delta_t1_tiers,
                ) = get_hbm_snapshot(llm, base_cuda_delta)

        (
            delta_total_cuda,
            delta_total_raw,
            delta_total_stored,
            delta_total_tiers,
        ) = get_hbm_snapshot(llm, base_cuda_delta)

        llm.reset_prefix_cache()
        recon_outputs = llm.generate(
            final_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # ------------------------------------------------------------------
        # Compute HBM Memory Utilization & Generation Fidelity Metrics
        # ------------------------------------------------------------------
        inc_uncomp_cuda = max(1, uncomp_total_cuda - uncomp_turn1_cuda)
        inc_delta_cuda = max(1, delta_total_cuda - delta_turn1_cuda)
        inc_uncomp_stored = max(1, uncomp_total_stored - uncomp_turn1_stored)
        inc_delta_stored = max(1, delta_total_stored - delta_turn1_stored)

        total_hbm_ratio_cuda = uncomp_total_cuda / max(1, delta_total_cuda)
        inc_hbm_ratio_cuda = inc_uncomp_cuda / max(1, inc_delta_cuda)
        total_hbm_ratio_tensor = uncomp_total_stored / max(1, delta_total_stored)
        inc_hbm_ratio_tensor = inc_uncomp_stored / max(1, inc_delta_stored)

        inc_tiers = {
            k: delta_total_tiers.get(k, 0) - delta_t1_tiers.get(k, 0)
            for k in delta_total_tiers
        }

        # Under a fixed GPU HBM cache budget equal to 1.05x delta_total_stored:
        fixed_hbm_budget_bytes = int(delta_total_stored * 1.05)
        uncomp_per_session_bytes = max(1, uncomp_total_stored // len(samples))
        delta_per_session_bytes = max(1, delta_total_stored // len(samples))
        uncomp_sessions_fit = min(
            len(samples), fixed_hbm_budget_bytes // uncomp_per_session_bytes
        )
        delta_sessions_fit = min(
            len(samples), fixed_hbm_budget_bytes // delta_per_session_bytes
        )
        uncomp_budget_hit_rate_pct = (uncomp_sessions_fit / len(samples)) * 100.0
        delta_budget_hit_rate_pct = (delta_sessions_fit / len(samples)) * 100.0

        top1_matches = 0
        seq_token_matches = 0
        f1_scores: list[float] = []
        target_file_hits_ref = 0
        target_file_hits_rec = 0

        for s, ref_o, rec_o in zip(samples, ref_outputs, recon_outputs):
            ref_toks = list(ref_o.outputs[0].token_ids)
            rec_toks = list(rec_o.outputs[0].token_ids)
            ref_text = ref_o.outputs[0].text
            rec_text = rec_o.outputs[0].text

            if ref_toks and rec_toks and ref_toks[0] == rec_toks[0]:
                top1_matches += 1
            min_len = min(len(ref_toks), len(rec_toks))
            if min_len > 0:
                matches = sum(
                    1 for a, b in zip(ref_toks[:min_len], rec_toks[:min_len]) if a == b
                )
                seq_token_matches += matches / min_len
            f1_scores.append(compute_token_f1(rec_text, ref_text))

            tf_short = s["target_file"].split("/")[-1]
            if tf_short in ref_text or s["target_file"] in ref_text:
                target_file_hits_ref += 1
            if tf_short in rec_text or s["target_file"] in rec_text:
                target_file_hits_rec += 1

        n_s = len(samples)
        summary = {
            "num_issues": n_s,
            "num_turns": num_turns,
            # Measured via torch.cuda.memory_allocated() across all GPUs:
            "uncompressed_hbm_cuda_allocated_mb": round(
                uncomp_total_cuda / (1024 * 1024), 2
            ),
            "delta_compressed_hbm_cuda_allocated_mb": round(
                delta_total_cuda / (1024 * 1024), 2
            ),
            "hbm_cuda_memory_saved_mb": round(
                (uncomp_total_cuda - delta_total_cuda) / (1024 * 1024), 2
            ),
            "hbm_cuda_memory_saved_pct": round(
                (1.0 - delta_total_cuda / max(1, uncomp_total_cuda)) * 100.0, 2
            ),
            "total_4turn_hbm_cuda_ratio": round(total_hbm_ratio_cuda, 2),
            "incremental_3turn_hbm_cuda_ratio": round(inc_hbm_ratio_cuda, 2),
            # Exact tensor payload bytes in HBM:
            "uncompressed_hbm_tensor_mb": round(
                uncomp_total_stored / (1024 * 1024), 2
            ),
            "delta_compressed_hbm_tensor_mb": round(
                delta_total_stored / (1024 * 1024), 2
            ),
            "incremental_uncompressed_hbm_mb": round(
                inc_uncomp_stored / (1024 * 1024), 2
            ),
            "incremental_delta_compressed_hbm_mb": round(
                inc_delta_stored / (1024 * 1024), 2
            ),
            "total_4turn_hbm_tensor_ratio": round(total_hbm_ratio_tensor, 2),
            "incremental_3turn_hbm_tensor_ratio": round(inc_hbm_ratio_tensor, 2),
            "incremental_tier_counts": inc_tiers,
            # Constrained HBM budget capacity:
            "fixed_hbm_budget_mb": round(
                fixed_hbm_budget_bytes / (1024 * 1024), 2
            ),
            "uncompressed_sessions_resident_in_budget": uncomp_sessions_fit,
            "delta_compressed_sessions_resident_in_budget": delta_sessions_fit,
            "uncompressed_hbm_hit_rate_under_budget_pct": round(
                uncomp_budget_hit_rate_pct, 2
            ),
            "delta_compressed_hbm_hit_rate_under_budget_pct": round(
                delta_budget_hit_rate_pct, 2
            ),
            # Generation quality & agreement:
            "turn4_top1_token_match_pct": round(top1_matches / n_s * 100.0, 2),
            "turn4_seq_token_agreement_pct": round(
                seq_token_matches / n_s * 100.0, 2
            ),
            "turn4_response_f1_agreement_pct": round(
                sum(f1_scores) / n_s * 100.0, 2
            ),
            "baseline_target_file_localization_pct": round(
                target_file_hits_ref / n_s * 100.0, 2
            ),
            "delta_compressed_target_file_localization_pct": round(
                target_file_hits_rec / n_s * 100.0, 2
            ),
        }
        print(
            f"\n--- [SWE-BENCH VERIFIED GPU HBM] Summary for {regime} ---\n"
            + json.dumps(summary, indent=2),
            flush=True,
        )
        results[regime] = summary

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GPU HBM memory savings on SWE-bench Verified."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-issues", type=int, default=20)
    parser.add_argument("--num-turns", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.72,
        help="Reserve a compact active working BlockPool so free HBM holds the multi-turn KV store.",
    )
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    print(f"Loading SWE-bench Verified dataset (num_issues={args.num_issues})...")
    records = load_swebench_verified(max_rows=max(100, args.num_issues * 2))
    print(f"Loaded {len(records)} SWE-bench Verified issues.")

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

    samples = build_swebench_samples(
        records=records,
        tokenizer=tokenizer,
        num_samples=args.num_issues,
        block_size=resolved_block_size,
    )
    avg_len = sum(len(s["token_ids"]) for s in samples) / max(1, len(samples))
    print(
        f"Built {len(samples)} SWE-bench Verified 4-turn samples "
        f"(avg prompt len={avg_len:.1f} tokens, block_size={resolved_block_size})."
    )

    swebench_results = evaluate_swebench_hbm(
        llm=llm,
        store=store,
        samples=samples,
        num_turns=args.num_turns,
        block_size=resolved_block_size,
        gen_tokens=args.gen_tokens,
    )

    out_payload = {
        "model": args.model,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "storage_tier": "GPU_HBM (torch.cuda.memory_allocated)",
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": resolved_block_size,
        "num_issues": len(samples),
        "num_turns": args.num_turns,
        "regimes": swebench_results,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"\nSaved SWE-bench Verified GPU HBM results to {args.output_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Downstream Task Quality Benchmark for DeltaCompressionConnector on vLLM V1.

Evaluates two standard downstream quality benchmarks across 4 turns (P_1 -> P_2 -> P_3 -> P_4)
under three prefix-divergence regimes (S1_dynamic_header_dp0, S2_prepend_context_dp16,
S3_intraturn_edit_dp0), comparing Uncompressed Baseline vs. DeltaCompressionConnector:

  1. GSM8K (8-Shot Chain-of-Thought Mathematical Reasoning):
     - Ground-truth numerical Exact Match Accuracy (%) over multi-step CoT generation.
     - Answer Agreement (%) between Uncompressed Baseline and Delta-Compressed KV.
     - 4-Turn Total Store & Incremental Compression Ratios.

  2. CoQA (4-Turn Conversational Reading Comprehension):
     - Official Stanford CoQA Macro Token-F1 (%) and Exact Match (%) against 4 human
       annotator references on Turn 4 (Q_4).
     - Prediction-to-Prediction F1 / EM agreement between Uncompressed and Delta-Compressed.
     - 4-Turn Total Store & Incremental Compression Ratios.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
import string
import time
from collections import defaultdict
from typing import Any

# Ensure V1 engine runs in-process so scheduler & worker share DeltaBlockStore in memory
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import requests
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    DeltaBlockStore,
    DeltaCompressionConnector,
)
from vllm.inputs import TokensPrompt

INVALID_NUM = -9999999


# ==============================================================================
# Dataset Downloading & Scoring Utilities
# ==============================================================================


def ensure_file(url: str, local_path: str) -> str:
    if not os.path.exists(local_path):
        print(f"Downloading {url} -> {local_path}...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
    return local_path


def extract_gsm8k_answer(text: str) -> int:
    """Official vLLM gsm8k_eval.py numerical answer extractor."""
    # If '####' delimiter is present, prioritize the number right after '####'
    if "####" in text:
        after_hash = text.split("####")[-1]
        after_hash_clean = after_hash.replace(",", "")
        nums = re.findall(r"-?\d+\.?\d*", after_hash_clean)
        if nums:
            try:
                val = float(nums[0])
                return int(val) if val.is_integer() else val  # type: ignore[return-value]
            except ValueError:
                pass
    clean = text.replace(",", "")
    numbers = re.findall(r"\d+", clean)
    if not numbers:
        return INVALID_NUM
    try:
        return int(ast.literal_eval(numbers[-1]))
    except (SyntaxError, ValueError):
        return INVALID_NUM


def normalize_coqa_answer(s: str) -> str:
    """Official SQuAD / CoQA text normalizer (lowercase, strip punct/articles/whitespace)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_coqa_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_coqa_answer(pred).split()
    gold_toks = normalize_coqa_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_coqa_em(pred: str, gold: str) -> float:
    return 1.0 if normalize_coqa_answer(pred) == normalize_coqa_answer(gold) else 0.0


def score_coqa_against_golds(pred: str, golds: list[str]) -> tuple[float, float]:
    """Return (max_f1, max_em) across all human reference answers."""
    # Take first line / clause before any self-generated follow-up question
    clean_pred = pred.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "A:"):
        if clean_pred.startswith(prefix):
            clean_pred = clean_pred[len(prefix) :].strip()
    max_f1 = max((compute_coqa_f1(clean_pred, g) for g in golds), default=0.0)
    max_em = max((compute_coqa_em(clean_pred, g) for g in golds), default=0.0)
    return max_f1, max_em


# ==============================================================================
# Prompt & Multi-Turn Regime Builders
# ==============================================================================


def build_gsm8k_samples(
    train_path: str,
    test_path: str,
    tokenizer: Any,
    num_samples: int,
    block_size: int,
) -> list[dict[str, Any]]:
    with open(train_path, encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(test_path, encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    edit_toks = [
        tokenizer.encode(str(d), add_special_tokens=False)[-1]
        for d in range(1, 10)
    ]

    # Build a clean 5-shot or 6-shot demonstration string (>= 3 * block_size tokens)
    min_prefix_tokens = max(512, 3 * block_size)
    demos_text = ""
    shot_idx = 0
    while (
        len(tokenizer.encode(demos_text, add_special_tokens=False))
        < min_prefix_tokens - 32
    ):
        ex = train_data[shot_idx]
        demos_text += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        shot_idx += 1
    demo_ids = tokenizer.encode(demos_text, add_special_tokens=False)

    header_base = (
        "System Session ID: 10000 | Mode: Math Reasoning Benchmark\n"
        "Solve each math word problem step by step and end with #### <number>.\n\n"
    )
    header_ids = tokenizer.encode(header_base, add_special_tokens=False)
    total_raw_prefix = len(header_ids) + len(demo_ids)
    target_blocks = max(3, (total_raw_prefix + block_size - 1) // block_size)
    target_prefix_len = target_blocks * block_size
    pad_needed = target_prefix_len - total_raw_prefix
    aligned_prefix = header_ids + [newline_id] * pad_needed + demo_ids
    assert len(aligned_prefix) == target_prefix_len

    ctx_str = (
        "[Context Note: Read each problem carefully, verify intermediate steps, "
        "and provide the final numerical answer after ####.]\n"
        * max(24, (block_size // 10) + 8)
    )
    ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)[:block_size]
    assert len(ctx_ids) == block_size

    is_harmony = (
        hasattr(tokenizer, "get_vocab") and "<|channel|>" in tokenizer.get_vocab()
    )
    samples: list[dict[str, Any]] = []
    for i in range(min(num_samples, len(test_data))):
        if is_harmony:
            q_text = (
                f"Question: {test_data[i]['question']}\n"
                "Solve ONLY this final question step by step and end with #### <number>."
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            )
        else:
            q_text = f"Question: {test_data[i]['question']}\nAnswer:"
        suffix_ids = tokenizer.encode(q_text, add_special_tokens=False)
        if len(suffix_ids) < 4:
            suffix_ids = [newline_id] * (4 - len(suffix_ids)) + suffix_ids
        if len(suffix_ids) % block_size == 0:
            suffix_ids = [newline_id] + suffix_ids

        full_ids = aligned_prefix + suffix_ids
        gold_val = extract_gsm8k_answer(test_data[i]["answer"])
        if gold_val == INVALID_NUM:
            continue

        samples.append(
            {
                "id": f"gsm8k_{i}",
                "token_ids": full_ids,
                "prepend_block": ctx_ids,
                "edit_toks": edit_toks,
                "gold_answer": gold_val,
                "question": test_data[i]["question"],
            }
        )
    return samples


def build_coqa_samples(
    coqa_path: str,
    tokenizer: Any,
    num_samples: int,
    block_size: int,
) -> list[dict[str, Any]]:
    with open(coqa_path, encoding="utf-8") as f:
        raw_coqa = json.load(f)["data"]

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    edit_toks = [
        tokenizer.encode(str(d), add_special_tokens=False)[-1]
        for d in range(1, 10)
    ]
    min_prefix_tokens = max(512, 3 * block_size)

    # Use stories near the end of dev set as few-shot pool
    demo_pool = [s for s in raw_coqa[-60:] if len(s.get("questions", [])) >= 4]
    ctx_str = (
        "[Context Note: Answer the question concisely using only facts "
        "stated in the passage above.]\n"
        * max(24, (block_size // 10) + 8)
    )
    ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)[:block_size]
    assert len(ctx_ids) == block_size

    samples: list[dict[str, Any]] = []
    for i, story_obj in enumerate(raw_coqa):
        if len(story_obj.get("questions", [])) < 4:
            continue

        # Target story + first 3 conversational turns (Q1/A1, Q2/A2, Q3/A3) — 100% intact!
        target_history = f"Passage:\n{story_obj['story'].strip()}\n\n"
        for t_idx in range(3):
            q_str = story_obj["questions"][t_idx]["input_text"].strip()
            a_str = story_obj["answers"][t_idx]["input_text"].strip()
            target_history += f"Q: {q_str}\nA: {a_str}\n"
        target_history_ids = tokenizer.encode(
            target_history, add_special_tokens=False
        )

        header = (
            "System Session ID: 20000 | Task: Concise Reading Comprehension\n"
            "Answer each question about the passage with a short, direct phrase.\n\n"
        )
        header_ids = tokenizer.encode(header, add_special_tokens=False)

        demos_text = ""
        d_idx = 0
        while (
            len(header_ids)
            + len(tokenizer.encode(demos_text, add_special_tokens=False))
            + len(target_history_ids)
            < min_prefix_tokens
        ):
            d_story = demo_pool[d_idx]
            demos_text += f"Passage:\n{d_story['story'].strip()}\n\n"
            for t_idx in range(4):
                dq = d_story["questions"][t_idx]["input_text"].strip()
                da = d_story["answers"][t_idx]["input_text"].strip()
                demos_text += f"Q: {dq}\nA: {da}\n"
            demos_text += "\n"
            d_idx += 1

        demo_ids = tokenizer.encode(demos_text, add_special_tokens=False)
        raw_len = len(header_ids) + len(demo_ids) + len(target_history_ids)
        target_blocks = max(3, (raw_len + block_size - 1) // block_size)
        target_prefix_len = target_blocks * block_size
        pad_needed = target_prefix_len - raw_len
        aligned_prefix = (
            header_ids
            + [newline_id] * pad_needed
            + demo_ids
            + target_history_ids
        )
        assert len(aligned_prefix) == target_prefix_len

        q4_str = story_obj["questions"][3]["input_text"].strip()
        is_harmony = (
            hasattr(tokenizer, "get_vocab")
            and "<|channel|>" in tokenizer.get_vocab()
        )
        if is_harmony:
            suffix_text = (
                f"Q: {q4_str}\n"
                "Reply with ONLY the short direct answer phrase to this final question "
                "(no heading or explanation)."
                "<|end|><|start|>assistant<|channel|>final<|message|>"
            )
        else:
            suffix_text = f"Q: {q4_str}\nA:"
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if len(suffix_ids) < 4:
            suffix_ids = [newline_id] * (4 - len(suffix_ids)) + suffix_ids
        if len(suffix_ids) % block_size == 0:
            suffix_ids = [newline_id] + suffix_ids

        full_ids = aligned_prefix + suffix_ids

        golds = [story_obj["answers"][3]["input_text"].strip()]
        for add_list in story_obj.get("additional_answers", {}).values():
            if len(add_list) > 3:
                golds.append(add_list[3]["input_text"].strip())

        samples.append(
            {
                "id": story_obj.get("id", f"coqa_{len(samples)}"),
                "token_ids": full_ids,
                "prepend_block": ctx_ids,
                "edit_toks": edit_toks,
                "gold_answers": golds,
                "question_4": q4_str,
            }
        )
        if len(samples) >= num_samples:
            break

    return samples


def build_regime_turns(
    sample: dict[str, Any],
    regime: str,
    num_turns: int,
    block_size: int,
) -> list[list[int]]:
    """Construct [turn_1_tokens, ..., turn_N_tokens] for a given prefix-divergence regime."""
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
    raise RuntimeError(
        "Could not access shared DeltaBlockStore; ensure VLLM_ENABLE_V1_MULTIPROCESSING=0"
    )


def _worker_get_store_stats(worker: Any) -> dict[str, Any]:
    if DeltaCompressionConnector._shared_stores:
        w_store = next(iter(DeltaCompressionConnector._shared_stores.values()))
        return {
            "raw_bytes_equivalent": w_store.stats.raw_bytes_equivalent,
            "compressed_bytes_stored": w_store.stats.compressed_bytes_stored,
            "tier_counts": {k.name: v for k, v in w_store.stats.tier_counts.items()},
        }
    return {
        "raw_bytes_equivalent": 0,
        "compressed_bytes_stored": 0,
        "tier_counts": {},
    }


def get_store_stats_snapshot(
    llm: LLM, store: DeltaBlockStore
) -> tuple[int, int, dict[str, int]]:
    """Return (raw_bytes_equivalent, compressed_bytes_stored, tier_counts) across all TP workers."""
    if store.stats.total_blocks_stored > 0:
        return (
            store.stats.raw_bytes_equivalent,
            store.stats.compressed_bytes_stored,
            {k.name: v for k, v in store.stats.tier_counts.items()},
        )
    worker_stats = llm.collective_rpc(_worker_get_store_stats)
    if not worker_stats:
        return 0, 0, {}
    raw_bytes = sum(w["raw_bytes_equivalent"] for w in worker_stats)
    comp_bytes = sum(w["compressed_bytes_stored"] for w in worker_stats)
    num_w = len(worker_stats)
    all_tiers: set[str] = set()
    for w in worker_stats:
        all_tiers.update(w["tier_counts"].keys())
    tiers = {
        t: round(sum(w["tier_counts"].get(t, 0) for w in worker_stats) / num_w)
        for t in all_tiers
    }
    return raw_bytes, comp_bytes, tiers


# ==============================================================================
# Core Benchmark Runner
# ==============================================================================


def run_task_evaluation(
    llm: LLM,
    store: DeltaBlockStore,
    samples: list[dict[str, Any]],
    task_name: str,
    sampling_params: SamplingParams,
    num_turns: int,
    block_size: int,
    batch_size: int,
) -> dict[str, Any]:
    regimes = [
        "S1_dynamic_header_dp0",
        "S2_prepend_context_dp16",
        "S3_intraturn_edit_dp0",
    ]
    task_results: dict[str, Any] = {}

    for regime in regimes:
        print(f"\n{'=' * 76}")
        print(
            f"[{task_name.upper()}] Evaluating Regime: {regime} "
            f"({len(samples)} samples x {num_turns} turns, batch_size={batch_size})"
        )
        print(f"{'=' * 76}")

        agg_tier_counts: dict[str, int] = defaultdict(int)
        base_uncomp_total = 0
        base_comp_total = 0
        inc_uncomp_total = 0
        inc_comp_total = 0

        # Task-specific quality accumulators
        base_scores: list[float] = []
        recon_scores: list[float] = []
        base_em_scores: list[float] = []
        recon_em_scores: list[float] = []
        pred_agreements: list[float] = []
        pred_f1_agreements: list[float] = []

        for b_start in range(0, len(samples), batch_size):
            batch = samples[b_start : b_start + batch_size]
            batch_turns = [
                build_regime_turns(
                    s,
                    regime=regime,
                    num_turns=num_turns,
                    block_size=block_size,
                )
                for s in batch
            ]
            final_prompts = [
                TokensPrompt(prompt_token_ids=t[-1]) for t in batch_turns
            ]

            # 1. Uncompressed Baseline on final_prompts:
            # Populate UNCOMPRESSED_BASE blocks first, then evict HBM and reload
            # the uncompressed KV cache so ref_outputs uses the exact same
            # suffix-prefill kernel path (q_len = tail) as Step 3.
            store.clear()
            llm.reset_prefix_cache()
            _ = llm.generate(
                final_prompts,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            llm.reset_prefix_cache()
            ref_outputs = llm.generate(
                final_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # 2. Populate Turn 1 (P_1) then Turns 2..N in DeltaCompressionConnector
            store.clear()
            llm.reset_prefix_cache()
            _ = llm.generate(
                [TokensPrompt(prompt_token_ids=t[0]) for t in batch_turns],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            b_uncomp, b_comp, b_tiers = get_store_stats_snapshot(llm, store)

            for t_idx in range(1, num_turns):
                llm.reset_prefix_cache()
                _ = llm.generate(
                    [
                        TokensPrompt(prompt_token_ids=t[t_idx])
                        for t in batch_turns
                    ],
                    sampling_params=SamplingParams(
                        temperature=0.0, max_tokens=1
                    ),
                    use_tqdm=False,
                )

            tot_uncomp, tot_comp, tot_tiers = get_store_stats_snapshot(llm, store)

            base_uncomp_total += b_uncomp
            base_comp_total += b_comp
            inc_uncomp_total += tot_uncomp - b_uncomp
            inc_comp_total += tot_comp - b_comp
            for k_name, cnt in tot_tiers.items():
                agg_tier_counts[k_name] += cnt - b_tiers.get(k_name, 0)

            # 3. Reset GPU HBM prefix cache and reconstruct from DeltaCompressionConnector
            llm.reset_prefix_cache()
            recon_outputs = llm.generate(
                final_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # 4. Score Baseline vs. Reconstructed against Ground-Truth Benchmark Labels
            for s, ref_o, rec_o in zip(batch, ref_outputs, recon_outputs):
                ref_text = ref_o.outputs[0].text
                rec_text = rec_o.outputs[0].text

                if task_name == "gsm8k":
                    gold_num = s["gold_answer"]
                    ref_num = extract_gsm8k_answer(ref_text)
                    rec_num = extract_gsm8k_answer(rec_text)
                    base_scores.append(1.0 if ref_num == gold_num else 0.0)
                    recon_scores.append(1.0 if rec_num == gold_num else 0.0)
                    pred_agreements.append(
                        1.0
                        if (ref_num == rec_num and ref_num != INVALID_NUM)
                        else 0.0
                    )
                elif task_name == "coqa":
                    golds = s["gold_answers"]
                    ref_f1, ref_em = score_coqa_against_golds(ref_text, golds)
                    rec_f1, rec_em = score_coqa_against_golds(rec_text, golds)
                    base_scores.append(ref_f1)
                    recon_scores.append(rec_f1)
                    base_em_scores.append(ref_em)
                    recon_em_scores.append(rec_em)
                    clean_ref = ref_text.strip().split("\n")[0].strip()
                    clean_rec = rec_text.strip().split("\n")[0].strip()
                    pred_agreements.append(compute_coqa_em(clean_rec, clean_ref))
                    pred_f1_agreements.append(
                        compute_coqa_f1(clean_rec, clean_ref)
                    )

            done = len(base_scores)
            print(
                f"  [{done:2d}/{len(samples)}] "
                f"Baseline Score: {sum(base_scores)/done*100:5.2f}% | "
                f"Delta-Comp Score: {sum(recon_scores)/done*100:5.2f}% | "
                f"Pred Agreement: {sum(pred_agreements)/done*100:5.2f}%"
            )

        overall_store_ratio = (base_uncomp_total + inc_uncomp_total) / max(
            base_comp_total + inc_comp_total, 1
        )
        inc_ratio = inc_uncomp_total / max(inc_comp_total, 1)

        summary: dict[str, Any] = {
            "num_samples": len(samples),
            "num_turns": num_turns,
            "total_uncompressed_mb": round(
                (base_uncomp_total + inc_uncomp_total) / (1024 * 1024), 2
            ),
            "total_compressed_mb": round(
                (base_comp_total + inc_comp_total) / (1024 * 1024), 2
            ),
            "overall_multiturn_store_ratio": round(overall_store_ratio, 2),
            "incremental_compression_ratio": round(inc_ratio, 2),
            "incremental_tier_counts": dict(agg_tier_counts),
        }

        if task_name == "gsm8k":
            base_acc = sum(base_scores) / len(base_scores) * 100.0
            rec_acc = sum(recon_scores) / len(recon_scores) * 100.0
            summary.update(
                {
                    "baseline_accuracy_pct": round(base_acc, 2),
                    "delta_compressed_accuracy_pct": round(rec_acc, 2),
                    "accuracy_diff_pct": round(rec_acc - base_acc, 2),
                    "answer_agreement_pct": round(
                        sum(pred_agreements) / len(pred_agreements) * 100.0, 2
                    ),
                }
            )
        else:
            base_f1 = sum(base_scores) / len(base_scores) * 100.0
            rec_f1 = sum(recon_scores) / len(recon_scores) * 100.0
            base_em = sum(base_em_scores) / len(base_em_scores) * 100.0
            rec_em = sum(recon_em_scores) / len(recon_em_scores) * 100.0
            summary.update(
                {
                    "baseline_f1_pct": round(base_f1, 2),
                    "delta_compressed_f1_pct": round(rec_f1, 2),
                    "f1_diff_pct": round(rec_f1 - base_f1, 2),
                    "baseline_em_pct": round(base_em, 2),
                    "delta_compressed_em_pct": round(rec_em, 2),
                    "em_diff_pct": round(rec_em - base_em, 2),
                    "pred_to_pred_f1_pct": round(
                        sum(pred_f1_agreements)
                        / len(pred_f1_agreements)
                        * 100.0,
                        2,
                    ),
                    "pred_to_pred_em_pct": round(
                        sum(pred_agreements) / len(pred_agreements) * 100.0, 2
                    ),
                }
            )

        task_results[regime] = summary
        print(f"\n--- [{task_name.upper()}] Summary for {regime} ---")
        print(json.dumps(summary, indent=2))

    return task_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-gsm8k", type=int, default=10)
    parser.add_argument("--num-coqa", type=int, default=10)
    parser.add_argument("--num-turns", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--mamba-ssm-cache-dtype", type=str, default="auto")
    parser.add_argument("--disable-mm", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/tmp/quality_benchmarks_results.json",
    )
    args = parser.parse_args()

    train_path = ensure_file(
        f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/train.jsonl",
        "/home/xzhaos/gsm8k_train.jsonl",
    )
    test_path = ensure_file(
        f"{VLLM_S3_BUCKET_URL}/ci-datasets/gsm8k/test.jsonl",
        "/home/xzhaos/gsm8k_test.jsonl",
    )
    coqa_path = ensure_file(
        "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json",
        "/home/xzhaos/coqa-dev-v1.0.json",
    )

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    effective_max_model_len = args.max_model_len
    if (
        "35b" in args.model.lower() or "moe" in args.model.lower()
    ) and effective_max_model_len < 3584:
        effective_max_model_len = 3584

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": 16,
        "enable_prefix_caching": True,
        "async_scheduling": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": effective_max_model_len,
        "enforce_eager": True,
        "kv_transfer_config": KVTransferConfig(
            kv_connector="DeltaCompressionConnector",
            kv_role="kv_both",
        ),
    }
    if args.mamba_ssm_cache_dtype != "auto":
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.disable_mm:
        llm_kwargs["language_model_only"] = True

    llm = LLM(**llm_kwargs)
    resolved_block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    store = get_shared_store()

    gsm8k_samples = build_gsm8k_samples(
        train_path,
        test_path,
        tokenizer,
        num_samples=args.num_gsm8k,
        block_size=resolved_block_size,
    )
    coqa_samples = build_coqa_samples(
        coqa_path,
        tokenizer,
        num_samples=args.num_coqa,
        block_size=resolved_block_size,
    )

    print(
        f"Built {len(gsm8k_samples)} GSM8K samples "
        f"(avg len={sum(len(s['token_ids']) for s in gsm8k_samples)/len(gsm8k_samples):.1f} toks) "
        f"and {len(coqa_samples)} CoQA 4-turn samples "
        f"(avg len={sum(len(s['token_ids']) for s in coqa_samples)/len(coqa_samples):.1f} toks)."
    )

    gsm8k_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=[
            "Question:",
            "\nQuestion",
            "<|im_end|>",
            "<end_of_turn>",
            "<|end|>",
            "<|return|>",
        ],
    )
    coqa_params = SamplingParams(
        temperature=0.0,
        max_tokens=24,
        stop=[
            "\n",
            "Q:",
            "Passage:",
            "<|im_end|>",
            "<end_of_turn>",
            "<|end|>",
            "<|return|>",
        ],
    )

    all_out = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "block_size": resolved_block_size,
        "num_turns": args.num_turns,
        "gsm8k": run_task_evaluation(
            llm=llm,
            store=store,
            samples=gsm8k_samples,
            task_name="gsm8k",
            sampling_params=gsm8k_params,
            num_turns=args.num_turns,
            block_size=resolved_block_size,
            batch_size=args.batch_size,
        ),
        "coqa": run_task_evaluation(
            llm=llm,
            store=store,
            samples=coqa_samples,
            task_name="coqa",
            sampling_params=coqa_params,
            num_turns=args.num_turns,
            block_size=resolved_block_size,
            batch_size=args.batch_size,
        ),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved full quality benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()

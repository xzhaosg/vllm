#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end ShareGPT GPU validation benchmark for DeltaCompressionConnector
on vLLM V1 with Google Gemma-3-1B-IT.

Evaluates three realistic multi-turn serving regimes across ShareGPT conversations:
  - Regime S1 (R_1a, delta_p = 0): Dynamic system/session metadata edit in Block 0
    followed by shared multi-turn ShareGPT conversation history.
  - Regime S2 (R_1b, delta_p = +16): Prepended 16-token memory/RAG context block
    shifting the entire ShareGPT conversation by +16 tokens (RoPE R(+16) + delta).
  - Regime S3 (R_2, delta_p = 0): Intra-conversation 2-token user turn edit in
    Block 2 followed by shared multi-turn conversation history.

For each regime, verifies both:
  1. Compression ratio & tier distribution across all 26 layers of Gemma-3-1B-IT.
  2. Generation fidelity after `llm.reset_prefix_cache()` forces vLLM V1 to
     reconstruct KV blocks from DeltaCompressionConnector into GPU HBM.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any

# Ensure V1 engine runs in-process so scheduler & worker share DeltaBlockStore in memory
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    DeltaBlockStore,
    DeltaCompressionConnector,
)
from vllm.inputs import TokensPrompt


def load_sharegpt_conversations(
    dataset_path: str,
    tokenizer: Any,
    num_samples: int = 50,
    min_tokens: int = 256,
    max_tokens: int = 512,
    block_size: int = 16,
) -> list[dict[str, Any]]:
    """Load and tokenize multi-turn ShareGPT conversations aligned to block boundaries."""
    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    selected: list[dict[str, Any]] = []
    for entry in raw_data:
        convs = entry.get("conversations", [])
        if len(convs) < 4:
            continue

        # Format multi-turn conversation
        formatted_turns = []
        valid = True
        for turn in convs:
            role = turn.get("from", "")
            val = turn.get("value", "").strip()
            if not val or not val.isascii():
                valid = False
                break
            speaker = "User" if role in ("human", "user") else "Assistant"
            formatted_turns.append(f"{speaker}: {val}")

        if not valid:
            continue

        full_text = "\n".join(formatted_turns) + "\nAssistant:"
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(token_ids) < min_tokens:
            continue

        # Truncate to a multiple of block_size + 4 trailing prompt tokens so that
        # all prefix blocks are full 16-token blocks cached by vLLM's block pool
        # and the final 4 tokens trigger next-token prefill/generation after KV load.
        usable_blocks = min(len(token_ids) // block_size, max_tokens // block_size)
        if usable_blocks < (min_tokens // block_size):
            continue

        target_len = (usable_blocks - 1) * block_size + 4
        trimmed_ids = token_ids[:target_len]

        selected.append(
            {
                "id": entry.get("id", f"conv_{len(selected)}"),
                "token_ids": trimmed_ids,
                "num_full_blocks": usable_blocks - 1,
                "num_turns": len(convs),
            }
        )
        if len(selected) >= num_samples:
            break

    return selected


def build_regime_pair(
    conv_tokens: list[int],
    regime: str,
    block_size: int = 16,
    vocab_ids: tuple[int, int] = (1050, 1051),
) -> tuple[list[int], list[int]]:
    """Construct (base_tokens, target_tokens) for a given ShareGPT regime."""
    t_a, t_b = vocab_ids
    if regime == "S1_dynamic_header_dp0":
        # Block 0 has a 2-token session/timestamp metadata edit (d_H = 2, delta_p = 0)
        base = list(conv_tokens)
        target = list(conv_tokens)
        target[4] = t_b if base[4] != t_b else t_a
        target[9] = t_b if base[9] != t_b else t_a
        return base, target

    elif regime == "S2_prepend_context_dp16":
        # Prepend a 16-token (1 full block) system/RAG context block before the conversation
        # shifting all subsequent blocks by delta_p = +16 tokens.
        prefix_block = [1000 + (i % 15) for i in range(block_size)]
        base = list(conv_tokens)
        target = prefix_block + list(conv_tokens)
        return base, target

    elif regime == "S3_intraturn_edit_dp0":
        # Edit 2 tokens inside an interior block representing an intra-turn edit
        base = list(conv_tokens)
        target = list(conv_tokens)
        edit_blk = 1 if block_size >= 128 else 2
        idx1 = edit_blk * block_size + 4
        idx2 = edit_blk * block_size + 9
        target[idx1] = t_b if base[idx1] != t_b else t_a
        target[idx2] = t_b if base[idx2] != t_b else t_a
        return base, target

    else:
        raise ValueError(f"Unknown regime: {regime}")


def get_shared_store() -> DeltaBlockStore:
    """Retrieve the shared DeltaBlockStore instance from the in-process V1 engine."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/xzhaos/gcs-home/hf_home/hf_cache/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/xzhaos/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--mamba-ssm-cache-dtype", type=str, default="auto")
    parser.add_argument("--disable-mm", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/tmp/sharegpt_l4_validation_results.json",
    )
    args = parser.parse_args()

    # Use V1 in-process mode so we can inspect connector stats directly in the same process
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Initializing vLLM V1 with DeltaCompressionConnector on NVIDIA L4 GPU...")
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "block_size": 16,
        "enable_prefix_caching": True,
        "async_scheduling": False,
        "gpu_memory_utilization": 0.6,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "kv_transfer_config": KVTransferConfig(
            kv_connector="DeltaCompressionConnector",
            kv_role="kv_both",
        ),
    }
    if args.mamba_ssm_cache_dtype != "auto":
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.disable_mm:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    llm = LLM(**llm_kwargs)
    resolved_block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
    min_tokens = max(args.min_tokens, 3 * resolved_block_size)
    max_tokens = max(args.max_tokens, 5 * resolved_block_size)

    print(
        f"Loading ShareGPT conversations from {args.dataset} "
        f"(resolved block_size={resolved_block_size}, min_tokens={min_tokens}, max_tokens={max_tokens})..."
    )
    conversations = load_sharegpt_conversations(
        args.dataset,
        tokenizer,
        num_samples=args.num_samples,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        block_size=resolved_block_size,
    )
    print(
        f"Selected {len(conversations)} multi-turn ShareGPT conversations "
        f"(avg tokens: {sum(len(c['token_ids']) for c in conversations) / len(conversations):.1f}, "
        f"avg full blocks: {sum(c['num_full_blocks'] for c in conversations) / len(conversations):.1f})."
    )

    store = get_shared_store()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.gen_tokens,
        logprobs=5,
    )

    regimes = [
        "S1_dynamic_header_dp0",
        "S2_prepend_context_dp16",
        "S3_intraturn_edit_dp0",
    ]

    all_results: dict[str, Any] = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "block_size": resolved_block_size,
        "num_conversations": len(conversations),
        "gen_tokens": args.gen_tokens,
        "regimes": {},
    }

    for regime in regimes:
        print(f"\n{'=' * 72}")
        print(f"Evaluating Regime: {regime} ({len(conversations)} conversations)")
        print(f"{'=' * 72}")

        top1_matches = 0
        top5_overlaps: list[float] = []
        seq_token_matches = 0
        total_seq_tokens = 0
        abs_logprob_diffs: list[float] = []

        agg_tier_counts: dict[str, int] = defaultdict(int)
        base_uncompressed_bytes_total = 0
        base_compressed_bytes_total = 0
        target_uncompressed_bytes_total = 0
        target_compressed_bytes_total = 0
        recon_times_ms: list[float] = []

        for idx, conv in enumerate(conversations):
            base_ids, target_ids = build_regime_pair(
                conv["token_ids"], regime=regime, block_size=resolved_block_size
            )

            # Step 1: Exact uncompressed ground-truth reference for target_ids
            store.clear()
            llm.reset_prefix_cache()
            ref_out = llm.generate(
                [TokensPrompt(prompt_token_ids=target_ids)],
                sampling_params=sampling_params,
                use_tqdm=False,
            )[0].outputs[0]
            ref_gen_ids = list(ref_out.token_ids)
            ref_first_logprobs = ref_out.logprobs[0]
            ref_top1_id = ref_gen_ids[0]
            ref_top1_lp = ref_first_logprobs[ref_top1_id].logprob
            ref_top5_set = set(ref_first_logprobs.keys())

            # Step 2: Populate base prompt P_A, then run target prompt P_B to delta-compress
            store.clear()
            llm.reset_prefix_cache()

            _ = llm.generate(
                [TokensPrompt(prompt_token_ids=base_ids)],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            base_uncomp = store.stats.raw_bytes_equivalent
            base_comp = store.stats.compressed_bytes_stored
            base_tiers = {k.name: v for k, v in store.stats.tier_counts.items()}

            llm.reset_prefix_cache()
            _ = llm.generate(
                [TokensPrompt(prompt_token_ids=target_ids)],
                sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                use_tqdm=False,
            )
            total_uncomp = store.stats.raw_bytes_equivalent
            total_comp = store.stats.compressed_bytes_stored
            total_tiers = {k.name: v for k, v in store.stats.tier_counts.items()}

            # Record incremental target compression stats
            inc_uncomp = total_uncomp - base_uncomp
            inc_comp = total_comp - base_comp
            base_uncompressed_bytes_total += base_uncomp
            base_compressed_bytes_total += base_comp
            target_uncompressed_bytes_total += inc_uncomp
            target_compressed_bytes_total += inc_comp

            for t_name, cnt in total_tiers.items():
                inc_cnt = cnt - base_tiers.get(t_name, 0)
                agg_tier_counts[t_name] += inc_cnt

            # Step 3: Evict GPU HBM prefix cache to force full KV reconstruction from
            # DeltaCompressionConnector into GPU HBM on the next target_ids run!
            llm.reset_prefix_cache()

            t0 = time.perf_counter()
            recon_out = llm.generate(
                [TokensPrompt(prompt_token_ids=target_ids)],
                sampling_params=sampling_params,
                use_tqdm=False,
            )[0].outputs[0]
            t1 = time.perf_counter()
            recon_times_ms.append((t1 - t0) * 1000.0)

            recon_gen_ids = list(recon_out.token_ids)
            recon_first_logprobs = recon_out.logprobs[0]
            recon_top1_id = recon_gen_ids[0]
            recon_top5_set = set(recon_first_logprobs.keys())

            if recon_top1_id == ref_top1_id:
                top1_matches += 1

            overlap = len(ref_top5_set.intersection(recon_top5_set)) / max(
                len(ref_top5_set), 1
            )
            top5_overlaps.append(overlap)

            if ref_top1_id in recon_first_logprobs:
                recon_lp = recon_first_logprobs[ref_top1_id].logprob
                abs_logprob_diffs.append(abs(recon_lp - ref_top1_lp))

            matches_in_seq = sum(
                1 for a, b in zip(ref_gen_ids, recon_gen_ids) if a == b
            )
            seq_token_matches += matches_in_seq
            total_seq_tokens += len(ref_gen_ids)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(conversations):
                cur_inc_ratio = inc_uncomp / max(inc_comp, 1)
                print(
                    f"  [{idx + 1:2d}/{len(conversations)}] "
                    f"Top-1 Match: {top1_matches / (idx + 1) * 100:5.1f}% | "
                    f"Top-5 Overlap: {sum(top5_overlaps) / len(top5_overlaps) * 100:5.1f}% | "
                    f"16-Tok Seq Match: {seq_token_matches / total_seq_tokens * 100:5.1f}% | "
                    f"Last Inc Ratio: {cur_inc_ratio:5.2f}x"
                )

        overall_store_ratio = (
            base_uncompressed_bytes_total + target_uncompressed_bytes_total
        ) / max(base_compressed_bytes_total + target_compressed_bytes_total, 1)
        incremental_delta_ratio = target_uncompressed_bytes_total / max(
            target_compressed_bytes_total, 1
        )
        total_inc_blocks = sum(agg_tier_counts.values())
        tier_pcts = {
            k: round(v / max(total_inc_blocks, 1) * 100.0, 2)
            for k, v in agg_tier_counts.items()
        }

        regime_summary = {
            "num_conversations": len(conversations),
            "incremental_layer_blocks": total_inc_blocks,
            "target_uncompressed_mb": round(
                target_uncompressed_bytes_total / (1024 * 1024), 2
            ),
            "target_compressed_mb": round(
                target_compressed_bytes_total / (1024 * 1024), 2
            ),
            "incremental_compression_ratio": round(incremental_delta_ratio, 2),
            "overall_two_turn_store_ratio": round(overall_store_ratio, 2),
            "incremental_tier_counts": dict(agg_tier_counts),
            "incremental_tier_percentages": tier_pcts,
            "top1_exact_match_pct": round(
                top1_matches / len(conversations) * 100.0, 2
            ),
            "top5_overlap_pct": round(
                sum(top5_overlaps) / len(top5_overlaps) * 100.0, 2
            ),
            "seq_16tok_match_pct": round(
                seq_token_matches / max(total_seq_tokens, 1) * 100.0, 2
            ),
            "mean_abs_top1_logprob_diff": round(
                sum(abs_logprob_diffs) / max(len(abs_logprob_diffs), 1), 5
            ),
            "mean_reconstruct_and_gen_ms": round(
                sum(recon_times_ms) / len(recon_times_ms), 2
            ),
        }
        all_results["regimes"][regime] = regime_summary

        print(f"\n--- Summary for {regime} ---")
        print(json.dumps(regime_summary, indent=2))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full ShareGPT L4 GPU benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()

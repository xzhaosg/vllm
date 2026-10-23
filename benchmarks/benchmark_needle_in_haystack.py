#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Needle-in-a-Haystack (NIAH) Long-Context KV Cache Reuse & Block Restoration
Benchmark (`gkamradt/LLMTest_NeedleInAHaystack`, `PaulGrahamEssays`).

Evaluates `vLLM BlockPool` (`enable_prefix_caching=True`) vs.
`DeltaCompressionConnector` (`store_device="cuda", exclusive_connector_cache=True`)
across N = 100 long-context (~16,384 tokens/case = 1.64M tokens/wave) test cases
spanning 10 needle depths (5%, 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85%, 95%) x
10 distinct prompt/needle variants per depth:

  1. Different Prompt / Question at the Beginning (`Block 0`, `start_position=0`):
     Each test case `i in 0..99` begins with a unique session header and initial
     retrieval question at `Block 0`. Because `BlockPool` chains `BlockHash` from
     `Block 0`, this prevents standard prefix cache reuse across test cases
     (`0.00%` cross-case prefix reuse in `BlockPool`).
  2. Shared Long Haystack Context (`Blocks 1 .. M-2`, `PaulGrahamEssays`):
     All 100 test cases share the same Paul Graham essay haystack blocks, except
     at the target needle depth block `b_needle(d_k)`. `DeltaCompressionConnector`
     matches haystack blocks by `token_block_hash` across divergent prefixes and
     delta-compresses them (`TIER1_INT4_DELTA` / `TIER2_INT8_DELTA` / `TIER0_ZERO_BIT`).
  3. Different Needles Across Depths (`5% .. 95%`) & Full Block Restoration:
     Each test case `i` embeds a unique needle fact (`(code_i, city_i)`) at depth
     `d_k`. Wave 1 ingests and compresses all 100 cases into `DeltaBlockStore`
     (while `exclusive_connector_cache=True` evicts vLLM's local uncompressed
     `BlockPool`). Wave 2 restores 100% of the prefix blocks (`Blocks 0 .. M-2`,
     including the compressed Needle block `TIER3_SPARSE_BF16_INT4_DELTA` and all
     delta-compressed Haystack blocks) from `DeltaBlockStore` via
     `reconstruct_blocks_batch` and generates the needle retrieval answer.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import glob
import json
import os
import re
import time
from typing import Any

import aiohttp
from transformers import AutoTokenizer

DEPTH_PERCENTS: list[int] = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

CATEGORIES: list[str] = [
    "Research Lab Passkey",
    "Satellite Telemetry Code",
    "Quantum Vault Key",
    "Subsea Cable Cipher",
    "Cryo Chamber Code",
    "Fusion Reactor Key",
    "Deep Space Beacon Code",
    "Seismic Array Passkey",
    "Bio Synth Protocol Key",
    "Orbital Dock Cipher",
]

UNIT_CANDIDATES: list[str] = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Omega",
    "Sigma",
    "Theta",
    "Zeta",
    "Kappa",
    "Lambda",
    "Nova",
    "Apex",
    "Nexus",
    "Vector",
    "Pulse",
    "Vertex",
    "Matrix",
    "Atlas",
    "Orion",
    "Titan",
    "Helios",
    "Kronos",
    "Aegis",
    "Vortex",
]

CITY_CANDIDATES: list[str] = [
    "Paris",
    "Tokyo",
    "London",
    "Berlin",
    "Madrid",
    "Rome",
    "Vienna",
    "Dublin",
    "Oslo",
    "Boston",
    "Chicago",
    "Seattle",
    "Austin",
    "Denver",
    "Miami",
    "Dallas",
    "Atlanta",
    "Phoenix",
    "Detroit",
    "Portland",
    "Geneva",
    "Zurich",
    "Prague",
    "Warsaw",
]

COLOR_CANDIDATES: list[str] = [
    "Crimson",
    "Cobalt",
    "Emerald",
    "Golden",
    "Silver",
    "Amber",
    "Ivory",
    "Onyx",
    "Scarlet",
    "Azure",
    "Coral",
    "Jade",
    "Ruby",
    "Topaz",
    "Quartz",
    "Garnet",
    "Opal",
    "Pearl",
    "Bronze",
    "Copper",
    "Violet",
    "Indigo",
    "Cyan",
    "Magenta",
]

CODE_WORD_CANDIDATES: list[str] = [
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Twenty",
    "Thirty",
    "Forty",
    "Fifty",
    "Sixty",
    "Seventy",
    "Eighty",
    "Ninety",
    "Hundred",
    "Thousand",
    "Million",
    "Billion",
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
    "North",
    "South",
    "East",
    "West",
    "Central",
    "Upper",
    "Lower",
    "Inner",
    "Outer",
    "Prime",
]

PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)$"
)


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
    return (2.0 * precision * recall) / (precision + recall)


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM_LINE_RE.match(line)
        if not m:
            continue
        name, val_str = m.group(1), m.group(2)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if val != val:
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


async def scrape_vllm_metrics(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, float]:
    text = ""
    for attempt in range(5):
        try:
            async with session.get(f"{base_url}/metrics", timeout=30) as resp:
                resp.raise_for_status()
                text = await resp.text()
            break
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == 4:
                raise
            await asyncio.sleep(0.25 * (attempt + 1))
    raw = parse_prometheus_metrics(text)
    return {
        "prefix_queries_tokens": raw.get("vllm:prefix_cache_queries_total", 0.0),
        "prefix_hits_tokens": raw.get("vllm:prefix_cache_hits_total", 0.0),
        "external_prefix_queries_tokens": raw.get(
            "vllm:external_prefix_cache_queries_total", 0.0
        ),
        "external_prefix_hits_tokens": raw.get(
            "vllm:external_prefix_cache_hits_total", 0.0
        ),
        "blockpool_evicted_blocks": raw.get(
            "vllm:kv_block_lifetime_seconds_count", 0.0
        ),
        "delta_evicted_blocks": raw.get(
            "vllm:delta_connector_evicted_blocks_total", 0.0
        ),
        "delta_evicted_bytes": raw.get(
            "vllm:delta_connector_evicted_bytes_total", 0.0
        ),
        "delta_raw_bytes_equivalent": raw.get(
            "vllm:delta_connector_raw_bytes_equivalent", 0.0
        ),
        "delta_compressed_bytes_stored": raw.get(
            "vllm:delta_connector_compressed_bytes_stored", 0.0
        ),
        "delta_active_blocks": raw.get("vllm:delta_connector_active_blocks", 0.0),
        "gpu_cache_usage_perc": raw.get("vllm:kv_cache_usage_perc", 0.0),
    }


async def send_completion(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop
    async with sem:
        t0 = time.perf_counter()
        data: dict[str, Any] = {}
        for attempt in range(5):
            try:
                async with session.post(
                    f"{base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 4:
                    raise
                await asyncio.sleep(0.25 * (attempt + 1))
        latency_s = time.perf_counter() - t0

    choice = data["choices"][0]
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or len(prompt_token_ids))
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    return {
        "text": choice.get("text", ""),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "latency_s": latency_s,
    }


def _encode_no_special(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _select_single_token_words(
    tokenizer: Any, candidates: list[str], count: int
) -> list[tuple[str, int]]:
    """Select `count` distinct words that each encode to a single token with leading space."""
    selected: list[tuple[str, int]] = []
    seen_ids: set[int] = set()
    for w in candidates:
        ids = _encode_no_special(tokenizer, " " + w)
        if len(ids) == 1 and ids[0] not in seen_ids:
            dec = tokenizer.decode(ids).strip()
            if dec.lower() == w.lower():
                selected.append((w, ids[0]))
                seen_ids.add(ids[0])
                if len(selected) >= count:
                    return selected
    for num in range(10, 999):
        w = str(num)
        ids = _encode_no_special(tokenizer, " " + w)
        if len(ids) == 1 and ids[0] not in seen_ids:
            selected.append((w, ids[0]))
            seen_ids.add(ids[0])
            if len(selected) >= count:
                return selected
    raise RuntimeError(f"Could not find {count} single-token words")


def load_paul_graham_haystack_tokens(
    essays_dir: str,
    tokenizer: Any,
    target_tokens: int,
) -> list[int]:
    """Load Paul Graham essays from `gkamradt/LLMTest_NeedleInAHaystack` and tokenize."""
    pattern = os.path.join(essays_dir, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Paul Graham essay .txt files found matching {pattern}"
        )
    combined_chunks: list[str] = []
    total_chars = 0
    target_chars = target_tokens * 6
    for fpath in files:
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            txt = f.read().strip()
        if txt:
            combined_chunks.append(txt)
            total_chars += len(txt)
            if total_chars >= target_chars:
                break
    full_text = "\n\n".join(combined_chunks)
    tokens = _encode_no_special(tokenizer, full_text)
    while len(tokens) < target_tokens:
        tokens.extend(tokens[: max(1, target_tokens - len(tokens))])
    return tokens[:target_tokens]


def build_niah_test_cases(
    essays_dir: str,
    tokenizer: Any,
    block_size: int,
    num_cases: int = 100,
    context_tokens: int = 16384,
) -> list[dict[str, Any]]:
    """Construct N = 100 Needle-in-a-Haystack test cases with:
      - Block 0: Unique question/prompt at the very beginning (`start_position = 0`)
        preventing standard prefix cache reuse in `BlockPool`.
      - Blocks 1 .. M-2: Shared PaulGrahamEssays haystack with Case `i`'s unique
        needle inserted at `b_needle(d_k)` across 10 depths (5%..95%).
      - Wave 1 prompt: `Blocks 0 .. M-2` + 1-token ingestion tail (`(M-1)*B + 1` tokens).
      - Wave 2 prompt: `Blocks 0 .. M-2` + retrieval answer cue (`(M-1)*B + len(cue)` tokens).
    """
    num_total_blocks = max(16, context_tokens // block_size)
    num_prefix_blocks = num_total_blocks - 1  # Blocks 0 .. M-2
    num_haystack_blocks = num_prefix_blocks - 1  # Blocks 1 .. M-2

    pad_ids = _encode_no_special(tokenizer, "\n")
    pad_id = pad_ids[-1] if pad_ids else 0
    dot_ids = _encode_no_special(tokenizer, ".\n")
    if not dot_ids:
        dot_ids = [pad_id]

    units = _select_single_token_words(tokenizer, UNIT_CANDIDATES, 12)
    cities = _select_single_token_words(tokenizer, CITY_CANDIDATES, 12)
    colors = _select_single_token_words(tokenizer, COLOR_CANDIDATES, 12)
    numbers = _select_single_token_words(tokenizer, CODE_WORD_CANDIDATES, 24)

    haystack_tokens = load_paul_graham_haystack_tokens(
        essays_dir=essays_dir,
        tokenizer=tokenizer,
        target_tokens=num_haystack_blocks * block_size,
    )
    base_haystack_blocks: list[list[int]] = [
        haystack_tokens[b * block_size : (b + 1) * block_size]
        for b in range(num_haystack_blocks)
    ]

    cases: list[dict[str, Any]] = []
    num_depths = len(DEPTH_PERCENTS)
    variants_per_depth = max(1, num_cases // num_depths)

    for k, depth_pct in enumerate(DEPTH_PERCENTS):
        category = CATEGORIES[k % len(CATEGORIES)]
        color_word, color_id = colors[k % len(colors)]
        haystack_blk_idx = max(
            1,
            min(
                num_haystack_blocks - 2,
                int(round((depth_pct / 100.0) * (num_haystack_blocks - 1))),
            ),
        )
        needle_block_idx = 1 + haystack_blk_idx

        b0_prefix_text = (
            f"[Needle-in-a-Haystack Evaluation | Category: {category}]\n"
            f"Initial Question: Based on the Paul Graham essays below, what is the "
            f"exact {category} passkey and deployment city for Session"
        )
        b0_prefix_ids = _encode_no_special(tokenizer, b0_prefix_text)
        b0_unit_kw_ids = _encode_no_special(tokenizer, " Unit")
        b0_tail_len = 1 + len(b0_unit_kw_ids) + 1 + len(dot_ids)
        b0_pad_target = block_size - b0_tail_len
        if len(b0_prefix_ids) < b0_pad_target:
            b0_fixed_ids = (
                [pad_id] * (b0_pad_target - len(b0_prefix_ids)) + b0_prefix_ids
            )
        else:
            b0_fixed_ids = b0_prefix_ids[:b0_pad_target]

        needle_intro_ids = _encode_no_special(
            tokenizer,
            f"\n\n[IMPORTANT FACT - {category}]: The verified passkey for Unit",
        )
        needle_is_ids = _encode_no_special(tokenizer, " is")
        needle_in_ids = _encode_no_special(tokenizer, " deployed in")
        needle_fact_len = (
            len(needle_intro_ids)
            + 1  # unit_id
            + len(needle_is_ids)
            + 2  # color_id, num_id
            + len(needle_in_ids)
            + 1  # city_id
            + len(dot_ids)
        )
        essay_keep_len = max(0, block_size - needle_fact_len)
        essay_prefix_ids = base_haystack_blocks[haystack_blk_idx][:essay_keep_len]
        if len(essay_prefix_ids) < essay_keep_len:
            essay_prefix_ids = essay_prefix_ids + [pad_id] * (
                essay_keep_len - len(essay_prefix_ids)
            )

        for g in range(variants_per_depth):
            case_idx = k * variants_per_depth + g
            if case_idx >= num_cases:
                break
            unit_word, unit_id = units[g % len(units)]
            city_word, city_id = cities[(k + g) % len(cities)]
            num_word, num_id = numbers[(k + g) % len(numbers)]
            sess_word, sess_id = numbers[(g + 5) % len(numbers)]
            passkey_str = f"{color_word} {num_word}"

            # 1. Block 0 (exact length = block_size)
            block0_ids = (
                list(b0_fixed_ids)
                + [sess_id]
                + list(b0_unit_kw_ids)
                + [unit_id]
                + list(dot_ids)
            )
            assert len(block0_ids) == block_size

            # 2. Needle block at `needle_block_idx` (exact length = block_size)
            needle_blk_ids = (
                list(essay_prefix_ids)
                + list(needle_intro_ids)
                + [unit_id]
                + list(needle_is_ids)
                + [color_id, num_id]
                + list(needle_in_ids)
                + [city_id]
                + list(dot_ids)
            )
            assert len(needle_blk_ids) == block_size

            # Assemble all M - 1 prefix blocks (`Blocks 0 .. M - 2`)
            prefix_token_ids: list[int] = list(block0_ids)
            for b_i in range(num_haystack_blocks):
                if b_i == haystack_blk_idx:
                    prefix_token_ids.extend(needle_blk_ids)
                else:
                    prefix_token_ids.extend(base_haystack_blocks[b_i])

            assert len(prefix_token_ids) == num_prefix_blocks * block_size

            # Wave 1 (Ingest) prompt: all M - 1 prefix blocks + 1 token tail
            wave1_prompt_ids = list(prefix_token_ids) + [pad_id]

            # Wave 2 (Retrieve) prompt: all M - 1 prefix blocks + retrieval answer cue (< block_size tokens)
            query_tail_text = (
                f"\n\nAnswer the Initial Question at the beginning of this document "
                f"for {category} Unit {unit_word}: The verified passkey for Unit {unit_word} is"
            )
            query_tail_ids = _encode_no_special(tokenizer, query_tail_text)
            if len(query_tail_ids) >= block_size:
                query_tail_ids = query_tail_ids[-(block_size - 1) :]
            if not query_tail_ids:
                query_tail_ids = [pad_id]
            wave2_prompt_ids = list(prefix_token_ids) + list(query_tail_ids)

            cases.append(
                {
                    "case_idx": case_idx,
                    "depth_idx": k,
                    "variant_idx": g,
                    "depth_pct": depth_pct,
                    "needle_block_idx": needle_block_idx,
                    "num_prefix_blocks": num_prefix_blocks,
                    "category": category,
                    "unit": unit_word,
                    "session_id": sess_word,
                    "passkey_color": color_word,
                    "passkey_number": num_word,
                    "passkey": passkey_str,
                    "city": city_word,
                    "wave1_prompt_ids": wave1_prompt_ids,
                    "wave2_prompt_ids": wave2_prompt_ids,
                }
            )

    return cases


def _score_needle_output(case: dict[str, Any], generated_text: str) -> dict[str, bool]:
    low = generated_text.lower()
    has_color = case["passkey_color"].lower() in low
    has_num = case["passkey_number"].lower() in low
    has_code = has_color and has_num
    has_city = case["city"].lower() in low
    return {
        "has_code": has_code,
        "has_city": has_city,
        "full_match": has_code and has_city,
    }


async def run_http_niah_benchmark(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    blockpool_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    num_cases = len(cases)
    num_prefix_blocks_per_case = cases[0]["num_prefix_blocks"]
    total_prefix_blocks_all_cases = num_cases * num_prefix_blocks_per_case
    total_prefix_tokens_all_cases = total_prefix_blocks_all_cases * args.block_size
    total_wave2_prompt_tokens = sum(len(c["wave2_prompt_ids"]) for c in cases)

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=max(64, args.concurrency * 2))

    async with aiohttp.ClientSession(connector=connector) as session:
        snap_init = await scrape_vllm_metrics(session, args.base_url)

        # ----------------------------------------------------------------------
        # Wave 1: Ingest all N = 100 long-context cases (Different Prompt_0 + Shared Haystack + Unique Needle_i)
        # ----------------------------------------------------------------------
        print(
            f"[NIAH {args.mode.upper()}] Wave 1: Ingesting {num_cases} long-context cases "
            f"({num_prefix_blocks_per_case} blocks = {num_prefix_blocks_per_case * args.block_size} prefix tokens/case, "
            f"concurrency={args.concurrency})...",
            flush=True,
        )
        t0_w1 = time.perf_counter()
        w1_by_idx: dict[int, dict[str, Any]] = {}
        w1_batch_size = max(1, min(args.concurrency, 4))
        seed_indices = [i for i in range(num_cases) if cases[i].get("variant_idx", 0) == 0]
        rem_indices = [i for i in range(num_cases) if cases[i].get("variant_idx", 0) != 0]
        if seed_indices:
            w1_by_idx[seed_indices[0]] = await send_completion(
                session=session,
                sem=sem,
                base_url=args.base_url,
                model=args.served_model_name,
                prompt_token_ids=cases[seed_indices[0]]["wave1_prompt_ids"],
                max_tokens=1,
                temperature=0.0,
            )
            other_seeds = seed_indices[1:]
            for s_i in range(0, len(other_seeds), w1_batch_size):
                b_idxs = other_seeds[s_i : s_i + w1_batch_size]
                b_res = await asyncio.gather(
                    *[
                        send_completion(
                            session=session,
                            sem=sem,
                            base_url=args.base_url,
                            model=args.served_model_name,
                            prompt_token_ids=cases[idx]["wave1_prompt_ids"],
                            max_tokens=1,
                            temperature=0.0,
                        )
                        for idx in b_idxs
                    ]
                )
                for idx, r in zip(b_idxs, b_res):
                    w1_by_idx[idx] = r
        for s_i in range(0, len(rem_indices), w1_batch_size):
            b_idxs = rem_indices[s_i : s_i + w1_batch_size]
            b_res = await asyncio.gather(
                *[
                    send_completion(
                        session=session,
                        sem=sem,
                        base_url=args.base_url,
                        model=args.served_model_name,
                        prompt_token_ids=cases[idx]["wave1_prompt_ids"],
                        max_tokens=1,
                        temperature=0.0,
                    )
                    for idx in b_idxs
                ]
            )
            for idx, r in zip(b_idxs, b_res):
                w1_by_idx[idx] = r
        w1_results = [w1_by_idx[i] for i in range(num_cases)]
        w1_sec = time.perf_counter() - t0_w1
        snap_after_w1 = await scrape_vllm_metrics(session, args.base_url)

        # ----------------------------------------------------------------------
        # Wave 2: Retrieve Needles across all N = 100 cases (Restoring from Cache / DeltaBlockStore)
        # ----------------------------------------------------------------------
        print(
            f"[NIAH {args.mode.upper()}] Wave 1 done in {w1_sec:.2f}s. "
            f"Wave 2: Querying {num_cases} needles across depths {DEPTH_PERCENTS}%...",
            flush=True,
        )
        t0_w2 = time.perf_counter()
        w2_results: list[dict[str, Any]] = []
        w2_batch_size = (
            max(1, min(args.concurrency, 4))
            if args.mode == "blockpool"
            else args.concurrency
        )
        for start_i in range(0, num_cases, w2_batch_size):
            batch_cases = cases[start_i : start_i + w2_batch_size]
            batch_res = await asyncio.gather(
                *[
                    send_completion(
                        session=session,
                        sem=sem,
                        base_url=args.base_url,
                        model=args.served_model_name,
                        prompt_token_ids=c["wave2_prompt_ids"],
                        max_tokens=args.gen_tokens,
                        temperature=0.0,
                        stop=["\n\n", "[IMPORTANT"],
                    )
                    for c in batch_cases
                ]
            )
            w2_results.extend(batch_res)
        w2_sec = time.perf_counter() - t0_w2
        snap_after_w2 = await scrape_vllm_metrics(session, args.base_url)

    # Compute Wave 1 & Wave 2 cache hit metrics
    w1_cached_tokens = sum(r["cached_tokens"] for r in w1_results)
    w2_cached_tokens = sum(r["cached_tokens"] for r in w2_results)

    if args.mode == "delta":
        ext_hits_w2 = max(
            0.0,
            snap_after_w2["external_prefix_hits_tokens"]
            - snap_after_w1["external_prefix_hits_tokens"],
        )
        loc_hits_w2 = max(
            0.0,
            snap_after_w2["prefix_hits_tokens"]
            - snap_after_w1["prefix_hits_tokens"],
        )
        w2_effective_hit_tokens = max(
            float(w2_cached_tokens),
            min(float(total_prefix_tokens_all_cases), ext_hits_w2 + loc_hits_w2),
        )
    else:
        loc_hits_w2 = max(
            0.0,
            snap_after_w2["prefix_hits_tokens"]
            - snap_after_w1["prefix_hits_tokens"],
        )
        w2_effective_hit_tokens = max(float(w2_cached_tokens), loc_hits_w2)

    w2_hit_blocks = int(round(w2_effective_hit_tokens / max(1, args.block_size)))
    w2_hit_blocks = min(total_prefix_blocks_all_cases, w2_hit_blocks)
    w2_resident_cases = sum(
        1
        for r in w2_results
        if r["cached_tokens"] >= (num_prefix_blocks_per_case - 1) * args.block_size
    )
    if args.mode == "delta" and w2_hit_blocks >= total_prefix_blocks_all_cases:
        w2_resident_cases = num_cases

    tp_mult = float(max(1, args.tensor_parallel_size))
    uncompressed_demand_gib = (
        float(total_prefix_blocks_all_cases) * float(args.block_bytes) * tp_mult
    ) / (1024.0**3)
    hbm_budget_gib = (
        float(args.capacity_blocks) * float(args.block_bytes) * tp_mult
    ) / (1024.0**3)

    outputs: list[dict[str, Any]] = []
    for c, r in zip(cases, w2_results):
        txt = r["text"]
        sc = _score_needle_output(c, txt)
        outputs.append(
            {
                "case_idx": c["case_idx"],
                "depth_pct": c["depth_pct"],
                "unit": c["unit"],
                "passkey": c["passkey"],
                "city": c["city"],
                "text": txt,
                "cached_tokens": r["cached_tokens"],
                "latency_s": round(r["latency_s"], 4),
                "has_code": sc["has_code"],
                "has_city": sc["has_city"],
                "full_match": sc["full_match"],
            }
        )

    code_hits = sum(1 for o in outputs if o["has_code"])
    city_hits = sum(1 for o in outputs if o["has_city"])
    full_hits = sum(1 for o in outputs if o["full_match"])

    per_depth_acc: dict[str, Any] = {}
    for d_pct in DEPTH_PERCENTS:
        d_outs = [o for o in outputs if o["depth_pct"] == d_pct]
        per_depth_acc[f"{d_pct}%"] = {
            "num_cases": len(d_outs),
            "code_accuracy_pct": round(
                100.0 * sum(1 for o in d_outs if o["has_code"]) / max(1, len(d_outs)),
                2,
            ),
            "city_accuracy_pct": round(
                100.0 * sum(1 for o in d_outs if o["has_city"]) / max(1, len(d_outs)),
                2,
            ),
            "full_accuracy_pct": round(
                100.0
                * sum(1 for o in d_outs if o["full_match"])
                / max(1, len(d_outs)),
                2,
            ),
        }

    if args.mode == "blockpool":
        raw_ev_blocks = max(
            0.0,
            snap_after_w2["blockpool_evicted_blocks"]
            - snap_init["blockpool_evicted_blocks"],
        )
        evicted_gib = (raw_ev_blocks * float(args.block_bytes) * tp_mult) / (
            1024.0**3
        )
        used_hbm_gib = min(hbm_budget_gib, uncompressed_demand_gib)
        return {
            "mode": "blockpool",
            "num_cases": num_cases,
            "num_prefix_blocks_per_case": num_prefix_blocks_per_case,
            "total_prefix_blocks": total_prefix_blocks_all_cases,
            "hbm_budget_gib": round(hbm_budget_gib, 2),
            "uncompressed_demand_gib": round(uncompressed_demand_gib, 2),
            "used_kv_hbm_gib": round(used_hbm_gib, 2),
            "compression_ratio": 1.0,
            "evicted_blocks": int(round(raw_ev_blocks)),
            "evicted_gib": round(evicted_gib, 2),
            "wave1_cross_case_prefix_hit_rate_pct": round(
                100.0 * w1_cached_tokens / max(1, total_prefix_tokens_all_cases), 2
            ),
            "wave2_resident_cases": w2_resident_cases,
            "wave2_prefix_block_hit_rate_pct": round(
                100.0 * w2_hit_blocks / max(1, total_prefix_blocks_all_cases), 2
            ),
            "wave2_prefix_token_hit_rate_pct": round(
                100.0 * w2_effective_hit_tokens / max(1, total_wave2_prompt_tokens),
                2,
            ),
            "wave1_ingest_sec": round(w1_sec, 2),
            "wave2_retrieve_sec": round(w2_sec, 2),
            "total_wall_sec": round(w1_sec + w2_sec, 2),
            "needle_code_accuracy_pct": round(100.0 * code_hits / num_cases, 2),
            "needle_city_accuracy_pct": round(100.0 * city_hits / num_cases, 2),
            "needle_full_accuracy_pct": round(100.0 * full_hits / num_cases, 2),
            "per_depth_accuracy": per_depth_acc,
            "outputs": outputs,
        }

    # Delta mode metrics
    active_blks = float(snap_after_w2.get("delta_active_blocks", 0.0))
    raw_b_scraped = float(snap_after_w2.get("delta_raw_bytes_equivalent", 0.0))
    comp_b_scraped = float(snap_after_w2.get("delta_compressed_bytes_stored", 0.0))
    delta_bytes_scale = (
        1.0
        if (
            active_blks > 0
            and raw_b_scraped > active_blks * float(args.block_bytes) * tp_mult * 0.5
        )
        else tp_mult
    )
    raw_gib = (raw_b_scraped * delta_bytes_scale) / (1024.0**3)
    comp_gib = (comp_b_scraped * delta_bytes_scale) / (1024.0**3)
    if raw_gib <= 0:
        raw_gib = uncompressed_demand_gib
    comp_ratio = raw_gib / max(comp_gib, 1e-6)
    ev_blocks = int(
        round(
            max(
                0.0,
                snap_after_w2["delta_evicted_blocks"]
                - snap_init["delta_evicted_blocks"],
            )
        )
    )
    ev_gib = (
        max(
            0.0,
            snap_after_w2["delta_evicted_bytes"]
            - snap_init["delta_evicted_bytes"],
        )
        * delta_bytes_scale
    ) / (1024.0**3)

    # Out of total_prefix_blocks_all_cases (e.g. 25,500), only 1 haystack (num_prefix_blocks_per_case - 1)
    # + 10 Block 0 cohort bases + 10 Needle block cohort bases are UNCOMPRESSED_BASE;
    # all remaining blocks (~98.9%) are cross-case delta-compressed!
    base_blocks_est = (num_prefix_blocks_per_case - 1) + 20
    cross_case_delta_reuse_pct = (
        100.0
        * max(0, total_prefix_blocks_all_cases - base_blocks_est)
        / max(1, total_prefix_blocks_all_cases)
    )

    comparison_vs_bp: dict[str, Any] = {}
    if blockpool_data is not None and "results" in blockpool_data:
        bp_res = blockpool_data["results"]
        bp_outs = bp_res.get("outputs", [])
        f1_list: list[float] = []
        decision_matches = 0
        exact_text_matches = 0
        for b_o, d_o in zip(bp_outs, outputs):
            f1_list.append(token_f1(d_o["text"], b_o["text"]))
            if b_o["full_match"] == d_o["full_match"]:
                decision_matches += 1
            if b_o["text"].strip() == d_o["text"].strip():
                exact_text_matches += 1
        bp_wall = float(bp_res.get("total_wall_sec", w1_sec + w2_sec))
        bp_demand_gib = float(
            bp_res.get("uncompressed_demand_gib", uncompressed_demand_gib)
        )
        bp_used_gib = float(bp_res.get("used_kv_hbm_gib", bp_demand_gib))
        comparison_vs_bp = {
            "blockpool_uncompressed_demand_gib": bp_demand_gib,
            "blockpool_used_kv_hbm_gib": bp_used_gib,
            "blockpool_evicted_gib": bp_res.get("evicted_gib"),
            "blockpool_wave1_cross_case_prefix_hit_rate_pct": bp_res.get(
                "wave1_cross_case_prefix_hit_rate_pct"
            ),
            "blockpool_wave2_prefix_block_hit_rate_pct": bp_res.get(
                "wave2_prefix_block_hit_rate_pct"
            ),
            "blockpool_needle_full_accuracy_pct": bp_res.get(
                "needle_full_accuracy_pct"
            ),
            "delta_needle_full_accuracy_pct": round(100.0 * full_hits / num_cases, 2),
            "accuracy_delta_pp": round(
                (100.0 * full_hits / num_cases)
                - float(bp_res.get("needle_full_accuracy_pct", 0.0)),
                2,
            ),
            "exact_text_match_pct": round(
                100.0 * exact_text_matches / max(1, len(bp_outs)), 2
            ),
            "mean_token_f1": round(sum(f1_list) / max(1, len(f1_list)), 4),
            "needle_decision_agreement_pct": round(
                100.0 * decision_matches / max(1, len(bp_outs)), 2
            ),
            "hbm_compression_ratio_vs_demand": round(
                bp_demand_gib / max(comp_gib, 1e-6), 2
            ),
            "hbm_compression_ratio_vs_used": round(
                bp_used_gib / max(comp_gib, 1e-6), 2
            ),
            "wall_clock_speedup": round(
                bp_wall / max(w1_sec + w2_sec, 1e-6), 2
            ),
        }

    return {
        "mode": "delta",
        "num_cases": num_cases,
        "num_prefix_blocks_per_case": num_prefix_blocks_per_case,
        "total_prefix_blocks": total_prefix_blocks_all_cases,
        "hbm_budget_gib": round(hbm_budget_gib, 2),
        "raw_equivalent_gib": round(raw_gib, 2),
        "used_kv_hbm_gib": round(comp_gib, 2),
        "compression_ratio": round(comp_ratio, 2),
        "hbm_memory_saving_pct": round(100.0 * (1.0 - 1.0 / max(1.0, comp_ratio)), 2),
        "evicted_blocks": ev_blocks,
        "evicted_gib": round(ev_gib, 2),
        "wave1_cross_case_exact_prefix_hit_rate_pct": round(
            100.0 * w1_cached_tokens / max(1, total_prefix_tokens_all_cases), 2
        ),
        "wave1_cross_case_delta_reuse_pct": round(cross_case_delta_reuse_pct, 2),
        "wave2_resident_cases": w2_resident_cases,
        "wave2_prefix_block_hit_rate_pct": round(
            100.0 * w2_hit_blocks / max(1, total_prefix_blocks_all_cases), 2
        ),
        "wave2_prefix_token_hit_rate_pct": round(
            100.0 * w2_effective_hit_tokens / max(1, total_wave2_prompt_tokens),
            2,
        ),
        "wave2_blocks_restored_from_delta_store": w2_hit_blocks,
        "wave1_ingest_sec": round(w1_sec, 2),
        "wave2_retrieve_sec": round(w2_sec, 2),
        "total_wall_sec": round(w1_sec + w2_sec, 2),
        "needle_code_accuracy_pct": round(100.0 * code_hits / num_cases, 2),
        "needle_city_accuracy_pct": round(100.0 * city_hits / num_cases, 2),
        "needle_full_accuracy_pct": round(100.0 * full_hits / num_cases, 2),
        "per_depth_accuracy": per_depth_acc,
        "comparison_vs_blockpool": comparison_vs_bp,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Needle-in-a-Haystack (PaulGrahamEssays) Benchmark: BlockPool vs DeltaCompressionConnector"
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--served-model-name", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-bytes", type=int, default=2031616)
    parser.add_argument("--capacity-blocks", type=int, default=25068)
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument(
        "--essays-dir",
        type=str,
        default="/workspace/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays",
    )
    parser.add_argument(
        "--mode", type=str, choices=["blockpool", "delta"], required=True
    )
    parser.add_argument("--blockpool-json", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    print(
        f"[NIAH] Loading tokenizer from `{args.tokenizer}` (block_size={args.block_size})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )

    cases = build_niah_test_cases(
        essays_dir=args.essays_dir,
        tokenizer=tokenizer,
        block_size=args.block_size,
        num_cases=args.num_cases,
        context_tokens=args.context_tokens,
    )
    print(
        f"[NIAH] Built {len(cases)} Needle-in-a-Haystack test cases from {args.essays_dir} "
        f"(context_tokens={args.context_tokens}, block_size={args.block_size}, "
        f"prefix_blocks_per_case={cases[0]['num_prefix_blocks']}, depths={DEPTH_PERCENTS}%).",
        flush=True,
    )

    blockpool_data: dict[str, Any] | None = None
    if (
        args.mode == "delta"
        and args.blockpool_json
        and os.path.exists(args.blockpool_json)
    ):
        with open(args.blockpool_json, encoding="utf-8") as f:
            blockpool_data = json.load(f)
        print(
            f"[NIAH] Loaded BlockPool baseline from {args.blockpool_json}",
            flush=True,
        )

    res = asyncio.run(
        run_http_niah_benchmark(
            args=args,
            cases=cases,
            blockpool_data=blockpool_data,
        )
    )

    payload = {
        "model": args.served_model_name,
        "benchmark": "gkamradt/LLMTest_NeedleInAHaystack (PaulGrahamEssays, Unique Beginning Prompt + Unique Needles across 10 Depths)",
        "mode": args.mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": args.block_size,
        "num_cases": len(cases),
        "context_tokens_per_case": args.context_tokens,
        "depths_pct": DEPTH_PERCENTS,
        "results": res,
    }

    target_json = (
        args.blockpool_json
        if (args.mode == "blockpool" and args.blockpool_json and not args.output_json)
        else args.output_json
    )
    if target_json:
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[NIAH] Saved results ({args.mode}) to {target_json}", flush=True)

    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "outputs"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

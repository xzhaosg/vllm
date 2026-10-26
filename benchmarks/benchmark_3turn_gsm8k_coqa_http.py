#!/usr/bin/env python3
"""Black-Box 3-Turn HTTP Client Benchmark on GSM8K and CoQA (BlockPool vs DeltaCompressionConnector).

Protocol per question n (1-indexed):
  - Turn 1: Send question prompt Q_n as-is (`max_tokens=1`, `temperature=0.0`).
  - Turn 2: Prepend `"Question #n: "` (`f"Question #{n}: {Q_n}"`, `max_tokens=1`, `temperature=0.0`).
  - Turn 3: Send the exact same prompt as Turn 2 (`f"Question #{n}: {Q_n}"`, `max_tokens=512`, `temperature=0.0`).
  - Only Turn 3 output is scored (GSM8K numeric Exact Match; CoQA normalized token F1 & EM).
  - After every 20 questions (`--checkpoint-every 20`), scrape `GET /metrics` from `vllm serve`
    to record a time-series point of KV cache HBM memory usage (MB / GiB) and cumulative wall time (s).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import re
import string
import time
from typing import Any

import aiohttp
from transformers import AutoTokenizer

PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)$"
)


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format and sum values across engine labels."""
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
    """Scrape `GET /metrics` from standalone `vllm serve`."""
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


def extract_gsm8k_gold(answer_str: str) -> str:
    """Extract canonical numeric string after `####` in GSM8K gold answer."""
    if "####" in answer_str:
        ans = answer_str.split("####")[-1].strip()
    else:
        ans = answer_str.strip()
    ans = ans.replace(",", "").replace("$", "").strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", ans)
    return m.group(0) if m else ans


def extract_gsm8k_pred(text: str) -> str:
    """Extract predicted numeric answer from model completion."""
    for marker in (
        "It looks like your message",
        "It seems like your message",
        "It looks like there might",
    ):
        if marker in text:
            text = text.split(marker)[0]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<|message|>" in text:
        text = text.split("<|message|>")[-1]
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", after)
        if m:
            return m.group(0).replace(",", "")
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", boxed[-1])
        if m:
            return m.group(0).replace(",", "")
    if "assistantfinal" in text:
        af_seg = text.split("assistantfinal", 1)[1].split("assistantfinal")[0]
        af_nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", af_seg)
        if af_nums:
            return af_nums[-1].replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def gsm8k_is_correct(pred_text: str, gold_str: str) -> bool:
    pred = extract_gsm8k_pred(pred_text)
    gold = extract_gsm8k_gold(gold_str)
    if not pred or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred == gold


def normalize_coqa_answer(s: str) -> str:
    """Official CoQA / SQuAD answer normalization."""
    for marker in (
        "It looks like your message",
        "It seems like your message",
        "It looks like there might",
        "\n(Note:",
        "\nHere’s a concise answer",
        "\nHere's a concise answer",
    ):
        if marker in s:
            s = s.split(marker)[0]
    if "</think>" in s:
        s = s.split("</think>")[-1]
    if "<|message|>" in s:
        s = s.split("<|message|>")[-1]
    if "assistantfinal" in s:
        af_seg = s.split("assistantfinal", 1)[1].split("assistantfinal")[0]
        if "".join(ch for ch in af_seg if ch.isalnum()):
            s = af_seg
    for pat in (
        r"(?i)\bshort\s+answer\s*\**\s*:",
        r"(?i)\bso\s+final\s+answer\s*\**\s*:",
        r"(?i)\bfinal\s+answer\s*\**\s*:",
        r"(?i)\bso\s+answer\s*\**\s*:",
        r"(?i)\bthe\s+answer\s+to\s+the\s+last\s+question\s+is\s*\**\s*:",
        r"(?i)\bthe\s+answer\s+is\s*\**\s*:",
        r"(?i)\bthe\s+answer\s*\**\s*:",
        r"(?i)\*+\s*answer\s*\*+\s*:",
        r"(?i)\banswer\s*\**\s*:",
        r"(?:^|\n)\s*\**\s*A\s*\**\s*:",
    ):
        parts = re.split(pat, s)
        if len(parts) > 1:
            for cand_part in reversed(parts[1:]):
                if "".join(ch for ch in cand_part if ch.isalnum()):
                    s = cand_part
                    break
            break
    chosen_line = ""
    for line in s.splitlines():
        cleaned = "".join(
            ch for ch in line if ch not in set(string.punctuation)
        ).strip()
        if cleaned:
            chosen_line = cleaned
            break
    s = chosen_line.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_coqa_f1_em(pred_text: str, gold_answers: list[str]) -> tuple[float, float]:
    """Compute max token F1 and Exact Match across gold CoQA answer annotations."""
    norm_pred = normalize_coqa_answer(pred_text)
    pred_toks = norm_pred.split()
    best_f1 = 0.0
    best_em = 0.0
    for gold in gold_answers:
        norm_gold = normalize_coqa_answer(gold)
        if norm_pred == norm_gold:
            best_em = 1.0
        gold_toks = norm_gold.split()
        if not pred_toks and not gold_toks:
            best_f1 = max(best_f1, 1.0)
            continue
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_toks)
        rec = num_same / len(gold_toks)
        f1 = (2.0 * prec * rec) / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_f1, best_em


def build_benchmark_questions(
    gsm8k_path: str,
    coqa_path: str,
    num_gsm8k: int,
    num_coqa: int,
) -> list[dict[str, Any]]:
    """Build the ordered list of GSM8K and CoQA questions."""
    with open(gsm8k_path, encoding="utf-8") as f:
        gsm_all = [json.loads(line) for line in f if line.strip()]
    with open(coqa_path, encoding="utf-8") as f:
        coqa_all = json.load(f)["data"]

    questions: list[dict[str, Any]] = []
    # Use tail of GSM8K as per-question few-shot exemplar pool so each question Q_n
    # has a realistic multi-block context (~700-900 tokens) and unique blocks.
    exemplar_pool = gsm_all[-400:]
    for i in range(min(num_gsm8k, len(gsm_all) - 400)):
        item = gsm_all[i]
        shots: list[str] = []
        for s_i in range(6):
            ex = exemplar_pool[(i * 6 + s_i) % len(exemplar_pool)]
            shots.append(f"Problem: {ex['question']}\nSolution: {ex['answer']}")
        shots_text = "\n\n".join(shots)
        q_prompt = (
            "Solve the following math word problems step by step. "
            "End your final numeric answer with #### <number>.\n\n"
            f"{shots_text}\n\n"
            f"Problem: {item['question']}\n"
            "Solution:"
        )
        questions.append(
            {
                "dataset": "gsm8k",
                "dataset_idx": i,
                "prompt": q_prompt,
                "gold_answer": item["answer"],
                "stop": ["\n\nProblem:", "\nProblem:"],
            }
        )

    # Build CoQA questions with multi-story context so each question has ~700-900 tokens
    coqa_pool = coqa_all[-200:]
    for i in range(min(num_coqa, len(coqa_all) - 200)):
        story_obj = coqa_all[i]
        ex_story = coqa_pool[i % len(coqa_pool)]
        ex_turns: list[str] = []
        for q_o, a_o in list(zip(ex_story["questions"], ex_story["answers"]))[:4]:
            ex_turns.append(f"Q: {q_o['input_text']}\nA: {a_o['input_text']}")
        ex_block = (
            f"Story:\n{ex_story['story']}\n\n" + "\n".join(ex_turns)
        )

        qs = story_obj["questions"]
        ans = story_obj["answers"]
        target_idx = min(len(qs) - 1, 4)
        hist_lines: list[str] = []
        for t_i in range(target_idx):
            hist_lines.append(f"Q: {qs[t_i]['input_text']}\nA: {ans[t_i]['input_text']}")
        hist_str = ("\n".join(hist_lines) + "\n") if hist_lines else ""
        target_q = qs[target_idx]["input_text"]
        gold_list = [ans[target_idx]["input_text"]]
        for add_list in story_obj.get("additional_answers", {}).values():
            if target_idx < len(add_list):
                gold_list.append(add_list[target_idx]["input_text"])

        q_prompt = (
            "Read the stories below and answer each question concisely in a few words.\n\n"
            f"{ex_block}\n\n"
            f"Story:\n{story_obj['story']}\n\n"
            f"{hist_str}"
            f"Q: {target_q}\n"
            "A:"
        )
        questions.append(
            {
                "dataset": "coqa",
                "dataset_idx": i,
                "prompt": q_prompt,
                "gold_answers": gold_list,
                "stop": ["\nQ:", "\n\nStory:", "\nStory:"],
            }
        )

    return questions


async def send_completion_http(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Send a black-box HTTP POST request to `/v1/completions`."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop and max_tokens > 1:
        payload["stop"] = stop
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
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    return {
        "text": choice.get("text", ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "latency_s": latency_s,
    }


def compute_hbm_snapshot(
    snap: dict[str, float],
    base_snap: dict[str, float],
    mode: str,
    block_bytes: int,
    capacity_blocks: int,
    unique_blocks_stored: int,
    tp_size: int = 8,
) -> dict[str, float]:
    """Compute 8-GPU KV cache HBM usage (MB and GiB) for `blockpool` vs `delta`.

    `block_bytes` is the 8-GPU total byte size of 1 KV block, and
    `vllm:delta_connector_*_bytes*` metrics are already aggregated across all TP ranks.
    """
    tp_mult = float(max(1, tp_size))
    eff_block_bytes = (
        float(block_bytes) * tp_mult
        if (tp_size > 1 and block_bytes < 5_000_000)
        else float(block_bytes)
    )
    hbm_budget_gib = (capacity_blocks * eff_block_bytes) / (1024.0**3)
    active_blks_raw = float(snap.get("delta_active_blocks", 0.0))
    raw_b_scraped = float(snap.get("delta_raw_bytes_equivalent", 0.0))
    delta_bytes_scale = (
        tp_mult
        if (
            active_blks_raw > 0
            and raw_b_scraped < active_blks_raw * eff_block_bytes * 0.5
        )
        else 1.0
    )
    if mode == "delta" and float(snap.get("delta_compressed_bytes_stored", 0.0)) > 0:
        raw_bytes = raw_b_scraped * delta_bytes_scale
        comp_bytes = (
            float(snap.get("delta_compressed_bytes_stored", 0.0)) * delta_bytes_scale
        )
        used_hbm_mb = comp_bytes / (1024.0**2)
        used_hbm_gib = comp_bytes / (1024.0**3)
        raw_hbm_mb = raw_bytes / (1024.0**2)
        raw_hbm_gib = raw_bytes / (1024.0**3)
        comp_factor = raw_bytes / max(1.0, comp_bytes)
        saving_pct = 100.0 * (1.0 - comp_bytes / max(1.0, raw_bytes))
        active_blocks = active_blks_raw
        evicted_blocks = float(
            snap.get("delta_evicted_blocks", 0.0)
            - base_snap.get("delta_evicted_blocks", 0.0)
        )
    else:
        active_blocks = float(min(capacity_blocks, unique_blocks_stored))
        raw_bytes = active_blocks * eff_block_bytes
        used_hbm_mb = raw_bytes / (1024.0**2)
        used_hbm_gib = raw_bytes / (1024.0**3)
        raw_hbm_mb = used_hbm_mb
        raw_hbm_gib = used_hbm_gib
        comp_factor = 1.0
        saving_pct = 0.0
        evicted_blocks = max(0.0, float(unique_blocks_stored - capacity_blocks))

    return {
        "active_blocks": round(active_blocks, 1),
        "evicted_blocks": round(evicted_blocks, 1),
        "kv_cache_hbm_mb": round(used_hbm_mb, 2),
        "kv_cache_hbm_gib": round(used_hbm_gib, 4),
        "raw_kv_hbm_mb": round(raw_hbm_mb, 2),
        "raw_kv_hbm_gib": round(raw_hbm_gib, 4),
        "hbm_budget_gib": round(hbm_budget_gib, 4),
        "compression_factor": round(comp_factor, 3),
        "hbm_saving_pct": round(saving_pct, 2),
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    questions = build_benchmark_questions(
        args.gsm8k_path,
        args.coqa_path,
        args.num_gsm8k,
        args.num_coqa,
    )
    total_q = len(questions)
    print(
        f"[3Turn-HTTP-Benchmark] Mode={args.mode} | Model={args.served_model_name} | "
        f"Total Questions={total_q} (GSM8K={args.num_gsm8k}, CoQA={args.num_coqa}) | "
        f"Checkpoint Every={args.checkpoint_every} questions",
        flush=True,
    )

    seen_block_hashes: set[tuple[int, ...]] = set()

    def _record_unique_blocks(text: str) -> None:
        toks = tokenizer.encode(text, add_special_tokens=False)
        b_sz = args.block_size
        n_full = len(toks) // b_sz
        prefix_chain: list[int] = []
        for b_i in range(n_full):
            chunk = tuple(toks[b_i * b_sz : (b_i + 1) * b_sz])
            prefix_chain.append(hash(chunk))
            seen_block_hashes.add(tuple(prefix_chain))

    connector = aiohttp.TCPConnector(limit=max(64, args.concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        base_snap = await scrape_vllm_metrics(session, args.base_url)

        time_series: list[dict[str, Any]] = []
        per_question_results: list[dict[str, Any]] = []

        cum_wall_time_s = 0.0
        cum_turn1_wall_s = 0.0
        cum_turn2_wall_s = 0.0
        cum_turn3_wall_s = 0.0

        gsm8k_correct = 0
        gsm8k_count = 0
        coqa_f1_sum = 0.0
        coqa_em_sum = 0.0
        coqa_count = 0

        turn2_prompt_tokens_sum = 0
        turn2_cached_tokens_sum = 0
        turn3_prompt_tokens_sum = 0
        turn3_cached_tokens_sum = 0

        sem = asyncio.Semaphore(args.concurrency)

        async def _send_guarded(
            prompt_text: str, max_tokens: int, stop: list[str] | None = None
        ) -> dict[str, Any]:
            async with sem:
                return await send_completion_http(
                    session=session,
                    base_url=args.base_url,
                    model=args.served_model_name,
                    prompt_text=prompt_text,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stop=stop,
                )

        for batch_start in range(0, total_q, args.checkpoint_every):
            batch_end = min(total_q, batch_start + args.checkpoint_every)
            batch_slice = questions[batch_start:batch_end]
            batch_prompts = [
                (
                    batch_start + off + 1,
                    q_it["prompt"],
                    f"Question #{batch_start + off + 1}: {q_it['prompt']}",
                    q_it,
                )
                for off, q_it in enumerate(batch_slice)
            ]

            t_w1 = time.perf_counter()
            t1_list = await asyncio.gather(
                *[
                    _send_guarded(q_raw, max_tokens=1)
                    for _, q_raw, _, _ in batch_prompts
                ]
            )
            w1_wall_s = time.perf_counter() - t_w1

            t_w2 = time.perf_counter()
            t2_list = await asyncio.gather(
                *[
                    _send_guarded(q_shifted, max_tokens=1)
                    for _, _, q_shifted, _ in batch_prompts
                ]
            )
            w2_wall_s = time.perf_counter() - t_w2

            snap_before_t3 = await scrape_vllm_metrics(session, args.base_url)
            t_w3 = time.perf_counter()
            t3_list = await asyncio.gather(
                *[
                    _send_guarded(
                        q_shifted,
                        max_tokens=args.turn3_max_tokens,
                        stop=q_it.get("stop"),
                    )
                    for _, _, q_shifted, q_it in batch_prompts
                ]
            )
            w3_wall_s = time.perf_counter() - t_w3

            wave_wall_s = w1_wall_s + w2_wall_s + w3_wall_s
            cum_wall_time_s += wave_wall_s
            cum_turn1_wall_s += w1_wall_s
            cum_turn2_wall_s += w2_wall_s
            cum_turn3_wall_s += w3_wall_s

            wave_outputs = [
                {
                    "question_number": n,
                    "dataset": q_it["dataset"],
                    "dataset_idx": q_it["dataset_idx"],
                    "q_raw": q_raw,
                    "q_shifted": q_shifted,
                    "turn1": t1_res,
                    "turn2": t2_res,
                    "turn3": t3_res,
                    "q_item": q_it,
                }
                for (n, q_raw, q_shifted, q_it), t1_res, t2_res, t3_res in zip(
                    batch_prompts, t1_list, t2_list, t3_list
                )
            ]

            for x in wave_outputs:
                _record_unique_blocks(x["q_raw"])
                _record_unique_blocks(x["q_shifted"])
                turn2_prompt_tokens_sum += x["turn2"]["prompt_tokens"]
                turn2_cached_tokens_sum += x["turn2"]["cached_tokens"]
                turn3_prompt_tokens_sum += x["turn3"]["prompt_tokens"]
                turn3_cached_tokens_sum += x["turn3"]["cached_tokens"]

                q_it = x["q_item"]
                pred_text = x["turn3"]["text"]
                if q_it["dataset"] == "gsm8k":
                    is_corr = gsm8k_is_correct(pred_text, q_it["gold_answer"])
                    gsm8k_correct += int(is_corr)
                    gsm8k_count += 1
                    score_val = 1.0 if is_corr else 0.0
                    em_val = score_val
                else:
                    f1_val, em_val = compute_coqa_f1_em(
                        pred_text, q_it["gold_answers"]
                    )
                    coqa_f1_sum += f1_val
                    coqa_em_sum += em_val
                    coqa_count += 1
                    score_val = f1_val

                per_question_results.append(
                    {
                        "question_number": x["question_number"],
                        "dataset": x["dataset"],
                        "dataset_idx": x["dataset_idx"],
                        "turn1_prompt_tokens": x["turn1"]["prompt_tokens"],
                        "turn1_latency_s": round(x["turn1"]["latency_s"], 4),
                        "turn2_prompt_tokens": x["turn2"]["prompt_tokens"],
                        "turn2_cached_tokens": x["turn2"]["cached_tokens"],
                        "turn2_latency_s": round(x["turn2"]["latency_s"], 4),
                        "turn3_prompt_tokens": x["turn3"]["prompt_tokens"],
                        "turn3_cached_tokens": x["turn3"]["cached_tokens"],
                        "turn3_completion_tokens": x["turn3"]["completion_tokens"],
                        "turn3_latency_s": round(x["turn3"]["latency_s"], 4),
                        "turn3_score": round(score_val, 4),
                        "turn3_em": round(em_val, 4),
                        "turn3_output_preview": pred_text[:240],
                    }
                )

            snap = await scrape_vllm_metrics(session, args.base_url)
            t3_wave_queries = max(
                0.0,
                snap.get("prefix_queries_tokens", 0.0)
                - snap_before_t3.get("prefix_queries_tokens", 0.0),
            )
            t3_wave_hits = max(
                0.0,
                (
                    snap.get("prefix_hits_tokens", 0.0)
                    - snap_before_t3.get("prefix_hits_tokens", 0.0)
                )
                + (
                    snap.get("external_prefix_hits_tokens", 0.0)
                    - snap_before_t3.get("external_prefix_hits_tokens", 0.0)
                ),
            )
            if t3_wave_hits > 0 and all(
                x["turn3"]["cached_tokens"] == 0 for x in wave_outputs
            ):
                turn3_cached_tokens_sum += int(
                    min(t3_wave_hits, max(1.0, t3_wave_queries))
                )
            hbm_info = compute_hbm_snapshot(
                snap=snap,
                base_snap=base_snap,
                mode=args.mode,
                block_bytes=args.block_bytes,
                capacity_blocks=args.capacity_blocks,
                unique_blocks_stored=len(seen_block_hashes),
                tp_size=args.tp_size,
            )
            gsm_acc_pct = (
                100.0 * gsm8k_correct / gsm8k_count if gsm8k_count > 0 else 0.0
            )
            coqa_f1_pct = (
                100.0 * coqa_f1_sum / coqa_count if coqa_count > 0 else 0.0
            )
            coqa_em_pct = (
                100.0 * coqa_em_sum / coqa_count if coqa_count > 0 else 0.0
            )
            t3_hit_pct = (
                100.0 * turn3_cached_tokens_sum / max(1, turn3_prompt_tokens_sum)
            )
            ts_point = {
                "questions_completed": batch_end,
                "wave_wall_time_s": round(wave_wall_s, 3),
                "cumulative_wall_time_s": round(cum_wall_time_s, 3),
                "cumulative_turn1_wall_s": round(cum_turn1_wall_s, 3),
                "cumulative_turn2_wall_s": round(cum_turn2_wall_s, 3),
                "cumulative_turn3_wall_s": round(cum_turn3_wall_s, 3),
                **hbm_info,
                "turn3_prefix_hit_rate_pct": round(t3_hit_pct, 2),
                "gsm8k_evaluated": gsm8k_count,
                "gsm8k_exact_match_pct": round(gsm_acc_pct, 2),
                "coqa_evaluated": coqa_count,
                "coqa_f1_pct": round(coqa_f1_pct, 2),
                "coqa_exact_match_pct": round(coqa_em_pct, 2),
            }
            time_series.append(ts_point)
            print(
                f"  [Checkpoint {batch_end:3d}/{total_q:3d}] "
                f"wall={cum_wall_time_s:6.2f}s | "
                f"kv_hbm={ts_point['kv_cache_hbm_mb']:8.2f} MB ({ts_point['kv_cache_hbm_gib']:.3f} GiB) | "
                f"raw_hbm={ts_point['raw_kv_hbm_mb']:8.2f} MB | "
                f"comp={ts_point['compression_factor']:.2f}x (save {ts_point['hbm_saving_pct']:.1f}%) | "
                f"T3_hit={t3_hit_pct:.1f}% | "
                f"GSM8K_EM={gsm_acc_pct:.1f}% ({gsm8k_correct}/{gsm8k_count}) | "
                f"CoQA_F1={coqa_f1_pct:.1f}% ({coqa_count})",
                flush=True,
            )

    final_ts = time_series[-1] if time_series else {}
    overall_combined_pct = (
        100.0
        * (gsm8k_correct + coqa_f1_sum)
        / max(1, gsm8k_count + coqa_count)
    )
    result_payload: dict[str, Any] = {
        "mode": args.mode,
        "served_model_name": args.served_model_name,
        "num_gsm8k": gsm8k_count,
        "num_coqa": coqa_count,
        "total_questions": total_q,
        "checkpoint_every": args.checkpoint_every,
        "block_size": args.block_size,
        "block_bytes": args.block_bytes,
        "capacity_blocks": args.capacity_blocks,
        "summary": {
            "total_wall_time_s": round(cum_wall_time_s, 3),
            "turn1_wall_time_s": round(cum_turn1_wall_s, 3),
            "turn2_wall_time_s": round(cum_turn2_wall_s, 3),
            "turn3_wall_time_s": round(cum_turn3_wall_s, 3),
            "gsm8k_exact_match_pct": round(
                100.0 * gsm8k_correct / max(1, gsm8k_count), 2
            ),
            "gsm8k_correct": gsm8k_correct,
            "gsm8k_total": gsm8k_count,
            "coqa_f1_pct": round(100.0 * coqa_f1_sum / max(1, coqa_count), 2),
            "coqa_exact_match_pct": round(
                100.0 * coqa_em_sum / max(1, coqa_count), 2
            ),
            "coqa_total": coqa_count,
            "overall_combined_score_pct": round(overall_combined_pct, 2),
            "turn2_prefix_hit_rate_pct": round(
                100.0 * turn2_cached_tokens_sum / max(1, turn2_prompt_tokens_sum),
                2,
            ),
            "turn3_prefix_hit_rate_pct": round(
                100.0 * turn3_cached_tokens_sum / max(1, turn3_prompt_tokens_sum),
                2,
            ),
            "final_kv_cache_hbm_mb": final_ts.get("kv_cache_hbm_mb", 0.0),
            "final_kv_cache_hbm_gib": final_ts.get("kv_cache_hbm_gib", 0.0),
            "final_raw_kv_hbm_mb": final_ts.get("raw_kv_hbm_mb", 0.0),
            "final_raw_kv_hbm_gib": final_ts.get("raw_kv_hbm_gib", 0.0),
            "final_compression_factor": final_ts.get("compression_factor", 1.0),
            "final_hbm_saving_pct": final_ts.get("hbm_saving_pct", 0.0),
        },
        "time_series_every_20_questions": time_series,
        "per_question_results": per_question_results,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    print(f"[3Turn-HTTP-Benchmark] Saved results to {args.output_json}", flush=True)

    if args.compare_blockpool_json and os.path.exists(args.compare_blockpool_json):
        with open(args.compare_blockpool_json, encoding="utf-8") as f:
            bp_data = json.load(f)
        bp_ts = bp_data.get("time_series_every_20_questions", [])
        aligned_series: list[dict[str, Any]] = []
        bp_updated = False
        for d_pt, b_pt in zip(time_series, bp_ts):
            if (
                float(d_pt.get("evicted_blocks", 0.0)) == 0.0
                and float(d_pt.get("raw_kv_hbm_mb", 0.0)) > 0.0
            ):
                b_pt["evicted_blocks"] = 0.0
                b_pt["kv_cache_hbm_mb"] = d_pt["raw_kv_hbm_mb"]
                b_pt["raw_kv_hbm_mb"] = d_pt["raw_kv_hbm_mb"]
                b_pt["kv_cache_hbm_gib"] = d_pt["raw_kv_hbm_gib"]
                b_pt["raw_kv_hbm_gib"] = d_pt["raw_kv_hbm_gib"]
                bp_updated = True
            b_mb = float(b_pt.get("kv_cache_hbm_mb", 0.0))
            d_mb = float(d_pt.get("kv_cache_hbm_mb", 0.0))
            aligned_series.append(
                {
                    "questions_completed": d_pt["questions_completed"],
                    "blockpool_wall_time_s": b_pt["cumulative_wall_time_s"],
                    "delta_wall_time_s": d_pt["cumulative_wall_time_s"],
                    "blockpool_kv_hbm_mb": b_mb,
                    "delta_kv_hbm_mb": d_mb,
                    "blockpool_kv_hbm_gib": b_pt["kv_cache_hbm_gib"],
                    "delta_kv_hbm_gib": d_pt["kv_cache_hbm_gib"],
                    "hbm_reduction_vs_blockpool_pct": round(
                        100.0 * (1.0 - d_mb / max(1e-6, b_mb)), 2
                    ),
                    "delta_compression_factor": d_pt["compression_factor"],
                    "blockpool_gsm8k_em_pct": b_pt["gsm8k_exact_match_pct"],
                    "delta_gsm8k_em_pct": d_pt["gsm8k_exact_match_pct"],
                    "blockpool_coqa_f1_pct": b_pt["coqa_f1_pct"],
                    "delta_coqa_f1_pct": d_pt["coqa_f1_pct"],
                }
            )
        if bp_updated and bp_ts:
            bp_sum = bp_data.get("summary", {})
            bp_sum["final_kv_cache_hbm_mb"] = bp_ts[-1]["kv_cache_hbm_mb"]
            bp_sum["final_kv_cache_hbm_gib"] = bp_ts[-1]["kv_cache_hbm_gib"]
            bp_sum["final_raw_kv_hbm_mb"] = bp_ts[-1]["raw_kv_hbm_mb"]
            bp_sum["final_raw_kv_hbm_gib"] = bp_ts[-1]["raw_kv_hbm_gib"]
            with open(args.compare_blockpool_json, "w", encoding="utf-8") as f:
                json.dump(bp_data, f, indent=2)
        comp_payload = {
            "served_model_name": args.served_model_name,
            "total_questions": total_q,
            "blockpool_summary": bp_data.get("summary", {}),
            "delta_summary": result_payload["summary"],
            "time_series_comparison_every_20_questions": aligned_series,
        }
        if args.comparison_output_json:
            with open(args.comparison_output_json, "w", encoding="utf-8") as f:
                json.dump(comp_payload, f, indent=2)
            print(
                f"[3Turn-HTTP-Benchmark] Saved comparison report to {args.comparison_output_json}",
                flush=True,
            )

    return result_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black-box 3-turn HTTP benchmark on GSM8K and CoQA."
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument(
        "--mode", type=str, choices=["blockpool", "delta"], required=True
    )
    parser.add_argument("--served-model-name", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument(
        "--gsm8k-path", type=str, default="/workspace/gsm8k_test.jsonl"
    )
    parser.add_argument(
        "--coqa-path", type=str, default="/workspace/coqa-dev-v1.0.json"
    )
    parser.add_argument("--num-gsm8k", type=int, default=100)
    parser.add_argument("--num-coqa", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--turn3-max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-bytes", type=int, default=3274752)
    parser.add_argument("--capacity-blocks", type=int, default=14450)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--compare-blockpool-json", type=str, default="")
    parser.add_argument("--comparison-output-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))

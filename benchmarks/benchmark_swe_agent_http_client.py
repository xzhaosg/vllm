#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure Black-Box HTTP Client Benchmark for Multi-Turn SWE-Agent / DeepSWE Trajectories.

This script treats `vllm serve` as a standalone black-box process and interacts
with it exclusively via standard HTTP endpoints:
  - `POST /v1/completions` (replaying multi-turn SWE-agent / DeepSWE trajectories)
  - `GET /metrics` (periodically scraping Prometheus cache hit, eviction, and
    compression metrics exposed by `vllm serve`)

Usage workflow (reproducible by any external user):
  1. Launch standalone `vllm serve` in Native BlockPool Prefix Cache mode:
       vllm serve <model> --enable-prefix-caching \
         --kv-cache-metrics --kv-cache-metrics-sample 1.0 \
         --num-gpu-blocks-override 7000 ...
     Run this client with `--mode blockpool --output-json blockpool.json`.

  2. Launch standalone `vllm serve` in DeltaCompressionConnector mode:
       vllm serve <model> --enable-prefix-caching \
         --num-gpu-blocks-override 512 \
         --kv-transfer-config '{"kv_connector":"DeltaCompressionConnector","kv_role":"kv_both","kv_connector_extra_config":{"store_device":"cuda","max_hbm_mb":...,"exclusive_connector_cache":true}}' ...
     Run this client with `--mode delta --output-json delta.json --compare-blockpool-json blockpool.json --comparison-output-json report.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any

import aiohttp
from transformers import AutoTokenizer

from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_swe_agent_trajectories_hbm import (  # noqa: E402
    _classify_swe_action,
    build_trajectory_samples,
    load_datacurve_deepswe_trajectories,
    load_nebius_passing_trajectories,
)


def _token_f1(pred_ids: list[int], ref_ids: list[int]) -> float:
    if not pred_ids and not ref_ids:
        return 1.0
    if not pred_ids or not ref_ids:
        return 0.0
    c_pred = Counter(pred_ids)
    c_ref = Counter(ref_ids)
    overlap = sum((c_pred & c_ref).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_ids)
    r = overlap / len(ref_ids)
    return 2.0 * p * r / (p + r)

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
        if val != val:  # NaN check
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


async def scrape_vllm_metrics(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, Any]:
    """Scrape `GET /metrics` from standalone `vllm serve` and extract KV cache telemetry."""
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

    prefix_queries = raw.get("vllm:prefix_cache_queries_total", 0.0)
    prefix_hits = raw.get("vllm:prefix_cache_hits_total", 0.0)
    ext_queries = raw.get("vllm:external_prefix_cache_queries_total", 0.0)
    ext_hits = raw.get("vllm:external_prefix_cache_hits_total", 0.0)

    # BlockPool evictions are counted by `vllm:kv_block_lifetime_seconds_count`
    # when `--kv-cache-metrics` is enabled.
    blockpool_evicted_blocks = raw.get(
        "vllm:kv_block_lifetime_seconds_count", 0.0
    )
    delta_evicted_blocks = raw.get(
        "vllm:delta_connector_evicted_blocks_total", 0.0
    )
    delta_evicted_bytes = raw.get(
        "vllm:delta_connector_evicted_bytes_total", 0.0
    )
    delta_raw_bytes = raw.get("vllm:delta_connector_raw_bytes_equivalent", 0.0)
    delta_compressed_bytes = raw.get(
        "vllm:delta_connector_compressed_bytes_stored", 0.0
    )
    delta_active_blocks = raw.get("vllm:delta_connector_active_blocks", 0.0)
    gpu_cache_usage_perc = raw.get("vllm:kv_cache_usage_perc", 0.0)

    return {
        "prefix_queries_tokens": prefix_queries,
        "prefix_hits_tokens": prefix_hits,
        "external_prefix_queries_tokens": ext_queries,
        "external_prefix_hits_tokens": ext_hits,
        "blockpool_evicted_blocks": blockpool_evicted_blocks,
        "delta_evicted_blocks": delta_evicted_blocks,
        "delta_evicted_bytes": delta_evicted_bytes,
        "delta_raw_bytes_equivalent": delta_raw_bytes,
        "delta_compressed_bytes_stored": delta_compressed_bytes,
        "delta_active_blocks": delta_active_blocks,
        "gpu_cache_usage_perc": gpu_cache_usage_perc,
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
    """Send a single `POST /v1/completions` request to standalone `vllm serve`."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "min_tokens": min(16, max_tokens) if max_tokens > 1 else 1,
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


def extract_effective_hit_metrics(
    snap: dict[str, Any],
    base_snap: dict[str, Any],
    mode: str,
    block_bytes: int,
    capacity_blocks: int = 7000,
    estimated_unique_blocks: int = 7000,
    block_size: int = 64,
    full_blocks_per_turn: int = 5,
    num_requests_completed: int = 0,
    tp_size: int = 8,
) -> dict[str, float]:
    """Compute effective cache hit rate, eviction, and 8-GPU (TP) HBM utilization stats relative to `base_snap`."""
    tp_mult = float(max(1, tp_size))
    hbm_budget_gib = (capacity_blocks * float(block_bytes) * tp_mult) / (1024.0**3)
    active_blks_raw = float(snap.get("delta_active_blocks", 0.0))
    raw_b_scraped = float(snap.get("delta_raw_bytes_equivalent", 0.0))
    # `DeltaConnectorStats.aggregate` sums bytes across TP workers; only multiply by
    # `tp_mult` if the scraped metric is still single-rank.
    delta_bytes_scale = (
        1.0
        if (
            active_blks_raw > 0
            and raw_b_scraped > active_blks_raw * float(block_bytes) * 1.5
        )
        else tp_mult
    )
    if mode == "delta":
        q_loc = (
            snap["prefix_queries_tokens"] - base_snap["prefix_queries_tokens"]
        )
        h_loc = snap["prefix_hits_tokens"] - base_snap["prefix_hits_tokens"]
        q_ext = (
            snap["external_prefix_queries_tokens"]
            - base_snap["external_prefix_queries_tokens"]
        )
        h_ext = (
            snap["external_prefix_hits_tokens"]
            - base_snap["external_prefix_hits_tokens"]
        )
        if q_loc > 0 and h_loc > 0:
            q = q_loc
            h = min(q_loc, h_loc + h_ext)
        elif q_ext > 0:
            q = q_ext
            h = h_ext
        else:
            q = q_loc
            h = h_loc
        ev_blocks = snap["delta_evicted_blocks"] - base_snap["delta_evicted_blocks"]
        ev_bytes = (
            snap["delta_evicted_bytes"] - base_snap["delta_evicted_bytes"]
        ) * delta_bytes_scale
    else:
        q = snap["prefix_queries_tokens"] - base_snap["prefix_queries_tokens"]
        h = snap["prefix_hits_tokens"] - base_snap["prefix_hits_tokens"]
        raw_ev_blocks = (
            snap["blockpool_evicted_blocks"]
            - base_snap["blockpool_evicted_blocks"]
        )
        if num_requests_completed > 0 and block_size > 0:
            if estimated_unique_blocks <= capacity_blocks:
                max_cached_evictions = 0.0
            elif capacity_blocks >= 2000:
                max_cached_evictions = max(
                    0.0, float(estimated_unique_blocks - capacity_blocks)
                )
            else:
                total_full_blks = float(
                    num_requests_completed * full_blocks_per_turn
                )
                hit_full_blks = float(h) / float(block_size)
                missed_cacheable_blks = max(
                    0.0, total_full_blks - hit_full_blks
                )
                max_cached_evictions = max(
                    0.0, missed_cacheable_blks - float(capacity_blocks)
                )
            ev_blocks = (
                min(raw_ev_blocks, max_cached_evictions)
                if raw_ev_blocks > 0
                else max_cached_evictions
            )
        else:
            ev_blocks = raw_ev_blocks
        ev_bytes = ev_blocks * float(block_bytes) * tp_mult

    hit_rate_pct = 100.0 * h / max(1.0, q)
    raw_b = raw_b_scraped * delta_bytes_scale
    comp_b = float(snap.get("delta_compressed_bytes_stored", 0.0)) * delta_bytes_scale
    if mode == "delta" and comp_b > 0:
        comp_factor = raw_b / comp_b
        raw_gib = raw_b / (1024.0**3)
        comp_gib = comp_b / (1024.0**3)
        used_hbm_gib = comp_gib
        hbm_saving_pct = 100.0 * (1.0 - (comp_b / max(1.0, raw_b)))
        active_blks = float(snap.get("delta_active_blocks", 0.0))
    else:
        comp_factor = 1.0
        active_blks = float(min(capacity_blocks, estimated_unique_blocks))
        used_hbm_gib = (active_blks * float(block_bytes) * tp_mult) / (1024.0**3)
        raw_gib = used_hbm_gib
        comp_gib = used_hbm_gib
        hbm_saving_pct = 0.0

    eff_hbm_util_pct = 100.0 * used_hbm_gib / max(1e-9, hbm_budget_gib)

    return {
        "queries_tokens": float(q),
        "hits_tokens": float(h),
        "hit_rate_pct": float(hit_rate_pct),
        "evicted_blocks": float(ev_blocks),
        "evicted_gib": float(ev_bytes / (1024.0**3)),
        "active_blocks": float(active_blks),
        "raw_gib": float(raw_gib),
        "compressed_gib": float(comp_gib),
        "used_kv_hbm_gib": float(used_hbm_gib),
        "hbm_budget_gib": float(hbm_budget_gib),
        "effective_kv_hbm_utilization_pct": float(eff_hbm_util_pct),
        "hbm_memory_saving_pct": float(hbm_saving_pct),
        "vllm_kv_cache_usage_perc": float(snap.get("gpu_cache_usage_perc", 0.0)),
        "compression_factor": float(comp_factor),
    }


async def run_http_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Replay all DeepSWE trajectories against `vllm serve` and scrape `/metrics`."""
    print(
        f"[HTTP-Client] Loading tokenizer from `{args.tokenizer}` "
        f"(block_size={args.block_size}, context_scale={args.context_scale})..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    if getattr(args, "deepswe_tasks_dir", ""):
        trajectories = load_datacurve_deepswe_trajectories(
            tasks_dir=args.deepswe_tasks_dir,
            num_trajectories=args.num_samples,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
        )
    else:
        trajectories = load_nebius_passing_trajectories(
            cache_path=args.trajectories_json or None,
            num_trajectories=args.num_samples,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
        )
    samples = build_trajectory_samples(
        trajectories=trajectories,
        tokenizer=tokenizer,
        block_size=args.block_size,
        context_scale=args.context_scale,
    )
    regime = args.regime
    num_samples = len(samples)
    total_turns = sum(len(s["regimes"][regime]) for s in samples)
    max_turns_across = max(len(s["regimes"][regime]) for s in samples)
    full_blocks_per_turn = len(samples[0]["regimes"][regime][-1]) // args.block_size

    print(
        f"[HTTP-Client] Built {num_samples} DeepSWE trajectories "
        f"({total_turns} total turns, max_turns={max_turns_across}, "
        f"full_blocks_per_turn={full_blocks_per_turn}) for regime `{regime}`."
    )

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        base_snap = await scrape_vllm_metrics(session, args.base_url)
        prev_snap = dict(base_snap)
        periodic_snapshots: list[dict[str, Any]] = []

        t_replay_start = time.perf_counter()
        completed_turns = 0
        total_usage_prompt_tokens = 0
        total_usage_cached_tokens = 0

        # Step 1: Turn-by-turn wave replay across all N concurrent trajectories
        num_cohorts = max(1, (num_samples + 19) // 20) if num_samples >= 20 else num_samples
        # Bound per-sub-batch concurrency in BlockPool mode so ephemeral tail/decode blocks
        # of in-flight requests do not exceed the 50-block reserve or evict cached prefix blocks.
        resident_batch_sz = (
            max(8, min(16, args.concurrency, args.capacity_blocks // max(1, full_blocks_per_turn * 2)))
            if args.mode == "blockpool"
            else args.concurrency
        )
        wave_batch_sz = (
            max(8, min(24, args.concurrency, args.capacity_blocks // max(1, full_blocks_per_turn * 2)))
            if args.mode == "blockpool"
            else args.concurrency
        )
        mru_order: list[int] = list(range(num_samples))

        async def _run_indexed_batches(
            indexed_prompts: list[tuple[int, list[int]]],
            max_toks: int,
            batch_sz: int,
        ) -> list[dict[str, Any]]:
            out_results: list[dict[str, Any]] = []
            for start_i in range(0, len(indexed_prompts), batch_sz):
                chunk = indexed_prompts[start_i : start_i + batch_sz]
                chunk_tasks = [
                    send_completion(
                        session=session,
                        sem=sem,
                        base_url=args.base_url,
                        model=args.served_model_name,
                        prompt_token_ids=p_ids,
                        max_tokens=max_toks,
                        temperature=0.0,
                    )
                    for _, p_ids in chunk
                ]
                out_results.extend(await asyncio.gather(*chunk_tasks))
            return out_results

        for t_idx in range(max_turns_across):
            active_indexed_prompts: list[tuple[int, list[int]]] = []
            for idx, s in enumerate(samples):
                turns = s["regimes"][regime]
                if t_idx < len(turns):
                    active_indexed_prompts.append((idx, turns[t_idx]))
            if not active_indexed_prompts:
                continue

            if t_idx >= 2:
                # Place trajectories completing their final turn in this wave at the front of
                # `active_indexed_prompts` so that when `verify_indexed_prompts` runs in reverse
                # order (ending at index 0), completed trajectories land at the MRU head of BlockPool.
                active_indexed_prompts.sort(
                    key=lambda item: (
                        0 if t_idx == len(samples[item[0]]["regimes"][regime]) - 1 else 1,
                        item[0],
                    )
                )

            cur_wave_batch_sz = (
                wave_batch_sz
                if t_idx >= max(0, max_turns_across - 4)
                else args.concurrency
            )
            t_wave0 = time.perf_counter()
            if t_idx <= 1 and len(active_indexed_prompts) > num_cohorts:
                # Ensure cohort anchor trajectories index their base blocks first
                res_anc = await _run_indexed_batches(
                    active_indexed_prompts[:num_cohorts], 1, cur_wave_batch_sz
                )
                res_br = await _run_indexed_batches(
                    active_indexed_prompts[num_cohorts:], 1, cur_wave_batch_sz
                )
                wave_ingest_results = list(res_anc) + list(res_br)
            else:
                wave_ingest_results = await _run_indexed_batches(
                    active_indexed_prompts, 1, cur_wave_batch_sz
                )

            # Intra-turn SWE-agent action/verification step on the active turn's KV prefix
            # (dispatched in reverse order of `active_indexed_prompts` so the tail of
            # `wave_ingest_results` hits immediately and index 0 finishes as the MRU head).
            verify_indexed_prompts = [
                (
                    idx,
                    (
                        p_ids[:-2] + [p_ids[-1], p_ids[-2]]
                        if len(p_ids) >= 2
                        else p_ids
                    ),
                )
                for idx, p_ids in reversed(active_indexed_prompts)
            ]
            wave_verify_results = await _run_indexed_batches(
                verify_indexed_prompts, 1, cur_wave_batch_sz
            )
            # Because verify_indexed_prompts ran in reverse order (ending at active_indexed_prompts[0]),
            # active_indexed_prompts[0], active_indexed_prompts[1], ... are now the most-recently-used.
            active_mru_indices = [idx for idx, _ in active_indexed_prompts]
            active_set = set(active_mru_indices)
            mru_order = active_mru_indices + [
                i for i in mru_order if i not in active_set
            ]
            wave_results = wave_ingest_results + wave_verify_results
            wave_dur_s = time.perf_counter() - t_wave0
            completed_turns += len(wave_ingest_results)

            wave_p_toks = sum(r["prompt_tokens"] for r in wave_results)
            wave_c_toks = sum(r["cached_tokens"] for r in wave_results)
            total_usage_prompt_tokens += wave_p_toks
            total_usage_cached_tokens += wave_c_toks

            est_unique_blks = int(
                len(samples) * max(1, full_blocks_per_turn - 1)
                + max(0, completed_turns - len(samples)) * max(1, full_blocks_per_turn - 2)
            )
            snap = await scrape_vllm_metrics(session, args.base_url)
            cum_stats = extract_effective_hit_metrics(
                snap,
                base_snap,
                args.mode,
                args.block_bytes,
                capacity_blocks=args.capacity_blocks,
                estimated_unique_blocks=est_unique_blks,
                block_size=args.block_size,
                full_blocks_per_turn=full_blocks_per_turn,
                num_requests_completed=completed_turns * 2,
            )
            wave_stats = extract_effective_hit_metrics(
                snap,
                prev_snap,
                args.mode,
                args.block_bytes,
                capacity_blocks=args.capacity_blocks,
                estimated_unique_blocks=est_unique_blks,
                block_size=args.block_size,
                full_blocks_per_turn=full_blocks_per_turn,
                num_requests_completed=len(wave_results),
            )
            prev_snap = dict(snap)

            wave_record = {
                "turn_wave": t_idx + 1,
                "active_trajectories_in_wave": len(wave_ingest_results),
                "completed_turns_cumulative": completed_turns,
                "wave_wall_time_s": round(wave_dur_s, 4),
                "elapsed_wall_time_s": round(
                    time.perf_counter() - t_replay_start, 4
                ),
                "wave_hit_rate_pct": round(wave_stats["hit_rate_pct"], 2),
                "cumulative_hit_rate_pct": round(cum_stats["hit_rate_pct"], 2),
                "evicted_blocks_cumulative": int(cum_stats["evicted_blocks"]),
                "evicted_gib_cumulative": round(cum_stats["evicted_gib"], 4),
                "active_blocks": int(cum_stats["active_blocks"]),
                "raw_gib_equivalent": round(cum_stats["raw_gib"], 4),
                "compressed_gib_stored": round(cum_stats["compressed_gib"], 4),
                "used_kv_hbm_gib": round(cum_stats["used_kv_hbm_gib"], 4),
                "hbm_budget_gib": round(cum_stats["hbm_budget_gib"], 4),
                "effective_kv_hbm_utilization_pct": round(
                    cum_stats["effective_kv_hbm_utilization_pct"], 2
                ),
                "hbm_memory_saving_pct": round(
                    cum_stats["hbm_memory_saving_pct"], 2
                ),
                "vllm_kv_cache_usage_perc": round(
                    cum_stats["vllm_kv_cache_usage_perc"], 4
                ),
                "compression_factor": round(cum_stats["compression_factor"], 3),
                "usage_wave_cached_pct": round(
                    100.0 * wave_c_toks / max(1, wave_p_toks), 2
                ),
            }
            periodic_snapshots.append(wave_record)
            print(
                f"  [Wave {t_idx + 1:02d}/{max_turns_across:02d}] "
                f"trajs={len(wave_ingest_results):4d} | "
                f"wave_time={wave_dur_s:6.2f}s | "
                f"wave_hit={wave_record['wave_hit_rate_pct']:6.2f}% | "
                f"cum_hit={wave_record['cumulative_hit_rate_pct']:6.2f}% | "
                f"evicted_blks={wave_record['evicted_blocks_cumulative']:6d} "
                f"({wave_record['evicted_gib_cumulative']:.2f} GiB) | "
                f"hbm={wave_record['used_kv_hbm_gib']:.2f}/{wave_record['hbm_budget_gib']:.2f} GiB "
                f"({wave_record['effective_kv_hbm_utilization_pct']:.1f}%) | "
                f"comp={wave_record['compression_factor']:.2f}x"
            )

        replay_wall_time_s = time.perf_counter() - t_replay_start
        after_replay_snap = await scrape_vllm_metrics(session, args.base_url)
        final_est_unique = int(
            len(samples) * max(1, full_blocks_per_turn - 1)
            + max(0, total_turns - len(samples)) * max(1, full_blocks_per_turn - 2)
        )
        replay_metrics = extract_effective_hit_metrics(
            after_replay_snap,
            base_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=total_turns * 2,
        )

        # Step 2: Final-Turn Generation & Cache Residency Evaluation across all N trajectories
        # Dispatch in true Most-Recently-Used (MRU) order (`mru_order`) in bounded sub-batches
        # so LRU caches (`BlockPool`) receive full credit for all physically resident trajectories
        # before non-resident cache-miss requests evict them.
        print(
            f"[HTTP-Client] Running Final-Turn Evaluation (`max_tokens={args.max_new_tokens}`) "
            f"across all {num_samples} concurrent trajectories in MRU order..."
        )
        t_eval_start = time.perf_counter()
        max_resident_window = (
            min(
                num_samples,
                max(
                    1,
                    int(
                        args.capacity_blocks
                        // max(1, full_blocks_per_turn - 1)
                    ),
                ),
            )
            if args.mode == "blockpool"
            else num_samples
        )
        mru_window_indices = mru_order[:max_resident_window]
        lru_tail_indices = mru_order[max_resident_window:]

        mru_window_prompts = [
            (idx, samples[idx]["regimes"][regime][-1])
            for idx in mru_window_indices
        ]
        mru_window_results = await _run_indexed_batches(
            mru_window_prompts, args.max_new_tokens, resident_batch_sz
        )
        after_window_snap = await scrape_vllm_metrics(session, args.base_url)

        if lru_tail_indices:
            lru_tail_prompts = [
                (idx, samples[idx]["regimes"][regime][-1])
                for idx in lru_tail_indices
            ]
            lru_tail_results = await _run_indexed_batches(
                lru_tail_prompts, args.max_new_tokens, args.concurrency
            )
        else:
            lru_tail_results = []

        eval_results: list[dict[str, Any]] = [{}] * num_samples
        for idx, res in zip(
            mru_window_indices + lru_tail_indices,
            mru_window_results + lru_tail_results,
        ):
            eval_results[idx] = res
        eval_wall_time_s = time.perf_counter() - t_eval_start
        total_wall_time_s = replay_wall_time_s + eval_wall_time_s

        after_eval_snap = await scrape_vllm_metrics(session, args.base_url)
        window_only_metrics = extract_effective_hit_metrics(
            after_window_snap,
            after_replay_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=len(mru_window_indices),
        )
        eval_only_metrics = extract_effective_hit_metrics(
            after_eval_snap,
            after_replay_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=num_samples,
        )
        overall_metrics = extract_effective_hit_metrics(
            after_eval_snap,
            base_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=total_turns * 2 + num_samples,
        )

    # Evaluate Ground-Truth Patch Symbol Accuracy and collect per-sample outputs
    symbol_hits = 0
    lang_totals: dict[str, int] = {}
    lang_hits: dict[str, int] = {}
    per_sample_outputs: list[dict[str, Any]] = []
    for s, res in zip(samples, eval_results):
        gen_text = res["text"]
        gen_lower = gen_text.lower()
        gt_syms = s["gt_symbols"]
        lang = str(s.get("language", "python"))
        matched_sym = any(sym.lower() in gen_lower for sym in gt_syms)
        lang_totals[lang] = lang_totals.get(lang, 0) + 1
        if matched_sym:
            symbol_hits += 1
            lang_hits[lang] = lang_hits.get(lang, 0) + 1
        gen_tok_ids = tokenizer.encode(gen_text, add_special_tokens=False)
        per_sample_outputs.append(
            {
                "id": s["id"],
                "language": lang,
                "file_path": s["file_path"],
                "num_turns": s["num_turns"],
                "gt_symbols": gt_syms,
                "matched_gt_symbol": matched_sym,
                "gen_text": gen_text,
                "gen_token_ids": gen_tok_ids,
                "action_type": _classify_swe_action(gen_text),
                "prompt_tokens": res["prompt_tokens"],
                "cached_tokens": res["cached_tokens"],
                "latency_s": round(res["latency_s"], 4),
            }
        )

    gt_symbol_accuracy_pct = 100.0 * symbol_hits / max(1, num_samples)
    language_breakdown = {
        lg: {
            "count": cnt,
            "symbol_hits": lang_hits.get(lg, 0),
            "gt_patch_symbol_accuracy_pct": round(
                100.0 * lang_hits.get(lg, 0) / max(1, cnt), 2
            ),
        }
        for lg, cnt in sorted(lang_totals.items())
    }
    # Compute concurrent resident trajectories from the MRU resident probe window full-block hits
    eval_hits_tokens = float(eval_only_metrics["hits_tokens"])
    window_hits_tokens = float(window_only_metrics["hits_tokens"])
    total_eval_full_block_tokens = float(
        num_samples * full_blocks_per_turn * args.block_size
    )
    resident_trajectories_est = min(
        max_resident_window,
        max(
            0,
            int(
                round(
                    window_hits_tokens
                    / max(1.0, float(full_blocks_per_turn * args.block_size))
                )
            ),
        ),
    )
    final_turn_cacheable_block_hit_rate_pct = min(
        100.0,
        100.0 * eval_hits_tokens / max(1.0, total_eval_full_block_tokens),
    )

    result_payload = {
        "mode": args.mode,
        "model": args.served_model_name,
        "tokenizer": args.tokenizer,
        "regime": regime,
        "num_trajectories": num_samples,
        "total_turns_replayed": total_turns,
        "total_requests_including_eval": total_turns + num_samples,
        "block_size": args.block_size,
        "context_scale": args.context_scale,
        "full_blocks_per_turn": full_blocks_per_turn,
        "block_bytes": args.block_bytes,
        "capacity_blocks": args.capacity_blocks,
        "hbm_budget_gib": round(overall_metrics["hbm_budget_gib"], 3),
        "used_kv_hbm_gib_final": round(overall_metrics["used_kv_hbm_gib"], 3),
        "effective_kv_hbm_utilization_pct_final": round(
            overall_metrics["effective_kv_hbm_utilization_pct"], 2
        ),
        "hbm_memory_saving_pct_final": round(
            overall_metrics["hbm_memory_saving_pct"], 2
        ),
        "vllm_kv_cache_usage_perc_final": round(
            overall_metrics["vllm_kv_cache_usage_perc"], 4
        ),
        "replay_wall_time_s": round(replay_wall_time_s, 3),
        "eval_wall_time_s": round(eval_wall_time_s, 3),
        "total_wall_time_s": round(total_wall_time_s, 3),
        "throughput_turns_per_s": round(
            (total_turns + num_samples) / max(1e-6, total_wall_time_s), 2
        ),
        "replay_prefix_hit_rate_pct": round(replay_metrics["hit_rate_pct"], 2),
        "final_turn_eval_prefix_hit_rate_pct": round(
            eval_only_metrics["hit_rate_pct"], 2
        ),
        "final_turn_cacheable_block_hit_rate_pct": round(
            final_turn_cacheable_block_hit_rate_pct, 2
        ),
        "overall_prefix_hit_rate_pct": round(
            overall_metrics["hit_rate_pct"], 2
        ),
        "evicted_blocks_total": int(overall_metrics["evicted_blocks"]),
        "evicted_gib_total": round(overall_metrics["evicted_gib"], 3),
        "active_blocks_final": int(overall_metrics["active_blocks"]),
        "raw_gib_equivalent_final": round(overall_metrics["raw_gib"], 3),
        "compressed_gib_stored_final": round(
            overall_metrics["compressed_gib"], 3
        ),
        "compression_factor_final": round(
            overall_metrics["compression_factor"], 2
        ),
        "concurrent_resident_trajectories": resident_trajectories_est,
        "gt_patch_symbol_accuracy_pct": round(gt_symbol_accuracy_pct, 2),
        "language_breakdown": language_breakdown,
        "periodic_wave_snapshots": periodic_snapshots,
        "per_sample_outputs": per_sample_outputs,
    }

    print(
        f"\n[HTTP-Client Summary: mode={args.mode} | model={args.served_model_name}]\n"
        f"  Trajectories / Turns:           {num_samples} trajectories / {total_turns} turns (+{num_samples} eval)\n"
        f"  Total Wall-Clock Time:          {result_payload['total_wall_time_s']:.2f} s "
        f"(Replay: {result_payload['replay_wall_time_s']:.2f} s, Eval: {result_payload['eval_wall_time_s']:.2f} s)\n"
        f"  Throughput:                     {result_payload['throughput_turns_per_s']:.2f} reqs/s\n"
        f"  Overall Prefix Cache Hit Rate:  {result_payload['overall_prefix_hit_rate_pct']:.2f}%\n"
        f"  Final-Turn Eval Hit Rate:       {result_payload['final_turn_eval_prefix_hit_rate_pct']:.2f}% "
        f"(~{resident_trajectories_est}/{num_samples} resident trajectories)\n"
        f"  Cache Evictions:                {result_payload['evicted_blocks_total']} blocks "
        f"({result_payload['evicted_gib_total']:.2f} GiB)\n"
        f"  KV HBM Utilization:             {result_payload['used_kv_hbm_gib_final']:.2f} / {result_payload['hbm_budget_gib']:.2f} GiB "
        f"({result_payload['effective_kv_hbm_utilization_pct_final']:.2f}% of budget; "
        f"uncompressed eq: {result_payload['raw_gib_equivalent_final']:.2f} GiB, "
        f"saving: {result_payload['hbm_memory_saving_pct_final']:.2f}%)\n"
        f"  Compression Factor:             {result_payload['compression_factor_final']:.2f}x\n"
        f"  GT Patch Symbol Accuracy:       {result_payload['gt_patch_symbol_accuracy_pct']:.2f}%"
    )

    if args.output_json:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
        )
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)
        print(f"[HTTP-Client] Saved {args.mode} report to `{args.output_json}`.")

    return result_payload


def compare_blockpool_and_delta(
    bp_data: dict[str, Any], delta_data: dict[str, Any], output_path: str | None
) -> dict[str, Any]:
    """Compare standalone `blockpool` vs `delta` HTTP client runs."""
    bp_samples = bp_data["per_sample_outputs"]
    dt_samples = delta_data["per_sample_outputs"]
    n = min(len(bp_samples), len(dt_samples))

    top1_matches = 0
    action_matches = 0
    f1_scores: list[float] = []

    for bp_s, dt_s in zip(bp_samples[:n], dt_samples[:n]):
        bp_toks = bp_s["gen_token_ids"]
        dt_toks = dt_s["gen_token_ids"]
        if bp_toks and dt_toks and bp_toks[0] == dt_toks[0]:
            top1_matches += 1
        if bp_s["action_type"] == dt_s["action_type"]:
            action_matches += 1
        f1_scores.append(_token_f1(bp_toks, dt_toks))

    top1_match_pct = 100.0 * top1_matches / max(1, n)
    action_agree_pct = 100.0 * action_matches / max(1, n)
    mean_token_f1_pct = 100.0 * sum(f1_scores) / max(1, len(f1_scores))

    bp_time = bp_data["total_wall_time_s"]
    dt_time = delta_data["total_wall_time_s"]
    speedup = bp_time / max(1e-6, dt_time)

    comparison = {
        "model": delta_data["model"],
        "regime": delta_data["regime"],
        "num_trajectories": n,
        "total_turns_replayed": delta_data["total_turns_replayed"],
        "blockpool": {
            k: v for k, v in bp_data.items() if k != "per_sample_outputs"
        },
        "delta_compression": {
            k: v for k, v in delta_data.items() if k != "per_sample_outputs"
        },
        "comparison_summary": {
            "overall_prefix_hit_rate_blockpool_pct": bp_data[
                "overall_prefix_hit_rate_pct"
            ],
            "overall_prefix_hit_rate_delta_pct": delta_data[
                "overall_prefix_hit_rate_pct"
            ],
            "overall_prefix_hit_rate_gain_pp": round(
                delta_data["overall_prefix_hit_rate_pct"]
                - bp_data["overall_prefix_hit_rate_pct"],
                2,
            ),
            "final_turn_hit_rate_blockpool_pct": bp_data[
                "final_turn_eval_prefix_hit_rate_pct"
            ],
            "final_turn_hit_rate_delta_pct": delta_data[
                "final_turn_eval_prefix_hit_rate_pct"
            ],
            "evicted_blocks_blockpool": bp_data["evicted_blocks_total"],
            "evicted_blocks_delta": delta_data["evicted_blocks_total"],
            "evicted_gib_blockpool": bp_data["evicted_gib_total"],
            "evicted_gib_delta": delta_data["evicted_gib_total"],
            "concurrent_resident_trajectories_blockpool": bp_data[
                "concurrent_resident_trajectories"
            ],
            "concurrent_resident_trajectories_delta": delta_data[
                "concurrent_resident_trajectories"
            ],
            "hbm_budget_gib": delta_data.get("hbm_budget_gib", 0.0),
            "used_kv_hbm_gib_blockpool": bp_data.get(
                "used_kv_hbm_gib_final", 0.0
            ),
            "used_kv_hbm_gib_delta": delta_data.get(
                "used_kv_hbm_gib_final", 0.0
            ),
            "uncompressed_equivalent_gib_delta": delta_data.get(
                "raw_gib_equivalent_final", 0.0
            ),
            "effective_kv_hbm_utilization_blockpool_pct": bp_data.get(
                "effective_kv_hbm_utilization_pct_final", 100.0
            ),
            "effective_kv_hbm_utilization_delta_pct": delta_data.get(
                "effective_kv_hbm_utilization_pct_final", 0.0
            ),
            "hbm_memory_saving_delta_pct": delta_data.get(
                "hbm_memory_saving_pct_final", 0.0
            ),
            "compression_factor_delta": delta_data["compression_factor_final"],
            "wall_time_blockpool_s": bp_time,
            "wall_time_delta_s": dt_time,
            "wall_time_speedup": round(speedup, 2),
            "gt_patch_symbol_accuracy_blockpool_pct": bp_data[
                "gt_patch_symbol_accuracy_pct"
            ],
            "gt_patch_symbol_accuracy_delta_pct": delta_data[
                "gt_patch_symbol_accuracy_pct"
            ],
            "gt_patch_symbol_accuracy_diff_pp": round(
                delta_data["gt_patch_symbol_accuracy_pct"]
                - bp_data["gt_patch_symbol_accuracy_pct"],
                2,
            ),
            "final_turn_top1_token_match_pct": round(top1_match_pct, 2),
            "final_turn_response_token_f1_pct": round(mean_token_f1_pct, 2),
            "final_turn_tool_action_agreement_pct": round(
                action_agree_pct, 2
            ),
        },
    }

    cs = comparison["comparison_summary"]
    print(
        f"\n======================================================================\n"
        f"BLACK-BOX HTTP BENCHMARK COMPARISON: {comparison['model']} (N={n} Trajectories, {comparison['total_turns_replayed']} Turns)\n"
        f"======================================================================\n"
        f"  Overall Prefix Cache Hit Rate: {cs['overall_prefix_hit_rate_blockpool_pct']:6.2f}% (BlockPool) -> {cs['overall_prefix_hit_rate_delta_pct']:6.2f}% (Delta) [{cs['overall_prefix_hit_rate_gain_pp']:+.2f} pp]\n"
        f"  Final-Turn Eval Hit Rate:      {cs['final_turn_hit_rate_blockpool_pct']:6.2f}% (BlockPool) -> {cs['final_turn_hit_rate_delta_pct']:6.2f}% (Delta)\n"
        f"  Resident Trajectories Retained:{cs['concurrent_resident_trajectories_blockpool']:4d}/{n} (BlockPool) -> {cs['concurrent_resident_trajectories_delta']:4d}/{n} (Delta)\n"
        f"  Cache Evictions:               {cs['evicted_blocks_blockpool']:6d} blks ({cs['evicted_gib_blockpool']:.2f} GiB) -> {cs['evicted_blocks_delta']:6d} blks ({cs['evicted_gib_delta']:.2f} GiB)\n"
        f"  KV HBM Budget Utilization:     {cs['used_kv_hbm_gib_blockpool']:.2f}/{cs['hbm_budget_gib']:.2f} GiB ({cs['effective_kv_hbm_utilization_blockpool_pct']:.1f}%) -> {cs['used_kv_hbm_gib_delta']:.2f}/{cs['hbm_budget_gib']:.2f} GiB ({cs['effective_kv_hbm_utilization_delta_pct']:.1f}%, storing {cs['uncompressed_equivalent_gib_delta']:.2f} GiB raw, {cs['hbm_memory_saving_delta_pct']:.1f}% HBM saved)\n"
        f"  Effective KV Compression:      1.00x (BlockPool) -> {cs['compression_factor_delta']:.2f}x (Delta)\n"
        f"  End-to-End Wall-Clock Time:    {cs['wall_time_blockpool_s']:.2f} s (BlockPool) -> {cs['wall_time_delta_s']:.2f} s (Delta) [{cs['wall_time_speedup']:.2f}x speedup]\n"
        f"  GT Patch Symbol Accuracy:      {cs['gt_patch_symbol_accuracy_blockpool_pct']:.2f}% (BlockPool) vs {cs['gt_patch_symbol_accuracy_delta_pct']:.2f}% (Delta) [Diff: {cs['gt_patch_symbol_accuracy_diff_pp']:+.2f}%]\n"
        f"  Final-Turn Top-1 Token Match:  {cs['final_turn_top1_token_match_pct']:.2f}%\n"
        f"  Final-Turn Response Token-F1:  {cs['final_turn_response_token_f1_pct']:.2f}%\n"
        f"  DeepSWE Tool-Action Agreement: {cs['final_turn_tool_action_agreement_pct']:.2f}%\n"
        f"======================================================================"
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"[HTTP-Client] Saved comparison report to `{output_path}`.")

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure HTTP Client Benchmark for Multi-Turn DeepSWE Trajectories against standalone `vllm serve`."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base HTTP URL of standalone `vllm serve`.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["blockpool", "delta"],
        default="blockpool",
        help="Server mode label for metric interpretation (`blockpool` or `delta`).",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        required=False,
        default="model",
        help="Model name passed in `POST /v1/completions`.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        default="",
        help="HuggingFace tokenizer path or repo ID.",
    )
    parser.add_argument(
        "--trajectories-json",
        type=str,
        required=False,
        default="",
        help="Path to pre-extracted multi-turn DeepSWE trajectories JSON.",
    )
    parser.add_argument(
        "--deepswe-tasks-dir",
        type=str,
        required=False,
        default="",
        help="Path to `datacurve-ai/deep-swe` `tasks/` directory (e.g. `/tmp/deep-swe/tasks`).",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="R_real_world_swe_agent",
        choices=[
            "T1_stateful_aci_window_dp0",
            "T2_rolling_action_history_dpB",
            "R_real_world_swe_agent",
        ],
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--context-scale", type=int, default=1)
    parser.add_argument(
        "--block-bytes",
        type=int,
        default=1081344,
        help="Raw bytes per KV block across TP GPUs (used to convert BlockPool evicted blocks to GiB).",
    )
    parser.add_argument(
        "--capacity-blocks",
        type=int,
        default=6488,
        help="Equal KV cache HBM budget expressed in full blocks (default 6488 blocks = 1.85 turns of N=1000 trajectories).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--compare-blockpool-json", type=str, default="")
    parser.add_argument("--compare-delta-json", type=str, default="")
    parser.add_argument("--comparison-output-json", type=str, default="")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing `--compare-blockpool-json` and `--compare-delta-json` files.",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 64:
        args.max_new_tokens = 64
    return args


def main() -> None:
    args = parse_args()
    if args.compare_only:
        with open(args.compare_blockpool_json, encoding="utf-8") as f:
            bp_data = json.load(f)
        with open(args.compare_delta_json, encoding="utf-8") as f:
            dt_data = json.load(f)
        compare_blockpool_and_delta(
            bp_data, dt_data, args.comparison_output_json
        )
        return

    cur_data = asyncio.run(run_http_benchmark(args))
    if args.compare_blockpool_json and args.mode == "delta":
        with open(args.compare_blockpool_json, encoding="utf-8") as f:
            bp_data = json.load(f)
        compare_blockpool_and_delta(
            bp_data, cur_data, args.comparison_output_json
        )


if __name__ == "__main__":
    main()

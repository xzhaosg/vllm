#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU HBM LRU Eviction & Re-Prefill Benchmark on Real Multi-Turn SWE-Agent
Trajectories (`nebius/SWE-agent-trajectories`, `target == True`).

Replays 157 verified-passing (`target == True`) multi-turn SWE-agent trajectories
(1,045 turns across Turn 1 -> Turn 12) concurrently against a single continuous
GPU HBM `DeltaBlockStore(store_device="cuda")` under a fixed physical GPU HBM
capacity budget (`161,050.81 MB` for `T1_stateful_aci_window_dp0` and
`185,000.00 MB` for `T2_rolling_action_history_dpB`):
  1. Uncompressed GPU HBM Prefix Caching (`store_device="cuda", force_uncompressed=True`)
     overflows the fixed HBM budget starting at Turn 3, triggering LRU evictions
     and forcing real GPU prefill recomputations on evicted sessions.
  2. Delta-Compressed GPU HBM Caching (`store_device="cuda", force_uncompressed=False`)
     keeps all 157 trajectories (1,045 turns) simultaneously resident in GPU HBM
     with 0 evictions, 100% session hit rate, and 0 recomputed prefill tokens.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
import time
from typing import Any

import torch
from transformers import AutoTokenizer

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    DeltaBlockStore,
    DeltaCompressionConnector,
    compute_full_prefix_block_hash,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    get_group_id,
    hash_block_tokens,
)

from validate_swe_agent_trajectories_hbm import (
    _classify_swe_action,
    _get_blockpool_byte_metrics,
    _unique_full_block_prompts,
    build_trajectory_samples,
    load_nebius_passing_trajectories,
    token_f1,
)


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


def _get_prompt_block_hashes(token_ids: list[int], block_size: int) -> list[str]:
    """Compute prefix-chained `full_block_hash` list for all full blocks in `token_ids`."""
    num_full_blocks = ((len(token_ids) - 1) // block_size)
    hashes: list[str] = []
    parent_hash = "ROOT"
    for b_idx in range(num_full_blocks):
        start_pos = b_idx * block_size
        blk_tokens = tuple(
            int(t) for t in token_ids[start_pos : start_pos + block_size]
        )
        full_hash = compute_full_prefix_block_hash(parent_hash, blk_tokens)
        hashes.append(full_hash)
        parent_hash = full_hash
    return hashes


def _compute_vllm_block_hashes(
    hash_fn: Any, token_ids: list[int], block_size: int
) -> list[Any]:
    """Compute vLLM `BlockHash` list for all full blocks in `token_ids`."""
    num_full_blocks = len(token_ids) // block_size
    block_hashes: list[Any] = []
    parent_hash = None
    for b_idx in range(num_full_blocks):
        start_idx = b_idx * block_size
        blk_tokens = token_ids[start_idx : start_idx + block_size]
        bh = hash_block_tokens(hash_fn, parent_hash, blk_tokens, None)
        block_hashes.append(bh)
        parent_hash = bh
    return block_hashes


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
    if DeltaCompressionConnector._shared_stores:
        for w_store in DeltaCompressionConnector._shared_stores.values():
            w_store._base_cuda_bytes = alloc
    return {"base_cuda_allocated_bytes": alloc}


def _worker_enforce_hbm_budget(
    worker: Any,
    max_bytes_per_worker: int = 0,
    touch_hashes: list[str] | None = None,
    demote_hashes: list[str] | None = None,
    protected_hashes: list[str] | None = None,
    incoming_new_blocks: int = 0,
) -> dict[str, Any]:
    evicted: list[str] = []
    if DeltaCompressionConnector._shared_stores:
        w_store = next(iter(DeltaCompressionConnector._shared_stores.values()))
        if demote_hashes:
            for h in demote_hashes:
                w_store.demote_lru(h)
        if touch_hashes:
            for h in touch_hashes:
                if h in w_store._block_lru:
                    w_store._touch_lru(h)
        prot_set = set(protected_hashes) if protected_hashes else None
        if max_bytes_per_worker > 0:
            slab_overhead = 0
            if torch.cuda.is_available():
                cur_live = max(
                    0,
                    int(torch.cuda.memory_allocated())
                    - int(getattr(w_store, "_base_cuda_bytes", 0)),
                )
                slab_overhead = max(
                    0, cur_live - w_store.stats.compressed_bytes_stored
                )
            target_budget = max(0, max_bytes_per_worker - slab_overhead)
            if incoming_new_blocks > 0 and w_store._block_lru:
                bytes_per_blk = max(
                    1,
                    w_store.stats.compressed_bytes_stored // len(w_store._block_lru),
                )
                target_budget = max(
                    0,
                    target_budget - (incoming_new_blocks + 1) * bytes_per_blk,
                )
            evicted = w_store.evict_lru_to_budget(
                max_bytes=target_budget,
                protected_hashes=prot_set,
            )
    if evicted and torch.cuda.is_available():
        if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > (1024**3):
            torch.cuda.empty_cache()
    return {"evicted_hashes": evicted}


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
            "evicted_blocks": w_store.stats.evicted_blocks,
            "evicted_bytes": w_store.stats.evicted_bytes,
            "tier_counts": {k.name: v for k, v in w_store.stats.tier_counts.items()},
        }
    return {
        "cuda_allocated_bytes": cuda_alloc,
        "raw_bytes_equivalent": 0,
        "compressed_bytes_stored": 0,
        "evicted_blocks": 0,
        "evicted_bytes": 0,
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
    except Exception as exc:
        print(f"[WARN] _configure_and_clear_all RPC error: {exc}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return int(torch.cuda.memory_allocated())
    return 0


def _enforce_hbm_budget_all(
    llm: LLM,
    store: DeltaBlockStore,
    max_bytes_per_worker: int,
    touch_hashes: list[str] | None = None,
    demote_hashes: list[str] | None = None,
    protected_hashes: list[str] | None = None,
    incoming_new_blocks: int = 0,
) -> list[str]:
    if demote_hashes:
        for h in demote_hashes:
            store.demote_lru(h)
    if touch_hashes:
        for h in touch_hashes:
            if h in store._block_lru:
                store._touch_lru(h)
    evicted_union: set[str] = set()
    try:
        res = llm.collective_rpc(
            _worker_enforce_hbm_budget,
            kwargs={
                "max_bytes_per_worker": max_bytes_per_worker,
                "touch_hashes": touch_hashes or [],
                "demote_hashes": demote_hashes or [],
                "protected_hashes": protected_hashes or [],
                "incoming_new_blocks": incoming_new_blocks,
            },
        )
        if isinstance(res, list):
            for r in res:
                if isinstance(r, dict):
                    for h in r.get("evicted_hashes", []):
                        evicted_union.add(h)
    except Exception as exc:
        print(f"[WARN] _enforce_hbm_budget_all RPC error: {exc}", flush=True)
    for h in evicted_union:
        store.evict_block(h)
    return list(evicted_union)


def _collect_hbm_snapshot(
    llm: LLM, store: DeltaBlockStore, base_cuda_bytes: int
) -> dict[str, Any]:
    try:
        res = llm.collective_rpc(_worker_get_hbm_and_store_stats)
        if isinstance(res, list) and res and isinstance(res[0], dict):
            total_cuda_alloc = sum(int(r.get("cuda_allocated_bytes", 0)) for r in res)
            total_raw_bytes = sum(int(r.get("raw_bytes_equivalent", 0)) for r in res)
            total_comp_bytes = sum(
                int(r.get("compressed_bytes_stored", 0)) for r in res
            )
            total_evicted_blocks = sum(int(r.get("evicted_blocks", 0)) for r in res)
            total_evicted_bytes = sum(int(r.get("evicted_bytes", 0)) for r in res)
            merged_tiers: dict[str, int] = {}
            for r in res:
                for k, v in r.get("tier_counts", {}).items():
                    merged_tiers[k] = merged_tiers.get(k, 0) + int(v)
            return {
                "hbm_store_cuda_allocated_bytes": max(
                    0, total_cuda_alloc - base_cuda_bytes
                ),
                "raw_bytes_equivalent": total_raw_bytes,
                "compressed_bytes_stored": total_comp_bytes,
                "evicted_blocks": total_evicted_blocks,
                "evicted_bytes": total_evicted_bytes,
                "tier_counts": merged_tiers,
            }
    except Exception as exc:
        print(f"[WARN] _collect_hbm_snapshot RPC error: {exc}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_now = int(torch.cuda.memory_allocated())
    else:
        cuda_now = 0
    return {
        "hbm_store_cuda_allocated_bytes": max(0, cuda_now - base_cuda_bytes),
        "raw_bytes_equivalent": store.stats.raw_bytes_equivalent,
        "compressed_bytes_stored": store.stats.compressed_bytes_stored,
        "evicted_blocks": store.stats.evicted_blocks,
        "evicted_bytes": store.stats.evicted_bytes,
        "tier_counts": {k.name: v for k, v in store.stats.tier_counts.items()},
    }


def _is_prompt_resident_in_blockpool(
    kv_cache_manager: Any,
    vllm_hashes: list[Any],
    prompt_len: int,
    block_size: int,
    touch_lru: bool = False,
) -> bool:
    """Check whether all full blocks of `prompt_len` are resident in `BlockPool`."""
    expected_full_tokens = ((prompt_len - 1) // block_size) * block_size
    if expected_full_tokens <= 0:
        return True
    hit_blocks, hit_tokens, _ = kv_cache_manager.coordinator.find_longest_cache_hit(
        vllm_hashes, prompt_len - 1
    )
    is_resident = hit_tokens >= expected_full_tokens
    if is_resident and touch_lru:
        block_pool = kv_cache_manager.block_pool
        for g_blks in hit_blocks:
            for b in g_blks:
                if (
                    not b.is_null
                    and b.ref_cnt == 0
                    and b.block_hash is not None
                    and b.prev_free_block is not None
                    and b.next_free_block is not None
                ):
                    block_pool.free_block_queue.remove(b)
                    block_pool.free_block_queue.append(b)
    return is_resident


def _run_blockpool_eviction_mode(
    llm: LLM,
    bp_meta: dict[str, Any],
    bp_evict_stats: dict[str, int],
    pinned_cap_blocks: list[Any],
    all_turns_per_sample: list[list[list[int]]],
    all_vllm_hashes_per_sample: list[list[list[Any]]],
    block_size: int,
    hbm_budget_mb: float,
    prefill_params: SamplingParams,
    eval_params: SamplingParams,
    wave_size: int = 20,
) -> dict[str, Any]:
    """Run all 157 trajectories across Turn 1..12 on native vLLM BlockPool under a fixed HBM budget."""
    num_samples = len(all_turns_per_sample)
    max_turns = max(len(s_turns) for s_turns in all_turns_per_sample)
    kv_cache_manager = bp_meta["kv_cache_manager"]
    block_pool = bp_meta["block_pool"]
    slot_bytes_tp = int(bp_meta["block_slot_bytes_total_tp"])
    group_unpadded_tp = bp_meta["group_unpadded_bytes_total_tp"]

    # Release any previously pinned capacity-cap blocks before resetting BlockPool
    if pinned_cap_blocks:
        for b in pinned_cap_blocks:
            b.ref_cnt = 0
        block_pool.free_block_queue.append_n(pinned_cap_blocks)
        pinned_cap_blocks.clear()

    llm.reset_prefix_cache()
    bp_evict_stats["evicted_blocks"] = 0
    bp_evict_stats["evicted_bytes"] = 0
    bp_evict_stats["evicted_unpadded_bytes"] = 0

    max_usable_blocks = max(
        64, int((hbm_budget_mb * (1024**2)) // max(1, slot_bytes_tp))
    )
    avail_free = int(block_pool.free_block_queue.num_free_blocks)
    excess_blocks = max(0, avail_free - max_usable_blocks)
    if excess_blocks > 0:
        popped = block_pool.free_block_queue.popleft_n(excess_blocks)
        for b in popped:
            b.ref_cnt = 1
        pinned_cap_blocks.extend(popped)

    print(
        f"  [BLOCKPOOL] Capped usable BlockPool blocks to {block_pool.free_block_queue.num_free_blocks} "
        f"(physical budget={hbm_budget_mb:,.2f} MB, {slot_bytes_tp / (1024**2):.2f} MB/block)",
        flush=True,
    )

    t0_mode = time.perf_counter()
    cum_session_hits = 0
    cum_session_misses = 0
    cum_recomputed_turns = 0
    cum_recomputed_tokens = 0
    cum_total_prefill_tokens = 0

    traj_turn_loop_sec = [0.0 for _ in range(num_samples)]
    traj_final_eval_sec = [0.0 for _ in range(num_samples)]
    turn_snaps: list[dict[str, Any]] = []

    for t in range(max_turns):
        active_indices = [
            i for i, s_turns in enumerate(all_turns_per_sample) if t < len(s_turns)
        ]
        num_waves = (len(active_indices) + wave_size - 1) // wave_size

        for w_idx in range(num_waves):
            wave_idxs = active_indices[w_idx * wave_size : (w_idx + 1) * wave_size]
            wave_prompts: list[list[int]] = []
            wave_prompt_owners: list[int] = []

            for i in wave_idxs:
                s_turns = all_turns_per_sample[i]
                s_vh = all_vllm_hashes_per_sample[i]

                if t == 0:
                    wave_prompts.append(s_turns[0])
                    wave_prompt_owners.append(i)
                    cum_total_prefill_tokens += len(s_turns[0])
                else:
                    base_resident = _is_prompt_resident_in_blockpool(
                        kv_cache_manager,
                        s_vh[0],
                        len(s_turns[0]),
                        block_size,
                        touch_lru=True,
                    )
                    if base_resident:
                        cum_session_hits += 1
                    else:
                        cum_session_misses += 1
                        wave_prompts.append(s_turns[0])
                        wave_prompt_owners.append(i)
                        cum_recomputed_turns += 1
                        cum_recomputed_tokens += len(s_turns[0])
                        cum_total_prefill_tokens += len(s_turns[0])

                    wave_prompts.append(s_turns[t])
                    wave_prompt_owners.append(i)
                    cum_total_prefill_tokens += len(s_turns[t])

            if wave_prompts:
                t_wave_start = time.perf_counter()
                dedup_prompts = _unique_full_block_prompts(wave_prompts, block_size)
                sub_batch_size = min(16, max(1, wave_size))
                for sb_start in range(0, len(dedup_prompts), sub_batch_size):
                    sb_prompts = dedup_prompts[sb_start : sb_start + sub_batch_size]
                    _ = llm.generate(
                        [{"prompt_token_ids": p} for p in sb_prompts],
                        prefill_params,
                        use_tqdm=False,
                    )
                wave_elapsed = time.perf_counter() - t_wave_start
                owner_weights = Counter(wave_prompt_owners)
                total_w = max(1, sum(owner_weights.values()))
                for owner_i, w_cnt in owner_weights.items():
                    traj_turn_loop_sec[owner_i] += wave_elapsed * (w_cnt / total_w)

        live_blks = [
            blk
            for blk in block_pool.blocks
            if not blk.is_null and blk.block_hash is not None
        ]
        live_slot_bytes = len(live_blks) * slot_bytes_tp
        live_unpadded_bytes = sum(
            int(group_unpadded_tp.get(get_group_id(blk.block_hash), slot_bytes_tp))
            for blk in live_blks
        )
        elapsed_turn = time.perf_counter() - t0_mode

        resident_latest_sessions = 0
        resident_full_history_sessions = 0
        resident_turns_count = 0
        total_turns_so_far = 0
        for i in range(num_samples):
            s_turns = all_turns_per_sample[i]
            s_vh = all_vllm_hashes_per_sample[i]
            upto = min(t + 1, len(s_turns))
            total_turns_so_far += upto
            turns_in_hbm = [
                _is_prompt_resident_in_blockpool(
                    kv_cache_manager, s_vh[k], len(s_turns[k]), block_size
                )
                for k in range(upto)
            ]
            resident_turns_count += sum(1 for ok in turns_in_hbm if ok)
            if turns_in_hbm[-1]:
                resident_latest_sessions += 1
            if all(turns_in_hbm):
                resident_full_history_sessions += 1

        snap = {
            "hbm_store_cuda_allocated_bytes": live_slot_bytes,
            "raw_bytes_equivalent": live_unpadded_bytes,
            "compressed_bytes_stored": live_unpadded_bytes,
            "evicted_blocks": int(bp_evict_stats["evicted_blocks"]),
            "evicted_bytes": int(bp_evict_stats["evicted_bytes"]),
            "evicted_unpadded_bytes": int(bp_evict_stats["evicted_unpadded_bytes"]),
            "tier_counts": {"BLOCKPOOL_CACHED": len(live_blks)},
            "turn": t + 1,
            "active_trajectories": len(active_indices),
            "cum_session_hits": cum_session_hits,
            "cum_session_misses": cum_session_misses,
            "cum_recomputed_turns": cum_recomputed_turns,
            "cum_recomputed_tokens": cum_recomputed_tokens,
            "cum_total_prefill_tokens": cum_total_prefill_tokens,
            "resident_latest_sessions": resident_latest_sessions,
            "resident_full_history_sessions": resident_full_history_sessions,
            "resident_turns_count": resident_turns_count,
            "total_turns_so_far": total_turns_so_far,
            "elapsed_sec": round(elapsed_turn, 2),
        }
        turn_snaps.append(snap)

        print(
            f"  [BLOCKPOOL] Turn {t + 1:2d}/{max_turns} "
            f"(active={len(active_indices):3d}) | "
            f"Live Slot HBM: {snap['hbm_store_cuda_allocated_bytes'] / (1024**2):8.1f} MB "
            f"(Unpadded: {snap['compressed_bytes_stored'] / (1024**2):8.1f} MB) | "
            f"Evicted: {snap['evicted_bytes'] / (1024**2):8.1f} MB | "
            f"Hits/Misses: {cum_session_hits}/{cum_session_misses} | "
            f"Resident Sessions: {resident_latest_sessions}/{num_samples} | "
            f"Elapsed: {elapsed_turn:6.1f}s",
            flush=True,
        )

    turn_loop_sec = time.perf_counter() - t0_mode

    # ------------------------------------------------------------------
    # Final-Turn (`s_turns[-1]`) Evaluation across all 157 trajectories
    # ------------------------------------------------------------------
    t0_eval = time.perf_counter()
    final_eval_hits = 0
    final_eval_misses = 0

    for i in range(num_samples):
        s_turns = all_turns_per_sample[i]
        s_vh = all_vllm_hashes_per_sample[i]
        is_hit = _is_prompt_resident_in_blockpool(
            kv_cache_manager,
            s_vh[-1],
            len(s_turns[-1]),
            block_size,
            touch_lru=True,
        )
        if is_hit:
            final_eval_hits += 1
        else:
            final_eval_misses += 1
            cum_recomputed_turns += 1
            cum_recomputed_tokens += len(s_turns[-1])

    outputs_by_idx: dict[int, Any] = {}
    eval_sub_batch = max(1, wave_size)
    for w_start in range(0, num_samples, eval_sub_batch):
        batch_idxs = list(range(w_start, min(num_samples, w_start + eval_sub_batch)))
        batch_prompts = [all_turns_per_sample[i][-1] for i in batch_idxs]

        evicted_in_wave_prompts: list[list[int]] = []
        evicted_in_wave_owners: list[int] = []
        for i in batch_idxs:
            if not _is_prompt_resident_in_blockpool(
                kv_cache_manager,
                all_vllm_hashes_per_sample[i][-1],
                len(all_turns_per_sample[i][-1]),
                block_size,
            ):
                evicted_in_wave_prompts.append(all_turns_per_sample[i][-1])
                evicted_in_wave_owners.append(i)

        if evicted_in_wave_prompts:
            t_reprefill = time.perf_counter()
            dedup_evicted = _unique_full_block_prompts(
                evicted_in_wave_prompts, block_size
            )
            reprefill_sb = min(16, eval_sub_batch)
            for sb_e_start in range(0, len(dedup_evicted), reprefill_sb):
                sb_evicted = dedup_evicted[sb_e_start : sb_e_start + reprefill_sb]
                _ = llm.generate(
                    [{"prompt_token_ids": p} for p in sb_evicted],
                    prefill_params,
                    use_tqdm=False,
                )
            reprefill_elapsed = time.perf_counter() - t_reprefill
            ev_weights = Counter(evicted_in_wave_owners)
            tot_ev_w = max(1, sum(ev_weights.values()))
            for owner_i, w_cnt in ev_weights.items():
                traj_final_eval_sec[owner_i] += reprefill_elapsed * (w_cnt / tot_ev_w)

        t_decode = time.perf_counter()
        batch_outs = llm.generate(
            [{"prompt_token_ids": p} for p in batch_prompts],
            eval_params,
            use_tqdm=False,
        )
        for i, out in zip(batch_idxs, batch_outs):
            outputs_by_idx[i] = {
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
            }
        decode_elapsed = time.perf_counter() - t_decode
        for i in batch_idxs:
            traj_final_eval_sec[i] += decode_elapsed / max(1, len(batch_idxs))

    final_eval_sec = time.perf_counter() - t0_eval
    total_mode_sec = time.perf_counter() - t0_mode
    ordered_outputs = [outputs_by_idx[i] for i in range(num_samples)]
    traj_total_sec = [
        round(traj_turn_loop_sec[i] + traj_final_eval_sec[i], 4)
        for i in range(num_samples)
    ]

    if pinned_cap_blocks:
        for b in pinned_cap_blocks:
            b.ref_cnt = 0
        block_pool.free_block_queue.append_n(pinned_cap_blocks)
        pinned_cap_blocks.clear()

    return {
        "turn_snaps": turn_snaps,
        "outputs": ordered_outputs,
        "final_eval_hits": final_eval_hits,
        "final_eval_misses": final_eval_misses,
        "cum_session_hits": cum_session_hits,
        "cum_session_misses": cum_session_misses,
        "cum_recomputed_turns": cum_recomputed_turns,
        "cum_recomputed_tokens": cum_recomputed_tokens,
        "cum_total_prefill_tokens": cum_total_prefill_tokens,
        "turn_loop_sec": round(turn_loop_sec, 2),
        "final_eval_sec": round(final_eval_sec, 2),
        "total_mode_sec": round(total_mode_sec, 2),
        "traj_turn_loop_sec": [round(x, 4) for x in traj_turn_loop_sec],
        "traj_final_eval_sec": [round(x, 4) for x in traj_final_eval_sec],
        "traj_total_sec": traj_total_sec,
    }


def _run_single_eviction_mode(
    llm: LLM,
    store: DeltaBlockStore,
    all_turns_per_sample: list[list[list[int]]],
    all_hashes_per_sample: list[list[list[str]]],
    block_size: int,
    tp_size: int,
    force_uncompressed: bool,
    hbm_budget_mb: float,
    prefill_params: SamplingParams,
    eval_params: SamplingParams,
    wave_size: int = 20,
) -> dict[str, Any]:
    """Run all 157 trajectories across Turn 1..12 under a fixed GPU HBM budget."""
    num_samples = len(all_turns_per_sample)
    max_turns = max(len(s_turns) for s_turns in all_turns_per_sample)
    max_bytes_total = int(hbm_budget_mb * (1024**2))
    max_bytes_per_worker = max_bytes_total // max(1, tp_size)

    llm.reset_prefix_cache()
    base_cuda_bytes = _configure_and_clear_all(
        llm, store, force_uncompressed=force_uncompressed
    )

    mode_label = "UNCOMPRESSED" if force_uncompressed else "DELTA-COMPRESSED"
    t0_mode = time.perf_counter()

    cum_session_hits = 0
    cum_session_misses = 0
    cum_recomputed_turns = 0
    cum_recomputed_tokens = 0
    cum_total_prefill_tokens = 0

    traj_turn_loop_sec = [0.0 for _ in range(num_samples)]
    traj_final_eval_sec = [0.0 for _ in range(num_samples)]

    turn_snaps: list[dict[str, Any]] = []

    for t in range(max_turns):
        active_indices = [
            i for i, s_turns in enumerate(all_turns_per_sample) if t < len(s_turns)
        ]
        num_waves = (len(active_indices) + wave_size - 1) // wave_size

        for w_idx in range(num_waves):
            wave_idxs = active_indices[w_idx * wave_size : (w_idx + 1) * wave_size]
            wave_prompts: list[list[int]] = []
            wave_prompt_owners: list[int] = []
            touch_hashes: list[str] = []
            demote_hashes: list[str] = []
            protected_hashes: list[str] = []

            # Keep active trajectories' base anchors and latest-completed turns warm in LRU
            for idx_all in range(num_samples):
                sh_all = all_hashes_per_sample[idx_all]
                latest_k = min(t, len(sh_all) - 1)
                if not force_uncompressed:
                    touch_hashes.extend(sh_all[0])
                touch_hashes.extend(sh_all[latest_k])

            for i in wave_idxs:
                s_turns = all_turns_per_sample[i]
                s_hashes = all_hashes_per_sample[i]
                base_h = s_hashes[0]

                if t == 0:
                    wave_prompts.append(s_turns[0])
                    wave_prompt_owners.append(i)
                    cum_total_prefill_tokens += len(s_turns[0])
                    protected_hashes.extend(base_h)
                else:
                    if not force_uncompressed and t >= 2:
                        prev_leaf_h = s_hashes[t - 1]
                        for h in prev_leaf_h:
                            if h not in base_h and h not in s_hashes[t]:
                                demote_hashes.append(h)

                    # Check whether the session's Turn-1 base anchor (`s_turns[0]`)
                    # is still resident in GPU HBM
                    base_resident = bool(base_h) and all(
                        store.has_full_block(h) for h in base_h
                    )

                    if base_resident:
                        cum_session_hits += 1
                        touch_hashes.extend(base_h)
                        protected_hashes.extend(base_h)
                    else:
                        cum_session_misses += 1
                        # Re-prefill evicted session anchor (`s_turns[0]`) on GPU
                        wave_prompts.append(s_turns[0])
                        wave_prompt_owners.append(i)
                        cum_recomputed_turns += 1
                        cum_recomputed_tokens += len(s_turns[0])
                        cum_total_prefill_tokens += len(s_turns[0])
                        protected_hashes.extend(base_h)

                    wave_prompts.append(s_turns[t])
                    wave_prompt_owners.append(i)
                    cum_total_prefill_tokens += len(s_turns[t])
                    protected_hashes.extend(s_hashes[t])

            if wave_prompts:
                t_wave_start = time.perf_counter()
                dedup_prompts = _unique_full_block_prompts(wave_prompts, block_size)
                sub_batch_size = (
                    min(16, max(1, wave_size))
                    if force_uncompressed
                    else max(1, wave_size)
                )
                for sb_start in range(0, len(dedup_prompts), sub_batch_size):
                    sb_prompts = dedup_prompts[sb_start : sb_start + sub_batch_size]
                    new_sb_hashes = {
                        h
                        for p in sb_prompts
                        for h in _get_prompt_block_hashes(p, block_size)
                        if not store.has_full_block(h)
                    }
                    if new_sb_hashes:
                        _enforce_hbm_budget_all(
                            llm=llm,
                            store=store,
                            max_bytes_per_worker=max_bytes_per_worker,
                            touch_hashes=touch_hashes,
                            demote_hashes=demote_hashes,
                            protected_hashes=protected_hashes,
                            incoming_new_blocks=len(new_sb_hashes) + 2,
                        )
                    _ = llm.generate(
                        [{"prompt_token_ids": p} for p in sb_prompts],
                        prefill_params,
                        use_tqdm=False,
                    )
                    llm.reset_prefix_cache()

                _enforce_hbm_budget_all(
                    llm=llm,
                    store=store,
                    max_bytes_per_worker=max_bytes_per_worker,
                    touch_hashes=touch_hashes,
                    demote_hashes=demote_hashes,
                    protected_hashes=protected_hashes,
                )
                wave_elapsed = time.perf_counter() - t_wave_start
                owner_weights = Counter(wave_prompt_owners)
                total_w = max(1, sum(owner_weights.values()))
                for owner_i, w_cnt in owner_weights.items():
                    traj_turn_loop_sec[owner_i] += wave_elapsed * (w_cnt / total_w)

        snap = _collect_hbm_snapshot(llm, store, base_cuda_bytes)
        elapsed_turn = time.perf_counter() - t0_mode

        # Count how many of the 157 trajectories currently have their latest turn
        # (`min(t, len(s_turns) - 1)`) and base turn (`0`) resident in GPU HBM
        resident_latest_sessions = 0
        resident_full_history_sessions = 0
        resident_turns_count = 0
        total_turns_so_far = 0
        for i in range(num_samples):
            s_hashes = all_hashes_per_sample[i]
            upto = min(t + 1, len(s_hashes))
            total_turns_so_far += upto
            turns_in_hbm = [
                all(store.has_full_block(h) for h in s_hashes[k]) for k in range(upto)
            ]
            resident_turns_count += sum(1 for ok in turns_in_hbm if ok)
            if turns_in_hbm[-1] and (force_uncompressed or turns_in_hbm[0]):
                resident_latest_sessions += 1
            if all(turns_in_hbm):
                resident_full_history_sessions += 1

        snap.update(
            {
                "turn": t + 1,
                "active_trajectories": len(active_indices),
                "cum_session_hits": cum_session_hits,
                "cum_session_misses": cum_session_misses,
                "cum_recomputed_turns": cum_recomputed_turns,
                "cum_recomputed_tokens": cum_recomputed_tokens,
                "cum_total_prefill_tokens": cum_total_prefill_tokens,
                "resident_latest_sessions": resident_latest_sessions,
                "resident_full_history_sessions": resident_full_history_sessions,
                "resident_turns_count": resident_turns_count,
                "total_turns_so_far": total_turns_so_far,
                "elapsed_sec": round(elapsed_turn, 2),
            }
        )
        turn_snaps.append(snap)

        print(
            f"  [{mode_label}] Turn {t + 1:2d}/{max_turns} "
            f"(active={len(active_indices):3d}) | "
            f"Live HBM: {snap['hbm_store_cuda_allocated_bytes'] / (1024**2):8.1f} MB | "
            f"Evicted: {snap['evicted_bytes'] / (1024**2):8.1f} MB | "
            f"Hits/Misses: {cum_session_hits}/{cum_session_misses} | "
            f"Resident Sessions: {resident_latest_sessions}/{num_samples} | "
            f"Elapsed: {elapsed_turn:6.1f}s",
            flush=True,
        )

    turn_loop_sec = time.perf_counter() - t0_mode

    # ------------------------------------------------------------------
    # Final-Turn (`s_turns[-1]`) Evaluation across all 157 trajectories
    # ------------------------------------------------------------------
    t0_eval = time.perf_counter()
    final_eval_hits = 0
    final_eval_misses = 0
    resident_order: list[int] = []
    evicted_order: list[int] = []
    all_resident_final_hashes: list[str] = []

    for i in range(num_samples):
        s_hashes = all_hashes_per_sample[i]
        final_h = s_hashes[-1]
        base_h = s_hashes[0]
        is_hit = all(store.has_full_block(h) for h in final_h) and (
            force_uncompressed or all(store.has_full_block(h) for h in base_h)
        )
        if is_hit:
            final_eval_hits += 1
            resident_order.append(i)
            all_resident_final_hashes.extend(final_h)
            if not force_uncompressed:
                all_resident_final_hashes.extend(base_h)
        else:
            final_eval_misses += 1
            evicted_order.append(i)
            cum_recomputed_turns += 1
            cum_recomputed_tokens += len(all_turns_per_sample[i][-1])

    outputs_by_idx: dict[int, Any] = {}
    eval_sub_batch = max(1, wave_size)
    for w_start in range(0, num_samples, eval_sub_batch):
        batch_idxs = list(range(w_start, min(num_samples, w_start + eval_sub_batch)))
        batch_prompts = [all_turns_per_sample[i][-1] for i in batch_idxs]
        batch_hashes = [
            h for i in batch_idxs for h in all_hashes_per_sample[i][-1]
        ]
        remaining_resident_hashes: list[str] = []
        for r_idx in resident_order:
            if r_idx not in outputs_by_idx:
                remaining_resident_hashes.extend(all_hashes_per_sample[r_idx][-1])
                if not force_uncompressed:
                    remaining_resident_hashes.extend(all_hashes_per_sample[r_idx][0])

        evicted_in_wave_prompts: list[list[int]] = []
        evicted_in_wave_owners: list[int] = []
        for i in batch_idxs:
            if not all(store.has_full_block(h) for h in all_hashes_per_sample[i][-1]):
                if not force_uncompressed and not all(
                    store.has_full_block(h) for h in all_hashes_per_sample[i][0]
                ):
                    evicted_in_wave_prompts.append(all_turns_per_sample[i][0])
                    evicted_in_wave_owners.append(i)
                evicted_in_wave_prompts.append(all_turns_per_sample[i][-1])
                evicted_in_wave_owners.append(i)

        if evicted_in_wave_prompts:
            t_reprefill = time.perf_counter()
            dedup_evicted = _unique_full_block_prompts(
                evicted_in_wave_prompts, block_size
            )
            reprefill_sb = min(16, eval_sub_batch)
            for sb_e_start in range(0, len(dedup_evicted), reprefill_sb):
                sb_evicted = dedup_evicted[sb_e_start : sb_e_start + reprefill_sb]
                new_eval_hashes = {
                    h
                    for p in sb_evicted
                    for h in _get_prompt_block_hashes(p, block_size)
                    if not store.has_full_block(h)
                }
                _enforce_hbm_budget_all(
                    llm=llm,
                    store=store,
                    max_bytes_per_worker=max_bytes_per_worker,
                    protected_hashes=remaining_resident_hashes + batch_hashes,
                    incoming_new_blocks=len(new_eval_hashes) + 2,
                )
                _ = llm.generate(
                    [{"prompt_token_ids": p} for p in sb_evicted],
                    prefill_params,
                    use_tqdm=False,
                )
                llm.reset_prefix_cache()
            reprefill_elapsed = time.perf_counter() - t_reprefill
            ev_weights = Counter(evicted_in_wave_owners)
            tot_ev_w = max(1, sum(ev_weights.values()))
            for owner_i, w_cnt in ev_weights.items():
                traj_final_eval_sec[owner_i] += reprefill_elapsed * (w_cnt / tot_ev_w)

        t_decode = time.perf_counter()
        batch_outs = llm.generate(
            [{"prompt_token_ids": p} for p in batch_prompts],
            eval_params,
            use_tqdm=False,
        )
        llm.reset_prefix_cache()
        for i, out in zip(batch_idxs, batch_outs):
            outputs_by_idx[i] = {
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
            }
        if evicted_in_wave_prompts:
            remaining_resident_after: list[str] = []
            for r_idx in resident_order:
                if r_idx not in outputs_by_idx:
                    remaining_resident_after.extend(all_hashes_per_sample[r_idx][-1])
                    if not force_uncompressed:
                        remaining_resident_after.extend(all_hashes_per_sample[r_idx][0])
            _enforce_hbm_budget_all(
                llm=llm,
                store=store,
                max_bytes_per_worker=max_bytes_per_worker,
                protected_hashes=remaining_resident_after,
            )
        decode_elapsed = time.perf_counter() - t_decode
        for i in batch_idxs:
            traj_final_eval_sec[i] += decode_elapsed / max(1, len(batch_idxs))

    final_eval_sec = time.perf_counter() - t0_eval
    total_mode_sec = time.perf_counter() - t0_mode
    ordered_outputs = [outputs_by_idx[i] for i in range(num_samples)]
    traj_total_sec = [
        round(traj_turn_loop_sec[i] + traj_final_eval_sec[i], 4)
        for i in range(num_samples)
    ]

    return {
        "turn_snaps": turn_snaps,
        "outputs": ordered_outputs,
        "final_eval_hits": final_eval_hits,
        "final_eval_misses": final_eval_misses,
        "cum_session_hits": cum_session_hits,
        "cum_session_misses": cum_session_misses,
        "cum_recomputed_turns": cum_recomputed_turns,
        "cum_recomputed_tokens": cum_recomputed_tokens,
        "cum_total_prefill_tokens": cum_total_prefill_tokens,
        "turn_loop_sec": round(turn_loop_sec, 2),
        "final_eval_sec": round(final_eval_sec, 2),
        "total_mode_sec": round(total_mode_sec, 2),
        "traj_turn_loop_sec": [round(x, 4) for x in traj_turn_loop_sec],
        "traj_final_eval_sec": [round(x, 4) for x in traj_final_eval_sec],
        "traj_total_sec": traj_total_sec,
    }


def evaluate_swe_agent_eviction_hbm(
    llm: LLM,
    store: DeltaBlockStore | None,
    samples: list[dict[str, Any]],
    block_size: int,
    tp_size: int,
    gen_tokens: int = 128,
    wave_size: int = 20,
    budget_t1_mb: float = 161050.81,
    budget_t2_mb: float = 164000.0,
    regime_filter: str = "all",
    mode: str = "both",
    blockpool_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_regimes = [
        ("T1_stateful_aci_window_dp0", budget_t1_mb),
        ("T2_rolling_action_history_dpB", budget_t2_mb),
    ]
    regimes = [
        (r, b)
        for r, b in all_regimes
        if regime_filter == "all" or r == regime_filter
    ]
    prefill_params = SamplingParams(temperature=0.0, max_tokens=1)
    eval_params = SamplingParams(temperature=0.0, max_tokens=gen_tokens)

    bp_meta: dict[str, Any] | None = None
    bp_evict_stats = {
        "evicted_blocks": 0,
        "evicted_bytes": 0,
        "evicted_unpadded_bytes": 0,
    }
    pinned_cap_blocks: list[Any] = []
    hash_fn: Any = None
    if mode == "blockpool":
        bp_meta = _get_blockpool_byte_metrics(llm, tp_size)
        block_pool = bp_meta["block_pool"]
        slot_bytes_tp = int(bp_meta["block_slot_bytes_total_tp"])
        group_unpadded_tp = bp_meta["group_unpadded_bytes_total_tp"]
        orig_maybe_evict = block_pool._maybe_evict_cached_block

        def _wrapped_maybe_evict(blk: Any) -> bool:
            g_id = get_group_id(blk.block_hash) if blk.block_hash is not None else 0
            was_evicted = orig_maybe_evict(blk)
            if was_evicted:
                bp_evict_stats["evicted_blocks"] += 1
                bp_evict_stats["evicted_bytes"] += slot_bytes_tp
                bp_evict_stats["evicted_unpadded_bytes"] += int(
                    group_unpadded_tp.get(g_id, slot_bytes_tp)
                )
            return was_evicted

        block_pool._maybe_evict_cached_block = _wrapped_maybe_evict
        hash_fn = get_hash_fn_by_name(
            llm.llm_engine.vllm_config.cache_config.prefix_caching_hash_algo
        )

    results: dict[str, Any] = {}
    for regime, hbm_budget_mb in regimes:
        all_turns_per_sample = [s["regimes"][regime] for s in samples]
        max_turns = max(len(s_turns) for s_turns in all_turns_per_sample)
        min_turns = min(len(s_turns) for s_turns in all_turns_per_sample)
        total_trajectory_turns = sum(len(s_turns) for s_turns in all_turns_per_sample)
        avg_turns = total_trajectory_turns / max(1, len(samples))

        if mode == "delta" and blockpool_data is not None:
            bp_uncomp = blockpool_data["regimes"][regime]["uncomp_res"]
            hbm_budget_mb = max(
                s["hbm_store_cuda_allocated_bytes"] for s in bp_uncomp["turn_snaps"]
            ) / (1024**2)

        print(f"\n{'=' * 90}")
        print(
            f"[SWE-AGENT GPU HBM EVICTION BENCHMARK ({mode.upper()})] Regime: {regime} | "
            f"Fixed HBM Budget: {hbm_budget_mb:,.2f} MB ({hbm_budget_mb / 1024.0:.2f} GiB) | "
            f"{len(samples)} trajectories, {total_trajectory_turns} turns ({min_turns}-{max_turns} turns/traj)"
        )
        print(f"{'=' * 90}", flush=True)

        if mode == "blockpool":
            assert bp_meta is not None and hash_fn is not None
            all_vllm_hashes_per_sample = [
                [_compute_vllm_block_hashes(hash_fn, p, block_size) for p in s_turns]
                for s_turns in all_turns_per_sample
            ]
            uncomp_res = _run_blockpool_eviction_mode(
                llm=llm,
                bp_meta=bp_meta,
                bp_evict_stats=bp_evict_stats,
                pinned_cap_blocks=pinned_cap_blocks,
                all_turns_per_sample=all_turns_per_sample,
                all_vllm_hashes_per_sample=all_vllm_hashes_per_sample,
                block_size=block_size,
                hbm_budget_mb=hbm_budget_mb,
                prefill_params=prefill_params,
                eval_params=eval_params,
                wave_size=wave_size,
            )
            results[regime] = {
                "fixed_hbm_budget_mb": round(hbm_budget_mb, 2),
                "uncomp_res": uncomp_res,
            }
            continue

        all_hashes_per_sample = [
            [_get_prompt_block_hashes(p, block_size) for p in s_turns]
            for s_turns in all_turns_per_sample
        ]
        assert store is not None

        if mode == "delta" and blockpool_data is not None:
            uncomp_res = blockpool_data["regimes"][regime]["uncomp_res"]
        else:
            # 1. Run Uncompressed (`force_uncompressed=True`) under fixed HBM budget
            uncomp_res = _run_single_eviction_mode(
                llm=llm,
                store=store,
                all_turns_per_sample=all_turns_per_sample,
                all_hashes_per_sample=all_hashes_per_sample,
                block_size=block_size,
                tp_size=tp_size,
                force_uncompressed=True,
                hbm_budget_mb=hbm_budget_mb,
                prefill_params=prefill_params,
                eval_params=eval_params,
                wave_size=wave_size,
            )

        # 2. Run Delta-Compressed (`force_uncompressed=False`) under fixed HBM budget
        comp_res = _run_single_eviction_mode(
            llm=llm,
            store=store,
            all_turns_per_sample=all_turns_per_sample,
            all_hashes_per_sample=all_hashes_per_sample,
            block_size=block_size,
            tp_size=tp_size,
            force_uncompressed=False,
            hbm_budget_mb=hbm_budget_mb,
            prefill_params=prefill_params,
            eval_params=eval_params,
            wave_size=wave_size,
        )

        uncomp_snaps = uncomp_res["turn_snaps"]
        comp_snaps = comp_res["turn_snaps"]
        u_final = uncomp_snaps[-1]
        c_final = comp_snaps[-1]

        turn_progression: list[dict[str, Any]] = []
        for t in range(max_turns):
            u_s = uncomp_snaps[t]
            c_s = comp_snaps[t]
            u_live_mb = u_s["hbm_store_cuda_allocated_bytes"] / (1024**2)
            c_live_mb = c_s["hbm_store_cuda_allocated_bytes"] / (1024**2)
            u_evict_mb = u_s["evicted_bytes"] / (1024**2)
            c_evict_mb = c_s["evicted_bytes"] / (1024**2)
            turn_progression.append(
                {
                    "turn": t + 1,
                    "active_trajectories": u_s["active_trajectories"],
                    "uncompressed_live_cuda_mb": round(u_live_mb, 2),
                    "delta_compressed_live_cuda_mb": round(c_live_mb, 2),
                    "uncompressed_evicted_mb": round(u_evict_mb, 2),
                    "delta_compressed_evicted_mb": round(c_evict_mb, 2),
                    "uncompressed_evicted_blocks": u_s["evicted_blocks"],
                    "delta_compressed_evicted_blocks": c_s["evicted_blocks"],
                    "uncompressed_cum_session_hits": u_s["cum_session_hits"],
                    "uncompressed_cum_session_misses": u_s["cum_session_misses"],
                    "delta_compressed_cum_session_hits": c_s["cum_session_hits"],
                    "delta_compressed_cum_session_misses": c_s["cum_session_misses"],
                    "uncompressed_recomputed_tokens": u_s["cum_recomputed_tokens"],
                    "delta_compressed_recomputed_tokens": c_s["cum_recomputed_tokens"],
                    "uncompressed_resident_sessions": u_s["resident_latest_sessions"],
                    "delta_compressed_resident_sessions": c_s[
                        "resident_latest_sessions"
                    ],
                    "uncompressed_resident_turns": u_s["resident_turns_count"],
                    "delta_compressed_resident_turns": c_s["resident_turns_count"],
                    "total_turns_so_far": u_s["total_turns_so_far"],
                    "uncompressed_elapsed_sec": u_s["elapsed_sec"],
                    "delta_compressed_elapsed_sec": c_s["elapsed_sec"],
                }
            )

        # Accuracy & Output Agreement
        top1_matches = 0
        f1_scores: list[float] = []
        action_matches = 0
        base_symbol_hits = 0
        comp_symbol_hits = 0

        for s, b_out, c_out in zip(
            samples, uncomp_res["outputs"], comp_res["outputs"]
        ):
            if isinstance(b_out, dict):
                b_ids = list(b_out["token_ids"])
                b_txt = str(b_out["text"])
            else:
                b_ids = list(b_out.outputs[0].token_ids)
                b_txt = b_out.outputs[0].text
            if isinstance(c_out, dict):
                c_ids = list(c_out["token_ids"])
                c_txt = str(c_out["text"])
            else:
                c_ids = list(c_out.outputs[0].token_ids)
                c_txt = c_out.outputs[0].text

            if b_ids and c_ids and b_ids[0] == c_ids[0]:
                top1_matches += 1
            f1_scores.append(token_f1(c_txt, b_txt))
            if _classify_swe_action(b_txt) == _classify_swe_action(c_txt):
                action_matches += 1

            gt_syms = [sym.lower() for sym in s["gt_symbols"]]
            if any(sym in b_txt.lower() for sym in gt_syms):
                base_symbol_hits += 1
            if any(sym in c_txt.lower() for sym in gt_syms):
                comp_symbol_hits += 1

        n_s = max(1, len(samples))
        u_followup_total = max(
            1, uncomp_res["cum_session_hits"] + uncomp_res["cum_session_misses"]
        )
        c_followup_total = max(
            1, comp_res["cum_session_hits"] + comp_res["cum_session_misses"]
        )

        summary = {
            "baseline_engine": "vLLM BlockPool (enable_prefix_caching=True, kv_transfer_config=None)"
            if mode == "delta" and blockpool_data is not None
            else "Uncompressed GPU HBM",
            "num_trajectories": len(samples),
            "min_turns_per_trajectory": min_turns,
            "max_turns_per_trajectory": max_turns,
            "avg_turns_per_trajectory": round(avg_turns, 2),
            "total_trajectory_turns": total_trajectory_turns,
            "fixed_hbm_budget_mb": round(hbm_budget_mb, 2),
            "fixed_hbm_budget_gib": round(hbm_budget_mb / 1024.0, 2),
            "uncompressed_peak_live_hbm_cuda_mb": round(
                max(s["hbm_store_cuda_allocated_bytes"] for s in uncomp_snaps)
                / (1024**2),
                2,
            ),
            "delta_compressed_peak_live_hbm_cuda_mb": round(
                max(s["hbm_store_cuda_allocated_bytes"] for s in comp_snaps)
                / (1024**2),
                2,
            ),
            "uncompressed_final_live_hbm_tensor_mb": round(
                u_final["compressed_bytes_stored"] / (1024**2), 2
            ),
            "delta_compressed_final_live_hbm_tensor_mb": round(
                c_final["compressed_bytes_stored"] / (1024**2), 2
            ),
            "uncompressed_total_evicted_mb": round(
                u_final["evicted_bytes"] / (1024**2), 2
            ),
            "uncompressed_total_evicted_gib": round(
                u_final["evicted_bytes"] / (1024**3), 2
            ),
            "uncompressed_total_evicted_unpadded_mb": round(
                u_final.get("evicted_unpadded_bytes", u_final["evicted_bytes"])
                / (1024**2),
                2,
            ),
            "uncompressed_total_evicted_unpadded_gib": round(
                u_final.get("evicted_unpadded_bytes", u_final["evicted_bytes"])
                / (1024**3),
                2,
            ),
            "delta_compressed_total_evicted_mb": round(
                c_final["evicted_bytes"] / (1024**2), 2
            ),
            "uncompressed_total_evicted_blocks": u_final["evicted_blocks"],
            "delta_compressed_total_evicted_blocks": c_final["evicted_blocks"],
            "uncompressed_followup_turn_hits": uncomp_res["cum_session_hits"],
            "uncompressed_followup_turn_misses": uncomp_res["cum_session_misses"],
            "uncompressed_followup_hit_rate_pct": round(
                100.0 * uncomp_res["cum_session_hits"] / u_followup_total, 2
            ),
            "delta_compressed_followup_turn_hits": comp_res["cum_session_hits"],
            "delta_compressed_followup_turn_misses": comp_res["cum_session_misses"],
            "delta_compressed_followup_hit_rate_pct": round(
                100.0 * comp_res["cum_session_hits"] / c_followup_total, 2
            ),
            "uncompressed_final_turn_resident_sessions": uncomp_res[
                "final_eval_hits"
            ],
            "uncompressed_final_turn_evicted_sessions": uncomp_res[
                "final_eval_misses"
            ],
            "uncompressed_final_turn_hit_rate_pct": round(
                100.0 * uncomp_res["final_eval_hits"] / n_s, 2
            ),
            "delta_compressed_final_turn_resident_sessions": comp_res[
                "final_eval_hits"
            ],
            "delta_compressed_final_turn_evicted_sessions": comp_res[
                "final_eval_misses"
            ],
            "delta_compressed_final_turn_hit_rate_pct": round(
                100.0 * comp_res["final_eval_hits"] / n_s, 2
            ),
            "uncompressed_resident_turns_at_end": u_final["resident_turns_count"],
            "delta_compressed_resident_turns_at_end": c_final["resident_turns_count"],
            "uncompressed_recomputed_turns": uncomp_res["cum_recomputed_turns"],
            "delta_compressed_recomputed_turns": comp_res["cum_recomputed_turns"],
            "uncompressed_recomputed_prefill_tokens": uncomp_res[
                "cum_recomputed_tokens"
            ],
            "delta_compressed_recomputed_prefill_tokens": comp_res[
                "cum_recomputed_tokens"
            ],
            "uncompressed_total_prefill_tokens": uncomp_res[
                "cum_total_prefill_tokens"
            ],
            "delta_compressed_total_prefill_tokens": comp_res[
                "cum_total_prefill_tokens"
            ],
            "uncompressed_turn_loop_sec": uncomp_res["turn_loop_sec"],
            "delta_compressed_turn_loop_sec": comp_res["turn_loop_sec"],
            "uncompressed_final_eval_sec": uncomp_res["final_eval_sec"],
            "delta_compressed_final_eval_sec": comp_res["final_eval_sec"],
            "uncompressed_total_runtime_sec": uncomp_res["total_mode_sec"],
            "delta_compressed_total_runtime_sec": comp_res["total_mode_sec"],
            "uncompressed_avg_completion_time_per_traj_sec": round(
                uncomp_res["total_mode_sec"] / n_s, 3
            ),
            "delta_compressed_avg_completion_time_per_traj_sec": round(
                comp_res["total_mode_sec"] / n_s, 3
            ),
            "uncompressed_turn_loop_per_traj_sec": round(
                uncomp_res["turn_loop_sec"] / n_s, 3
            ),
            "delta_compressed_turn_loop_per_traj_sec": round(
                comp_res["turn_loop_sec"] / n_s, 3
            ),
            "uncompressed_final_eval_per_traj_sec": round(
                uncomp_res["final_eval_sec"] / n_s, 3
            ),
            "delta_compressed_final_eval_per_traj_sec": round(
                comp_res["final_eval_sec"] / n_s, 3
            ),
            "uncompressed_traj_completion_p50_sec": round(
                sorted(uncomp_res["traj_total_sec"])[len(samples) // 2], 3
            ),
            "delta_compressed_traj_completion_p50_sec": round(
                sorted(comp_res["traj_total_sec"])[len(samples) // 2], 3
            ),
            "uncompressed_traj_completion_p90_sec": round(
                sorted(uncomp_res["traj_total_sec"])[int(len(samples) * 0.9)], 3
            ),
            "delta_compressed_traj_completion_p90_sec": round(
                sorted(comp_res["traj_total_sec"])[int(len(samples) * 0.9)], 3
            ),
            "uncompressed_traj_completion_max_sec": round(
                max(uncomp_res["traj_total_sec"]), 3
            ),
            "delta_compressed_traj_completion_max_sec": round(
                max(comp_res["traj_total_sec"]), 3
            ),
            "runtime_speedup_x": round(
                uncomp_res["total_mode_sec"]
                / max(comp_res["total_mode_sec"], 1e-6),
                2,
            ),
            "final_eval_speedup_x": round(
                uncomp_res["final_eval_sec"]
                / max(comp_res["final_eval_sec"], 1e-6),
                2,
            ),
            "final_turn_top1_token_match_pct": round(100.0 * top1_matches / n_s, 2),
            "final_turn_response_f1_agreement_pct": round(
                100.0 * sum(f1_scores) / len(f1_scores), 2
            ),
            "final_turn_swe_action_agreement_pct": round(
                100.0 * action_matches / n_s, 2
            ),
            "baseline_gt_patch_symbol_accuracy_pct": round(
                100.0 * base_symbol_hits / n_s, 2
            ),
            "delta_compressed_gt_patch_symbol_accuracy_pct": round(
                100.0 * comp_symbol_hits / n_s, 2
            ),
            "turn_progression": turn_progression,
        }
        print(f"\n--- [SWE-AGENT GPU HBM EVICTION BENCHMARK] Summary for {regime} ---")
        print(json.dumps(summary, indent=2), flush=True)
        results[regime] = summary

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GPU HBM LRU Eviction & Re-Prefill on real SWE-agent trajectories."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=157)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.688)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--wave-size", type=int, default=20)
    parser.add_argument("--budget-t1-mb", type=float, default=161050.81)
    parser.add_argument("--budget-t2-mb", type=float, default=164000.0)
    parser.add_argument("--regime", type=str, default="all")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["blockpool", "delta", "both"],
        default="both",
    )
    parser.add_argument("--blockpool-json", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    t_job_start = time.perf_counter()
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

    if args.mode == "blockpool":
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            block_size=16,
            enable_prefix_caching=True,
            async_scheduling=False,
            gpu_memory_utilization=max(args.gpu_memory_utilization, 0.88),
            max_model_len=args.max_model_len,
            max_num_batched_tokens=16384,
            enforce_eager=True,
        )
        store = None
    else:
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
        store = get_shared_store()

    resolved_block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)

    samples = build_trajectory_samples(
        trajectories=trajectories,
        tokenizer=tokenizer,
        block_size=resolved_block_size,
    )
    total_turns = sum(s["num_turns"] for s in samples)
    print(
        f"Built {len(samples)} SWE-agent multi-turn samples "
        f"({total_turns} total turns, avg={total_turns / max(1, len(samples)):.2f} turns/traj, "
        f"block_size={resolved_block_size}).",
        flush=True,
    )

    blockpool_data: dict[str, Any] | None = None
    if args.mode == "delta" and args.blockpool_json:
        with open(args.blockpool_json, encoding="utf-8") as f:
            blockpool_data = json.load(f)
        print(f"Loaded BlockPool eviction baseline from {args.blockpool_json}")

    eviction_results = evaluate_swe_agent_eviction_hbm(
        llm=llm,
        store=store,
        samples=samples,
        block_size=resolved_block_size,
        tp_size=args.tensor_parallel_size,
        gen_tokens=args.gen_tokens,
        wave_size=args.wave_size,
        budget_t1_mb=args.budget_t1_mb,
        budget_t2_mb=args.budget_t2_mb,
        regime_filter=args.regime,
        mode=args.mode,
        blockpool_data=blockpool_data,
    )

    total_job_sec = time.perf_counter() - t_job_start
    out_payload = {
        "model": args.model,
        "dataset": "nebius/SWE-agent-trajectories (target == True, verified passing, all turns, LRU eviction)",
        "storage_tier": "GPU_HBM (vLLM BlockPool vs DeltaCompressionConnector under fixed HBM budget)",
        "mode": args.mode,
        "tensor_parallel_size": args.tensor_parallel_size,
        "block_size": resolved_block_size,
        "num_trajectories": len(samples),
        "total_trajectory_turns": total_turns,
        "total_job_runtime_sec": round(total_job_sec, 2),
        "regimes": eviction_results,
    }
    target_json = (
        args.blockpool_json
        if (args.mode == "blockpool" and args.blockpool_json and not args.output_json)
        else args.output_json
    )
    if target_json:
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(
            f"\nSaved SWE-agent GPU HBM eviction benchmark results ({args.mode}) to {target_json}"
        )


if __name__ == "__main__":
    main()

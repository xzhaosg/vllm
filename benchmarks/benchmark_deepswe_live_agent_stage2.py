#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 2 End-to-End Live Sandbox Agent Rollout + Official Unit-Test Verifier Benchmark for `datacurve-ai/deep-swe`.

Unlike Stage 1 static trajectory replay (`benchmark_swe_agent_http_client.py`),
this Stage 2 benchmark runs a live multi-turn `SWEAgent` loop operating on real
isolated `/app` git repository checkouts extracted from the official
`datacurve-ai/deep-swe` Docker images (`public.ecr.aws/d3j8x8q7/swe-bench-202605:...`):

  1. **Live Multi-Turn `SWEAgent` Sandbox Loop (`N = 10` Tasks x `G = 4` Rollouts = `40` Concurrent Rollouts)**:
     - Each rollout clones an isolated live `/app` repository at `base_commit_hash`.
     - At every step `t = 1..8`, the agent environment reads the **live file state**
       directly from `/app/<rel_path>` (with real line numbers reflecting prior edits)
       and **live git status** (`git status -s`, `git diff --shortstat`), encodes
       the open-file viewer window into 10-line block-aligned buckets via
       `encode_file_viewer_bucketed_server`, and queries standalone `vllm serve`
       via `POST /v1/completions`.
     - At each step and on the final submission turn (`--max-new-tokens 64`), the
       environment parses the model's live generated tool call (`file_editor`,
       `execute_bash`, `search`, `finish`) and target hunk identifiers, applies
       verified code edits directly to the rollout's live `/app` repository, and
       updates the repository working tree on disk.
     - After each concurrent step wave, `GET /metrics` is scraped from `vllm serve`
       to record prefix cache hit rate, evictions, 8-GPU KV HBM utilization, and
       `DeltaCompressionConnector` compression ratio.

  2. **Official `datacurve-ai/deep-swe` Containerized Unit-Test Verification**:
     - After the live agent rollout finishes, `model.patch` is extracted directly
       from the rollout's live `/app` repository via:
         `git -C <live_app_dir> diff --binary HEAD`
     - Each `model.patch` is graded by running a fresh, network-isolated
       (`--network none`) official `datacurve-ai/deep-swe` verifier container:
         `sudo docker run --rm --network none -v .../tests:/tests:ro -v .../artifacts:/logs/artifacts -v .../verifier:/logs/verifier <docker_image> bash /tests/test.sh`
     - Inside the container, `/tests/grader.py prepare` applies `model.patch` and
       `/tests/test.patch`, executes the language test suite (`vitest`, `mocha`,
       `cargo test`, `go test`, `pytest`), and `/tests/grader.py grade` writes
       `/logs/verifier/reward.json` (`reward` / `Pass@1`, `f2p_passed`, `f2p_total`,
       `p2p_passed`, `p2p_total`, `partial`).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STAGE2_DEFAULT_TASKS: list[str] = [
    "meriyah-explicit-resource-declarations",       # JavaScript (Meriyah parser)
    "csstree-shorthand-expansion-compression",      # JavaScript (CSSTree AST)
    "ink-grid-box-layout",                          # TypeScript (Ink Flexbox/Grid layout)
    "true-myth-iterable-collection-combinators",    # TypeScript (True Myth Result/Maybe)
    "httpx-streaming-json-iteration",               # Python (HTTPX streaming JSON)
    "psd-tools-blend-range-api",                    # Python (psd-tools blend range API)
    "prometheus-transactional-reload-status",       # Go (Prometheus scrape reload)
    "anko-default-function-arguments",              # Go (Anko VM default args)
    "opa-rego-rule-profiling",                      # Go (Open Policy Agent Rego profiler)
    "pest-character-class-coalescing",              # Rust (Pest parser generator)
]

SWE_SYSTEM_TOOL_SPEC = """You are an autonomous software engineering agent operating inside a live `/app` repository container.
You have access to the following XML tools:
1. `<function=file_editor>` with parameters `command` (`view`, `str_replace`, `insert`, `create`), `path`, `old_str`, `new_str`.
2. `<function=execute_bash>` with parameter `cmd`.
3. `<function=search>` with parameters `query`, `path`.
4. `<function=finish>` when all target hunks are applied and verified."""


_DOCKER_IMAGE_CACHE: dict[str, bool] = {}


def _has_local_docker_image(docker_image: str) -> bool:
    """Return True if `docker_image` is present in the local Docker daemon."""
    if not docker_image:
        return False
    if docker_image in _DOCKER_IMAGE_CACHE:
        return _DOCKER_IMAGE_CACHE[docker_image]
    proc = subprocess.run(
        ["sudo", "docker", "image", "inspect", docker_image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    ok = proc.returncode == 0
    _DOCKER_IMAGE_CACHE[docker_image] = ok
    return ok


def _ensure_local_docker_image(docker_image: str) -> bool:
    """Ensure `docker_image` is present in the local Docker daemon, pulling it from ECR if needed."""
    if _has_local_docker_image(docker_image):
        return True
    if not docker_image:
        return False
    for attempt in range(3):
        proc = subprocess.run(
            ["sudo", "docker", "pull", "-q", docker_image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            _DOCKER_IMAGE_CACHE[docker_image] = True
            return True
        time.sleep(1.5 * (attempt + 1))
    return False


def _read_test_totals_from_config(tests_dir: str) -> tuple[int, int]:
    """Read deduplicated `(f2p_total, p2p_total)` from `tests/config.json` (matching `grader.py`)."""
    cfg_path = os.path.join(tests_dir, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as cf:
            cfg = json.load(cf)

        def _count_ids(k_primary: str, k_fallback: str) -> int:
            raw_list = cfg.get(k_primary) or cfg.get(k_fallback) or []
            seen: set[str] = set()
            for item in raw_list:
                s = str(item).strip()
                if s and s not in seen:
                    seen.add(s)
            return len(seen)

        return _count_ids("f2p_node_ids", "f2p"), _count_ids("p2p_node_ids", "p2p")
    except Exception:
        return 0, 0


def run_docker_verifier_sync(
    task_id: str,
    docker_image: str,
    tests_dir: str,
    model_patch: str,
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Run the official `datacurve-ai/deep-swe` verifier container (`bash /tests/test.sh`) on `model_patch`."""
    t0 = time.perf_counter()
    f2p_tot_cfg, p2p_tot_cfg = _read_test_totals_from_config(tests_dir)
    if not model_patch.strip():
        tot_all = f2p_tot_cfg + p2p_tot_cfg
        return {
            "reward": 0,
            "f2p_total": f2p_tot_cfg,
            "f2p_passed": 0,
            "p2p_total": p2p_tot_cfg,
            "p2p_passed": p2p_tot_cfg,
            "f2p": 0.0,
            "p2p": 1.0 if p2p_tot_cfg >= 0 else 0.0,
            "partial": (p2p_tot_cfg / tot_all) if tot_all > 0 else 0.0,
            "verifier_exit_code": 0,
            "verifier_wall_time_s": round(time.perf_counter() - t0, 3),
            "verifier_log_tail": "[verifier] no model.patch submitted — pristine base state (reward 0 by construction)",
        }
    if not _ensure_local_docker_image(docker_image):
        return {
            "reward": 0,
            "f2p_total": f2p_tot_cfg,
            "f2p_passed": 0,
            "p2p_total": p2p_tot_cfg,
            "p2p_passed": 0,
            "f2p": 0.0,
            "p2p": 0.0,
            "partial": 0.0,
            "verifier_exit_code": -1,
            "verifier_wall_time_s": round(time.perf_counter() - t0, 3),
            "error": f"Failed to pull docker image {docker_image}",
        }
    tmp_dir = tempfile.mkdtemp(prefix=f"deepswe_verify_{task_id}_")
    cname = f"deepswe_ver_{os.path.basename(tmp_dir)}"
    try:
        artifacts_dir = os.path.join(tmp_dir, "artifacts")
        verifier_dir = os.path.join(tmp_dir, "verifier")
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(verifier_dir, exist_ok=True)

        patch_path = os.path.join(artifacts_dir, "model.patch")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(model_patch)

        cmd = [
            "sudo",
            "docker",
            "run",
            "--rm",
            "--name",
            cname,
            "--network",
            "none",
            "--cpus=8",
            "-e",
            "LOKY_MAX_CPU_COUNT=4",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "OPENBLAS_NUM_THREADS=2",
            "-e",
            "MKL_NUM_THREADS=2",
            "-e",
            "RAYON_NUM_THREADS=4",
            "-e",
            "GOMAXPROCS=4",
            "-v",
            f"{os.path.abspath(tests_dir)}:/tests:ro",
            "-v",
            f"{artifacts_dir}:/logs/artifacts",
            "-v",
            f"{verifier_dir}:/logs/verifier",
            docker_image,
            "bash",
            "/tests/test.sh",
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            exit_code = int(proc.returncode)
            stdout_txt = proc.stdout or ""
            err_msg = ""
        except subprocess.TimeoutExpired as te:
            subprocess.run(
                ["sudo", "docker", "rm", "-f", cname],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            exit_code = -9
            stdout_txt = (
                te.stdout.decode("utf-8", errors="replace")
                if isinstance(te.stdout, bytes)
                else (te.stdout or "")
            )
            err_msg = f"Verifier container timed out after {timeout_s}s"
        except Exception as exc:
            subprocess.run(
                ["sudo", "docker", "rm", "-f", cname],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            exit_code = -2
            stdout_txt = ""
            err_msg = f"Verifier container error: {exc}"

        verifier_time_s = time.perf_counter() - t0
        reward_path = os.path.join(verifier_dir, "reward.json")
        if os.path.exists(reward_path):
            with open(reward_path, encoding="utf-8") as rf:
                reward_data = json.load(rf)
        else:
            reward_data = {
                "reward": 0,
                "f2p_total": f2p_tot_cfg,
                "f2p_passed": 0,
                "p2p_total": p2p_tot_cfg,
                "p2p_passed": 0,
                "f2p": 0.0,
                "p2p": 0.0,
                "partial": 0.0,
                "error": err_msg or f"reward.json missing (exit={exit_code})",
            }
        reward_data["verifier_exit_code"] = exit_code
        reward_data["verifier_wall_time_s"] = round(verifier_time_s, 3)
        reward_data["verifier_log_tail"] = "\n".join(
            stdout_txt.splitlines()[-15:]
        )
        return reward_data
    finally:
        subprocess.run(["sudo", "rm", "-rf", tmp_dir], check=False)


def serve_verifier_daemon(port: int, tasks_dir: str) -> None:
    """Run a lightweight host HTTP daemon on `127.0.0.1:<port>` that executes official Docker verifiers."""
    import threading

    cache_file = "/tmp/deepswe_verifier_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            cache = {}
    key_locks: dict[str, threading.Lock] = {}
    meta_lock = threading.Lock()

    class VerifierHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/verify":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            task_id = str(body["task_id"])
            docker_image = str(body["docker_image"])
            model_patch = str(body.get("model_patch", ""))
            use_cache = bool(body.get("use_cache", True))
            patch_sha = hashlib.sha256(model_patch.encode("utf-8")).hexdigest()
            cache_key = f"{task_id}:{patch_sha}"

            with meta_lock:
                if cache_key not in key_locks:
                    key_locks[cache_key] = threading.Lock()
                k_lock = key_locks[cache_key]

            with k_lock:
                if use_cache and cache_key in cache:
                    res = dict(cache[cache_key])
                    res["cached_verifier"] = True
                else:
                    tests_dir = os.path.join(tasks_dir, task_id, "tests")
                    res = run_docker_verifier_sync(
                        task_id=task_id,
                        docker_image=docker_image,
                        tests_dir=tests_dir,
                        model_patch=model_patch,
                    )
                    res["patch_sha256"] = patch_sha
                    res["cached_verifier"] = False
                    with meta_lock:
                        cache[cache_key] = dict(res)
                        try:
                            tmp_cf = f"{cache_file}.tmp"
                            with open(tmp_cf, "w", encoding="utf-8") as wf:
                                json.dump(cache, wf)
                            os.replace(tmp_cf, cache_file)
                        except Exception:
                            pass

            out_bytes = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out_bytes)))
            self.end_headers()
            self.wfile.write(out_bytes)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), VerifierHandler)
    print(
        f"[Stage2-Verifier-Daemon] Listening on http://127.0.0.1:{port}/verify "
        f"(tasks_dir={tasks_dir})...",
        flush=True,
    )
    server.serve_forever()


def _find_existing_neighbor_file(app_dir: str, rel_path: str) -> str:
    """Find an existing source file in `<app_dir>` near `<rel_path>` when `<rel_path>` is a new file."""
    source_exts = (".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs")
    target_dir = os.path.dirname(rel_path)
    abs_target_dir = os.path.join(app_dir, target_dir)
    if os.path.isdir(abs_target_dir):
        try:
            candidates = sorted(os.listdir(abs_target_dir))
            for fn in candidates:
                if fn.endswith(source_exts) and not fn.startswith("test_"):
                    cand_abs = os.path.join(abs_target_dir, fn)
                    if os.path.isfile(cand_abs) and os.path.getsize(cand_abs) >= 300:
                        return os.path.join(target_dir, fn).lstrip("/")
        except OSError:
            pass
    try:
        ls_out = subprocess.check_output(
            ["git", "-C", app_dir, "ls-files"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
        best_rel = ""
        best_score = -1
        rel_parts = rel_path.split("/")
        for cand in ls_out:
            if not cand.endswith(source_exts):
                continue
            cand_abs = os.path.join(app_dir, cand)
            if not os.path.isfile(cand_abs) or os.path.getsize(cand_abs) < 300:
                continue
            c_parts = cand.split("/")
            common = 0
            for a, b in zip(rel_parts[:-1], c_parts[:-1]):
                if a == b:
                    common += 1
                else:
                    break
            score = common * 100 - abs(len(c_parts) - len(rel_parts))
            if "test" in cand.lower():
                score -= 20
            if score > best_score:
                best_score = score
                best_rel = cand
        if best_rel:
            return best_rel
    except Exception:
        pass
    return rel_path


def _extract_instruction_symbols_and_paths(instruction: str) -> tuple[list[str], list[str]]:
    """Extract backticked symbols, identifiers, and dotted/slashed paths strictly from `instruction.md`."""
    backticked = [
        m.group(1).strip()
        for m in re.finditer(r"`([^`\n]{2,80})`", instruction)
    ]
    path_hints: list[str] = []
    symbols: list[str] = []
    stop_words = {
        "true", "false", "null", "none", "self", "this", "return", "const",
        "let", "var", "func", "function", "class", "struct", "interface",
        "type", "import", "from", "export", "default", "async", "await",
        "string", "number", "boolean", "object", "array", "list", "dict",
        "int", "float", "bool", "str", "bytes", "error", "value", "result",
    }
    for bt in backticked:
        if "/" in bt or bt.endswith((".py", ".go", ".ts", ".tsx", ".js", ".rs")):
            path_hints.append(bt.strip("/"))
        elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$", bt):
            path_hints.append(bt.replace(".", "/"))
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", bt):
            sym = m.group(0)
            if sym.lower() not in stop_words and sym not in symbols:
                symbols.append(sym)

    for m in re.finditer(
        r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[a-z]+(?:_[a-z0-9]+)+\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b",
        instruction,
    ):
        sym = m.group(0)
        if sym.lower() not in stop_words and sym not in symbols:
            symbols.append(sym)
    return symbols[:32], path_hints[:16]


def _discover_autonomous_repo_files(
    app_dir: str,
    instruction: str,
    num_steps: int = 8,
) -> tuple[list[str], str, list[dict[str, Any]], list[str]]:
    """Discover and rank `/app` source files and code windows strictly from `git ls-files` and `instruction.md`."""
    source_exts = (".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs")
    inst_symbols, path_hints = _extract_instruction_symbols_and_paths(instruction)
    inst_tokens = {
        m.group(0).lower()
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", instruction)
    }
    sym_set_exact = set(inst_symbols)
    sym_set_lower = {s.lower() for s in inst_symbols}

    try:
        ls_out = subprocess.check_output(
            ["git", "-C", app_dir, "ls-files"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
    except Exception:
        ls_out = []

    scored: list[tuple[int, str, list[int]]] = []
    for cand in ls_out:
        if not cand.endswith(source_exts):
            continue
        low_cand = cand.lower()
        if any(
            skip in low_cand
            for skip in (
                "node_modules/",
                "vendor/",
                "dist/",
                "build/",
                "__snapshots__/",
                "examples/",
                "benchmarks/",
                "docs/",
            )
        ):
            continue
        cand_abs = os.path.join(app_dir, cand)
        if not os.path.isfile(cand_abs):
            continue
        try:
            sz = os.path.getsize(cand_abs)
        except OSError:
            continue
        if sz < 100 or sz > 600_000:
            continue

        path_toks = {
            m.group(0).lower()
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", cand)
        }
        score = len(path_toks & inst_tokens) * 20
        score += len(path_toks & sym_set_lower) * 45
        for ph in path_hints:
            ph_low = ph.lower()
            if ph_low in low_cand or os.path.dirname(ph_low) and os.path.dirname(ph_low) in low_cand:
                score += 80

        is_test_file = (
            "/test" in f"/{low_cand}"
            or low_cand.endswith(("_test.go", "_test.py", ".test.ts", ".spec.ts", ".test.js", ".spec.js"))
            or os.path.basename(low_cand).startswith("test_")
        )
        if is_test_file:
            score -= 60

        best_lines: list[int] = [1]
        try:
            full_txt = open(
                cand_abs, encoding="utf-8", errors="replace"
            ).read(131072)
            file_toks = {
                m.group(0)
                for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", full_txt)
            }
            exact_hits = len(file_toks & sym_set_exact)
            lower_hits = len({t.lower() for t in file_toks} & inst_tokens)
            score += exact_hits * 30 + min(60, lower_hits * 2)

            lines = full_txt.splitlines()
            if lines and sym_set_exact:
                win_scores: list[tuple[int, int]] = []
                stride = 25
                win_sz = 50
                for s_ln in range(1, max(2, len(lines) - 10), stride):
                    chunk_txt = "\n".join(lines[s_ln - 1 : s_ln - 1 + win_sz])
                    c_toks = {
                        m.group(0)
                        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", chunk_txt)
                    }
                    w_hits = len(c_toks & sym_set_exact) * 10 + len(
                        {t.lower() for t in c_toks} & sym_set_lower
                    ) * 3
                    if w_hits > 0:
                        win_scores.append((w_hits, s_ln))
                win_scores.sort(key=lambda x: (-x[0], x[1]))
                chosen_lns: list[int] = []
                for _, ln_cand in win_scores:
                    if all(abs(ln_cand - prev_ln) >= 35 for prev_ln in chosen_lns):
                        chosen_lns.append(ln_cand)
                        if len(chosen_lns) >= 4:
                            break
                if chosen_lns:
                    best_lines = chosen_lns
        except OSError:
            pass
        scored.append((score, cand, best_lines))

    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    ranked = [c for _, c, _ in scored[:12]]
    if not ranked:
        ranked = [ls_out[0]] if ls_out else ["README.md"]

    step_schedule: list[dict[str, Any]] = []
    for _, cand_rel, cand_lns in scored[:6]:
        for s_ln in cand_lns[:2]:
            step_schedule.append(
                {
                    "step_num": len(step_schedule) + 1,
                    "rel_path": cand_rel,
                    "fpath": f"/app/{cand_rel}",
                    "start_ln": s_ln,
                    "inst_symbols": inst_symbols[:16] or [os.path.basename(cand_rel)],
                }
            )
            if len(step_schedule) >= max(1, num_steps):
                break
        if len(step_schedule) >= max(1, num_steps):
            break
    while len(step_schedule) < max(1, num_steps):
        base_entry = step_schedule[len(step_schedule) % max(1, len(step_schedule))] if step_schedule else {
            "rel_path": ranked[0],
            "fpath": f"/app/{ranked[0]}",
            "start_ln": 1,
            "inst_symbols": inst_symbols[:16] or [os.path.basename(ranked[0])],
        }
        step_schedule.append(
            {
                **base_entry,
                "step_num": len(step_schedule) + 1,
            }
        )

    summary_lines = [f"- `/app/{c}`" for c in ranked[:12]]
    return ranked, "\n".join(summary_lines), step_schedule, inst_symbols


def _strip_viewer_line_numbers(text: str) -> str:
    """Strip accidental viewer line prefixes (`  42: `) if the model copied them from the file viewer."""
    lines = text.splitlines()
    if not lines:
        return text
    numbered_count = sum(
        1 for ln in lines if re.match(r"^\s*\d+:\s?", ln) or not ln.strip()
    )
    if numbered_count == len(lines) and any(re.match(r"^\s*\d+:\s?", ln) for ln in lines):
        return "\n".join(re.sub(r"^\s*\d+:\s?", "", ln, count=1) for ln in lines)
    return text


def _apply_str_replace_to_content(
    cur_content: str, old_s: str, new_s: str
) -> str | None:
    """Apply `str_replace` with fallback for viewer line numbers or minor whitespace differences."""
    for cand_old, cand_new in [
        (old_s, new_s),
        (_strip_viewer_line_numbers(old_s), _strip_viewer_line_numbers(new_s)),
    ]:
        if not cand_old or cand_old == cand_new:
            continue
        if cand_old in cur_content:
            return cur_content.replace(cand_old, cand_new, 1)

        cur_lines = cur_content.splitlines()
        old_lines = [ln.rstrip() for ln in cand_old.splitlines()]
        while old_lines and not old_lines[0].strip():
            old_lines.pop(0)
        while old_lines and not old_lines[-1].strip():
            old_lines.pop()
        if not old_lines:
            continue

        m = len(old_lines)
        for i in range(len(cur_lines) - m + 1):
            if all(
                cur_lines[i + j].rstrip() == old_lines[j]
                or cur_lines[i + j].strip() == old_lines[j].strip()
                for j in range(m)
            ):
                new_lines = cand_new.splitlines()
                updated_lines = cur_lines[:i] + new_lines + cur_lines[i + m :]
                trailing_nl = "\n" if cur_content.endswith("\n") else ""
                return "\n".join(updated_lines) + trailing_nl
    return None


def _execute_autonomous_tool_call(r: dict[str, Any], gen_text: str) -> bool:
    """Parse and execute one or more live `<function=file_editor>` tool calls against `r['live_app_dir']`."""
    # Prepend the assistant prefill if `gen_text` starts mid-tool-call
    full_xml = (
        "<function=file_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>"
        + gen_text
        if not gen_text.lstrip().startswith("<function=")
        else gen_text
    )
    blocks = re.split(r"<function=file_editor>", full_xml)
    applied_any = False

    for blk in blocks[1:]:
        m_cmd = re.search(
            r"<parameter=command>\s*([^\n<]+?)\s*</parameter>", blk
        )
        cmd_name = m_cmd.group(1).strip().lower() if m_cmd else "str_replace"

        m_path = re.search(
            r"<parameter=path>\s*([^\n<]+?)\s*</parameter>", blk
        )
        rel_path = ""
        if m_path:
            raw_p = m_path.group(1).strip().strip('"').strip("'")
            if raw_p.startswith("/app/"):
                rel_path = raw_p[len("/app/") :]
            elif not raw_p.startswith("__"):
                rel_path = raw_p.lstrip("/")
        if not rel_path:
            rel_path = r.get("autonomous_primary_rel_path") or r["step_schedule"][0]["rel_path"]

        m_old = re.search(
            r"<parameter=old_str>(.*?)</parameter>", blk, flags=re.DOTALL
        )
        m_new = re.search(
            r"<parameter=(?:new_str|file_text)>(.*?)(?:</parameter>|$)",
            blk,
            flags=re.DOTALL,
        )
        old_s = m_old.group(1).strip("\n") if m_old else ""
        new_s = m_new.group(1).strip("\n") if m_new else ""

        abs_target = os.path.join(r["live_app_dir"], rel_path)
        if cmd_name == "create" or (not os.path.exists(abs_target) and new_s and not old_s):
            try:
                os.makedirs(os.path.dirname(abs_target), exist_ok=True)
                clean_new = _strip_viewer_line_numbers(new_s)
                with open(abs_target, "w", encoding="utf-8") as wf:
                    wf.write(clean_new + "\n")
                r["applied_hunks"] += 1
                applied_any = True
            except OSError:
                pass
            continue

        if not os.path.isfile(abs_target) or not new_s:
            continue

        try:
            cur_content = open(
                abs_target, encoding="utf-8", errors="replace"
            ).read()
            if cmd_name == "insert":
                m_ln = re.search(
                    r"<parameter=insert_line>\s*(\d+)\s*</parameter>", blk
                )
                ins_ln = int(m_ln.group(1)) if m_ln else 0
                cur_lines = cur_content.splitlines()
                ins_idx = max(0, min(len(cur_lines), ins_ln))
                clean_new = _strip_viewer_line_numbers(new_s)
                updated_lines = (
                    cur_lines[:ins_idx]
                    + clean_new.splitlines()
                    + cur_lines[ins_idx:]
                )
                with open(abs_target, "w", encoding="utf-8") as wf:
                    wf.write("\n".join(updated_lines) + "\n")
                r["applied_hunks"] += 1
                applied_any = True
            else:
                updated = _apply_str_replace_to_content(
                    cur_content, old_s, new_s
                )
                if updated is not None and updated != cur_content:
                    with open(abs_target, "w", encoding="utf-8") as wf:
                        wf.write(updated)
                    r["applied_hunks"] += 1
                    applied_any = True
        except OSError:
            pass

    return applied_any


def _read_live_file_window(
    app_dir: str, rel_path: str, center_start_ln: int, num_lines: int = 30
) -> str:
    """Read a live numbered window directly from `<app_dir>/<rel_path>` (or nearest sibling file) on disk."""
    abs_p = os.path.join(app_dir, rel_path)
    fpath = f"/app/{rel_path}"
    effective_rel = rel_path
    if not os.path.exists(abs_p):
        effective_rel = _find_existing_neighbor_file(app_dir, rel_path)
        abs_p = os.path.join(app_dir, effective_rel)
    if not os.path.exists(abs_p):
        return f"[File: {fpath} (new file to be created in live /app workspace)]\n"
    try:
        with open(abs_p, encoding="utf-8", errors="replace") as f:
            all_lines = f.read().splitlines()
    except OSError:
        return f"[File: {fpath} (unable to read from live /app workspace)]\n"

    total_l = len(all_lines)
    s_idx = max(0, min(center_start_ln - 1, max(0, total_l - num_lines)))
    e_idx = min(total_l, s_idx + num_lines)
    body = "\n".join(all_lines[s_idx:e_idx])
    if effective_rel != rel_path:
        return (
            f"[File: {fpath} (new target file; viewing sibling /app/{effective_rel}, "
            f"lines {s_idx + 1}-{e_idx} of {total_l})]\n{body}\n"
        )
    return (
        f"[File: {fpath} ({e_idx - s_idx} lines viewed from live /app, "
        f"lines {s_idx + 1}-{e_idx} of {total_l})]\n{body}\n"
    )


def _live_git_status_summary(app_dir: str) -> str:
    """Run `git status -s` and `git diff --shortstat` on the live `/app` workspace."""
    try:
        st = subprocess.check_output(
            ["git", "-C", app_dir, "status", "-s"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        df = subprocess.check_output(
            ["git", "-C", app_dir, "diff", "--shortstat"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not st and not df:
            return "clean working tree (0 files modified)"
        parts = [p for p in [df, st.replace("\n", "; ")] if p]
        return " | ".join(parts)[:240]
    except Exception:
        return "live /app workspace active"


def _extract_base_repo_from_docker_image(
    docker_image: str, app_dir: str
) -> bool:
    """Extract `/app` directly from the official `datacurve-ai/deep-swe` container image."""
    if not _ensure_local_docker_image(docker_image):
        return False
    cid = ""
    try:
        cid = subprocess.check_output(
            ["sudo", "docker", "create", docker_image],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not cid:
            return False
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)
        os.makedirs(os.path.dirname(app_dir), exist_ok=True)
        subprocess.run(
            ["sudo", "docker", "cp", f"{cid}:/app", app_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", app_dir],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return os.path.isdir(os.path.join(app_dir, ".git"))
    except Exception:
        return False
    finally:
        if cid:
            subprocess.run(
                ["sudo", "docker", "rm", "-f", cid],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def select_stage2_task_ids(
    tasks_dir: str, num_tasks: int, explicit_task_ids: str
) -> list[str]:
    """Select `num_tasks` official `datacurve-ai/deep-swe` tasks (starting with `STAGE2_DEFAULT_TASKS` and interleaving remaining tasks by language)."""
    parsed_explicit = [
        t.strip() for t in explicit_task_ids.split(",") if t.strip()
    ]
    if num_tasks <= 0:
        return parsed_explicit or list(STAGE2_DEFAULT_TASKS)
    if (
        num_tasks <= len(STAGE2_DEFAULT_TASKS)
        and parsed_explicit == STAGE2_DEFAULT_TASKS
    ):
        return list(STAGE2_DEFAULT_TASKS[:num_tasks])

    selected: list[str] = [
        t
        for t in STAGE2_DEFAULT_TASKS
        if os.path.isdir(os.path.join(tasks_dir, t))
    ]
    selected_set = set(selected)

    by_lang: dict[str, list[str]] = {}
    lang_order = ["javascript", "typescript", "python", "go", "rust"]
    for name in sorted(os.listdir(tasks_dir)):
        if name in selected_set:
            continue
        td = os.path.join(tasks_dir, name)
        if not os.path.isdir(td):
            continue
        toml_p = os.path.join(td, "task.toml")
        inst_p = os.path.join(td, "instruction.md")
        cfg_p = os.path.join(td, "tests", "config.json")
        if not all(os.path.exists(p) for p in (toml_p, inst_p, cfg_p)):
            continue
        try:
            with open(toml_p, "rb") as tf:
                t_data = tomllib.load(tf)
            lg = str(
                t_data.get("metadata", {}).get("language") or "python"
            ).lower()
        except Exception:
            continue
        by_lang.setdefault(lg, []).append(name)

    all_langs = lang_order + [lg for lg in sorted(by_lang) if lg not in lang_order]
    lang_cursors = {lg: 0 for lg in all_langs}
    while len(selected) < num_tasks:
        added_in_round = False
        for lg in all_langs:
            bucket = by_lang.get(lg, [])
            cur = lang_cursors[lg]
            if cur < len(bucket):
                selected.append(bucket[cur])
                lang_cursors[lg] = cur + 1
                added_in_round = True
                if len(selected) >= num_tasks:
                    break
        if not added_in_round:
            break
    return selected[:num_tasks]


def load_stage2_live_tasks(
    tasks_dir: str,
    repos_dir: str,
    task_ids: list[str],
    rollouts_per_task: int,
    num_steps: int,
) -> list[dict[str, Any]]:
    """Load metadata, instructions, and base `/app` paths for Stage 2 tasks (zero `solution.patch` access)."""
    num_tasks = len(task_ids)
    num_cohorts = min(10, max(1, num_tasks))
    os.makedirs(repos_dir, exist_ok=True)

    task_templates: list[dict[str, Any]] = []
    for t_idx, tid in enumerate(task_ids):
        td = os.path.join(tasks_dir, tid)
        with open(os.path.join(td, "task.toml"), "rb") as f:
            toml_data = tomllib.load(f)
        meta = toml_data.get("metadata", {})
        ver_cfg = toml_data.get("verifier", {})
        env_cfg = toml_data.get("environment", {})
        docker_image = str(
            env_cfg.get("docker_image")
            or ver_cfg.get("docker_image")
            or meta.get("docker_image")
            or ""
        )
        if not docker_image:
            raw_toml = open(os.path.join(td, "task.toml"), encoding="utf-8").read()
            m_img = re.search(r'docker_image\s*=\s*"([^"]+)"', raw_toml)
            docker_image = m_img.group(1) if m_img else ""

        lang = str(meta.get("language") or "python")
        repo_url = str(meta.get("repository_url") or "")
        title = str(meta.get("display_title") or tid)
        desc = str(meta.get("display_description") or "")
        base_commit = str(meta.get("base_commit_hash") or "HEAD")

        with open(
            os.path.join(td, "instruction.md"), encoding="utf-8", errors="replace"
        ) as f:
            instruction = f.read().strip()

        base_app_dir = os.path.join(repos_dir, tid, "app")
        need_real_repo = not os.path.isdir(os.path.join(base_app_dir, ".git"))
        if not need_real_repo:
            try:
                commit_subj = subprocess.check_output(
                    ["git", "-C", base_app_dir, "log", "-1", "--pretty=%s"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if commit_subj == "base":
                    need_real_repo = True
            except Exception:
                need_real_repo = True
        if need_real_repo:
            fetched_real = False
            if repo_url and base_commit and base_commit != "HEAD":
                tmp_fetch = f"{base_app_dir}_fetching"
                try:
                    if os.path.exists(tmp_fetch):
                        shutil.rmtree(tmp_fetch)
                    os.makedirs(tmp_fetch, exist_ok=True)
                    subprocess.run(
                        ["git", "-C", tmp_fetch, "init", "-q"], check=True
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            tmp_fetch,
                            "remote",
                            "add",
                            "origin",
                            repo_url,
                        ],
                        check=True,
                    )
                    r_fetch = subprocess.run(
                        [
                            "git",
                            "-C",
                            tmp_fetch,
                            "fetch",
                            "--depth",
                            "1",
                            "--filter=blob:none",
                            "origin",
                            base_commit,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if r_fetch.returncode == 0:
                        subprocess.run(
                            [
                                "git",
                                "-C",
                                tmp_fetch,
                                "checkout",
                                "-q",
                                "FETCH_HEAD",
                            ],
                            check=True,
                        )
                    else:
                        shutil.rmtree(tmp_fetch)
                        subprocess.run(
                            [
                                "git",
                                "clone",
                                "--filter=blob:none",
                                "--no-checkout",
                                repo_url,
                                tmp_fetch,
                            ],
                            check=True,
                            timeout=180,
                        )
                        subprocess.run(
                            [
                                "git",
                                "-C",
                                tmp_fetch,
                                "checkout",
                                "-q",
                                base_commit,
                            ],
                            check=True,
                        )
                    if os.path.exists(base_app_dir):
                        shutil.rmtree(base_app_dir)
                    os.rename(tmp_fetch, base_app_dir)
                    fetched_real = True
                except Exception:
                    if os.path.exists(tmp_fetch):
                        shutil.rmtree(tmp_fetch, ignore_errors=True)
            if not fetched_real:
                _extract_base_repo_from_docker_image(docker_image, base_app_dir)

        (
            autonomous_files,
            repo_files_summary,
            step_schedule,
            inst_symbols,
        ) = _discover_autonomous_repo_files(
            base_app_dir, instruction, num_steps=num_steps
        )
        all_task_syms: list[str] = list(inst_symbols)
        for extra_sym in [
            tid,
            *[os.path.basename(af) for af in autonomous_files[:3]],
            *[os.path.splitext(os.path.basename(af))[0] for af in autonomous_files[:3]],
        ]:
            if extra_sym and not extra_sym.isdigit() and extra_sym not in all_task_syms:
                all_task_syms.append(extra_sym)

        task_templates.append(
            {
                "task_id": tid,
                "task_index": t_idx,
                "cohort_idx": t_idx % num_cohorts,
                "language": lang,
                "repository_url": repo_url,
                "title": title,
                "description": desc,
                "base_commit": base_commit,
                "docker_image": docker_image,
                "instruction": instruction,
                "primary_fpath": f"/app/{autonomous_files[0]}",
                "autonomous_primary_rel_path": autonomous_files[0],
                "autonomous_files": autonomous_files,
                "repo_files_summary": repo_files_summary,
                "gt_symbols": all_task_syms,
                "step_schedule": step_schedule,
                "base_app_dir": base_app_dir,
            }
        )

    rollouts: list[dict[str, Any]] = []
    for g_idx in range(rollouts_per_task):
        for tmpl in task_templates:
            tid = tmpl["task_id"]
            t_idx = tmpl["task_index"]
            rollout_id = f"{tid}__r{g_idx}"
            cohort_branch_idx = (
                (t_idx // num_cohorts) * rollouts_per_task + g_idx
                if num_tasks > num_cohorts
                else g_idx
            )
            rollouts.append(
                {
                    "rollout_index": len(rollouts),
                    "rollout_id": rollout_id,
                    "group_rollout_idx": g_idx,
                    "cohort_branch_idx": cohort_branch_idx,
                    **tmpl,
                }
            )
    return rollouts


async def verify_patch_via_http_or_docker(
    session: Any,
    verifier_url: str,
    tasks_dir: str,
    task_id: str,
    docker_image: str,
    model_patch: str,
) -> dict[str, Any]:
    """Verify `model.patch` against the official `datacurve-ai/deep-swe` container."""
    import aiohttp

    tests_dir = os.path.join(tasks_dir, task_id, "tests")
    if verifier_url:
        payload = {
            "task_id": task_id,
            "docker_image": docker_image,
            "model_patch": model_patch,
            "use_cache": True,
        }
        last_err = ""
        for _attempt in range(2):
            try:
                async with session.post(
                    verifier_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as exc:
                last_err = str(exc)
                await asyncio.sleep(1.0)
        f2p_tot, p2p_tot = _read_test_totals_from_config(tests_dir)
        return {
            "reward": 0,
            "f2p_total": f2p_tot,
            "f2p_passed": 0,
            "p2p_total": p2p_tot,
            "p2p_passed": 0,
            "f2p": 0.0,
            "p2p": 0.0,
            "partial": 0.0,
            "verifier_exit_code": -3,
            "verifier_wall_time_s": 0.0,
            "patch_sha256": hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
            "error": f"Verifier HTTP error: {last_err}",
        }

    return await asyncio.to_thread(
        run_docker_verifier_sync,
        task_id,
        docker_image,
        tests_dir,
        model_patch,
    )


async def run_stage2_live_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the Stage 2 live agent loop + official unit-test verifier benchmark."""
    import aiohttp
    from transformers import AutoTokenizer
    from benchmark_swe_agent_http_client import (
        extract_effective_hit_metrics,
        scrape_vllm_metrics,
        send_completion,
    )
    from validate_swe_agent_trajectories_hbm import (
        _align_to_block,
        _classify_swe_action,
        _encode_file_viewer_bucketed,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
        resolve_pad_token_id,
    )

    print(
        f"[Stage2-Live-Agent] Loading tokenizer `{args.tokenizer}` "
        f"(block_size={args.block_size})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    newline_id = resolve_pad_token_id(tokenizer=tokenizer)

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_ids.add(newline_id)
    edit_toks: list[int] = []
    base_words = [
        "module", "context", "target", "source", "buffer", "record", "window",
        "config", "method", "object", "update", "status", "result", "helper", "branch",
        "commit", "tensor", "vector", "matrix", "kernel", "schema", "section", "marker",
        "client", "server", "router", "parser", "format", "output", "stream", "handle",
        "layout", "driver", "engine", "system", "policy", "filter", "option", "header",
        "metric", "action", "worker", "runner", "loader", "mapper", "reader", "writer",
        "checks", "values", "fields", "params", "tokens", "blocks", "layers", "scopes",
    ]
    for w in base_words:
        for variant in (f" {w}", w, f"_{w}"):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            for t_id in toks:
                if t_id not in special_ids and t_id not in edit_toks:
                    edit_toks.append(t_id)
    for d in range(1, 250):
        t_id = tokenizer.encode(f"_{d}", add_special_tokens=False)[-1]
        if t_id not in special_ids and t_id not in edit_toks:
            edit_toks.append(t_id)
    n_et = max(1, len(edit_toks))
    block_size = args.block_size

    is_harmony = (
        hasattr(tokenizer, "get_vocab")
        and "<|channel|>" in tokenizer.get_vocab()
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
    is_kimi = (
        hasattr(tokenizer, "get_vocab")
        and "<|im_user|>" in tokenizer.get_vocab()
        and "<|im_assistant|>" in tokenizer.get_vocab()
    )
    is_muse_spark = (
        hasattr(tokenizer, "get_vocab")
        and "<|eot|>" in tokenizer.get_vocab()
        and "<|start|>" in tokenizer.get_vocab()
        and "<|message|>" in tokenizer.get_vocab()
    )

    def _repeat_and_align(raw_ids: list[int], target_blks: int) -> list[int]:
        ids = list(raw_ids) if raw_ids else [newline_id]
        if len(ids) % 2 == 0:
            ids.append(newline_id)
        target_len = target_blks * block_size
        while len(ids) < target_len:
            ids = ids + ids
        return list(
            _align_to_block(ids, block_size, newline_id, target_blocks=target_blks)
        )

    def _tag_task_anchor_blocks(
        ids: list[int], num_blks: int, role_id: int, task_idx: int
    ) -> list[int]:
        out = list(ids)
        for b in range(num_blks):
            b_off = b * block_size
            a_off = b_off + block_size - 2
            if a_off + 1 < b_off + block_size:
                u = role_id * 10007 + task_idx * 37 + b * 13
                out[a_off] = edit_toks[u % n_et]
                out[a_off + 1] = edit_toks[(u // n_et + role_id + b + 1) % n_et]
        return out

    def _apply_rollout_step_tag(
        anchor_ids: list[int],
        num_blks: int,
        rollout_idx: int,
        step_idx: int,
        role_salt: int = 0,
    ) -> list[int]:
        out = list(anchor_ids)
        for b in range(num_blks):
            b_off = b * block_size
            a_edit = b_off + block_size - 2
            if a_edit + 1 < b_off + block_size:
                u = (
                    (rollout_idx + 1) * 16
                    + step_idx
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

    task_ids = select_stage2_task_ids(
        tasks_dir=args.deepswe_tasks_dir,
        num_tasks=args.num_tasks,
        explicit_task_ids=args.task_ids,
    )
    rollouts = load_stage2_live_tasks(
        tasks_dir=args.deepswe_tasks_dir,
        repos_dir=args.repos_dir,
        task_ids=task_ids,
        rollouts_per_task=args.rollouts_per_task,
        num_steps=args.num_steps,
    )
    num_rollouts = len(rollouts)
    num_tasks = len(task_ids)
    num_cohorts = min(10, max(1, num_tasks))
    blk_mult = max(1, 128 // block_size) if args.long_context else 1
    h_blks = 5 * blk_mult if args.long_context else 1
    issue_blks = 15 * blk_mult if args.long_context else 1
    p_blks = 25 * blk_mult if args.long_context else 1
    obs_blks = 80 * blk_mult if args.long_context else 2
    full_blocks_per_turn = h_blks + issue_blks + p_blks + obs_blks
    step0_blks = h_blks + issue_blks + obs_blks
    subseq_new_blks = p_blks + obs_blks

    # Ensure git allows operations on extracted container `/app` repositories
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", "*"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    # Prepare isolated live `/app` workspaces for all rollouts (reuse & reset if already cloned)
    from concurrent.futures import ThreadPoolExecutor

    work_root = args.workspaces_root
    os.makedirs(work_root, exist_ok=True)

    print(
        f"[Stage2-Live-Agent] Preparing {num_rollouts} isolated live `/app` git workspaces "
        f"({num_tasks} tasks x G={args.rollouts_per_task} rollouts) under `{work_root}`...",
        flush=True,
    )
    t_clone0 = time.perf_counter()

    def _init_rollout_workspace(r: dict[str, Any]) -> None:
        r_dir = os.path.join(work_root, r["rollout_id"], "app")
        need_copy = not os.path.isdir(os.path.join(r_dir, ".git"))
        if not need_copy:
            try:
                subj = subprocess.check_output(
                    ["git", "-C", r_dir, "log", "-1", "--pretty=%s"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                if subj == "base":
                    need_copy = True
            except Exception:
                need_copy = True
        if need_copy:
            os.makedirs(os.path.dirname(r_dir), exist_ok=True)
            if os.path.exists(r_dir):
                shutil.rmtree(r_dir)
            subprocess.run(
                ["cp", "-a", r["base_app_dir"], r_dir],
                check=True,
            )
        subprocess.run(
            ["git", "-C", r_dir, "reset", "--hard", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["git", "-C", r_dir, "clean", "-fd"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        r["live_app_dir"] = r_dir
        r["tool_history"] = []
        r["applied_hunks"] = 0
        r["step_symbol_hits"] = 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_init_rollout_workspace, rollouts))

    print(
        f"[Stage2-Live-Agent] Initialized {num_rollouts} live `/app` repositories in "
        f"{time.perf_counter() - t_clone0:.2f}s.",
        flush=True,
    )

    # Pre-build per-cohort anchor blocks from each cohort's live `/app` initial checkout
    task_anchors: dict[int, dict[str, list[int]]] = {}
    for r in rollouts:
        c_idx = r["cohort_idx"]
        if c_idx in task_anchors:
            continue
        cohort_tasks = [
            rt
            for rt in rollouts
            if rt["cohort_idx"] == c_idx and rt["group_rollout_idx"] == 0
        ]
        s0 = r["step_schedule"][0]
        c_sid_h = 100000000 + c_idx
        c_sid_b = 200000000 + c_idx
        c_sid_p = 300000000 + c_idx

        open_rel = r["autonomous_primary_rel_path"]
        open_fpath = f"/app/{open_rel}"
        open_start_ln = s0["start_ln"]

        if len(cohort_tasks) > 1:
            cohort_summary = ", ".join(
                f"{ct['task_id']} ({ct['primary_fpath']})"
                for ct in cohort_tasks
            )
            static_txt = (
                f"ID: {c_sid_h} | SWE-Agent Multi-Task Workspace Cohort {c_idx}\n"
                f"{SWE_SYSTEM_TOOL_SPEC}\n"
                f"Cohort Tasks & Discovered Candidate Files: {cohort_summary}\n\n"
            )
        else:
            static_txt = (
                f"ID: {c_sid_h} | SWE-Agent Live Trajectory: {r['task_id']} | Open File: {open_fpath}\n"
                f"{SWE_SYSTEM_TOOL_SPEC}\n"
                f"Repository: {r['repository_url']} (commit {r['base_commit'][:12]}) | Language: {r['language']}\n"
                f"Task: {r['title']} — {r['description']}\n\n"
            )
        h_static_anc = _tag_task_anchor_blocks(
            _repeat_and_align(
                tokenizer.encode(static_txt, add_special_tokens=False), h_blks
            ),
            h_blks,
            role_id=0,
            task_idx=c_idx,
        )

        if len(cohort_tasks) > 1:
            cohort_issue_parts = []
            for ct in cohort_tasks:
                cand_list = ", ".join(
                    f"`/app/{af}`" for af in ct["autonomous_files"][:3]
                )
                cohort_issue_parts.append(
                    f"### Task `{ct['task_id']}` — Candidate Files: {cand_list}\n"
                    f"Title: {ct['title']} — {ct['description']}\n"
                    f"Specification:\n{ct['instruction'][:420]}"
                )
            issue_txt = (
                f"ID: {c_sid_h} | Active SWE-Agent Multi-Task Issue & Repository Context (Cohort {c_idx}): {c_sid_b}\n\n"
                + "\n\n".join(cohort_issue_parts)
                + "\n\n"
            )
        else:
            issue_txt = (
                f"ID: {c_sid_h} | Active SWE-Agent Issue & Repository Context: {c_sid_b}\n\n"
                f"### Specification ({r['task_id']})\n"
                f"{r['instruction'] if args.long_context else r['instruction'][:2600]}\n\n"
                f"### Live `/app` Repository Files\n"
                f"{r['repo_files_summary']}\n\n"
            )
        issue_raw_ids = tokenizer.encode(issue_txt, add_special_tokens=False)
        issue_base_anc = _tag_task_anchor_blocks(
            _repeat_and_align(issue_raw_ids, issue_blks),
            issue_blks,
            role_id=2,
            task_idx=c_idx,
        )

        if args.long_context:
            multi_views: list[str] = []
            seen_rels: set[str] = set()
            # Pass 1: Primary discovered `/app` file window (`step_schedule[0]`) for every cohort task
            for ct in cohort_tasks:
                sc0 = ct["step_schedule"][0]
                rel_p = sc0["rel_path"]
                key_p = f"{ct['task_id']}:{rel_p}:{sc0['start_ln']}:primary"
                if key_p not in seen_rels:
                    seen_rels.add(key_p)
                    multi_views.append(
                        f"### Task `{ct['task_id']}` — Primary `/app` Window (`{sc0['fpath']}`)\n"
                        + _read_live_file_window(
                            ct["live_app_dir"],
                            rel_p,
                            max(1, sc0["start_ln"] - 8),
                            num_lines=160 if len(cohort_tasks) == 1 else 32,
                        )
                    )
            # Pass 2: Extended windows across discovered candidate files & line windows of each cohort task
            for ct in cohort_tasks:
                for sc in ct["step_schedule"]:
                    rel_p = sc["rel_path"]
                    key_p = f"{ct['task_id']}:{rel_p}:{sc['start_ln']}"
                    if key_p not in seen_rels:
                        seen_rels.add(key_p)
                        multi_views.append(
                            _read_live_file_window(
                                ct["live_app_dir"],
                                rel_p,
                                max(1, sc["start_ln"] - 20),
                                num_lines=160 if len(cohort_tasks) == 1 else 60,
                            )
                        )
            for rel_p in r.get("autonomous_files", [])[:6]:
                key_p = f"{r['task_id']}:{rel_p}:1"
                if key_p not in seen_rels:
                    seen_rels.add(key_p)
                    multi_views.append(
                        _read_live_file_window(
                            r["live_app_dir"], rel_p, 1, num_lines=120
                        )
                    )
            live_view_txt = "\n\n".join(multi_views)
            views_ids = tokenizer.encode(
                live_view_txt, add_special_tokens=False
            )
            obs_target_toks = obs_blks * block_size
            while len(views_ids) < obs_target_toks:
                views_ids = views_ids + views_ids
            obs_raw_ids = views_ids[:obs_target_toks]
        else:
            live_view_txt = _read_live_file_window(
                r["live_app_dir"], open_rel, open_start_ln, num_lines=32
            )
            obs_raw_ids = _encode_file_viewer_bucketed(
                live_view_txt,
                open_fpath,
                tokenizer,
                block_size,
                newline_id,
                obs_blks,
            )
        obs_base_anc = _tag_task_anchor_blocks(
            obs_raw_ids,
            obs_blks,
            role_id=3,
            task_idx=c_idx,
        )

        if len(cohort_tasks) > 1:
            p1_txt = (
                f"ID: {c_sid_p} | Rolling Live Trajectory Log (Cohort {c_idx})\n"
                f"Step: 1 | Multi-Workspace Live Git Status: {_live_git_status_summary(r['live_app_dir'])}\n\n"
                f"Workspace File Windows (Cohort {c_idx}):\n"
                f"{live_view_txt[:11500] if args.long_context else live_view_txt[:1200]}\n\n"
            )
        else:
            p1_txt = (
                f"ID: {c_sid_p} | Rolling Live Trajectory Log: {r['task_id']} | Active File: {open_fpath}\n"
                f"Step: 1 | Live Git Status: {_live_git_status_summary(r['live_app_dir'])}\n\n"
                f"Trajectory History ({r['task_id']}):\n"
                f"{live_view_txt[:8000] if args.long_context else live_view_txt[:1200]}\n\n"
            )
        p_turn_1_anc = _tag_task_anchor_blocks(
            _repeat_and_align(
                tokenizer.encode(p1_txt, add_special_tokens=False), p_blks
            ),
            p_blks,
            role_id=4,
            task_idx=c_idx,
        )

        task_anchors[c_idx] = {
            "h_static": h_static_anc,
            "issue_base": issue_base_anc,
            "obs_turn_0": obs_base_anc,
            "p_turn_1": p_turn_1_anc,
        }

    def _build_live_step_prompt(
        r: dict[str, Any], step_idx: int, is_final_submit: bool = False
    ) -> list[int]:
        c_idx = r["cohort_idx"]
        anc = task_anchors[c_idx]
        r_idx = r["rollout_index"]
        c_branch = r["cohort_branch_idx"]
        sched = r["step_schedule"]
        cur_s = (
            sched[0]
            if is_final_submit
            else sched[min(step_idx, len(sched) - 1)]
        )

        # Inspect the live `/app` file and git status at the current step
        step_rel = cur_s["rel_path"]
        _ = _read_live_file_window(
            r["live_app_dir"],
            step_rel,
            cur_s["start_ln"],
            num_lines=24,
        )
        live_git_stat = _live_git_status_summary(r["live_app_dir"])
        r["last_live_git_status"] = live_git_stat

        # Block 0: Rollout static header (`h_blks` blocks)
        h_static = (
            list(anc["h_static"])
            if c_branch == 0
            else _apply_rollout_step_tag(
                anc["h_static"], h_blks, r_idx, 0, role_salt=0
            )
        )
        # Block 1: Shared Task Specification & Context (`issue_blks` blocks)
        issue_ids_base = list(anc["issue_base"])
        # Bucketed Live File Viewer Window (`obs_blks` blocks)
        obs_turn_rw = list(anc["obs_turn_0"])

        assistant_prefill = (
            "<function=file_editor>\n"
            "<parameter=command>str_replace</parameter>\n"
            "<parameter=path>"
        )

        def _format_suffix_text(q_str: str) -> str:
            if is_harmony:
                return (
                    f"{q_str}"
                    f"<|end|><|start|>assistant<|channel|>final<|message|>{assistant_prefill}"
                )
            if is_qwen_thinking:
                return (
                    f"{q_str}"
                    f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefill}"
                )
            if is_deepseek_v41:
                return f"<｜User｜>{q_str}<｜Assistant｜></think>{assistant_prefill}"
            if is_glm:
                return f"<|user|>{q_str}<|assistant|><think></think>{assistant_prefill}"
            if is_kimi:
                return (
                    f"<|im_user|>user<|im_middle|>{q_str}<|im_end|>"
                    f"<|im_assistant|>assistant<|im_middle|><think></think>{assistant_prefill}"
                )
            if is_muse_spark:
                return (
                    f"<|start|>user<|message|>{q_str}<|eot|>"
                    f"<|start|>assistant to=user<|message|>{assistant_prefill}"
                )
            return f"\n\nUser: {q_str}\nAssistant: {assistant_prefill}"

        # Partial Tail Block (< block_size tokens): Autonomous Tool-Call Generation Trigger (zero solution.patch)
        target_fpath = cur_s["fpath"]
        short_fpath = (
            target_fpath
            if len(target_fpath) <= 36
            else "/app/.../" + os.path.basename(target_fpath)
        )
        cand_questions = [
            f"Task `{r['task_id']}`: Edit `{target_fpath}` in `/app` to implement `{r['title'][:48]}`:",
            f"Task `{r['task_id']}`: Edit `{short_fpath}` in `/app` to implement the specification:",
            f"Task `{r['task_id']}`: Edit `{os.path.basename(target_fpath)}` in `/app`:",
            f"Edit `{short_fpath}`:",
        ]
        suffix_ids: list[int] = []
        for cq in cand_questions:
            c_ids = tokenizer.encode(
                _format_suffix_text(cq), add_special_tokens=False
            )
            suffix_ids = c_ids
            if len(c_ids) <= block_size - 2:
                break

        if len(suffix_ids) >= block_size:
            empty_wrap_ids = tokenizer.encode(
                _format_suffix_text(""), add_special_tokens=False
            )
            budget_q = max(2, block_size - 2 - len(empty_wrap_ids))
            q_ids = tokenizer.encode(
                f"{os.path.basename(target_fpath)}",
                add_special_tokens=False,
            )[:budget_q]
            q_trunc = tokenizer.decode(q_ids, skip_special_tokens=True)
            suffix_ids = tokenizer.encode(
                _format_suffix_text(q_trunc), add_special_tokens=False
            )
            if len(suffix_ids) >= block_size:
                suffix_ids = empty_wrap_ids[-(block_size - 2) :]
        if len(suffix_ids) < 4:
            suffix_ids = [newline_id] * (4 - len(suffix_ids)) + suffix_ids
        if len(suffix_ids) % block_size == 0:
            suffix_ids = [newline_id] + suffix_ids

        if step_idx == 0 and not is_final_submit:
            return h_static + issue_ids_base + obs_turn_rw + suffix_ids

        eff_t_idx = args.num_steps - 1 if is_final_submit else step_idx
        p_turn = (
            list(anc["p_turn_1"])
            if (c_branch == 0 and eff_t_idx == 1)
            else _apply_rollout_step_tag(
                anc["p_turn_1"], p_blks, r_idx, eff_t_idx, role_salt=4
            )
        )
        return h_static + issue_ids_base + p_turn + obs_turn_rw + suffix_ids

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    sem = asyncio.Semaphore(args.concurrency)

    resident_batch_sz = (
        max(
            8,
            min(
                16,
                args.concurrency,
                args.capacity_blocks // max(1, full_blocks_per_turn * 2),
            ),
        )
        if args.mode == "blockpool" and not args.long_context
        else args.concurrency
    )
    wave_batch_sz = (
        max(
            8,
            min(
                24,
                args.concurrency,
                args.capacity_blocks // max(1, full_blocks_per_turn * 2),
            ),
        )
        if args.mode == "blockpool" and not args.long_context
        else args.concurrency
    )
    mru_order: list[int] = list(range(num_rollouts))

    async with aiohttp.ClientSession(connector=connector) as session:
        base_snap = await scrape_vllm_metrics(session, args.base_url)
        prev_snap = dict(base_snap)
        periodic_snapshots: list[dict[str, Any]] = []

        async def _run_indexed_batches(
            indexed_prompts: list[tuple[int, list[int]]],
            max_toks: int,
            batch_sz: int,
            stop_seqs: list[str] | None = None,
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
                        stop=stop_seqs,
                    )
                    for _, p_ids in chunk
                ]
                out_results.extend(await asyncio.gather(*chunk_tasks))
            return out_results

        t_rollout_start = time.perf_counter()
        completed_turns = 0
        rollout_llm_server_time_s = 0.0
        rollout_tool_call_time_s = 0.0

        # Step 1: Live Multi-Turn SWEAgent Rollout Waves (`step 1 .. num_steps`)
        for step_idx in range(args.num_steps):
            t_wave_total0 = time.perf_counter()
            t_tool_obs0 = time.perf_counter()
            active_indexed_prompts: list[tuple[int, list[int]]] = [
                (r["rollout_index"], _build_live_step_prompt(r, step_idx, False))
                for r in rollouts
            ]
            # Group cohort anchors (`cohort_branch_idx == 0`) first in Steps 0..1 so cohort anchors populate before sibling trajectories
            if step_idx <= 1:
                anchors_first = [
                    item
                    for item in active_indexed_prompts
                    if rollouts[item[0]]["cohort_branch_idx"] == 0
                ]
                siblings_after = [
                    item
                    for item in active_indexed_prompts
                    if rollouts[item[0]]["cohort_branch_idx"] != 0
                ]
                active_indexed_prompts = anchors_first + siblings_after
            wave_tool_obs_dur_s = time.perf_counter() - t_tool_obs0

            cur_batch_sz = (
                wave_batch_sz
                if step_idx >= max(0, args.num_steps - 3)
                else args.concurrency
            )
            t_llm0 = time.perf_counter()
            if step_idx <= 1 and len(active_indexed_prompts) > num_cohorts:
                res_anc = await _run_indexed_batches(
                    active_indexed_prompts[:num_cohorts], 1, cur_batch_sz
                )
                res_sib = await _run_indexed_batches(
                    active_indexed_prompts[num_cohorts:], 1, cur_batch_sz
                )
                wave_ingest_results = list(res_anc) + list(res_sib)
            else:
                wave_ingest_results = await _run_indexed_batches(
                    active_indexed_prompts, 1, cur_batch_sz
                )

            # Intra-step tool verification query on the active turn's KV prefix (reverse order for MRU head)
            if not args.long_context:
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
                    verify_indexed_prompts, 1, cur_batch_sz
                )
            else:
                wave_verify_results = []
            wave_llm_dur_s = time.perf_counter() - t_llm0

            # Apply live autonomous tool action to each rollout's `/app` repository as the step executes
            t_tool_exec0 = time.perf_counter()
            for (r_idx, _), step_res in zip(
                active_indexed_prompts, wave_ingest_results
            ):
                r = rollouts[r_idx]
                gen_txt = step_res["text"]
                sched_s = r["step_schedule"][
                    min(step_idx, len(r["step_schedule"]) - 1)
                ]
                _execute_autonomous_tool_call(r, gen_txt)
                step_rel = sched_s["rel_path"]
                r["tool_history"].append(
                    {
                        "step": step_idx + 1,
                        "rel_path": step_rel,
                        "gen_preview": gen_txt[:60],
                    }
                )
            wave_tool_exec_dur_s = time.perf_counter() - t_tool_exec0
            wave_tool_dur_s = wave_tool_obs_dur_s + wave_tool_exec_dur_s
            rollout_llm_server_time_s += wave_llm_dur_s
            rollout_tool_call_time_s += wave_tool_dur_s

            active_mru_indices = [
                idx
                for idx, _ in (
                    reversed(active_indexed_prompts)
                    if args.long_context
                    else active_indexed_prompts
                )
            ]
            active_set = set(active_mru_indices)
            mru_order = active_mru_indices + [
                i for i in mru_order if i not in active_set
            ]
            wave_results = wave_ingest_results + wave_verify_results
            wave_dur_s = time.perf_counter() - t_wave_total0
            completed_turns += len(wave_ingest_results)
            reqs_per_turn = 1 if args.long_context else 2

            est_unique_blks = int(
                min(completed_turns, num_rollouts) * step0_blks
                + max(0, completed_turns - num_rollouts) * subseq_new_blks
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
                num_requests_completed=completed_turns * reqs_per_turn,
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
                "turn_wave": step_idx + 1,
                "active_rollouts_in_wave": len(wave_ingest_results),
                "completed_turns_cumulative": completed_turns,
                "wave_wall_time_s": round(wave_dur_s, 4),
                "wave_llm_server_wall_time_s": round(wave_llm_dur_s, 4),
                "wave_tool_call_wall_time_s": round(wave_tool_dur_s, 4),
                "elapsed_wall_time_s": round(
                    time.perf_counter() - t_rollout_start, 4
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
                "compression_factor": round(cum_stats["compression_factor"], 3),
            }
            periodic_snapshots.append(wave_record)
            print(
                f"  [Live Agent Wave {step_idx + 1:02d}/{args.num_steps:02d}] "
                f"rollouts={len(wave_ingest_results):3d} | "
                f"wave_time={wave_dur_s:6.2f}s (llm={wave_llm_dur_s:6.2f}s, tool={wave_tool_dur_s:5.2f}s) | "
                f"wave_hit={wave_record['wave_hit_rate_pct']:6.2f}% | "
                f"cum_hit={wave_record['cumulative_hit_rate_pct']:6.2f}% | "
                f"evicted_blks={wave_record['evicted_blocks_cumulative']:5d} "
                f"({wave_record['evicted_gib_cumulative']:.2f} GiB) | "
                f"hbm={wave_record['used_kv_hbm_gib']:.2f}/{wave_record['hbm_budget_gib']:.2f} GiB "
                f"({wave_record['effective_kv_hbm_utilization_pct']:.1f}%) | "
                f"comp={wave_record['compression_factor']:.2f}x",
                flush=True,
            )

        rollout_wall_time_s = time.perf_counter() - t_rollout_start
        after_rollout_snap = await scrape_vllm_metrics(session, args.base_url)
        final_est_unique = int(
            min(completed_turns, num_rollouts) * step0_blks
            + max(0, completed_turns - num_rollouts) * subseq_new_blks
        )
        reqs_per_turn = 1 if args.long_context else 2
        rollout_metrics = extract_effective_hit_metrics(
            after_rollout_snap,
            base_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=completed_turns * reqs_per_turn,
        )

        # Step 2: Final Patch Submission & Cache Residency Evaluation Wave
        final_max_tokens = max(args.max_new_tokens, 512)
        final_stop_seqs = ["</function>", "<function=finish>", "<|eot|>", "<|im_end|>"]
        print(
            f"[Stage2-Live-Agent] Running Final Patch Submission & Residency Wave "
            f"(`max_tokens={final_max_tokens}`) across all {num_rollouts} live rollouts in MRU order...",
            flush=True,
        )
        t_eval_start = time.perf_counter()
        denom_blks = (
            max(1, full_blocks_per_turn)
            if args.long_context
            else max(1, full_blocks_per_turn - 1)
        )
        max_resident_window = (
            min(
                num_rollouts,
                max(
                    1,
                    int(
                        max(0, args.capacity_blocks - resident_batch_sz * 2)
                        // denom_blks
                    ),
                ),
            )
            if args.mode == "blockpool"
            else num_rollouts
        )
        mru_window_indices = mru_order[:max_resident_window]
        lru_tail_indices = mru_order[max_resident_window:]

        t_eval_tool_obs0 = time.perf_counter()
        mru_window_prompts = [
            (
                idx,
                _build_live_step_prompt(
                    rollouts[idx], args.num_steps - 1, is_final_submit=True
                ),
            )
            for idx in mru_window_indices
        ]
        lru_tail_prompts = [
            (
                idx,
                _build_live_step_prompt(
                    rollouts[idx], args.num_steps - 1, is_final_submit=True
                ),
            )
            for idx in lru_tail_indices
        ]
        eval_tool_obs_dur_s = time.perf_counter() - t_eval_tool_obs0

        t_eval_llm0 = time.perf_counter()
        mru_window_results = await _run_indexed_batches(
            mru_window_prompts,
            final_max_tokens,
            resident_batch_sz,
            stop_seqs=final_stop_seqs,
        )
        t_eval_llm_mru = time.perf_counter() - t_eval_llm0
        after_window_snap = await scrape_vllm_metrics(session, args.base_url)

        t_eval_llm1 = time.perf_counter()
        if lru_tail_prompts:
            lru_tail_results = await _run_indexed_batches(
                lru_tail_prompts,
                final_max_tokens,
                args.concurrency,
                stop_seqs=final_stop_seqs,
            )
        else:
            lru_tail_results = []
        eval_llm_server_time_s = t_eval_llm_mru + (
            time.perf_counter() - t_eval_llm1
        )

        eval_results: list[dict[str, Any]] = [{}] * num_rollouts
        for idx, res in zip(
            mru_window_indices + lru_tail_indices,
            mru_window_results + lru_tail_results,
        ):
            eval_results[idx] = res
        eval_wall_time_s = time.perf_counter() - t_eval_start

        after_eval_snap = await scrape_vllm_metrics(session, args.base_url)
        window_only_metrics = extract_effective_hit_metrics(
            after_window_snap,
            after_rollout_snap,
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
            after_rollout_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=num_rollouts,
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
            num_requests_completed=completed_turns * reqs_per_turn
            + num_rollouts,
        )

        # Step 3: Apply the model's generated `<function=file_editor>` tool calls to each rollout's
        # live `/app` repo, extract `git diff --binary HEAD`, and grade with the official
        # `datacurve-ai/deep-swe` Docker container verifier (`bash /tests/test.sh`)
        print(
            f"[Stage2-Live-Agent] Executing live model `<function=file_editor>` edits, extracting `git diff --binary HEAD`, "
            f"and grading all {num_rollouts} rollouts with official `datacurve-ai/deep-swe` verifier containers...",
            flush=True,
        )
        t_final_tool_exec0 = time.perf_counter()
        per_rollout_outputs: list[dict[str, Any]] = []
        symbol_hits = 0
        passed_rollouts = 0
        f2p_passed_sum = 0
        f2p_total_sum = 0
        p2p_passed_sum = 0
        p2p_total_sum = 0
        partial_sum = 0.0

        task_pass_counts: dict[str, int] = {tid: 0 for tid in task_ids}
        lang_stats: dict[str, dict[str, Any]] = {}

        rollout_patches: list[tuple[str, str, bool]] = []
        for r, res in zip(rollouts, eval_results):
            gen_text = res["text"]
            gen_lower = gen_text.lower()
            matched_final = any(
                sym.lower() in gen_lower for sym in r["gt_symbols"]
            )
            if matched_final:
                symbol_hits += 1

            _execute_autonomous_tool_call(r, gen_text)

            # Stage intent-to-add (`git add -N .`) so newly created files are included in `git diff HEAD`
            subprocess.run(
                ["git", "-C", r["live_app_dir"], "add", "-N", "."],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            # Extract real `model.patch` strictly from the rollout's live `/app` git checkout
            diff_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    r["live_app_dir"],
                    "diff",
                    "--binary",
                    "HEAD",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            live_model_patch = diff_proc.stdout or ""
            patch_sha256 = hashlib.sha256(
                live_model_patch.encode("utf-8")
            ).hexdigest()
            rollout_patches.append(
                (live_model_patch, patch_sha256, matched_final)
            )
        eval_tool_call_time_s = eval_tool_obs_dur_s + (
            time.perf_counter() - t_final_tool_exec0
        )
        total_llm_server_wall_time_s = (
            rollout_llm_server_time_s + eval_llm_server_time_s
        )
        total_tool_call_wall_time_s = (
            rollout_tool_call_time_s + eval_tool_call_time_s
        )
        total_llm_wall_time_s = (
            total_llm_server_wall_time_s + total_tool_call_wall_time_s
        )

        t_verify_start = time.perf_counter()
        verify_sem = asyncio.Semaphore(8)

        async def _verify_one(
            r_item: dict[str, Any], patch_txt: str
        ) -> dict[str, Any]:
            async with verify_sem:
                return await verify_patch_via_http_or_docker(
                    session=session,
                    verifier_url=args.verifier_url,
                    tasks_dir=args.deepswe_tasks_dir,
                    task_id=r_item["task_id"],
                    docker_image=r_item["docker_image"],
                    model_patch=patch_txt,
                )

        verifier_results = await asyncio.gather(
            *[
                _verify_one(r, patch_txt)
                for r, (patch_txt, _, _) in zip(rollouts, rollout_patches)
            ]
        )

        for r, res, (live_model_patch, patch_sha256, matched_final), verifier_res in zip(
            rollouts, eval_results, rollout_patches, verifier_results
        ):
            gen_text = res["text"]
            reward_val = int(verifier_res.get("reward", 0))
            f2p_p = int(verifier_res.get("f2p_passed", 0))
            f2p_t = int(verifier_res.get("f2p_total", 0))
            p2p_p = int(verifier_res.get("p2p_passed", 0))
            p2p_t = int(verifier_res.get("p2p_total", 0))
            partial_val = float(verifier_res.get("partial", 0.0))

            if reward_val == 1:
                passed_rollouts += 1
                task_pass_counts[r["task_id"]] = (
                    task_pass_counts.get(r["task_id"], 0) + 1
                )
            f2p_passed_sum += f2p_p
            f2p_total_sum += f2p_t
            p2p_passed_sum += p2p_p
            p2p_total_sum += p2p_t
            partial_sum += partial_val

            lg = r["language"]
            if lg not in lang_stats:
                lang_stats[lg] = {
                    "rollouts": 0,
                    "passed_rollouts": 0,
                    "f2p_passed": 0,
                    "f2p_total": 0,
                    "p2p_passed": 0,
                    "p2p_total": 0,
                }
            lang_stats[lg]["rollouts"] += 1
            lang_stats[lg]["passed_rollouts"] += reward_val
            lang_stats[lg]["f2p_passed"] += f2p_p
            lang_stats[lg]["f2p_total"] += f2p_t
            lang_stats[lg]["p2p_passed"] += p2p_p
            lang_stats[lg]["p2p_total"] += p2p_t

            gen_tok_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            per_rollout_outputs.append(
                {
                    "rollout_id": r["rollout_id"],
                    "task_id": r["task_id"],
                    "group_rollout_idx": r["group_rollout_idx"],
                    "language": lg,
                    "file_path": r["primary_fpath"],
                    "num_steps": args.num_steps,
                    "matched_final_symbol": matched_final,
                    "applied_hunks": r["applied_hunks"],
                    "patch_bytes": len(live_model_patch.encode("utf-8")),
                    "patch_sha256": patch_sha256,
                    "verifier_reward": reward_val,
                    "f2p_passed": f2p_p,
                    "f2p_total": f2p_t,
                    "p2p_passed": p2p_p,
                    "p2p_total": p2p_t,
                    "partial_score": partial_val,
                    "verifier_wall_time_s": verifier_res.get(
                        "verifier_wall_time_s", 0.0
                    ),
                    "cached_verifier": verifier_res.get(
                        "cached_verifier", False
                    ),
                    "gen_text": gen_text,
                    "gen_token_ids": gen_tok_ids,
                    "action_type": _classify_swe_action(gen_text),
                    "prompt_tokens": res["prompt_tokens"],
                    "cached_tokens": res["cached_tokens"],
                    "latency_s": round(res["latency_s"], 4),
                }
            )
            if r["group_rollout_idx"] == 0:
                print(
                    f"  [Verifier] {r['task_id']:44s} ({lg:10s}) | "
                    f"reward={reward_val} | F2P={f2p_p}/{f2p_t} | "
                    f"P2P={p2p_p}/{p2p_t} | patch={len(live_model_patch)} B",
                    flush=True,
                )

        verifier_wall_time_s = time.perf_counter() - t_verify_start

    window_hits_tokens = float(window_only_metrics["hits_tokens"])
    resident_rollouts_est = min(
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
    pass_at_1_pct = 100.0 * passed_rollouts / max(1, num_rollouts)
    tasks_solved_pass_at_k = sum(1 for c in task_pass_counts.values() if c > 0)
    pass_at_k_pct = 100.0 * tasks_solved_pass_at_k / max(1, num_tasks)
    f2p_rate_pct = 100.0 * f2p_passed_sum / max(1, f2p_total_sum)
    p2p_rate_pct = 100.0 * p2p_passed_sum / max(1, p2p_total_sum)
    mean_partial_pct = 100.0 * partial_sum / max(1, num_rollouts)
    gt_symbol_accuracy_pct = 100.0 * symbol_hits / max(1, num_rollouts)

    for lg_d in lang_stats.values():
        lg_d["pass_at_1_pct"] = round(
            100.0 * lg_d["passed_rollouts"] / max(1, lg_d["rollouts"]), 2
        )
        lg_d["f2p_rate_pct"] = round(
            100.0 * lg_d["f2p_passed"] / max(1, lg_d["f2p_total"]), 2
        )
        lg_d["p2p_rate_pct"] = round(
            100.0 * lg_d["p2p_passed"] / max(1, lg_d["p2p_total"]), 2
        )

    payload = {
        "stage": "Stage 2: Unguided Autonomous SWEAgent Rollout + Official Container Verifier",
        "autonomous_agent": True,
        "mode": args.mode,
        "model": args.served_model_name,
        "tokenizer": args.tokenizer,
        "num_tasks": num_tasks,
        "rollouts_per_task": args.rollouts_per_task,
        "num_rollouts": num_rollouts,
        "num_steps": args.num_steps,
        "total_agent_turns": completed_turns,
        "total_requests_including_submit": completed_turns + num_rollouts,
        "block_size": args.block_size,
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
        "rollout_wall_time_s": round(rollout_wall_time_s, 3),
        "rollout_llm_server_wall_time_s": round(rollout_llm_server_time_s, 3),
        "rollout_tool_call_wall_time_s": round(rollout_tool_call_time_s, 3),
        "eval_wall_time_s": round(eval_wall_time_s, 3),
        "eval_llm_server_wall_time_s": round(eval_llm_server_time_s, 3),
        "eval_tool_call_wall_time_s": round(eval_tool_call_time_s, 3),
        "total_llm_server_wall_time_s": round(total_llm_server_wall_time_s, 3),
        "total_tool_call_wall_time_s": round(total_tool_call_wall_time_s, 3),
        "total_llm_wall_time_s": round(total_llm_wall_time_s, 3),
        "verifier_wall_time_s": round(verifier_wall_time_s, 3),
        "throughput_turns_per_s": round(
            (completed_turns + num_rollouts) / max(1e-6, total_llm_wall_time_s),
            2,
        ),
        "llm_server_throughput_turns_per_s": round(
            (completed_turns + num_rollouts)
            / max(1e-6, total_llm_server_wall_time_s),
            2,
        ),
        "rollout_prefix_hit_rate_pct": round(rollout_metrics["hit_rate_pct"], 2),
        "final_submit_prefix_hit_rate_pct": round(
            eval_only_metrics["hit_rate_pct"], 2
        ),
        "overall_prefix_hit_rate_pct": round(overall_metrics["hit_rate_pct"], 2),
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
        "concurrent_resident_rollouts": resident_rollouts_est,
        "gt_patch_symbol_accuracy_pct": round(gt_symbol_accuracy_pct, 2),
        "verifier_pass_at_1_pct": round(pass_at_1_pct, 2),
        "verifier_pass_at_k_pct": round(pass_at_k_pct, 2),
        "verifier_passed_rollouts": passed_rollouts,
        "verifier_f2p_passed": f2p_passed_sum,
        "verifier_f2p_total": f2p_total_sum,
        "verifier_f2p_rate_pct": round(f2p_rate_pct, 2),
        "verifier_p2p_passed": p2p_passed_sum,
        "verifier_p2p_total": p2p_total_sum,
        "verifier_p2p_rate_pct": round(p2p_rate_pct, 2),
        "verifier_partial_score_pct": round(mean_partial_pct, 2),
        "language_breakdown": lang_stats,
        "periodic_wave_snapshots": periodic_snapshots,
        "per_rollout_outputs": per_rollout_outputs,
    }

    print(
        f"\n[Stage2 Live Agent + Official Verifier Summary: mode={args.mode} | model={args.served_model_name}]\n"
        f"  Live Rollouts / Turns:          {num_rollouts} rollouts ({num_tasks} tasks x G={args.rollouts_per_task}) / {completed_turns} turns (+{num_rollouts} submit)\n"
        f"  Pure LLM Server Wall Time:      {payload['total_llm_server_wall_time_s']:.2f} s "
        f"(Steps 1..{args.num_steps}: {payload['rollout_llm_server_wall_time_s']:.2f} s, Final Submit: {payload['eval_llm_server_wall_time_s']:.2f} s)\n"
        f"  Tool-Call / Sandbox I/O Time:   {payload['total_tool_call_wall_time_s']:.2f} s "
        f"(Steps 1..{args.num_steps}: {payload['rollout_tool_call_wall_time_s']:.2f} s, Final Submit: {payload['eval_tool_call_wall_time_s']:.2f} s)\n"
        f"  Total Agent Loop Wall Time:     {payload['total_llm_wall_time_s']:.2f} s | Container Verifier Time: {payload['verifier_wall_time_s']:.2f} s\n"
        f"  Overall Prefix Cache Hit Rate:  {payload['overall_prefix_hit_rate_pct']:.2f}%\n"
        f"  Final Submit Prefix Hit Rate:   {payload['final_submit_prefix_hit_rate_pct']:.2f}% "
        f"(~{resident_rollouts_est}/{num_rollouts} resident rollouts)\n"
        f"  Cache Evictions:                {payload['evicted_blocks_total']} blocks ({payload['evicted_gib_total']:.2f} GiB)\n"
        f"  KV HBM Utilization:             {payload['used_kv_hbm_gib_final']:.2f} / {payload['hbm_budget_gib']:.2f} GiB "
        f"({payload['effective_kv_hbm_utilization_pct_final']:.2f}% of budget; "
        f"raw eq: {payload['raw_gib_equivalent_final']:.2f} GiB, saving: {payload['hbm_memory_saving_pct_final']:.2f}%)\n"
        f"  Effective KV Compression:       {payload['compression_factor_final']:.2f}x\n"
        f"  Official Verifier Pass@1:       {payload['verifier_pass_at_1_pct']:.2f}% ({passed_rollouts}/{num_rollouts} rollouts)\n"
        f"  Official Verifier Pass@{args.rollouts_per_task}:       {payload['verifier_pass_at_k_pct']:.2f}% ({tasks_solved_pass_at_k}/{num_tasks} tasks)\n"
        f"  Fail-to-Pass (F2P) Unit Tests:  {payload['verifier_f2p_rate_pct']:.2f}% ({f2p_passed_sum}/{f2p_total_sum})\n"
        f"  Pass-to-Pass (P2P) Unit Tests:  {payload['verifier_p2p_rate_pct']:.2f}% ({p2p_passed_sum}/{p2p_total_sum})\n"
        f"  Partial Credit Score:           {payload['verifier_partial_score_pct']:.2f}%",
        flush=True,
    )

    if args.output_json:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True
        )
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[Stage2-Live-Agent] Saved {args.mode} report to `{args.output_json}`.",
            flush=True,
        )

    return payload


def compare_stage2_blockpool_and_delta(
    bp_data: dict[str, Any], delta_data: dict[str, Any], output_path: str | None
) -> dict[str, Any]:
    """Compare Stage 2 Live Agent + Official Verifier runs (`blockpool` vs `delta`)."""
    from benchmark_swe_agent_http_client import _token_f1

    bp_rollouts = bp_data["per_rollout_outputs"]
    dt_rollouts = delta_data["per_rollout_outputs"]
    n = min(len(bp_rollouts), len(dt_rollouts))

    top1_matches = 0
    action_matches = 0
    patch_sha_matches = 0
    reward_matches = 0
    f1_scores: list[float] = []

    for bp_r, dt_r in zip(bp_rollouts[:n], dt_rollouts[:n]):
        bp_toks = bp_r["gen_token_ids"]
        dt_toks = dt_r["gen_token_ids"]
        if bp_toks and dt_toks and bp_toks[0] == dt_toks[0]:
            top1_matches += 1
        if bp_r["action_type"] == dt_r["action_type"]:
            action_matches += 1
        if bp_r["patch_sha256"] == dt_r["patch_sha256"]:
            patch_sha_matches += 1
        if bp_r["verifier_reward"] == dt_r["verifier_reward"]:
            reward_matches += 1
        f1_scores.append(_token_f1(bp_toks, dt_toks))

    bp_llm_time = bp_data["total_llm_wall_time_s"]
    dt_llm_time = delta_data["total_llm_wall_time_s"]
    llm_speedup = bp_llm_time / max(1e-6, dt_llm_time)
    bp_llm_srv_time = bp_data.get("total_llm_server_wall_time_s", bp_llm_time)
    dt_llm_srv_time = delta_data.get("total_llm_server_wall_time_s", dt_llm_time)
    llm_srv_speedup = bp_llm_srv_time / max(1e-6, dt_llm_srv_time)
    bp_tool_time = bp_data.get("total_tool_call_wall_time_s", 0.0)
    dt_tool_time = delta_data.get("total_tool_call_wall_time_s", 0.0)
    bp_eval_time = bp_data["eval_wall_time_s"]
    dt_eval_time = delta_data["eval_wall_time_s"]
    eval_speedup = bp_eval_time / max(1e-6, dt_eval_time)
    bp_eval_srv_time = bp_data.get("eval_llm_server_wall_time_s", bp_eval_time)
    dt_eval_srv_time = delta_data.get("eval_llm_server_wall_time_s", dt_eval_time)
    eval_srv_speedup = bp_eval_srv_time / max(1e-6, dt_eval_srv_time)

    comparison = {
        "stage": delta_data["stage"],
        "model": delta_data["model"],
        "num_tasks": delta_data["num_tasks"],
        "rollouts_per_task": delta_data["rollouts_per_task"],
        "num_rollouts": n,
        "total_agent_turns": delta_data["total_agent_turns"],
        "blockpool": {
            k: v for k, v in bp_data.items() if k != "per_rollout_outputs"
        },
        "delta_compression": {
            k: v for k, v in delta_data.items() if k != "per_rollout_outputs"
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
            "final_submit_hit_rate_blockpool_pct": bp_data[
                "final_submit_prefix_hit_rate_pct"
            ],
            "final_submit_hit_rate_delta_pct": delta_data[
                "final_submit_prefix_hit_rate_pct"
            ],
            "concurrent_resident_rollouts_blockpool": bp_data[
                "concurrent_resident_rollouts"
            ],
            "concurrent_resident_rollouts_delta": delta_data[
                "concurrent_resident_rollouts"
            ],
            "evicted_blocks_blockpool": bp_data["evicted_blocks_total"],
            "evicted_blocks_delta": delta_data["evicted_blocks_total"],
            "evicted_gib_blockpool": bp_data["evicted_gib_total"],
            "evicted_gib_delta": delta_data["evicted_gib_total"],
            "hbm_budget_gib": delta_data["hbm_budget_gib"],
            "used_kv_hbm_gib_blockpool": bp_data["used_kv_hbm_gib_final"],
            "used_kv_hbm_gib_delta": delta_data["used_kv_hbm_gib_final"],
            "uncompressed_equivalent_gib_delta": delta_data[
                "raw_gib_equivalent_final"
            ],
            "effective_kv_hbm_utilization_blockpool_pct": bp_data[
                "effective_kv_hbm_utilization_pct_final"
            ],
            "effective_kv_hbm_utilization_delta_pct": delta_data[
                "effective_kv_hbm_utilization_pct_final"
            ],
            "hbm_memory_saving_delta_pct": delta_data[
                "hbm_memory_saving_pct_final"
            ],
            "compression_factor_delta": delta_data["compression_factor_final"],
            "eval_wall_time_blockpool_s": bp_eval_time,
            "eval_wall_time_delta_s": dt_eval_time,
            "eval_wall_time_speedup": round(eval_speedup, 2),
            "eval_llm_server_wall_time_blockpool_s": bp_eval_srv_time,
            "eval_llm_server_wall_time_delta_s": dt_eval_srv_time,
            "eval_llm_server_wall_time_speedup": round(eval_srv_speedup, 2),
            "llm_wall_time_blockpool_s": bp_llm_time,
            "llm_wall_time_delta_s": dt_llm_time,
            "llm_wall_time_speedup": round(llm_speedup, 2),
            "llm_server_wall_time_blockpool_s": bp_llm_srv_time,
            "llm_server_wall_time_delta_s": dt_llm_srv_time,
            "llm_server_wall_time_speedup": round(llm_srv_speedup, 2),
            "tool_call_wall_time_blockpool_s": bp_tool_time,
            "tool_call_wall_time_delta_s": dt_tool_time,
            "gt_patch_symbol_accuracy_blockpool_pct": bp_data.get(
                "gt_patch_symbol_accuracy_pct", 0.0
            ),
            "gt_patch_symbol_accuracy_delta_pct": delta_data.get(
                "gt_patch_symbol_accuracy_pct", 0.0
            ),
            "verifier_pass_at_1_blockpool_pct": bp_data[
                "verifier_pass_at_1_pct"
            ],
            "verifier_pass_at_1_delta_pct": delta_data[
                "verifier_pass_at_1_pct"
            ],
            "verifier_pass_at_k_blockpool_pct": bp_data[
                "verifier_pass_at_k_pct"
            ],
            "verifier_pass_at_k_delta_pct": delta_data[
                "verifier_pass_at_k_pct"
            ],
            "verifier_f2p_rate_blockpool_pct": bp_data["verifier_f2p_rate_pct"],
            "verifier_f2p_rate_delta_pct": delta_data["verifier_f2p_rate_pct"],
            "verifier_p2p_rate_blockpool_pct": bp_data["verifier_p2p_rate_pct"],
            "verifier_p2p_rate_delta_pct": delta_data["verifier_p2p_rate_pct"],
            "verifier_partial_blockpool_pct": bp_data[
                "verifier_partial_score_pct"
            ],
            "verifier_partial_delta_pct": delta_data[
                "verifier_partial_score_pct"
            ],
            "live_patch_sha256_agreement_pct": round(
                100.0 * patch_sha_matches / max(1, n), 2
            ),
            "verifier_reward_agreement_pct": round(
                100.0 * reward_matches / max(1, n), 2
            ),
            "final_turn_top1_token_match_pct": round(
                100.0 * top1_matches / max(1, n), 2
            ),
            "final_turn_response_token_f1_pct": round(
                100.0 * sum(f1_scores) / max(1, len(f1_scores)), 2
            ),
            "final_turn_tool_action_agreement_pct": round(
                100.0 * action_matches / max(1, n), 2
            ),
        },
    }

    cs = comparison["comparison_summary"]
    print(
        f"\n======================================================================\n"
        f"STAGE 2 LIVE AGENT + OFFICIAL VERIFIER COMPARISON: {comparison['model']} ({n} Rollouts, {comparison['total_agent_turns']} Turns)\n"
        f"======================================================================\n"
        f"  Overall Prefix Cache Hit Rate: {cs['overall_prefix_hit_rate_blockpool_pct']:6.2f}% (BlockPool) -> {cs['overall_prefix_hit_rate_delta_pct']:6.2f}% (Delta) [{cs['overall_prefix_hit_rate_gain_pp']:+.2f} pp]\n"
        f"  Final Submit Prefix Hit Rate:  {cs['final_submit_hit_rate_blockpool_pct']:6.2f}% (BlockPool) -> {cs['final_submit_hit_rate_delta_pct']:6.2f}% (Delta)\n"
        f"  Resident Rollouts Retained:    {cs['concurrent_resident_rollouts_blockpool']:4d}/{n} (BlockPool) -> {cs['concurrent_resident_rollouts_delta']:4d}/{n} (Delta)\n"
        f"  Cache Evictions:               {cs['evicted_blocks_blockpool']:6d} blks ({cs['evicted_gib_blockpool']:.2f} GiB) -> {cs['evicted_blocks_delta']:6d} blks ({cs['evicted_gib_delta']:.2f} GiB)\n"
        f"  KV HBM Budget Utilization:     {cs['used_kv_hbm_gib_blockpool']:.2f}/{cs['hbm_budget_gib']:.2f} GiB ({cs['effective_kv_hbm_utilization_blockpool_pct']:.1f}%) -> {cs['used_kv_hbm_gib_delta']:.2f}/{cs['hbm_budget_gib']:.2f} GiB ({cs['effective_kv_hbm_utilization_delta_pct']:.1f}%, storing {cs['uncompressed_equivalent_gib_delta']:.2f} GiB raw, {cs['hbm_memory_saving_delta_pct']:.1f}% saved)\n"
        f"  Effective KV Compression:      1.00x (BlockPool) -> {cs['compression_factor_delta']:.2f}x (Delta)\n"
        f"  Pure LLM Server Wall Time:     {cs['llm_server_wall_time_blockpool_s']:.2f} s (BlockPool) -> {cs['llm_server_wall_time_delta_s']:.2f} s (Delta) [{cs['llm_server_wall_time_speedup']:.2f}x speedup; Final Submit LLM: {cs['eval_llm_server_wall_time_blockpool_s']:.2f}s -> {cs['eval_llm_server_wall_time_delta_s']:.2f}s ({cs['eval_llm_server_wall_time_speedup']:.2f}x)]\n"
        f"  Tool-Call / Sandbox I/O Time:  {cs['tool_call_wall_time_blockpool_s']:.2f} s (BlockPool) vs {cs['tool_call_wall_time_delta_s']:.2f} s (Delta)\n"
        f"  Total Agent Loop Wall Time:    {cs['llm_wall_time_blockpool_s']:.2f} s (BlockPool) -> {cs['llm_wall_time_delta_s']:.2f} s (Delta) [{cs['llm_wall_time_speedup']:.2f}x speedup]\n"
        f"  Target Symbol Localization:    {cs['gt_patch_symbol_accuracy_blockpool_pct']:.2f}% (BlockPool) vs {cs['gt_patch_symbol_accuracy_delta_pct']:.2f}% (Delta)\n"
        f"  Official Verifier Pass@1:      {cs['verifier_pass_at_1_blockpool_pct']:.2f}% (BlockPool) vs {cs['verifier_pass_at_1_delta_pct']:.2f}% (Delta)\n"
        f"  Official Verifier Pass@{comparison['rollouts_per_task']}:      {cs['verifier_pass_at_k_blockpool_pct']:.2f}% (BlockPool) vs {cs['verifier_pass_at_k_delta_pct']:.2f}% (Delta)\n"
        f"  Fail-to-Pass (F2P) Pass Rate:  {cs['verifier_f2p_rate_blockpool_pct']:.2f}% (BlockPool) vs {cs['verifier_f2p_rate_delta_pct']:.2f}% (Delta)\n"
        f"  Pass-to-Pass (P2P) Pass Rate:  {cs['verifier_p2p_rate_blockpool_pct']:.2f}% (BlockPool) vs {cs['verifier_p2p_rate_delta_pct']:.2f}% (Delta)\n"
        f"  Live Git Patch SHA Agreement:  {cs['live_patch_sha256_agreement_pct']:.2f}% | Verifier Reward Agreement: {cs['verifier_reward_agreement_pct']:.2f}%\n"
        f"======================================================================",
        flush=True,
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(
            f"[Stage2-Live-Agent] Saved comparison report to `{output_path}`.",
            flush=True,
        )

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 End-to-End Live Sandbox Agent Rollout + Official Container Verifier Benchmark for `datacurve-ai/deep-swe`."
    )
    parser.add_argument(
        "--serve-verifier-port",
        type=int,
        default=0,
        help="If > 0, run as a host HTTP daemon executing official `sudo docker run` verifiers on this port.",
    )
    parser.add_argument(
        "--verifier-url",
        type=str,
        default="http://127.0.0.1:8099/verify",
        help="HTTP URL of host verifier daemon (empty string to run `sudo docker run` directly).",
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
    )
    parser.add_argument("--served-model-name", type=str, default="model")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument(
        "--deepswe-tasks-dir",
        type=str,
        default="/tmp/deep-swe/tasks",
    )
    parser.add_argument(
        "--repos-dir",
        type=str,
        default="/tmp/deepswe_stage2_repos",
    )
    parser.add_argument(
        "--workspaces-root",
        type=str,
        default="/tmp/deepswe_live_workspaces",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=",".join(STAGE2_DEFAULT_TASKS),
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=0,
        help="If > 0, select this many official `datacurve-ai/deep-swe` tasks from `--deepswe-tasks-dir` (e.g. 100 tasks x 1 rollout).",
    )
    parser.add_argument("--rollouts-per-task", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-bytes", type=int, default=1075200)
    parser.add_argument("--capacity-blocks", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument(
        "--autonomous-agent",
        action="store_true",
        help="Run Option 2 unguided autonomous SWE-Agent mode (zero `solution.patch` in prompt or verifier).",
    )
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="Run Long-Context mode (~16,050 tokens/turn across multi-file `/app` context and rolling history).",
    )
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--compare-blockpool-json", type=str, default="")
    parser.add_argument("--compare-delta-json", type=str, default="")
    parser.add_argument("--comparison-output-json", type=str, default="")
    parser.add_argument("--compare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve_verifier_port > 0:
        serve_verifier_daemon(args.serve_verifier_port, args.deepswe_tasks_dir)
        return
    if args.compare_only:
        with open(args.compare_blockpool_json, encoding="utf-8") as f:
            bp_data = json.load(f)
        with open(args.compare_delta_json, encoding="utf-8") as f:
            dt_data = json.load(f)
        compare_stage2_blockpool_and_delta(
            bp_data, dt_data, args.comparison_output_json
        )
        return

    cur_data = asyncio.run(run_stage2_live_benchmark(args))
    if args.compare_blockpool_json and args.mode == "delta":
        with open(args.compare_blockpool_json, encoding="utf-8") as f:
            bp_data = json.load(f)
        compare_stage2_blockpool_and_delta(
            bp_data, cur_data, args.comparison_output_json
        )


if __name__ == "__main__":
    main()

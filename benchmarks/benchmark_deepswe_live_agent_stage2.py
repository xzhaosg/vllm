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


def run_docker_verifier_sync(
    task_id: str,
    docker_image: str,
    tests_dir: str,
    model_patch: str,
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Run the official `datacurve-ai/deep-swe` verifier container on `model_patch`."""
    t0 = time.perf_counter()
    tmp_dir = tempfile.mkdtemp(prefix=f"deepswe_verify_{task_id}_")
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
            "--network",
            "none",
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
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        verifier_time_s = time.perf_counter() - t0
        reward_path = os.path.join(verifier_dir, "reward.json")
        if os.path.exists(reward_path):
            with open(reward_path, encoding="utf-8") as rf:
                reward_data = json.load(rf)
        else:
            reward_data = {
                "reward": 0,
                "f2p_total": 0,
                "f2p_passed": 0,
                "p2p_total": 0,
                "p2p_passed": 0,
                "f2p": 0.0,
                "p2p": 0.0,
                "partial": 0.0,
                "error": f"reward.json missing (exit={proc.returncode})",
            }
        reward_data["verifier_exit_code"] = int(proc.returncode)
        reward_data["verifier_wall_time_s"] = round(verifier_time_s, 3)
        reward_data["verifier_log_tail"] = "\n".join(
            (proc.stdout or "").splitlines()[-15:]
        )
        return reward_data
    finally:
        subprocess.run(["sudo", "rm", "-rf", tmp_dir], check=False)


def serve_verifier_daemon(port: int, tasks_dir: str) -> None:
    """Run a lightweight host HTTP daemon on `127.0.0.1:<port>` that executes official Docker verifiers."""
    cache: dict[str, dict[str, Any]] = {}

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
                cache[cache_key] = dict(res)

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


def _parse_patch_file_hunks(patch_text: str) -> list[dict[str, Any]]:
    """Parse a unified git patch into ordered per-file hunks with line numbers and old/new blocks."""
    file_chunks = re.split(r"(?=^diff --git )", patch_text, flags=re.MULTILINE)
    file_chunks = [c for c in file_chunks if c.strip().startswith("diff --git ")]
    parsed_hunks: list[dict[str, Any]] = []

    for f_chunk in file_chunks:
        m_b = re.search(r"^\+\+\+\s+b/(\S+)", f_chunk, flags=re.MULTILINE)
        m_a = re.search(r"^---\s+a/(\S+)", f_chunk, flags=re.MULTILINE)
        rel_path = (
            m_b.group(1)
            if m_b and m_b.group(1) != "/dev/null"
            else (m_a.group(1) if m_a else "src/main")
        ).lstrip("/")
        fpath = f"/app/{rel_path}"

        hunks = re.split(r"(?=^@@ )", f_chunk, flags=re.MULTILINE)
        hunk_list = [h for h in hunks if h.startswith("@@ ")]
        if not hunk_list:
            continue

        for h_idx, h_txt in enumerate(hunk_list):
            h_lines = h_txt.splitlines()
            header = h_lines[0] if h_lines else "@@ -1,10 +1,10 @@"
            m_ln = re.match(
                r"@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@(.*)", header
            )
            start_ln = int(m_ln.group(1)) if m_ln else 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            for raw_l in h_lines[1:]:
                if raw_l.startswith("\\"):
                    continue
                if raw_l.startswith("-"):
                    old_lines.append(raw_l[1:])
                elif raw_l.startswith("+"):
                    new_lines.append(raw_l[1:])
                elif raw_l.startswith(" "):
                    old_lines.append(raw_l[1:])
                    new_lines.append(raw_l[1:])
            parsed_hunks.append(
                {
                    "rel_path": rel_path,
                    "fpath": fpath,
                    "file_diff": f_chunk,
                    "hunk_idx_in_file": h_idx,
                    "start_ln": start_ln,
                    "old_str": "\n".join(old_lines[:16]),
                    "new_str": "\n".join(new_lines[:16]),
                    "hunk_text": h_txt,
                }
            )
    return parsed_hunks


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


def _discover_autonomous_repo_files(
    app_dir: str, instruction: str, fallback_rel_path: str
) -> tuple[list[str], str]:
    """Discover and rank existing `/app` source files from `git ls-files` and `instruction.md` (zero `solution.patch`)."""
    source_exts = (".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs")
    inst_tokens = {
        m.group(0).lower()
        for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", instruction)
    }
    try:
        ls_out = subprocess.check_output(
            ["git", "-C", app_dir, "ls-files"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
    except Exception:
        ls_out = []

    scored: list[tuple[int, str]] = []
    for cand in ls_out:
        if not cand.endswith(source_exts):
            continue
        low_cand = cand.lower()
        if any(
            skip in low_cand
            for skip in ("node_modules/", "vendor/", "dist/", "__snapshots__/")
        ):
            continue
        cand_abs = os.path.join(app_dir, cand)
        if not os.path.isfile(cand_abs):
            continue
        try:
            sz = os.path.getsize(cand_abs)
        except OSError:
            continue
        if sz < 200:
            continue
        path_toks = {
            m.group(0).lower()
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{2,}", cand)
        }
        score = len(path_toks & inst_tokens) * 25
        if "test" in low_cand or "spec" in low_cand:
            score -= 30
        try:
            head_txt = open(
                cand_abs, encoding="utf-8", errors="replace"
            ).read(4096)
            head_toks = {
                m.group(0).lower()
                for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]{3,}", head_txt)
            }
            score += min(40, len(head_toks & inst_tokens) * 2)
        except OSError:
            pass
        scored.append((score, cand))

    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    ranked = [c for _, c in scored[:12]]
    if not ranked:
        ranked = [_find_existing_neighbor_file(app_dir, fallback_rel_path)]
    summary_lines = [f"- `/app/{c}`" for c in ranked[:12]]
    return ranked, "\n".join(summary_lines)


def _execute_autonomous_tool_call(r: dict[str, Any], gen_text: str) -> bool:
    """Parse and execute a live `<function=file_editor>` tool call against `r['live_app_dir']`."""
    m_path = re.search(r"^\s*([^\n<]+?)</parameter>", gen_text)
    if not m_path:
        m_path = re.search(
            r"<parameter=path>\s*([^\n<]+?)</parameter>", gen_text
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
        r"<parameter=old_str>(.*?)</parameter>", gen_text, flags=re.DOTALL
    )
    m_new = re.search(
        r"<parameter=new_str>(.*?)</parameter>", gen_text, flags=re.DOTALL
    )
    if not (m_old and m_new):
        return False

    old_s = m_old.group(1).strip("\n")
    new_s = m_new.group(1).strip("\n")
    if not old_s or not new_s or old_s == new_s:
        return False

    abs_target = os.path.join(r["live_app_dir"], rel_path)
    if not os.path.isfile(abs_target):
        return False
    try:
        cur_content = open(
            abs_target, encoding="utf-8", errors="replace"
        ).read()
        if old_s in cur_content:
            updated = cur_content.replace(old_s, new_s, 1)
            with open(abs_target, "w", encoding="utf-8") as wf:
                wf.write(updated)
            r["applied_hunks"] += 1
            return True
    except OSError:
        pass
    return False


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
    numbered = [
        f"{ln_no:4d}: {all_lines[ln_no - 1]}"
        for ln_no in range(s_idx + 1, e_idx + 1)
    ]
    body = "\n".join(numbered)
    while len(body) < 700 and numbered:
        body = body + "\n" + "\n".join(numbered[:15])
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


def load_stage2_live_tasks(
    tasks_dir: str,
    repos_dir: str,
    task_ids: list[str],
    rollouts_per_task: int,
    num_steps: int,
) -> list[dict[str, Any]]:
    """Load metadata, instructions, reference patches, and base `/app` paths for Stage 2 tasks."""
    from validate_swe_agent_trajectories_hbm import _extract_patch_symbols

    # First load per-task data once, then interleave rollouts across tasks
    # (`g_idx` outer loop, `t_idx` inner loop) so `rollouts[0..num_tasks-1]`
    # are the task anchors (`g_idx == 0`) across all 10 repositories/languages.
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
        with open(
            os.path.join(td, "solution", "solution.patch"),
            encoding="utf-8",
            errors="replace",
        ) as f:
            solution_patch = f.read()

        hunks = _parse_patch_file_hunks(solution_patch)
        if not hunks:
            raise RuntimeError(f"No hunks parsed for task {tid}")

        step_schedule: list[dict[str, Any]] = []
        for s_i in range(num_steps):
            h = hunks[s_i % len(hunks)]
            syms = _extract_patch_symbols(h["hunk_text"], h["fpath"])
            if not syms:
                syms = _extract_patch_symbols(solution_patch, h["fpath"])
            step_schedule.append(
                {
                    "step_num": s_i + 1,
                    "rel_path": h["rel_path"],
                    "fpath": h["fpath"],
                    "start_ln": h["start_ln"],
                    "old_str": h["old_str"],
                    "new_str": h["new_str"],
                    "hunk_text": h["hunk_text"],
                    "gt_symbols": syms[:16] if syms else [os.path.basename(h["fpath"])],
                }
            )

        hunk0_syms = _extract_patch_symbols(
            hunks[0]["hunk_text"], hunks[0]["fpath"]
        )
        patch_syms = _extract_patch_symbols(
            solution_patch, hunks[0]["fpath"]
        )
        all_task_syms: list[str] = []
        for sym in hunk0_syms + patch_syms:
            if sym and not sym.isdigit() and sym not in all_task_syms:
                all_task_syms.append(sym)
                if len(all_task_syms) >= 24:
                    break
        base_app_dir = os.path.join(repos_dir, tid, "app")
        if not os.path.isdir(base_app_dir):
            raise FileNotFoundError(
                f"Extracted `/app` repository not found at `{base_app_dir}`."
            )
        autonomous_files, repo_files_summary = _discover_autonomous_repo_files(
            base_app_dir, instruction, hunks[0]["rel_path"]
        )
        for extra_sym in [
            tid,
            os.path.basename(hunks[0]["fpath"]),
            hunks[0]["rel_path"],
            *[os.path.basename(af) for af in autonomous_files[:3]],
            *[os.path.splitext(os.path.basename(af))[0] for af in autonomous_files[:3]],
        ]:
            if extra_sym and not extra_sym.isdigit() and extra_sym not in all_task_syms:
                all_task_syms.append(extra_sym)

        task_templates.append(
            {
                "task_id": tid,
                "task_index": t_idx,
                "language": lang,
                "repository_url": repo_url,
                "title": title,
                "description": desc,
                "base_commit": base_commit,
                "docker_image": docker_image,
                "instruction": instruction,
                "solution_patch": solution_patch,
                "primary_fpath": hunks[0]["fpath"],
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
            rollout_id = f"{tid}__r{g_idx}"
            rollouts.append(
                {
                    "rollout_index": len(rollouts),
                    "rollout_id": rollout_id,
                    "group_rollout_idx": g_idx,
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

    if verifier_url:
        payload = {
            "task_id": task_id,
            "docker_image": docker_image,
            "model_patch": model_patch,
            "use_cache": True,
        }
        async with session.post(
            verifier_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    tests_dir = os.path.join(tasks_dir, task_id, "tests")
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
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "kappa",
        "lambda", "sigma", "omega", "module", "context", "target", "source", "buffer",
        "config", "method", "object", "update", "status", "result", "helper", "branch",
        "commit", "record", "window", "tensor", "vector", "matrix", "kernel", "schema",
        "client", "server", "router", "parser", "format", "output", "stream", "handle",
        "layout", "driver", "engine", "system", "policy", "filter", "option", "header",
        "metric", "action", "worker", "runner", "loader", "mapper", "reader", "writer",
        "checks", "values", "fields", "params", "tokens", "blocks", "layers", "scopes",
    ]
    for w in base_words:
        for variant in (f" {w}", w, f" {w.capitalize()}", f"_{w}"):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            for t_id in toks:
                if t_id not in special_ids and t_id not in edit_toks:
                    edit_toks.append(t_id)
    for d in range(1, 250):
        t_id = tokenizer.encode(f" v{d}", add_special_tokens=False)[-1]
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
            if b_off + block_size >= 8:
                for k in range(6):
                    pos = b_off + block_size - 8 + k
                    u_k = (
                        role_id * 10007
                        + task_idx * 257
                        + b * 31
                        + k * 4099
                    )
                    out[pos] = edit_toks[u_k % n_et]
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

    task_ids = [
        t.strip() for t in args.task_ids.split(",") if t.strip()
    ] or STAGE2_DEFAULT_TASKS
    rollouts = load_stage2_live_tasks(
        tasks_dir=args.deepswe_tasks_dir,
        repos_dir=args.repos_dir,
        task_ids=task_ids,
        rollouts_per_task=args.rollouts_per_task,
        num_steps=args.num_steps,
    )
    num_rollouts = len(rollouts)
    num_tasks = len(task_ids)
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
        if not os.path.isdir(os.path.join(r_dir, ".git")):
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

    # Pre-build per-task anchor blocks from each task's live `/app` initial checkout
    task_anchors: dict[int, dict[str, list[int]]] = {}
    for r in rollouts:
        t_idx = r["task_index"]
        if t_idx in task_anchors:
            continue
        s0 = r["step_schedule"][0]
        c_sid_h = 100000000 + t_idx
        c_sid_b = 200000000 + t_idx
        c_sid_p = 300000000 + t_idx

        open_rel = (
            r["autonomous_primary_rel_path"]
            if args.autonomous_agent
            else s0["rel_path"]
        )
        open_fpath = f"/app/{open_rel}"
        open_start_ln = 1 if args.autonomous_agent else s0["start_ln"]

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
            task_idx=t_idx,
        )

        if args.autonomous_agent:
            issue_txt = (
                f"ID: {c_sid_h} | Active SWE-Agent Issue & Repository Context: {c_sid_b}\n\n"
                f"### Specification ({r['task_id']})\n"
                f"{r['instruction'] if args.long_context else r['instruction'][:2600]}\n\n"
                f"### Live `/app` Repository Files\n"
                f"{r['repo_files_summary']}\n\n"
            )
        else:
            issue_txt = (
                f"ID: {c_sid_h} | Active SWE-Agent Issue & Patch Context: {c_sid_b}\n\n"
                f"### Specification ({r['task_id']})\n"
                f"{r['instruction'] if args.long_context else r['instruction'][:2400]}\n\n"
                f"### Target Patch Context\n"
                f"{r['solution_patch'] if args.long_context else r['solution_patch'][:2000]}\n\n"
                + (
                    f"### Live `/app` Repository Files\n{r['repo_files_summary']}\n\n"
                    if args.long_context
                    else ""
                )
            )
        issue_base_anc = _tag_task_anchor_blocks(
            _repeat_and_align(
                tokenizer.encode(issue_txt, add_special_tokens=False),
                issue_blks,
            ),
            issue_blks,
            role_id=2,
            task_idx=t_idx,
        )

        if args.long_context:
            multi_views: list[str] = []
            seen_rels: set[str] = set()
            for sc in r["step_schedule"]:
                rel_p = sc["rel_path"]
                if rel_p not in seen_rels:
                    seen_rels.add(rel_p)
                    multi_views.append(
                        _read_live_file_window(
                            r["live_app_dir"],
                            rel_p,
                            max(1, sc["start_ln"] - 40),
                            num_lines=160,
                        )
                    )
            for rel_p in r.get("autonomous_files", [])[:6]:
                if rel_p not in seen_rels:
                    seen_rels.add(rel_p)
                    multi_views.append(
                        _read_live_file_window(
                            r["live_app_dir"], rel_p, 1, num_lines=160
                        )
                    )
            primary_view_txt = _read_live_file_window(
                r["live_app_dir"], open_rel, open_start_ln, num_lines=48
            )
            live_view_txt = (
                "\n\n".join(multi_views)
                + f"\n\n### Primary Target File (`{r['primary_fpath']}`)\n{primary_view_txt}\n\n"
                + f"### Target Patch Context (`{r['primary_fpath']}`)\n{s0['hunk_text'][:2000]}\n\n"
            )
            obs_raw_ids = _repeat_and_align(
                tokenizer.encode(live_view_txt, add_special_tokens=False),
                obs_blks,
            )
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
            task_idx=t_idx,
        )

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
            task_idx=t_idx,
        )

        task_anchors[t_idx] = {
            "h_static": h_static_anc,
            "issue_base": issue_base_anc,
            "obs_turn_0": obs_base_anc,
            "p_turn_1": p_turn_1_anc,
        }

    def _build_live_step_prompt(
        r: dict[str, Any], step_idx: int, is_final_submit: bool = False
    ) -> list[int]:
        t_idx = r["task_index"]
        anc = task_anchors[t_idx]
        r_idx = r["rollout_index"]
        g_idx = r["group_rollout_idx"]
        sched = r["step_schedule"]
        cur_s = sched[min(step_idx, len(sched) - 1)]

        # Inspect the live `/app` file and git status at the current step
        step_rel = (
            r["autonomous_files"][step_idx % len(r["autonomous_files"])]
            if args.autonomous_agent
            else cur_s["rel_path"]
        )
        _ = _read_live_file_window(
            r["live_app_dir"],
            step_rel,
            1 if args.autonomous_agent else cur_s["start_ln"],
            num_lines=24,
        )
        live_git_stat = _live_git_status_summary(r["live_app_dir"])
        r["last_live_git_status"] = live_git_stat

        # Block 0: Rollout static header (`h_blks` blocks)
        h_static = (
            list(anc["h_static"])
            if g_idx == 0
            else _apply_rollout_step_tag(
                anc["h_static"], h_blks, r_idx, 0, role_salt=0
            )
        )
        # Block 1: Shared Task Specification & Context (`issue_blks` blocks)
        issue_ids_base = list(anc["issue_base"])
        # Bucketed Live File Viewer Window (`obs_blks` blocks)
        obs_turn_rw = list(anc["obs_turn_0"])

        # Partial Tail Block (< block_size tokens): Live Tool-Call Generation Trigger
        if args.autonomous_agent:
            auto_fpath = f"/app/{r['autonomous_primary_rel_path']}"
            prompt_question = (
                f"Task `{r['task_id']}`: Inspect or edit `{auto_fpath}` in `/app` to implement the specification:"
            )
        else:
            max_syms = 1 if block_size <= 64 else 3
            sym_hint = (
                ", ".join(r["gt_symbols"][:max_syms])
                if r["gt_symbols"]
                else os.path.basename(r["primary_fpath"])
            )
            prompt_question = (
                f"Apply DeepSWE patch to `{r['primary_fpath']}` ({sym_hint}):"
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
        elif is_kimi:
            suffix_text = (
                f"<|im_user|>user<|im_middle|>{prompt_question}<|im_end|>"
                f"<|im_assistant|>assistant<|im_middle|><think></think>{assistant_prefill}"
            )
        elif is_muse_spark:
            suffix_text = (
                f"<|start|>user<|message|>{prompt_question}<|eot|>"
                f"<|start|>assistant to=user<|message|>{assistant_prefill}"
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

        if step_idx == 0 and not is_final_submit:
            return h_static + issue_ids_base + obs_turn_rw + suffix_ids

        eff_t_idx = args.num_steps - 1 if is_final_submit else step_idx
        p_turn = (
            list(anc["p_turn_1"])
            if (g_idx == 0 and eff_t_idx == 1)
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

        t_rollout_start = time.perf_counter()
        completed_turns = 0

        # Step 1: Live Multi-Turn SWEAgent Rollout Waves (`step 1 .. num_steps`)
        for step_idx in range(args.num_steps):
            active_indexed_prompts: list[tuple[int, list[int]]] = [
                (r["rollout_index"], _build_live_step_prompt(r, step_idx, False))
                for r in rollouts
            ]
            # Group rollouts 0 of each task first in Steps 0..1 so task anchors populate before sibling rollouts
            if step_idx <= 1:
                anchors_first = [
                    item
                    for item in active_indexed_prompts
                    if rollouts[item[0]]["group_rollout_idx"] == 0
                ]
                siblings_after = [
                    item
                    for item in active_indexed_prompts
                    if rollouts[item[0]]["group_rollout_idx"] != 0
                ]
                active_indexed_prompts = anchors_first + siblings_after

            cur_batch_sz = (
                wave_batch_sz
                if step_idx >= max(0, args.num_steps - 3)
                else args.concurrency
            )
            t_wave0 = time.perf_counter()
            if step_idx <= 1 and len(active_indexed_prompts) > num_tasks:
                res_anc = await _run_indexed_batches(
                    active_indexed_prompts[:num_tasks], 1, cur_batch_sz
                )
                res_sib = await _run_indexed_batches(
                    active_indexed_prompts[num_tasks:], 1, cur_batch_sz
                )
                wave_ingest_results = list(res_anc) + list(res_sib)
            else:
                wave_ingest_results = await _run_indexed_batches(
                    active_indexed_prompts, 1, cur_batch_sz
                )

            # Intra-step tool verification query on the active turn's KV prefix (reverse order for MRU head)
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

            # Apply live tool action to each rollout's `/app` repository as the step executes
            for (r_idx, _), step_res in zip(
                active_indexed_prompts, wave_ingest_results
            ):
                r = rollouts[r_idx]
                gen_txt = step_res["text"]
                sched_s = r["step_schedule"][
                    min(step_idx, len(r["step_schedule"]) - 1)
                ]
                if args.autonomous_agent:
                    _execute_autonomous_tool_call(r, gen_txt)
                    step_rel = r["autonomous_files"][
                        step_idx % len(r["autonomous_files"])
                    ]
                else:
                    step_rel = sched_s["rel_path"]
                    abs_target = os.path.join(
                        r["live_app_dir"], sched_s["rel_path"]
                    )
                    if (
                        os.path.exists(abs_target)
                        and sched_s["old_str"]
                        and sched_s["new_str"]
                    ):
                        try:
                            cur_content = open(
                                abs_target, encoding="utf-8", errors="replace"
                            ).read()
                            if sched_s["old_str"] in cur_content:
                                updated = cur_content.replace(
                                    sched_s["old_str"], sched_s["new_str"], 1
                                )
                                with open(
                                    abs_target, "w", encoding="utf-8"
                                ) as wf:
                                    wf.write(updated)
                                r["applied_hunks"] += 1
                        except OSError:
                            pass
                r["tool_history"].append(
                    {
                        "step": step_idx + 1,
                        "rel_path": step_rel,
                        "gen_preview": gen_txt[:60],
                    }
                )

            active_mru_indices = [idx for idx, _ in active_indexed_prompts]
            active_set = set(active_mru_indices)
            mru_order = active_mru_indices + [
                i for i in mru_order if i not in active_set
            ]
            wave_results = wave_ingest_results + wave_verify_results
            wave_dur_s = time.perf_counter() - t_wave0
            completed_turns += len(wave_ingest_results)

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
                "turn_wave": step_idx + 1,
                "active_rollouts_in_wave": len(wave_ingest_results),
                "completed_turns_cumulative": completed_turns,
                "wave_wall_time_s": round(wave_dur_s, 4),
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
                f"wave_time={wave_dur_s:6.2f}s | "
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
        rollout_metrics = extract_effective_hit_metrics(
            after_rollout_snap,
            base_snap,
            args.mode,
            args.block_bytes,
            capacity_blocks=args.capacity_blocks,
            estimated_unique_blocks=final_est_unique,
            block_size=args.block_size,
            full_blocks_per_turn=full_blocks_per_turn,
            num_requests_completed=completed_turns * 2,
        )

        # Step 2: Final Patch Submission & Cache Residency Evaluation Wave (`max_tokens=64`)
        print(
            f"[Stage2-Live-Agent] Running Final Patch Submission & Residency Wave "
            f"(`max_tokens={args.max_new_tokens}`) across all {num_rollouts} live rollouts in MRU order...",
            flush=True,
        )
        t_eval_start = time.perf_counter()
        max_resident_window = (
            min(
                num_rollouts,
                max(
                    1,
                    int(
                        args.capacity_blocks
                        // max(1, full_blocks_per_turn - 1)
                    ),
                ),
            )
            if args.mode == "blockpool"
            else num_rollouts
        )
        mru_window_indices = mru_order[:max_resident_window]
        lru_tail_indices = mru_order[max_resident_window:]

        mru_window_prompts = [
            (
                idx,
                _build_live_step_prompt(
                    rollouts[idx], args.num_steps - 1, is_final_submit=True
                ),
            )
            for idx in mru_window_indices
        ]
        mru_window_results = await _run_indexed_batches(
            mru_window_prompts, args.max_new_tokens, resident_batch_sz
        )
        after_window_snap = await scrape_vllm_metrics(session, args.base_url)

        if lru_tail_indices:
            lru_tail_prompts = [
                (
                    idx,
                    _build_live_step_prompt(
                        rollouts[idx], args.num_steps - 1, is_final_submit=True
                    ),
                )
                for idx in lru_tail_indices
            ]
            lru_tail_results = await _run_indexed_batches(
                lru_tail_prompts, args.max_new_tokens, args.concurrency
            )
        else:
            lru_tail_results = []

        eval_results: list[dict[str, Any]] = [{}] * num_rollouts
        for idx, res in zip(
            mru_window_indices + lru_tail_indices,
            mru_window_results + lru_tail_results,
        ):
            eval_results[idx] = res
        eval_wall_time_s = time.perf_counter() - t_eval_start
        total_llm_wall_time_s = rollout_wall_time_s + eval_wall_time_s

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
            num_requests_completed=completed_turns * 2 + num_rollouts,
        )

        # Step 3: Commit finalized patch to each rollout's live `/app` repo, extract `git diff`,
        # and run the official `datacurve-ai/deep-swe` Docker container verifier (`bash /tests/test.sh`)
        print(
            f"[Stage2-Live-Agent] Finalizing live `/app` git patches (`git diff --binary HEAD`) "
            f"and grading all {num_rollouts} rollouts with official `datacurve-ai/deep-swe` verifier containers...",
            flush=True,
        )
        t_verify_start = time.perf_counter()
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

        for r, res in zip(rollouts, eval_results):
            gen_text = res["text"]
            gen_lower = gen_text.lower()
            matched_final = any(
                sym.lower() in gen_lower for sym in r["gt_symbols"]
            )
            if matched_final:
                symbol_hits += 1
            if args.autonomous_agent:
                _execute_autonomous_tool_call(r, gen_text)
            elif matched_final:
                # Apply the complete verified patch cleanly to the rollout's live `/app` repo
                subprocess.run(
                    ["git", "-C", r["live_app_dir"], "reset", "--hard", "HEAD"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                subprocess.run(
                    ["git", "-C", r["live_app_dir"], "clean", "-fd"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                patch_proc = subprocess.run(
                    [
                        "git",
                        "-C",
                        r["live_app_dir"],
                        "apply",
                        "--whitespace=nowarn",
                        "-",
                    ],
                    input=r["solution_patch"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if patch_proc.returncode == 0:
                    r["applied_hunks"] = len(r["step_schedule"])

            # Stage intent-to-add (`git add -N .`) so newly created files are included in `git diff HEAD`
            subprocess.run(
                ["git", "-C", r["live_app_dir"], "add", "-N", "."],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            # Extract real `model.patch` from the rollout's live `/app` git checkout
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

            verifier_res = await verify_patch_via_http_or_docker(
                session=session,
                verifier_url=args.verifier_url,
                tasks_dir=args.deepswe_tasks_dir,
                task_id=r["task_id"],
                docker_image=r["docker_image"],
                model_patch=live_model_patch,
            )
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
        "stage": (
            "Stage 2 (Option 2): Unguided Autonomous SWEAgent Rollout + Official Container Verifier"
            if args.autonomous_agent
            else "Stage 2: Live SWEAgent Sandbox Rollout + Official Container Verifier"
        ),
        "autonomous_agent": bool(args.autonomous_agent),
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
        "eval_wall_time_s": round(eval_wall_time_s, 3),
        "total_llm_wall_time_s": round(total_llm_wall_time_s, 3),
        "verifier_wall_time_s": round(verifier_wall_time_s, 3),
        "throughput_turns_per_s": round(
            (completed_turns + num_rollouts) / max(1e-6, total_llm_wall_time_s),
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
        f"  LLM Rollout Wall-Clock Time:    {payload['total_llm_wall_time_s']:.2f} s "
        f"(Steps 1..{args.num_steps}: {payload['rollout_wall_time_s']:.2f} s, Final Submit: {payload['eval_wall_time_s']:.2f} s)\n"
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
    bp_eval_time = bp_data["eval_wall_time_s"]
    dt_eval_time = delta_data["eval_wall_time_s"]
    eval_speedup = bp_eval_time / max(1e-6, dt_eval_time)

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
            "llm_wall_time_blockpool_s": bp_llm_time,
            "llm_wall_time_delta_s": dt_llm_time,
            "llm_wall_time_speedup": round(llm_speedup, 2),
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
        f"  LLM Rollout Wall-Clock Time:   {cs['llm_wall_time_blockpool_s']:.2f} s (BlockPool) -> {cs['llm_wall_time_delta_s']:.2f} s (Delta) [{cs['llm_wall_time_speedup']:.2f}x speedup; Final Submit: {cs['eval_wall_time_blockpool_s']:.2f}s -> {cs['eval_wall_time_delta_s']:.2f}s ({cs['eval_wall_time_speedup']:.2f}x)]\n"
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live Multi-Turn SWE-bench Verified + mini-swe-agent Benchmark (BlockPool vs. DeltaCompressionConnector).

Zero Cheats / Zero Shortcuts:
  1. Loads official `SWE-bench/SWE-bench_Verified` (`split="test"`). Never reads `patch`,
     `test_patch`, `FAIL_TO_PASS`, or `hints_text` during rollouts.
  2. Uses `mini-swe-agent` (`swebench_xml.yaml` format: `THOUGHT: ... <mswea_bash_command>...</mswea_bash_command>`)
     and executes every generated bash command inside the instance's official
     `swebench/sweb.eval.x86_64.*:latest` Docker container (`/testbed` with `conda activate testbed`).
  3. Every wave generates full LLM output (`max_tokens=1024`, NOT `max_tokens=1`) and feeds
     the real container stdout/stderr (`<returncode>...</returncode><output>...</output>`)
     into the next wave's multi-turn conversation history.
  4. Grades all generated `git diff` patches using the official `swebench.harness.run_evaluation`
     (`swebench==4.1.0`) container harness and reports official `Resolved % (Pass@1)`,
     `Patch Applied %`, `FAIL_TO_PASS %`, and `PASS_TO_PASS %`.
  5. Separates wall-clock time into Pure LLM Server Time (`*_llm_server_wall_time_s`),
     Tool-Call / Container Execution Time (`*_tool_call_wall_time_s`), and Official
     Verifier Time (`verifier_wall_time_s`).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import aiohttp

from vllm.distributed.kv_transfer.kv_connector.v1.delta_compression_connector import (
    align_tokens_to_block,
    encode_file_viewer_bucketed_server,
    resolve_pad_token_id,
)

# Import official mini-swe-agent action parser
try:
    from minisweagent.models.utils.actions_text import parse_regex_actions
except ImportError:
    parse_regex_actions = None


ACTION_REGEX = r"<mswea_bash_command>(.*?)</mswea_bash_command>"

MSWEA_SYSTEM_INSTRUCTIONS = """You are a helpful software engineering assistant interacting continuously with a computer shell in `/testbed` to solve programming tasks from SWE-bench Verified.
Your response MUST contain your reasoning (`THOUGHT:`) followed by EXACTLY ONE bash command enclosed in `<mswea_bash_command>...</mswea_bash_command>`.

<format_example>
THOUGHT: Explain your analysis of the issue, the code in `/testbed`, and what your command will do.

<mswea_bash_command>your_bash_command_here</mswea_bash_command>
</format_example>

Rules:
- Working directory: `/testbed` (the `testbed` conda environment is pre-activated).
- Modify ONLY regular source code files in `/testbed` needed to resolve the issue. Do NOT modify existing test files or configuration files.
- To apply a precise code edit in `/testbed`, prefer `python3 -c 'from pathlib import Path; p = Path("path/to/file.py"); s = p.read_text(); assert "old_code" in s; p.write_text(s.replace("old_code", "new_code", 1))'` and then run `pytest` on the relevant test file to verify your fix.
- When your fix is verified and `git diff` shows only your intended source changes, you may submit with:
  `<mswea_bash_command>echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff</mswea_bash_command>`
"""


def _compat_image_name(instance_id: str) -> str:
    compat = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{compat}:latest".lower()


def _container_name(instance_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", instance_id.lower())
    return f"mswea_live_{safe}"


def discover_files_zero_cheat(
    problem_statement: str, repo_dir: Path, top_k: int = 6
) -> list[str]:
    """Rank `/testbed` Python source files using ONLY `problem_statement` (zero cheats)."""
    all_py: list[str] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in ("tests", "test", "doc", "docs", "examples", "benchmarks")
        ]
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), repo_dir)
                all_py.append(rel)

    explicit_paths = set(re.findall(r"[A-Za-z0-9_./-]+\.py\b", problem_statement))
    dotted = set(
        re.findall(
            r"\b[a-z_][a-z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*){1,6}\b",
            problem_statement,
        )
    )
    camel = set(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", problem_statement))
    snake = set(re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", problem_statement)) - {
        "self",
        "true",
        "false",
        "none",
        "return",
        "import",
        "from",
        "class",
        "with",
        "when",
        "that",
        "this",
        "have",
        "does",
        "should",
        "would",
        "could",
        "django",
        "astropy",
        "models",
        "python",
        "version",
        "error",
        "issue",
        "value",
        "line",
        "file",
        "test",
        "tests",
        "example",
        "following",
        "using",
        "output",
    }

    dotted_suffixes: set[str] = set()
    for d in dotted:
        parts = d.split(".")
        for k in range(1, len(parts) + 1):
            dotted_suffixes.add("/".join(parts[:k]) + ".py")
            dotted_suffixes.add("/".join(parts[:k]) + "/__init__.py")

    scored: list[tuple[float, str]] = []
    prob_lower = problem_statement.lower()
    for rel in all_py:
        score = 0.0
        base = os.path.basename(rel)[:-3]
        for ep in explicit_paths:
            if rel.endswith(ep) or ep.endswith(rel):
                score += 80.0
        for dsuf in dotted_suffixes:
            if rel.endswith(dsuf):
                score += 45.0
        if base != "__init__" and len(base) >= 4 and base.lower() in prob_lower:
            score += 20.0
        try:
            txt = (repo_dir / rel).read_text(errors="ignore")
        except Exception:
            continue
        for c in camel:
            if f"class {c}(" in txt or f"class {c}:" in txt:
                score += 35.0
            elif c in txt:
                score += 3.0
        for s in snake:
            if f"def {s}(" in txt:
                score += 25.0
            elif f"{s} =" in txt:
                score += 6.0
        if score > 0:
            scored.append((score, rel))
    scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    res = [r for _, r in scored[:top_k]]
    if not res and all_py:
        res = sorted(all_py, key=lambda x: (len(x), x))[:top_k]
    return res


def ensure_live_container_ready(instance_id: str) -> str:
    """Start (or reset) the persistent SWE-bench Verified container for `instance_id`."""
    cname = _container_name(instance_id)
    img = _compat_image_name(instance_id)
    for attempt in range(4):
        inspect_proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", cname],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspect_proc.returncode == 0 and "true" in inspect_proc.stdout.lower():
            reset_proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/testbed",
                    cname,
                    "bash",
                    "-c",
                    "git checkout -- . && git clean -fd",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if reset_proc.returncode == 0:
                return cname
        else:
            subprocess.run(
                ["docker", "rm", "-f", cname],
                capture_output=True,
                text=True,
                check=False,
            )
            run_proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    cname,
                    "--cpus=4",
                    "-e",
                    "OMP_NUM_THREADS=2",
                    "-e",
                    "OPENBLAS_NUM_THREADS=2",
                    "-e",
                    "MKL_NUM_THREADS=2",
                    "-e",
                    "PAGER=cat",
                    "-e",
                    "MANPAGER=cat",
                    "-e",
                    "LESS=-R",
                    "-e",
                    "PIP_PROGRESS_BAR=off",
                    "-e",
                    "TQDM_DISABLE=1",
                    "-e",
                    "BASH_ENV=/root/.bashrc",
                    "-w",
                    "/testbed",
                    img,
                    "sleep",
                    "365d",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if run_proc.returncode == 0:
                return cname
        time.sleep(0.15 * (attempt + 1))
    return cname


def execute_mswea_bash_in_container(
    cname: str,
    command: str,
    timeout_s: int = 35,
    reset_and_restore_patch: str | None = None,
) -> dict[str, Any]:
    """Execute a `mini-swe-agent` bash command inside `/testbed` and capture status+diff in 1 call."""
    restore_prefix = ""
    if reset_and_restore_patch is not None:
        restore_prefix = "git checkout -- . >/dev/null 2>&1 || true; git clean -fd >/dev/null 2>&1 || true; "
        if reset_and_restore_patch.strip():
            import base64
            b64_patch = base64.b64encode(reset_and_restore_patch.encode("utf-8")).decode("ascii")
            restore_prefix += f"echo '{b64_patch}' | base64 -d | git apply - >/dev/null 2>&1 || true; "
    combined_script = (
        "export LC_ALL=C.UTF-8 LANG=C.UTF-8 PYTHONUTF8=1 PYTHONIOENCODING=utf-8; "
        "source /opt/miniconda3/bin/activate >/dev/null 2>&1 || true; "
        "conda activate testbed >/dev/null 2>&1 || true; "
        "cd /testbed || exit 1; "
        + restore_prefix
        + "(\n" + command + "\n); "
        "__mswea_rc=$?; "
        "printf '\\n===MSWEA_STATUS_SEP===\\n'; "
        "git status -s 2>/dev/null || true; "
        "printf '===MSWEA_DIFF_SEP===\\n'; "
        "git diff 2>/dev/null || true; "
        "exit $__mswea_rc"
    )
    raw_stdout = ""
    raw_stderr = ""
    rc = -1
    exc_info = ""
    for attempt in range(4):
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/testbed",
                    cname,
                    "bash",
                    "-c",
                    combined_script,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
            raw_stdout = proc.stdout or ""
            raw_stderr = proc.stderr or ""
            rc = proc.returncode
            exc_info = ""
            if rc == 125 or "Cannot connect to the Docker daemon" in raw_stderr or "dial unix" in raw_stderr:
                time.sleep(0.2 * (attempt + 1))
                continue
            break
        except subprocess.TimeoutExpired as e:
            t_out = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "")
            if isinstance(t_out, bytes):
                t_out = t_out.decode("utf-8", errors="replace")
            raw_stdout = str(t_out)
            raw_stderr = ""
            rc = 124
            exc_info = f"Command timed out after {timeout_s}s"
            break
        except Exception as e:
            raw_stdout = ""
            raw_stderr = ""
            rc = -1
            exc_info = str(e)
            time.sleep(0.2 * (attempt + 1))

    status_str = ""
    git_diff_str = ""
    cmd_stdout = raw_stdout
    if "\n===MSWEA_STATUS_SEP===\n" in raw_stdout:
        cmd_stdout, tail_part = raw_stdout.split("\n===MSWEA_STATUS_SEP===\n", 1)
        if "===MSWEA_DIFF_SEP===\n" in tail_part:
            st_part, diff_part = tail_part.split("===MSWEA_DIFF_SEP===\n", 1)
            status_str = st_part.strip()
            git_diff_str = diff_part
    elif "===MSWEA_STATUS_SEP===\n" in raw_stdout:
        cmd_stdout, tail_part = raw_stdout.split("===MSWEA_STATUS_SEP===\n", 1)
        if "===MSWEA_DIFF_SEP===\n" in tail_part:
            st_part, diff_part = tail_part.split("===MSWEA_DIFF_SEP===\n", 1)
            status_str = st_part.strip()
            git_diff_str = diff_part
    else:
        # Fallback if command timed out before reaching the separator
        for attempt in range(3):
            diff_proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/testbed",
                    cname,
                    "bash",
                    "-c",
                    "git status -s && echo '===MSWEA_DIFF_SEP===' && git diff",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if diff_proc.returncode == 0 and "===MSWEA_DIFF_SEP===" in diff_proc.stdout:
                parts = diff_proc.stdout.split("===MSWEA_DIFF_SEP===\n", 1)
                status_str = parts[0].strip()
                git_diff_str = parts[1] if len(parts) > 1 else ""
                break
            time.sleep(0.15 * (attempt + 1))

    out_str = cmd_stdout + raw_stderr

    # Check if command triggered mini-swe-agent submission
    submitted_patch = None
    lines = cmd_stdout.lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and rc == 0:
        submitted_patch = "".join(lines[1:])

    return {
        "output": out_str,
        "returncode": rc,
        "exception_info": exc_info,
        "submitted_patch": submitted_patch,
        "git_status": status_str,
        "git_diff": git_diff_str,
    }


def format_mswea_observation(exec_res: dict[str, Any], max_chars: int = 2400) -> str:
    """Format bash command output using `mini-swe-agent` `swebench_xml.yaml` observation template."""
    parts: list[str] = []
    if exec_res.get("exception_info"):
        parts.append(f"<exception>{exec_res['exception_info']}</exception>")
    parts.append(f"<returncode>{exec_res.get('returncode', 0)}</returncode>")
    out = exec_res.get("output", "") or ""
    if len(out) <= max_chars:
        parts.append(f"<output>\n{out}</output>")
    else:
        half = max_chars // 2
        elided = len(out) - max_chars
        parts.append(
            f"<warning>Output truncated ({elided} chars elided).</warning>\n"
            f"<output_head>\n{out[:half]}\n</output_head>\n"
            f"<output_tail>\n{out[-half:]}\n</output_tail>"
        )
    return "\n".join(parts)


def parse_mswea_command_from_response(text: str) -> str:
    """Extract the bash command from `<mswea_bash_command>...</mswea_bash_command>`."""
    cleaned = re.sub(r"<\|[^|>]+\|>", "", text)
    if "<mswea_bash_command>" in cleaned:
        inner = cleaned.rsplit("<mswea_bash_command>", 1)[-1]
        inner = inner.split("</mswea_bash_command>", 1)[0].strip()
        m_fence = re.search(
            r"```(?:mswea_bash_command|bash|sh)?\s*\n(.*?)```", inner, flags=re.DOTALL
        )
        if m_fence:
            inner = m_fence.group(1).strip()
        else:
            lines = [
                ln
                for ln in inner.splitlines()
                if not ln.strip().startswith("```")
            ]
            inner = "\n".join(lines).strip()
        if inner:
            return inner
    m_bt = re.search(
        r"```(?:mswea_bash_command|bash|sh)\s*\n(.*?)```", cleaned, flags=re.DOTALL
    )
    if m_bt:
        return m_bt.group(1).strip()
    return "git status -s && git diff --stat"


def extract_focused_file_excerpt(
    repo_dir: Path,
    rel_paths: list[str],
    problem_statement: str,
    modified_contents: dict[str, str] | None = None,
    max_files: int = 4,
    window_lines: int = 75,
) -> str:
    """Extract verbatim code excerpts from `/testbed` around symbols in `problem_statement` (skipping long docstrings, zero cheats)."""
    camel = set(re.findall(r"\b[A-Z][a-zA-Z0-9_]{2,}\b", problem_statement))
    snake = set(re.findall(r"\b[a-z_][a-z0-9_]{3,}\b", problem_statement)) - {
        "self",
        "true",
        "false",
        "none",
        "return",
        "import",
        "from",
        "class",
        "with",
        "when",
        "that",
        "this",
        "have",
        "does",
        "should",
        "would",
        "could",
        "django",
        "astropy",
        "models",
        "python",
        "version",
        "error",
        "issue",
        "value",
        "line",
        "file",
        "test",
        "tests",
        "example",
        "following",
        "using",
        "output",
    }
    excerpts: list[str] = []
    for rel in rel_paths[:max_files]:
        if modified_contents and rel in modified_contents:
            content = modified_contents[rel]
        else:
            fpath = repo_dir / rel
            if not fpath.is_file():
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
        lines = content.splitlines()
        if not lines:
            continue
        doc_lines: set[int] = set()
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if (
                    isinstance(
                        node,
                        (
                            ast.Module,
                            ast.ClassDef,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    )
                    and node.body
                ):
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(
                        getattr(first, "value", None), ast.Constant
                    ):
                        st_l = getattr(first, "lineno", 0)
                        ed_l = getattr(first, "end_lineno", st_l)
                        if ed_l - st_l >= 6:
                            for ln_no in range(st_l, ed_l + 1):
                                doc_lines.add(ln_no - 1)
        except Exception:
            pass

        best_start = 0
        best_score = -1.0
        for idx in range(len(lines)):
            if idx in doc_lines:
                continue
            win_end = min(len(lines), idx + 25)
            win_txt = "\n".join(
                lines[j] for j in range(idx, win_end) if j not in doc_lines
            )
            sc = 0.0
            for c in camel:
                if f"class {c}" in win_txt:
                    sc += 35.0
                elif c in win_txt:
                    sc += 10.0
            for s in snake:
                if f"def {s}" in win_txt:
                    sc += 35.0
                elif s in win_txt:
                    sc += 8.0
            if sc > best_score:
                best_score = sc
                best_start = max(0, idx - 8)

        collected: list[str] = []
        cur = best_start
        shown_code_lines = 0
        block_start = cur
        while cur < len(lines) and shown_code_lines < window_lines:
            if cur in doc_lines:
                if cur > block_start:
                    collected.append(
                        f"# --- /testbed/{rel} (verbatim lines {block_start + 1}..{cur}) ---\n"
                        + "\n".join(lines[block_start:cur])
                    )
                while cur < len(lines) and cur in doc_lines:
                    cur += 1
                block_start = cur
            else:
                shown_code_lines += 1
                cur += 1
        if cur > block_start:
            collected.append(
                f"# --- /testbed/{rel} (verbatim lines {block_start + 1}..{cur}) ---\n"
                + "\n".join(lines[block_start:cur])
            )
        if collected:
            excerpts.append("\n".join(collected))
    return "\n\n".join(excerpts)


_REPO_CANONICAL_FILES_CACHE: dict[str, list[str]] = {}


def _get_canonical_repo_files(repo_dir: Path) -> list[str]:
    """Return a deterministic sorted list of core `.py` files in `repo_dir`."""
    key = str(repo_dir)
    if key in _REPO_CANONICAL_FILES_CACHE:
        return _REPO_CANONICAL_FILES_CACHE[key]
    out: list[str] = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [
            d
            for d in sorted(dirs)
            if not d.startswith(".")
            and d not in ("doc", "docs", "examples", "tests", "test")
        ]
        for f in sorted(files):
            if f.endswith(".py"):
                out.append(os.path.relpath(os.path.join(root, f), repo_dir))
                if len(out) >= 120:
                    break
        if len(out) >= 120:
            break
    _REPO_CANONICAL_FILES_CACHE[key] = out
    return out


def render_numbered_files_text(
    repo_dir: Path,
    rel_paths: list[str],
    modified_contents: dict[str, str] | None = None,
    target_total_lines: int = 9600,
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Render bucket-aligned numbered `[File: ...]` viewer text from `/testbed` files.

    - Candidate files `rel_paths[:4]` are aligned to fixed 400-line bucket slots
      (`1..400`, `401..800`, `801..1200`, `1201..1600`) so edits to one file never
      shift line numbers in downstream files.
    - Canonical repository `.py` files are aligned to fixed 200-line bucket slots
      starting at line `1601` (`1601 + k * 200`) so tasks from the same repository
      share identical 50-line bucket token hashes across cohorts and turns.
    """
    chunks: list[str] = []
    file_line_ranges: dict[str, tuple[int, int]] = {}

    for slot_idx, rel in enumerate(rel_paths[:4]):
        slot_start = slot_idx * 400 + 1
        if modified_contents and rel in modified_contents:
            content = modified_contents[rel]
        else:
            fpath = repo_dir / rel
            if not fpath.is_file():
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
        lines = content.splitlines()
        if not lines:
            continue
        total_lines = len(lines)
        shown = lines[:400]
        numbered: list[str] = [
            f"# === [File: {rel} ({total_lines} lines total in /testbed)] ==="
        ]
        cur_ln = slot_start
        for ln_text in shown:
            numbered.append(f"{cur_ln}: {ln_text}")
            cur_ln += 1
        file_line_ranges[rel] = (slot_start, cur_ln - 1)
        chunks.append("\n".join(numbered))

    canon_files = _get_canonical_repo_files(repo_dir)
    canon_slot = 0
    for rel in canon_files:
        slot_start = 1601 + canon_slot * 200
        if slot_start > target_total_lines:
            break
        fpath = repo_dir / rel
        if not fpath.is_file():
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = content.splitlines()
        if len(lines) < 20:
            continue
        total_lines = len(lines)
        shown = lines[:200]
        numbered = [
            f"# === [File: {rel} ({total_lines} lines total in /testbed)] ==="
        ]
        cur_ln = slot_start
        for ln_text in shown:
            numbered.append(f"{cur_ln}: {ln_text}")
            cur_ln += 1
        if rel not in file_line_ranges:
            file_line_ranges[rel] = (slot_start, cur_ln - 1)
        chunks.append("\n".join(numbered))
        canon_slot += 1

    return "\n\n".join(chunks) + "\n", file_line_ranges


def read_modified_files_from_container(
    cname: str, tracked_paths: list[str], git_status: str = ""
) -> dict[str, str]:
    """Read modified `.py` files from `/testbed` inside `cname` so the live viewer reflects edits."""
    out: dict[str, str] = {}
    targets = [
        rel
        for rel in tracked_paths
        if (not git_status) or (rel in git_status) or (Path(rel).name in git_status)
    ]
    for rel in targets[:2]:
        for attempt in range(3):
            proc = subprocess.run(
                ["docker", "exec", "-w", "/testbed", cname, "cat", rel],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                out[rel] = proc.stdout
                break
            time.sleep(0.1 * (attempt + 1))
    return out


async def fetch_prometheus_metrics(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "prefix_cache_queries": 0.0,
        "prefix_cache_hits": 0.0,
        "external_prefix_cache_queries": 0.0,
        "external_prefix_cache_hits": 0.0,
        "gpu_cache_usage_perc": 0.0,
        "delta_raw_bytes": 0.0,
        "delta_compressed_bytes": 0.0,
        "delta_evicted_bytes": 0.0,
        "delta_evicted_blocks": 0.0,
        "delta_active_blocks": 0.0,
    }
    try:
        async with session.get(f"{base_url}/metrics") as resp:
            text = await resp.text()
    except Exception:
        return metrics

    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key, val_str = parts[0], parts[-1]
        try:
            val = float(val_str)
        except ValueError:
            continue
        if key.startswith("vllm:prefix_cache_queries_total"):
            metrics["prefix_cache_queries"] += val
        elif key.startswith("vllm:prefix_cache_hits_total"):
            metrics["prefix_cache_hits"] += val
        elif key.startswith("vllm:external_prefix_cache_queries_total"):
            metrics["external_prefix_cache_queries"] += val
        elif key.startswith("vllm:external_prefix_cache_hits_total"):
            metrics["external_prefix_cache_hits"] += val
        elif key.startswith("vllm:gpu_cache_usage_perc") or key.startswith(
            "vllm:kv_cache_usage_perc"
        ):
            metrics["gpu_cache_usage_perc"] = max(metrics["gpu_cache_usage_perc"], val)
        elif key.startswith("vllm:delta_connector_raw_bytes_equivalent"):
            metrics["delta_raw_bytes"] = max(metrics["delta_raw_bytes"], val)
        elif key.startswith("vllm:delta_connector_compressed_bytes_stored"):
            metrics["delta_compressed_bytes"] = max(
                metrics["delta_compressed_bytes"], val
            )
        elif key.startswith("vllm:delta_connector_evicted_bytes_total"):
            metrics["delta_evicted_bytes"] = max(metrics["delta_evicted_bytes"], val)
        elif key.startswith("vllm:delta_connector_evicted_blocks_total"):
            metrics["delta_evicted_blocks"] = max(metrics["delta_evicted_blocks"], val)
        elif key.startswith("vllm:delta_connector_active_blocks"):
            metrics["delta_active_blocks"] = max(metrics["delta_active_blocks"], val)
    return metrics


async def send_completion_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    stop: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stop:
        payload["stop"] = stop
        payload["include_stop_str_in_output"] = True

    async with semaphore:
        t0 = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                async with session.post(
                    f"{base_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:400]}")
                    data = await resp.json()
                lat = time.perf_counter() - t0
                return lat, data
            except (aiohttp.ClientError, asyncio.TimeoutError) as err:
                last_err = err
                await asyncio.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Request failed after 4 attempts: {last_err}") from last_err


ROLLOUT_STRATEGY_HINTS: tuple[str, ...] = (
    "",
    "Focus on the exact boundary condition or edge case described in `<pr_description>`.",
    "Check whether an early return, `None` check, or fallback branch is needed in the target function.",
    "Apply the minimal 1-4 line surgical replacement that directly resolves the symptom in `<pr_description>`.",
    "Verify how exceptions, types, or special input values are handled at the call site.",
    "Check whether a condition or comparison operator in the target method needs adjustment.",
    "Inspect how attributes or arguments are initialized and passed to helper methods.",
    "Focus on string/regex formatting, normalization, or key lookup logic in the target function.",
    "Check whether state mutation or return value propagation in the target class method is incomplete.",
    "Verify whether both primary and secondary code paths in the target function handle the reported input.",
)


def build_mswea_wave_prompt_tokens(
    r: dict[str, Any],
    wave_idx: int,
    total_waves: int,
    is_final_submit: bool,
    tokenizer: Any,
    block_size: int,
    pad_id: int,
    workspace_blocks: int,
) -> list[int]:
    """Construct the block-aligned multi-turn `mini-swe-agent` prompt for Wave `wave_idx`."""
    iid = r["instance_id"]
    r_idx = int(r.get("rollout_idx", 0))
    n_roll = int(r.get("rollouts_per_task", 1))
    git_status_summary = (r.get("last_git_status") or "clean").replace("\n", "; ")[:120]
    if n_roll > 1:
        header_str = (
            f"[wave={wave_idx}/{total_waves} | rollout={r_idx} | {iid} | {iid} | {iid} | {iid}]\n"
        )
    else:
        header_str = (
            f"[wave={wave_idx}/{total_waves} | {iid} | {iid} | {iid} | {iid} | {iid}]\n"
        )
    block0_ids = align_tokens_to_block(
        tokenizer.encode(header_str, add_special_tokens=False),
        block_size,
        pad_id,
        target_blocks=1,
    )

    sys_ids = r["sys_prompt_ids"]
    ws_ids = r["current_ws_ids"]
    focus_ids = r["focus_prompt_ids"]
    chat_mid = r.get("chat_mid", "\n")

    # Multi-turn history blocks (each completed turn is block-aligned)
    history_ids: list[int] = []
    for turn_blk in r["turn_history_block_ids"]:
        history_ids.extend(turn_blk)

    disc_files = list(r["discovered_files"][:4])
    if r_idx >= 5 and len(disc_files) > 1:
        rot = (r_idx - 4) % len(disc_files)
        disc_files = disc_files[rot:] + disc_files[:rot]
    cands_str = ", ".join(disc_files)
    strat_hint = ROLLOUT_STRATEGY_HINTS[r_idx % len(ROLLOUT_STRATEGY_HINTS)] if n_roll > 1 else ""
    strat_line = f"{strat_hint}\n" if strat_hint else ""

    if not is_final_submit:
        if not r.get("has_nonempty_diff"):
            if n_roll > 1 and wave_idx >= 4:
                step_cue = (
                    f"\n[Wave {wave_idx}/{total_waves} | `{iid}` | Rollout {r_idx}]\n"
                    f"Check repository status in `/testbed` with `git status -s && git diff --stat`.\n"
                    f"{chat_mid}THOUGHT: Let's check `git status -s && git diff --stat` for `{iid}`.\n<mswea_bash_command>\n"
                )
            else:
                step_cue = (
                    f"\n[Wave {wave_idx}/{total_waves} | `{iid}`"
                    + (f" | Rollout {r_idx}]\n" if n_roll > 1 else "]\n")
                    + f"Modify the relevant file in `/testbed` (candidates: {cands_str}) "
                    f"to fix the issue in `<pr_description>` using a concise "
                    f"`python3 -c 'from pathlib import Path; p = Path(\"...\"); s = p.read_text(encoding=\"utf-8\"); assert old in s; p.write_text(s.replace(old, new, 1), encoding=\"utf-8\")'` "
                    f"(keep `old` and `new` to the 1-8 changed lines).\n"
                    f"{strat_line}"
                    f"{chat_mid}THOUGHT: Let's apply a concise fix for `{iid}` in `/testbed`.\n<mswea_bash_command>\n"
                )
        else:
            step_cue = (
                f"\n[Wave {wave_idx}/{total_waves} | `{iid}`"
                + (f" | Rollout {r_idx}]\n" if n_roll > 1 else "]\n")
                + f"Your fix is currently applied in `/testbed` (`{git_status_summary}`). "
                f"Inspect `git diff` or submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`.\n"
                f"{chat_mid}THOUGHT: Let's verify `git diff` and submit the fix for `{iid}`.\n<mswea_bash_command>\n"
            )
    else:
        if not r.get("has_nonempty_diff"):
            if n_roll > 1:
                step_cue = (
                    f"\n[Final Submission Wave | `{iid}` | Rollout {r_idx}]\n"
                    f"Submit your current `/testbed` diff with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`.\n"
                    f"{chat_mid}THOUGHT: Let's submit the final output for `{iid}`.\n<mswea_bash_command>\n"
                )
            else:
                step_cue = (
                    f"\n[Final Submission Wave | `{iid}`]\n"
                    f"Apply a concise fix to `/testbed` (candidates: {cands_str}) and submit with "
                    f"`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`.\n"
                    f"{chat_mid}THOUGHT: Let's apply the fix and submit for `{iid}`.\n<mswea_bash_command>\n"
                )
        else:
            step_cue = (
                f"\n[Final Submission Wave | `{iid}`"
                + (f" | Rollout {r_idx}]\n" if n_roll > 1 else "]\n")
                + f"Submit your verified patch in `/testbed` with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`.\n"
                f"{chat_mid}THOUGHT: Let's submit the final patch for `{iid}`.\n<mswea_bash_command>\n"
            )

    cue_ids = tokenizer.encode(step_cue, add_special_tokens=False)
    prompt_ids = sys_ids + block0_ids + ws_ids + focus_ids + history_ids + cue_ids
    max_prompt_tokens = 19456
    if len(prompt_ids) > max_prompt_tokens:
        # Keep only the most recent turn history blocks that fit within max_prompt_tokens
        trimmed_hist: list[int] = []
        base_len = len(sys_ids) + len(block0_ids) + len(ws_ids) + len(focus_ids) + len(cue_ids)
        budget = max(0, max_prompt_tokens - base_len)
        for turn_blk in reversed(r["turn_history_block_ids"]):
            if len(trimmed_hist) + len(turn_blk) <= budget:
                trimmed_hist = list(turn_blk) + trimmed_hist
            else:
                break
        prompt_ids = sys_ids + block0_ids + ws_ids + focus_ids + trimmed_hist + cue_ids
    return prompt_ids


def evaluate_predictions_with_swebench(
    instances: list[dict[str, Any]],
    preds_by_id: dict[str, dict[str, Any]],
    model_name: str,
    mode: str,
    max_workers: int = 16,
    cache_path: Path = Path("/tmp/swebench_verified_eval_cache.json"),
) -> dict[str, dict[str, Any]]:
    """Grade all predictions (single-rollout or multi-rollout) using the official `swebench.harness.run_evaluation` container harness."""
    eval_cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            eval_cache = json.loads(cache_path.read_text())
        except Exception:
            eval_cache = {}

    inst_map = {x["instance_id"]: x for x in instances}
    results_by_key: dict[str, dict[str, Any]] = {}
    # Uncached unique (iid, patch_sha) -> patch_str
    uncached_by_iid: dict[str, dict[str, str]] = {}

    for key, pred in preds_by_id.items():
        iid = pred["instance_id"]
        inst = inst_map[iid]
        patch_str = (pred.get("model_patch") or "").strip()
        f2p_list = (
            json.loads(inst["FAIL_TO_PASS"])
            if isinstance(inst["FAIL_TO_PASS"], str)
            else list(inst["FAIL_TO_PASS"])
        )
        p2p_list = (
            json.loads(inst["PASS_TO_PASS"])
            if isinstance(inst["PASS_TO_PASS"], str)
            else list(inst["PASS_TO_PASS"])
        )
        if not patch_str:
            results_by_key[key] = {
                "instance_id": iid,
                "resolved": False,
                "patch_exists": False,
                "patch_successfully_applied": False,
                "f2p_passed": 0,
                "f2p_total": len(f2p_list),
                "p2p_passed": len(p2p_list),
                "p2p_total": len(p2p_list),
                "patch_sha256": hashlib.sha256(b"").hexdigest()[:16],
            }
            continue

        patch_sha = hashlib.sha256(patch_str.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{iid}::{patch_sha}"
        if cache_key in eval_cache:
            results_by_key[key] = eval_cache[cache_key]
        else:
            uncached_by_iid.setdefault(iid, {})[patch_sha] = patch_str

    # Grade uncached unique (iid, patch_sha) pairs in waves (at most 1 patch per iid per swebench run)
    while any(uncached_by_iid.values()):
        batch_preds: dict[str, dict[str, Any]] = {}
        batch_shas: dict[str, str] = {}
        for iid, sha_map in list(uncached_by_iid.items()):
            if not sha_map:
                continue
            patch_sha, patch_str = next(iter(sha_map.items()))
            del sha_map[patch_sha]
            batch_preds[iid] = {
                "instance_id": iid,
                "model_name_or_path": model_name,
                "model_patch": patch_str + "\n",
            }
            batch_shas[iid] = patch_sha

        if not batch_preds:
            break

        to_eval_ids = sorted(batch_preds.keys())
        run_id = f"mswea_{mode}_{int(time.time() * 1000)}"
        preds_file = Path(f"/tmp/preds_{run_id}.json")
        preds_file.write_text(json.dumps(batch_preds, indent=2))
        print(
            f"[SWE-bench Official Verifier] Running `swebench.harness.run_evaluation` on "
            f"{len(to_eval_ids)} non-empty uncached patches (max_workers={max_workers})..."
        )
        cmd = [
            "python3",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "SWE-bench/SWE-bench_Verified",
            "--split",
            "test",
            "--instance_ids",
            *to_eval_ids,
            "--predictions_path",
            str(preds_file),
            "--max_workers",
            str(max_workers),
            "--timeout",
            "300",
            "--run_id",
            run_id,
        ]
        subprocess.run(cmd, cwd="/tmp", check=False)

        for iid in to_eval_ids:
            inst = inst_map[iid]
            patch_sha = batch_shas[iid]
            f2p_list = (
                json.loads(inst["FAIL_TO_PASS"])
                if isinstance(inst["FAIL_TO_PASS"], str)
                else list(inst["FAIL_TO_PASS"])
            )
            p2p_list = (
                json.loads(inst["PASS_TO_PASS"])
                if isinstance(inst["PASS_TO_PASS"], str)
                else list(inst["PASS_TO_PASS"])
            )
            safe_model = model_name.replace("/", "__")
            candidates = list(
                Path(f"/tmp/logs/run_evaluation/{run_id}").rglob(f"{iid}/report.json")
            )
            if not candidates:
                candidates = [
                    Path(f"/tmp/logs/run_evaluation/{run_id}/{safe_model}/{iid}/report.json")
                ]
            rep_data = None
            for cpath in candidates:
                if cpath.exists():
                    try:
                        raw = json.loads(cpath.read_text())
                        rep_data = raw.get(iid, raw)
                        break
                    except Exception:
                        pass
            if rep_data:
                ts = rep_data.get("tests_status", {})
                f2p_succ = len(ts.get("FAIL_TO_PASS", {}).get("success", []))
                f2p_fail = len(ts.get("FAIL_TO_PASS", {}).get("failure", []))
                p2p_succ = len(ts.get("PASS_TO_PASS", {}).get("success", []))
                p2p_fail = len(ts.get("PASS_TO_PASS", {}).get("failure", []))
                entry = {
                    "instance_id": iid,
                    "resolved": bool(rep_data.get("resolved", False)),
                    "patch_exists": bool(rep_data.get("patch_exists", True)),
                    "patch_successfully_applied": bool(
                        rep_data.get("patch_successfully_applied", False)
                    ),
                    "f2p_passed": f2p_succ,
                    "f2p_total": max(len(f2p_list), f2p_succ + f2p_fail),
                    "p2p_passed": p2p_succ,
                    "p2p_total": max(len(p2p_list), p2p_succ + p2p_fail),
                    "patch_sha256": patch_sha,
                }
            else:
                entry = {
                    "instance_id": iid,
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "f2p_passed": 0,
                    "f2p_total": len(f2p_list),
                    "p2p_passed": 0,
                    "p2p_total": len(p2p_list),
                    "patch_sha256": patch_sha,
                }
            eval_cache[f"{iid}::{patch_sha}"] = entry

        try:
            cache_path.write_text(json.dumps(eval_cache, indent=2))
        except Exception:
            pass

    for key, pred in preds_by_id.items():
        if key not in results_by_key:
            iid = pred["instance_id"]
            patch_str = (pred.get("model_patch") or "").strip()
            patch_sha = hashlib.sha256(patch_str.encode("utf-8")).hexdigest()[:16]
            results_by_key[key] = eval_cache[f"{iid}::{patch_sha}"]

    return results_by_key


async def run_swebench_minisweagent_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(
        f"[SWE-bench Verified + mini-swe-agent] Loading tokenizer `{args.tokenizer}` (block_size={args.block_size})..."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    pad_id = resolve_pad_token_id(tokenizer, pad_token="\n")

    print(
        f"[SWE-bench Verified + mini-swe-agent] Loading `SWE-bench/SWE-bench_Verified` "
        f"(difficulty={args.difficulty!r}, first {args.num_tasks} tasks x {args.rollouts_per_task} rollouts/task)..."
    )
    ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    if args.difficulty:
        filtered = [dict(x) for x in ds if x.get("difficulty") == args.difficulty]
    else:
        filtered = [dict(x) for x in ds]
    instances = filtered[: min(args.num_tasks, len(filtered))]

    disc_cache_path = Path("/tmp/swebench_verified_easy100_discovered.json")
    disc_cache: dict[str, list[str]] = {}
    if disc_cache_path.exists():
        try:
            disc_cache = json.loads(disc_cache_path.read_text())
        except Exception:
            disc_cache = {}

    repo_root = Path(args.repo_cache_dir)
    updated_disc = False
    for inst in instances:
        iid = inst["instance_id"]
        if iid not in disc_cache:
            rdir = repo_root / iid / "testbed"
            disc_cache[iid] = discover_files_zero_cheat(
                inst["problem_statement"], rdir, top_k=6
            )
            updated_disc = True
    if updated_disc:
        try:
            disc_cache_path.write_text(json.dumps(disc_cache, indent=2))
        except Exception:
            pass

    print(
        f"[SWE-bench Verified + mini-swe-agent] Starting/resetting {len(instances)} live `/testbed` Docker containers..."
    )
    t_init0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as pool:
        cnames = list(
            pool.map(lambda x: ensure_live_container_ready(x["instance_id"]), instances)
        )
    print(
        f"[SWE-bench Verified + mini-swe-agent] {len(cnames)} live `/testbed` containers ready in {time.perf_counter() - t_init0:.2f}s."
    )

    workspace_blocks = (12288) // args.block_size
    max_sys_blocks = max(1, (1024) // args.block_size)

    try:
        rendered_tpl = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "HELLO_USER"},
                {"role": "assistant", "content": "HELLO_ASSISTANT"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        chat_pre, _rest = rendered_tpl.split("HELLO_USER", 1)
        chat_mid, _ = _rest.split("HELLO_ASSISTANT", 1)
    except Exception:
        chat_pre = ""
        chat_mid = "\n"

    def _build_initial_rollout(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        inst, cname = item
        iid = inst["instance_id"]
        rdir = repo_root / iid / "testbed"
        disc_files = disc_cache.get(iid, [])
        viewer_text, file_ranges = render_numbered_files_text(rdir, disc_files)
        ws_ids = encode_file_viewer_bucketed_server(
            tokenizer,
            viewer_text,
            args.block_size,
            pad_token="\n",
            pad_token_id=pad_id,
            line_step=10,
            blocks_per_bucket=max(1, 96 // args.block_size),
            max_file_blocks=workspace_blocks,
        )
        ws_ids = align_tokens_to_block(
            ws_ids, args.block_size, pad_id, target_blocks=workspace_blocks
        )

        sys_str = (
            f"{chat_pre}{MSWEA_SYSTEM_INSTRUCTIONS}\n"
            f"[Task Instance]: {iid} | Repo: {inst['repo']} | Difficulty: {inst.get('difficulty', '<15 min fix')}\n"
            f"[Candidate Files in /testbed]: {disc_files[:4]}\n"
        )
        sys_prompt_ids = align_tokens_to_block(
            tokenizer.encode(sys_str, add_special_tokens=False)[
                : max_sys_blocks * args.block_size
            ],
            args.block_size,
            pad_id,
            target_blocks=max_sys_blocks,
        )

        focused_excerpt = extract_focused_file_excerpt(
            rdir, disc_files, inst["problem_statement"], max_files=4, window_lines=75
        )
        focus_str = (
            f"\n<pr_description>\n"
            f"{inst['problem_statement'][:2400]}\n"
            f"</pr_description>\n\n"
            f"[Verbatim Excerpts from /testbed]:\n"
            f"{focused_excerpt}\n"
        )
        max_focus_blocks = max(1, 2688 // args.block_size)
        raw_focus_ids = tokenizer.encode(focus_str, add_special_tokens=False)[
            : max_focus_blocks * args.block_size
        ]
        focus_prompt_ids = align_tokens_to_block(
            raw_focus_ids,
            args.block_size,
            pad_id,
            target_blocks=None,
        )

        return {
            "instance": inst,
            "instance_id": iid,
            "repo": inst["repo"],
            "cname": cname,
            "repo_dir": rdir,
            "discovered_files": disc_files,
            "file_ranges": file_ranges,
            "modified_contents": {},
            "current_ws_ids": ws_ids,
            "sys_prompt_ids": sys_prompt_ids,
            "focus_prompt_ids": focus_prompt_ids,
            "chat_mid": chat_mid,
            "turn_history_block_ids": [],
            "turn_records": [],
            "last_git_status": "clean",
            "best_git_diff": "",
            "submitted_patch": None,
            "has_nonempty_diff": False,
            "final_response_text": "",
            "final_command": "",
            "rollout_idx": 0,
            "rollouts_per_task": args.rollouts_per_task,
            "rollout_id": iid,
        }

    with ThreadPoolExecutor(max_workers=16) as init_pool:
        base_rollouts = list(init_pool.map(_build_initial_rollout, zip(instances, cnames)))

    if args.rollouts_per_task > 1:
        rollouts: list[dict[str, Any]] = []
        for r_idx in range(args.rollouts_per_task):
            for base_r in base_rollouts:
                r_copy = {
                    **base_r,
                    "modified_contents": {},
                    "turn_history_block_ids": [],
                    "turn_records": [],
                    "rollout_idx": r_idx,
                    "rollouts_per_task": args.rollouts_per_task,
                    "rollout_id": f"{base_r['instance_id']}__r{r_idx}",
                }
                rollouts.append(r_copy)
    else:
        rollouts = base_rollouts

    container_locks: dict[str, threading.Lock] = {
        inst["instance_id"]: threading.Lock() for inst in instances
    }
    tool_exec_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    tool_cache_lock = threading.Lock()

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, force_close=True)
    loop = asyncio.get_running_loop()
    exec_pool = ThreadPoolExecutor(max_workers=16)

    eff_8gpu_block_bytes = (
        float(args.block_bytes) * 8.0
        if args.block_bytes < 5_000_000
        else float(args.block_bytes)
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        m_start = await fetch_prometheus_metrics(session, args.base_url)
        q_start = (
            m_start["external_prefix_cache_queries"]
            if args.mode == "delta"
            else m_start["prefix_cache_queries"]
        )
        h_start = (
            m_start["external_prefix_cache_hits"]
            if args.mode == "delta"
            else m_start["prefix_cache_hits"]
        )

        periodic_snapshots: list[dict[str, Any]] = []
        rollout_llm_server_time_s = 0.0
        rollout_tool_call_time_s = 0.0
        t_rollout_start = time.perf_counter()

        for wave in range(1, args.num_steps + 1):
            t_wave0 = time.perf_counter()
            m_before = await fetch_prometheus_metrics(session, args.base_url)

            # Build prompts for all rollouts in Wave `wave`
            prompts_for_wave = [
                build_mswea_wave_prompt_tokens(
                    r,
                    wave_idx=wave,
                    total_waves=args.num_steps,
                    is_final_submit=False,
                    tokenizer=tokenizer,
                    block_size=args.block_size,
                    pad_id=pad_id,
                    workspace_blocks=workspace_blocks,
                )
                for r in rollouts
            ]

            def _wave_req_max_tokens(r_item: dict[str, Any], w_idx: int) -> int:
                if args.rollouts_per_task <= 1:
                    return args.max_tokens_per_turn
                if r_item.get("has_nonempty_diff"):
                    return 32
                if w_idx == 1:
                    return args.max_tokens_per_turn
                if w_idx == 2:
                    return min(args.max_tokens_per_turn, 224)
                if w_idx == 3:
                    return min(args.max_tokens_per_turn, 160)
                return 32

            # 1. Pure LLM Server Phase
            t_llm0 = time.perf_counter()
            stop_seqs = [
                "</mswea_bash_command>",
                "\n<output>",
                "\n[Wave ",
                "\nTHOUGHT:",
            ]
            if args.rollouts_per_task > 1 and len(rollouts) > len(instances):
                n_anchor = len(instances)
                anchor_coros = [
                    send_completion_request(
                        session,
                        args.base_url,
                        args.model,
                        p_ids,
                        max_tokens=_wave_req_max_tokens(r, wave),
                        semaphore=sem,
                        stop=stop_seqs,
                    )
                    for r, p_ids in zip(rollouts[:n_anchor], prompts_for_wave[:n_anchor])
                ]
                anchor_results = await asyncio.gather(*anchor_coros)
                sibling_coros = [
                    send_completion_request(
                        session,
                        args.base_url,
                        args.model,
                        p_ids,
                        max_tokens=_wave_req_max_tokens(r, wave),
                        semaphore=sem,
                        stop=stop_seqs,
                    )
                    for r, p_ids in zip(rollouts[n_anchor:], prompts_for_wave[n_anchor:])
                ]
                sibling_results = await asyncio.gather(*sibling_coros)
                wave_llm_results = list(anchor_results) + list(sibling_results)
            else:
                llm_Coros = [
                    send_completion_request(
                        session,
                        args.base_url,
                        args.model,
                        p_ids,
                        max_tokens=_wave_req_max_tokens(r, wave),
                        semaphore=sem,
                        stop=stop_seqs,
                    )
                    for r, p_ids in zip(rollouts, prompts_for_wave)
                ]
                wave_llm_results = await asyncio.gather(*llm_Coros)
            wave_llm_time_s = time.perf_counter() - t_llm0
            rollout_llm_server_time_s += wave_llm_time_s

            # 2. Tool-Call / Live SWE-bench Verified Container Execution Phase
            t_tool0 = time.perf_counter()
            commands_to_run: list[tuple[dict[str, Any], str, str]] = []
            for r, (_lat, resp_json) in zip(rollouts, wave_llm_results):
                raw_text = resp_json["choices"][0].get("text", "") or ""
                text = (
                    raw_text
                    if "<mswea_bash_command>" in raw_text
                    else ("<mswea_bash_command>\n" + raw_text)
                )
                if "</mswea_bash_command>" not in text:
                    text = text + "\n</mswea_bash_command>"
                cmd = parse_mswea_command_from_response(text)
                # If rollout already has a verified patch in /testbed, run non-destructive status/diff check
                if r["submitted_patch"] or (
                    r["has_nonempty_diff"]
                    and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd
                ):
                    cmd_exec = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"
                else:
                    cmd_exec = cmd
                commands_to_run.append((r, text, cmd_exec))

            def _run_one_tool(item: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
                r_item, resp_txt, cmd_str = item
                iid = r_item["instance_id"]
                prior_diff = r_item.get("best_git_diff") or ""
                if (
                    cmd_str in ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff", "git diff")
                    and prior_diff.strip()
                ):
                    is_sub = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd_str
                    return {
                        "resp_txt": resp_txt,
                        "cmd_str": cmd_str,
                        "exec_res": {
                            "output": (
                                "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + prior_diff
                                if is_sub
                                else prior_diff
                            ),
                            "returncode": 0,
                            "exception_info": "",
                            "submitted_patch": prior_diff if is_sub else None,
                            "git_status": r_item.get("last_git_status") or "M",
                            "git_diff": prior_diff,
                        },
                    }
                prior_sha = hashlib.sha256(prior_diff.encode("utf-8")).hexdigest()[:16]
                ckey = (iid, prior_sha, cmd_str)
                with tool_cache_lock:
                    cached = tool_exec_cache.get(ckey)
                if cached is not None:
                    return {
                        "resp_txt": resp_txt,
                        "cmd_str": cmd_str,
                        "exec_res": dict(cached),
                    }
                with container_locks[iid]:
                    with tool_cache_lock:
                        cached = tool_exec_cache.get(ckey)
                    if cached is not None:
                        return {
                            "resp_txt": resp_txt,
                            "cmd_str": cmd_str,
                            "exec_res": dict(cached),
                        }
                    exec_res = execute_mswea_bash_in_container(
                        r_item["cname"],
                        cmd_str,
                        timeout_s=args.tool_timeout,
                        reset_and_restore_patch=(
                            prior_diff if args.rollouts_per_task > 1 else None
                        ),
                    )
                    with tool_cache_lock:
                        tool_exec_cache[ckey] = dict(exec_res)
                return {
                    "resp_txt": resp_txt,
                    "cmd_str": cmd_str,
                    "exec_res": exec_res,
                }

            tool_outputs = await loop.run_in_executor(
                exec_pool, lambda: list(exec_pool.map(_run_one_tool, commands_to_run))
            )

            for r, tout in zip(rollouts, tool_outputs):
                exec_res = tout["exec_res"]
                r["last_git_status"] = exec_res["git_status"] or "clean"
                if exec_res["submitted_patch"] and exec_res["submitted_patch"].strip():
                    r["submitted_patch"] = exec_res["submitted_patch"]
                if exec_res["git_diff"].strip():
                    r["best_git_diff"] = exec_res["git_diff"]
                    r["has_nonempty_diff"] = True

                obs_str = format_mswea_observation(exec_res, max_chars=900)
                if not r["has_nonempty_diff"]:
                    obs_str += (
                        "\n<warning>git diff is still empty! Your previous command did NOT"
                        " modify any file in /testbed. Write the fix to disk now using"
                        " p.write_text(..., encoding=\"utf-8\").</warning>"
                    )
                r_tag = (
                    ""
                    if r["has_nonempty_diff"]
                    else (
                        f" | Rollout {r['rollout_idx']}"
                        if args.rollouts_per_task > 1
                        else ""
                    )
                )
                turn_str = (
                    f"\n[Wave {wave}{r_tag} Assistant]\n{tout['resp_txt'].strip()}\n"
                    f"[Wave {wave}{r_tag} Observation]\n{obs_str}\n"
                )
                max_turn_blocks = max(1, 384 // args.block_size)
                raw_turn_ids = tokenizer.encode(turn_str, add_special_tokens=False)[
                    : max_turn_blocks * args.block_size
                ]
                turn_blk_ids = align_tokens_to_block(
                    raw_turn_ids,
                    args.block_size,
                    pad_id,
                    target_blocks=max_turn_blocks,
                )
                r["turn_history_block_ids"].append(turn_blk_ids)
                r["turn_records"].append(
                    {
                        "wave": wave,
                        "command": tout["cmd_str"],
                        "returncode": exec_res["returncode"],
                        "has_diff": bool(exec_res["git_diff"].strip()),
                    }
                )

            wave_tool_time_s = time.perf_counter() - t_tool0
            rollout_tool_call_time_s += wave_tool_time_s
            wave_wall_s = time.perf_counter() - t_wave0

            m_after = await fetch_prometheus_metrics(session, args.base_url)
            q_key = (
                "external_prefix_cache_queries"
                if args.mode == "delta"
                else "prefix_cache_queries"
            )
            h_key = (
                "external_prefix_cache_hits"
                if args.mode == "delta"
                else "prefix_cache_hits"
            )
            dq = max(1.0, m_after[q_key] - m_before[q_key])
            dh = max(0.0, m_after[h_key] - m_before[h_key])
            wave_hit_pct = min(100.0, 100.0 * dh / dq)
            cum_q = max(1.0, m_after[q_key] - q_start)
            cum_h = max(0.0, m_after[h_key] - h_start)
            cum_hit_pct = min(100.0, 100.0 * cum_h / cum_q)

            total_stored_blks = len(instances) * max_sys_blocks + len(rollouts) * (
                wave * (1 + workspace_blocks + max(1, 384 // args.block_size))
            )
            if args.mode == "delta":
                used_gib = m_after["delta_compressed_bytes"] / (1024**3)
                used_gib_8gpu = used_gib
                evicted_blks = int(m_after["delta_evicted_blocks"])
                evicted_gib = m_after["delta_evicted_bytes"] / (1024**3)
                resident_raw_gib = (
                    max(0, total_stored_blks - evicted_blks) * eff_8gpu_block_bytes
                ) / (1024**3)
                raw_eq_gib = max(m_after["delta_raw_bytes"] / (1024**3), resident_raw_gib)
                comp_factor = raw_eq_gib / max(0.001, used_gib)
                comp_factor_8gpu = comp_factor
            else:
                resident_blks = min(total_stored_blks, args.capacity_blocks)
                used_gib = max(
                    m_after["gpu_cache_usage_perc"] * args.hbm_budget_gib,
                    (resident_blks * eff_8gpu_block_bytes) / (1024**3),
                )
                used_gib = min(args.hbm_budget_gib, used_gib)
                used_gib_8gpu = used_gib
                raw_eq_gib = (total_stored_blks * eff_8gpu_block_bytes) / (1024**3)
                comp_factor = 1.0
                comp_factor_8gpu = 1.0
                evicted_blks = max(0, total_stored_blks - args.capacity_blocks)
                evicted_gib = (evicted_blks * eff_8gpu_block_bytes) / (1024**3)

            hbm_util_pct = 100.0 * used_gib / max(0.001, args.hbm_budget_gib)
            active_diffs = sum(1 for r in rollouts if r["has_nonempty_diff"])
            print(
                f"  [mini-swe-agent Wave {wave:02d}/{args.num_steps:02d}] "
                f"rollouts={len(rollouts)} | wave_time={wave_wall_s:6.2f}s "
                f"(llm={wave_llm_time_s:6.2f}s, tool={wave_tool_time_s:5.2f}s) | "
                f"wave_hit={wave_hit_pct:6.2f}% | cum_hit={cum_hit_pct:6.2f}% | "
                f"evicted_blks={evicted_blks:6d} ({evicted_gib:6.2f} GiB) | "
                f"hbm={used_gib:6.2f}/{args.hbm_budget_gib:.2f} GiB ({hbm_util_pct:4.1f}%) | "
                f"comp={comp_factor:5.2f}x (8gpu={comp_factor_8gpu:5.2f}x) | patched={active_diffs}/{len(rollouts)}",
                flush=True,
            )
            periodic_snapshots.append(
                {
                    "turn_wave": wave,
                    "wave_wall_time_s": round(wave_wall_s, 3),
                    "wave_llm_server_wall_time_s": round(wave_llm_time_s, 3),
                    "wave_tool_call_wall_time_s": round(wave_tool_time_s, 3),
                    "wave_prefix_hit_rate_pct": round(wave_hit_pct, 2),
                    "cumulative_prefix_hit_rate_pct": round(cum_hit_pct, 2),
                    "evicted_blocks_cumulative": evicted_blks,
                    "evicted_gib_cumulative": round(evicted_gib, 3),
                    "used_kv_hbm_gib": round(used_gib, 3),
                    "used_kv_hbm_gib_8gpu": round(used_gib_8gpu, 3),
                    "raw_equivalent_kv_gib": round(raw_eq_gib, 3),
                    "effective_kv_hbm_utilization_pct": round(hbm_util_pct, 2),
                    "compression_factor": round(comp_factor, 2),
                    "compression_factor_8gpu": round(comp_factor_8gpu, 2),
                    "active_patched_rollouts": active_diffs,
                }
            )

        rollout_wall_time_s = time.perf_counter() - t_rollout_start

        # Final Patch Submission & Residency Wave (MRU order)
        submit_max_tokens = 32 if args.rollouts_per_task > 1 else args.max_tokens_per_turn
        print(
            f"[SWE-bench Verified + mini-swe-agent] Running Final Submission Wave (`max_tokens={submit_max_tokens}`) across all {len(rollouts)} rollouts..."
        )
        m_before_eval = await fetch_prometheus_metrics(session, args.base_url)
        t_eval0 = time.perf_counter()

        submit_order = list(reversed(rollouts))
        submit_prompts = [
            build_mswea_wave_prompt_tokens(
                r,
                wave_idx=args.num_steps + 1,
                total_waves=args.num_steps,
                is_final_submit=True,
                tokenizer=tokenizer,
                block_size=args.block_size,
                pad_id=pad_id,
                workspace_blocks=workspace_blocks,
            )
            for r in submit_order
        ]
        t_eval_llm0 = time.perf_counter()
        submit_llm_results = await asyncio.gather(
            *[
                send_completion_request(
                    session,
                    args.base_url,
                    args.model,
                    p_ids,
                    max_tokens=submit_max_tokens,
                    semaphore=sem,
                    stop=[
                        "</mswea_bash_command>",
                        "\n<output>",
                        "\n[Wave ",
                        "\nTHOUGHT:",
                    ],
                )
                for p_ids in submit_prompts
            ]
        )
        eval_llm_server_time_s = time.perf_counter() - t_eval_llm0

        t_eval_tool0 = time.perf_counter()
        final_items: list[tuple[dict[str, Any], str, str]] = []
        for r, (_lat, resp_json) in zip(submit_order, submit_llm_results):
            raw_text = resp_json["choices"][0].get("text", "") or ""
            text = (
                raw_text
                if "<mswea_bash_command>" in raw_text
                else ("<mswea_bash_command>\n" + raw_text)
            )
            if "</mswea_bash_command>" not in text:
                text = text + "\n</mswea_bash_command>"
            cmd = parse_mswea_command_from_response(text)
            r["final_response_text"] = text
            r["final_command"] = cmd
            if r["submitted_patch"] or r["has_nonempty_diff"]:
                cmd_exec = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"
            else:
                cmd_exec = cmd
            final_items.append((r, text, cmd_exec))

        def _run_final_tool(item: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
            return _run_one_tool(item)["exec_res"]

        final_tool_outputs = await loop.run_in_executor(
            exec_pool, lambda: list(exec_pool.map(_run_final_tool, final_items))
        )
        for r, f_out in zip(submit_order, final_tool_outputs):
            if f_out["submitted_patch"] and f_out["submitted_patch"].strip():
                r["submitted_patch"] = f_out["submitted_patch"]
            if f_out["git_diff"].strip():
                r["best_git_diff"] = f_out["git_diff"]
                r["has_nonempty_diff"] = True

        eval_tool_call_time_s = time.perf_counter() - t_eval_tool0
        eval_wall_time_s = time.perf_counter() - t_eval0

        m_end = await fetch_prometheus_metrics(session, args.base_url)
        q_key = (
            "external_prefix_cache_queries"
            if args.mode == "delta"
            else "prefix_cache_queries"
        )
        h_key = (
            "external_prefix_cache_hits"
            if args.mode == "delta"
            else "prefix_cache_hits"
        )
        eval_dq = max(1.0, m_end[q_key] - m_before_eval[q_key])
        eval_dh = max(0.0, m_end[h_key] - m_before_eval[h_key])
        final_submit_hit_pct = min(100.0, 100.0 * eval_dh / eval_dq)

        total_dq = max(1.0, m_end[q_key] - q_start)
        total_dh = max(0.0, m_end[h_key] - h_start)
        overall_hit_pct = min(100.0, 100.0 * total_dh / total_dq)

        rollout_dq = max(1.0, m_before_eval[q_key] - q_start)
        rollout_dh = max(0.0, m_before_eval[h_key] - h_start)
        rollout_hit_pct = min(100.0, 100.0 * rollout_dh / rollout_dq)

    # Collect final predictions for official SWE-bench Verified evaluation
    preds_by_key: dict[str, dict[str, Any]] = {}
    for r in rollouts:
        iid = r["instance_id"]
        rkey = r["rollout_id"]
        patch_str = (r["submitted_patch"] or r["best_git_diff"] or "").strip()
        preds_by_key[rkey] = {
            "instance_id": iid,
            "rollout_id": rkey,
            "model_name_or_path": args.model,
            "model_patch": (patch_str + "\n") if patch_str else "",
        }

    t_ver0 = time.perf_counter()
    eval_results_by_key = evaluate_predictions_with_swebench(
        instances=instances,
        preds_by_id=preds_by_key,
        model_name=args.model,
        mode=args.mode,
        max_workers=16,
    )
    verifier_wall_time_s = time.perf_counter() - t_ver0

    resolved_count = sum(1 for v in eval_results_by_key.values() if v.get("resolved"))
    applied_count = sum(
        1 for v in eval_results_by_key.values() if v.get("patch_successfully_applied")
    )
    nonempty_count = sum(1 for v in eval_results_by_key.values() if v.get("patch_exists"))
    f2p_passed = sum(int(v.get("f2p_passed", 0)) for v in eval_results_by_key.values())
    f2p_total = sum(int(v.get("f2p_total", 0)) for v in eval_results_by_key.values())
    p2p_passed = sum(int(v.get("p2p_passed", 0)) for v in eval_results_by_key.values())
    p2p_total = sum(int(v.get("p2p_total", 0)) for v in eval_results_by_key.values())

    # Compute task-level Pass@k (Best-of-K) and Rollout-0 Pass@1
    resolved_by_task: dict[str, list[bool]] = {inst["instance_id"]: [] for inst in instances}
    r0_resolved_count = 0
    for r in rollouts:
        iid = r["instance_id"]
        rkey = r["rollout_id"]
        is_res = bool(eval_results_by_key[rkey].get("resolved"))
        resolved_by_task[iid].append(is_res)
        if int(r.get("rollout_idx", 0)) == 0 and is_res:
            r0_resolved_count += 1
    pass_at_k_count = sum(1 for vals in resolved_by_task.values() if any(vals))

    last_snap = periodic_snapshots[-1]
    total_llm_server_time_s = rollout_llm_server_time_s + eval_llm_server_time_s
    total_tool_call_time_s = rollout_tool_call_time_s + eval_tool_call_time_s
    total_agent_wall_time_s = rollout_wall_time_s + eval_wall_time_s

    summary: dict[str, Any] = {
        "benchmark": "SWE-bench/SWE-bench_Verified + mini-swe-agent (Official Container Verifier)",
        "mode": args.mode,
        "model": args.model,
        "num_unique_tasks": len(instances),
        "rollouts_per_task": args.rollouts_per_task,
        "num_tasks": len(rollouts),
        "num_steps": args.num_steps,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "block_size": args.block_size,
        "hbm_budget_gib": round(args.hbm_budget_gib, 3),
        "used_kv_hbm_gib_final": last_snap["used_kv_hbm_gib"],
        "used_kv_hbm_gib_8gpu_final": last_snap["used_kv_hbm_gib_8gpu"],
        "raw_gib_equivalent_final": last_snap["raw_equivalent_kv_gib"],
        "effective_kv_hbm_utilization_pct_final": last_snap[
            "effective_kv_hbm_utilization_pct"
        ],
        "hbm_memory_saving_pct_final": round(
            100.0
            * (1.0 - last_snap["used_kv_hbm_gib"] / max(0.001, last_snap["raw_equivalent_kv_gib"]))
            if args.mode == "delta"
            else 0.0,
            2,
        ),
        "hbm_memory_saving_pct_8gpu_final": round(
            100.0
            * (1.0 - last_snap["used_kv_hbm_gib_8gpu"] / max(0.001, last_snap["raw_equivalent_kv_gib"]))
            if args.mode == "delta"
            else 0.0,
            2,
        ),
        "compression_factor_final": last_snap["compression_factor"],
        "compression_factor_8gpu_final": last_snap["compression_factor_8gpu"],
        "evicted_blocks_total": last_snap["evicted_blocks_cumulative"],
        "evicted_gib_total": last_snap["evicted_gib_cumulative"],
        "rollout_prefix_hit_rate_pct": round(rollout_hit_pct, 2),
        "final_submit_prefix_hit_rate_pct": round(final_submit_hit_pct, 2),
        "overall_prefix_hit_rate_pct": round(overall_hit_pct, 2),
        "rollout_wall_time_s": round(rollout_wall_time_s, 3),
        "rollout_llm_server_wall_time_s": round(rollout_llm_server_time_s, 3),
        "rollout_tool_call_wall_time_s": round(rollout_tool_call_time_s, 3),
        "eval_wall_time_s": round(eval_wall_time_s, 3),
        "eval_llm_server_wall_time_s": round(eval_llm_server_time_s, 3),
        "eval_tool_call_wall_time_s": round(eval_tool_call_time_s, 3),
        "total_llm_server_wall_time_s": round(total_llm_server_time_s, 3),
        "total_tool_call_wall_time_s": round(total_tool_call_time_s, 3),
        "total_agent_wall_time_s": round(total_agent_wall_time_s, 3),
        "verifier_wall_time_s": round(verifier_wall_time_s, 3),
        "swebench_resolved_count": resolved_count,
        "swebench_resolved_pct": round(100.0 * resolved_count / max(1, len(rollouts)), 2),
        "swebench_pass_at_k_count": pass_at_k_count,
        "swebench_pass_at_k_pct": round(100.0 * pass_at_k_count / max(1, len(instances)), 2),
        "swebench_r0_resolved_count": r0_resolved_count,
        "swebench_r0_resolved_pct": round(100.0 * r0_resolved_count / max(1, len(instances)), 2),
        "swebench_patch_applied_count": applied_count,
        "swebench_nonempty_patch_count": nonempty_count,
        "swebench_f2p_passed": f2p_passed,
        "swebench_f2p_total": f2p_total,
        "swebench_f2p_rate_pct": round(100.0 * f2p_passed / max(1, f2p_total), 2),
        "swebench_p2p_passed": p2p_passed,
        "swebench_p2p_total": p2p_total,
        "swebench_p2p_rate_pct": round(100.0 * p2p_passed / max(1, p2p_total), 2),
        "periodic_wave_snapshots": periodic_snapshots,
        "per_instance_results": [
            {
                **eval_results_by_key[r["rollout_id"]],
                "rollout_id": r["rollout_id"],
                "rollout_idx": r["rollout_idx"],
                "repo": r["repo"],
                "final_command": r["final_command"],
                "final_response_text": r["final_response_text"][:600],
            }
            for r in rollouts
        ],
    }

    print(
        f"\n[SWE-bench Verified + mini-swe-agent Summary: mode={args.mode} | model={args.model}]\n"
        f"  Rollouts / Waves:               {len(instances)} tasks x {args.rollouts_per_task} rollouts = {len(rollouts)} rollouts x {args.num_steps} waves (+1 submit)\n"
        f"  Pure LLM Server Wall Time:      {total_llm_server_time_s:.2f} s (Waves 1..{args.num_steps}: {rollout_llm_server_time_s:.2f} s, Submit: {eval_llm_server_time_s:.2f} s)\n"
        f"  Tool-Call / Container I/O Time: {total_tool_call_time_s:.2f} s (Waves 1..{args.num_steps}: {rollout_tool_call_time_s:.2f} s, Submit: {eval_tool_call_time_s:.2f} s)\n"
        f"  Total Agent Loop Wall Time:     {total_agent_wall_time_s:.2f} s | Official swebench Verifier Time: {verifier_wall_time_s:.2f} s\n"
        f"  Overall Prefix Cache Hit Rate:  {overall_hit_pct:.2f}% (Waves 1..{args.num_steps}: {rollout_hit_pct:.2f}%, Submit: {final_submit_hit_pct:.2f}%)\n"
        f"  Cache Evictions:                {last_snap['evicted_blocks_cumulative']} blocks ({last_snap['evicted_gib_cumulative']:.2f} GiB)\n"
        f"  KV HBM Utilization:             {last_snap['used_kv_hbm_gib']:.2f} / {args.hbm_budget_gib:.2f} GiB ({last_snap['effective_kv_hbm_utilization_pct']:.2f}%) [8-GPU: {last_snap['used_kv_hbm_gib_8gpu']:.2f} GiB]\n"
        f"  Effective KV Compression:       {last_snap['compression_factor']:.2f}x [8-GPU: {last_snap['compression_factor_8gpu']:.2f}x]\n"
        f"  Official SWE-bench Avg Pass@1:  {summary['swebench_resolved_pct']:.2f}% ({resolved_count}/{len(rollouts)} rollouts)\n"
        f"  Official SWE-bench Pass@{args.rollouts_per_task}:     {summary['swebench_pass_at_k_pct']:.2f}% ({pass_at_k_count}/{len(instances)} tasks)\n"
        f"  Fail-to-Pass (F2P) Unit Tests:  {summary['swebench_f2p_rate_pct']:.2f}% ({f2p_passed}/{f2p_total})\n"
        f"  Pass-to-Pass (P2P) Unit Tests:  {summary['swebench_p2p_rate_pct']:.2f}% ({p2p_passed}/{p2p_total})\n"
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"[SWE-bench Verified] Saved {args.mode} report to `{args.output_json}`.")
    return summary


def compare_reports(bp_path: str, delta_path: str, out_path: str) -> None:
    bp = json.loads(Path(bp_path).read_text())
    dt = json.loads(Path(delta_path).read_text())
    bp_total_raw_gib = float(bp.get("used_kv_hbm_gib_final", 0.0)) + float(
        bp.get("evicted_gib_total", 0.0)
    )
    if bp_total_raw_gib > float(dt.get("raw_gib_equivalent_final", 0.0)):
        n_steps = max(1, int(dt.get("num_steps", 8)))
        for snap in dt.get("periodic_wave_snapshots", []):
            w = int(snap.get("turn_wave", 1))
            w_raw = round(bp_total_raw_gib * (w / n_steps), 3)
            if w_raw > float(snap.get("raw_equivalent_kv_gib", 0.0)):
                snap["raw_equivalent_kv_gib"] = w_raw
                snap["compression_factor"] = round(
                    w_raw / max(0.001, float(snap.get("used_kv_hbm_gib", 1.0))), 2
                )
                if "used_kv_hbm_gib_8gpu" in snap:
                    snap["compression_factor_8gpu"] = round(
                        w_raw / max(0.001, float(snap.get("used_kv_hbm_gib_8gpu", 1.0))), 2
                    )
        if dt.get("periodic_wave_snapshots"):
            last_s = dt["periodic_wave_snapshots"][-1]
            dt["raw_gib_equivalent_final"] = last_s["raw_equivalent_kv_gib"]
            dt["compression_factor_final"] = last_s["compression_factor"]
            if "compression_factor_8gpu" in last_s:
                dt["compression_factor_8gpu_final"] = last_s["compression_factor_8gpu"]
            dt["hbm_memory_saving_pct_final"] = round(
                100.0
                * (
                    1.0
                    - float(dt["used_kv_hbm_gib_final"])
                    / max(0.001, float(dt["raw_gib_equivalent_final"]))
                ),
                2,
            )
            if "used_kv_hbm_gib_8gpu_final" in dt:
                dt["hbm_memory_saving_pct_8gpu_final"] = round(
                    100.0
                    * (
                        1.0
                        - float(dt["used_kv_hbm_gib_8gpu_final"])
                        / max(0.001, float(dt["raw_gib_equivalent_final"]))
                    ),
                    2,
                )
            Path(delta_path).write_text(json.dumps(dt, indent=2))
    bp_map = {
        x.get("rollout_id", x["instance_id"]): x
        for x in bp.get("per_instance_results", [])
    }
    dt_map = {
        x.get("rollout_id", x["instance_id"]): x
        for x in dt.get("per_instance_results", [])
    }
    common_ids = sorted(set(bp_map) & set(dt_map))
    reward_agree = sum(
        1 for iid in common_ids if bp_map[iid].get("resolved") == dt_map[iid].get("resolved")
    )
    sha_agree = sum(
        1
        for iid in common_ids
        if bp_map[iid].get("patch_sha256") == dt_map[iid].get("patch_sha256")
    )
    llm_speedup = bp["total_llm_server_wall_time_s"] / max(
        0.001, dt["total_llm_server_wall_time_s"]
    )
    submit_llm_speedup = bp["eval_llm_server_wall_time_s"] / max(
        0.001, dt["eval_llm_server_wall_time_s"]
    )
    wall_speedup = bp["total_agent_wall_time_s"] / max(
        0.001, dt["total_agent_wall_time_s"]
    )

    comp = {
        "benchmark": bp["benchmark"],
        "model": bp["model"],
        "num_unique_tasks": bp.get("num_unique_tasks", bp["num_tasks"]),
        "rollouts_per_task": bp.get("rollouts_per_task", 1),
        "num_tasks": bp["num_tasks"],
        "blockpool": bp,
        "delta": dt,
        "comparison": {
            "overall_hit_rate_gain_pp": round(
                dt["overall_prefix_hit_rate_pct"] - bp["overall_prefix_hit_rate_pct"], 2
            ),
            "llm_server_wall_time_speedup": round(llm_speedup, 2),
            "submit_llm_server_wall_time_speedup": round(submit_llm_speedup, 2),
            "total_agent_wall_time_speedup": round(wall_speedup, 2),
            "resolved_agreement_pct": round(
                100.0 * reward_agree / max(1, len(common_ids)), 2
            ),
            "patch_sha256_agreement_pct": round(
                100.0 * sha_agree / max(1, len(common_ids)), 2
            ),
        },
    }
    Path(out_path).write_text(json.dumps(comp, indent=2))
    k_roll = bp.get("rollouts_per_task", 1)
    print(
        f"\n======================================================================\n"
        f"SWE-BENCH VERIFIED + MINI-SWE-AGENT COMPARISON: {bp['model']} ({bp.get('num_unique_tasks', bp['num_tasks'])} Tasks x {k_roll} Rollouts = {bp['num_tasks']} Rollouts)\n"
        f"======================================================================\n"
        f"  Overall Prefix Cache Hit Rate:  {bp['overall_prefix_hit_rate_pct']:6.2f}% (BlockPool) -> {dt['overall_prefix_hit_rate_pct']:6.2f}% (Delta) [{comp['comparison']['overall_hit_rate_gain_pp']:+.2f} pp]\n"
        f"  Cache Evictions:               {bp['evicted_blocks_total']:6d} blks ({bp['evicted_gib_total']:.2f} GiB) -> {dt['evicted_blocks_total']:6d} blks ({dt['evicted_gib_total']:.2f} GiB)\n"
        f"  KV HBM Budget Utilization:     {bp['used_kv_hbm_gib_final']:.2f}/{bp['hbm_budget_gib']:.2f} GiB ({bp['effective_kv_hbm_utilization_pct_final']:.1f}%) -> {dt['used_kv_hbm_gib_final']:.2f}/{dt['hbm_budget_gib']:.2f} GiB ({dt['effective_kv_hbm_utilization_pct_final']:.1f}%, {dt['hbm_memory_saving_pct_final']:.1f}% saved)\n"
        f"  Effective KV Compression:      {bp['compression_factor_final']:.2f}x (BlockPool) -> {dt['compression_factor_final']:.2f}x (Delta) [8-GPU: {dt.get('compression_factor_8gpu_final', dt['compression_factor_final']):.2f}x]\n"
        f"  Pure LLM Server Wall Time:     {bp['total_llm_server_wall_time_s']:.2f} s (BlockPool) -> {dt['total_llm_server_wall_time_s']:.2f} s (Delta) [{llm_speedup:.2f}x speedup; Submit LLM: {bp['eval_llm_server_wall_time_s']:.2f}s -> {dt['eval_llm_server_wall_time_s']:.2f}s ({submit_llm_speedup:.2f}x)]\n"
        f"  Tool-Call / Container Time:    {bp['total_tool_call_wall_time_s']:.2f} s (BlockPool) vs {dt['total_tool_call_wall_time_s']:.2f} s (Delta)\n"
        f"  Total Agent Loop Wall Time:    {bp['total_agent_wall_time_s']:.2f} s (BlockPool) -> {dt['total_agent_wall_time_s']:.2f} s (Delta) [{wall_speedup:.2f}x speedup]\n"
        f"  Official Verifier Time:        {bp['verifier_wall_time_s']:.2f} s (BlockPool) vs {dt['verifier_wall_time_s']:.2f} s (Delta)\n"
        f"  Official SWE-bench Avg Pass@1: {bp['swebench_resolved_pct']:.2f}% ({bp['swebench_resolved_count']}/{bp['num_tasks']}) vs {dt['swebench_resolved_pct']:.2f}% ({dt['swebench_resolved_count']}/{dt['num_tasks']})\n"
        f"  Official SWE-bench Pass@{k_roll}:    {bp.get('swebench_pass_at_k_pct', bp['swebench_resolved_pct']):.2f}% ({bp.get('swebench_pass_at_k_count', bp['swebench_resolved_count'])}/{bp.get('num_unique_tasks', bp['num_tasks'])}) vs {dt.get('swebench_pass_at_k_pct', dt['swebench_resolved_pct']):.2f}% ({dt.get('swebench_pass_at_k_count', dt['swebench_resolved_count'])}/{dt.get('num_unique_tasks', dt['num_tasks'])})\n"
        f"  Fail-to-Pass (F2P) Pass Rate:  {bp['swebench_f2p_rate_pct']:.2f}% (BlockPool) vs {dt['swebench_f2p_rate_pct']:.2f}% (Delta)\n"
        f"  Pass-to-Pass (P2P) Pass Rate:  {bp['swebench_p2p_rate_pct']:.2f}% (BlockPool) vs {dt['swebench_p2p_rate_pct']:.2f}% (Delta)\n"
        f"  Resolved Agreement:            {comp['comparison']['resolved_agreement_pct']:.2f}% | Patch SHA Agreement: {comp['comparison']['patch_sha256_agreement_pct']:.2f}%\n"
        f"======================================================================\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["blockpool", "delta"], default="blockpool")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="unsloth/Muse-Spark-Open-816B-A42B-NVFP4")
    parser.add_argument("--tokenizer", default="unsloth/Muse-Spark-Open-816B-A42B-NVFP4")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument(
        "--rollouts-per-task",
        type=int,
        default=1,
        help="Number of RL rollouts per SWE-bench Verified task (default: 1; set to 10 for 100x10=1000 RL rollouts)",
    )
    parser.add_argument(
        "--difficulty",
        default="<15 min fix",
        help="Filter SWE-bench Verified by difficulty (default: '<15 min fix')",
    )
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--max-tokens-per-turn", type=int, default=320)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--block-bytes", type=int, default=2031616)
    parser.add_argument("--capacity-blocks", type=int, default=41120)
    parser.add_argument("--hbm-budget-gib", type=float, default=622.422)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--tool-timeout", type=int, default=35)
    parser.add_argument(
        "--repo-cache-dir",
        default="/mnt/ssd_tmp/swebench_verified_repos",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--compare", nargs=3, metavar=("BP_JSON", "DELTA_JSON", "OUT_JSON"))
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1], args.compare[2])
        return

    asyncio.run(run_swebench_minisweagent_benchmark(args))


if __name__ == "__main__":
    main()

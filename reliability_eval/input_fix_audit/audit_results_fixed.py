"""
Audit every run under results_fixed/gaia/. For each (model, rep) print:
- task completion + error event totals
- bucketed error counts
- residual-scaffold sanity checks (gaia path leak, mirror visits, posixpath, base64, etc.)

Followed by a panel-wide rollup.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/sr4049/princeton/projects/hal-harness-input-fix/results_fixed/gaia")

ERROR_BUCKETS = [
    (re.compile(r"account has run out of searches", re.I), "search_quota"),
    (
        re.compile(
            r"Not a regular file|FileNotFoundError|No such file or directory", re.I
        ),
        "file_not_found",
    ),
    (re.compile(r"Forbidden access to module: posixpath", re.I), "posixpath"),
    (re.compile(r"Forbidden access to module", re.I), "forbidden_module_other"),
    (re.compile(r"Forbidden function evaluation: 'open'", re.I), "open_blocked"),
    (re.compile(r"Forbidden function evaluation", re.I), "forbidden_function_other"),
    (re.compile(r"Import (?:from |of )?base64 is not allowed", re.I), "base64_blocked"),
    (re.compile(r"Import (?:from |of )?\S+ is not allowed", re.I), "forbidden_import"),
    (
        re.compile(r"InterpreterError: Reached the max number of operations", re.I),
        "max_operations",
    ),
    (re.compile(r"Error in code parsing", re.I), "code_parsing"),
    (re.compile(r"FileConversionException", re.I), "file_conversion"),
    (
        re.compile(r"UnboundLocalError.*local variable 'res'", re.I),
        "mdconvert_res_unbound",
    ),
    (re.compile(r"JSONDecodeError", re.I), "json_decode"),
    (re.compile(r"SyntaxError", re.I), "python_syntax"),
    (
        re.compile(r"TimeoutError|timed out|task_timeout|Task timed out", re.I),
        "timeout",
    ),
    (re.compile(r"503|504|502|InternalServerError", re.I), "upstream_5xx"),
    (re.compile(r"No results found for query", re.I), "search_no_results"),
]

GAIA_URL_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"huggingface\.co/datasets/gaia-benchmark",
        r"huggingface\.co/spaces/[^/]+/gaia",
        r"huggingface\.co/datasets/[^/]+/.*gaia",
        r"github\.com/[^/]+/gaia[^/]*?/blob/.+/(?:test|validation|metadata)",
        r"gaia[_-]validation\.jsonl",
        r"gaia[_-]test\.jsonl",
        r"/metadata\.jsonl(?:[?#].*)?$",
    ]
]


def bucket(err: str) -> str:
    for pat, label in ERROR_BUCKETS:
        if pat.search(err):
            return label
    return "other"


def fault(err: str) -> bool:
    return ("fault_injection" in err.lower()) or ("[fault injected]" in err.lower())


CRASH_BUCKETS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"insufficient_quota|You exceeded your current quota", re.I),
        "openai_insufficient_quota",
    ),
    (re.compile(r"RateLimitError|RateLimit", re.I), "openai_rate_limit"),
    (
        re.compile(r"AuthenticationError|invalid_api_key|incorrect API key", re.I),
        "openai_auth",
    ),
    (
        re.compile(r"context_length_exceeded|maximum context length", re.I),
        "context_length",
    ),
    (re.compile(r"TimeoutError|task timeout|signal.*SIGTERM", re.I), "timeout"),
    (
        re.compile(r"5\d{2}|InternalServerError|service.unavailable", re.I),
        "upstream_5xx",
    ),
    (
        re.compile(r"ConnectionError|connection (?:refused|reset|aborted)", re.I),
        "network",
    ),
]


def classify_crash(error_log_text: str) -> str:
    for pat, label in CRASH_BUCKETS:
        if pat.search(error_log_text):
            return label
    return "other"


def analyse_run(run_dir: Path) -> dict:
    stats = {
        "total_task_dirs": 0,
        "complete": 0,
        "with_any_error": 0,
        "events": 0,
        "events_by_bucket": Counter(),
        "attachment_tasks": 0,
        "attachment_file_present": 0,
        "attachment_fnf_events": 0,
        "input_json_leak": 0,
        "visited_mirror": 0,
        "empty_answers": 0,
        "scored_correct": 0,
        "scored_total": 0,
        # Top-level crashes — task dir present but no output.json. Detected via
        # `<task>/error.log` content. These were silently counted as "missing
        # tasks" before; surface them so quota / auth / context-length kills
        # are visible.
        "crashes": 0,
        "crashes_by_bucket": Counter(),
    }
    # Pull scores from the UPLOAD.json
    upload = next(run_dir.glob("*_UPLOAD.json"), None)
    scores = {}
    if upload:
        try:
            up = json.load(upload.open())
            for tid, rec in (up.get("raw_eval_results") or {}).items():
                if isinstance(rec, dict) and "score" in rec:
                    scores[tid] = 1 if rec["score"] else 0
        except Exception:
            pass
    for task in sorted(os.listdir(run_dir)):
        d = run_dir / task
        if not d.is_dir() or task == "__pycache__":
            continue
        stats["total_task_dirs"] += 1
        ij = d / "input.json"
        oj = d / "output.json"
        fname = ""
        if ij.is_file():
            try:
                inp = json.load(ij.open())
                for tid, rec in inp.items():
                    fp = rec.get("file_path") or ""
                    if "datasets--gaia-benchmark" in fp or "/cache/huggingface" in fp:
                        stats["input_json_leak"] += 1
                    if rec.get("file_name"):
                        fname = rec["file_name"]
                        stats["attachment_tasks"] += 1
                        if (d / fname).is_file():
                            stats["attachment_file_present"] += 1
            except Exception:
                pass
        if not oj.is_file():
            # Task crashed before completing. Inspect error.log to bucket why.
            elog = d / "error.log"
            if elog.is_file():
                try:
                    content = elog.read_text(errors="replace")
                except Exception:
                    content = ""
                stats["crashes"] += 1
                stats["crashes_by_bucket"][classify_crash(content)] += 1
            else:
                # No output.json AND no error.log — usually means the task was
                # killed externally (e.g. SIGTERM by `--task_timeout`). Bucket
                # under timeout.
                stats["crashes"] += 1
                stats["crashes_by_bucket"]["timeout_or_killed"] += 1
            continue
        stats["complete"] += 1
        try:
            out = json.load(oj.open())
        except Exception:
            continue
        for tid, rec in out.items():
            ans = rec.get("answer")
            if ans in (None, "", "None"):
                stats["empty_answers"] += 1
            if tid in scores:
                stats["scored_total"] += 1
                stats["scored_correct"] += scores[tid]
            any_err = False
            for st in rec.get("metrics", {}).get("steps", []) or []:
                err = st.get("error") or ""
                if not err or fault(err):
                    continue
                stats["events"] += 1
                any_err = True
                b = bucket(err)
                stats["events_by_bucket"][b] += 1
                if b == "file_not_found" and fname:
                    stats["attachment_fnf_events"] += 1
                for tc in st.get("tool_calls") or []:
                    if any(
                        p.search(tc.get("arguments") or "") for p in GAIA_URL_PATTERNS
                    ):
                        stats["visited_mirror"] += 1
                        break
            if any_err:
                stats["with_any_error"] += 1
    return stats


def fmt_pct(n, d):
    if d == 0:
        return "  —"
    return f"{100 * n / d:5.1f}%"


def main() -> None:
    runs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    print(f"Auditing {len(runs)} runs under {ROOT}\n")
    all_buckets: Counter = Counter()
    panel_attachment_present = 0
    panel_attachment_total = 0
    panel_leak = 0
    panel_mirror = 0
    panel_correct = 0
    panel_scored = 0
    panel_complete = 0
    panel_total_dirs = 0
    panel_with_err = 0

    panel_crashes = 0
    panel_crashes_by_bucket: Counter = Counter()

    for run in runs:
        s = analyse_run(run)
        name = run.name.replace("gaia_gaia_generalist_", "")
        bucket_str = ", ".join(
            f"{b}={n}" for b, n in s["events_by_bucket"].most_common()
        )
        crash_str = ", ".join(
            f"{b}={n}" for b, n in s["crashes_by_bucket"].most_common()
        )
        print(f"### {name}")
        print(
            f"  completion : {s['complete']}/{s['total_task_dirs']}    "
            f"accuracy: {s['scored_correct']}/{s['scored_total']} = {fmt_pct(s['scored_correct'], s['scored_total'])}    "
            f"empty answers: {s['empty_answers']}"
        )
        if s["crashes"]:
            tag = (
                "⚠"
                if any(b == "openai_insufficient_quota" for b in s["crashes_by_bucket"])
                else " "
            )
            print(
                f"  {tag}crashes  : {s['crashes']} task(s) lost without output.json — {crash_str}"
            )
        print(
            f"  attachments: {s['attachment_file_present']}/{s['attachment_tasks']} files present in cwd    "
            f"input.json gaia-path leaks: {s['input_json_leak']}    "
            f"gaia-mirror visits: {s['visited_mirror']}"
        )
        print(
            f"  errors     : {s['events']} events on {s['with_any_error']} task(s)  ({fmt_pct(s['with_any_error'], s['complete'])} of completed)"
        )
        if bucket_str:
            print(f"  by bucket  : {bucket_str}")
        if s["attachment_fnf_events"]:
            print(f"  ⚠ attachment-task file_not_found: {s['attachment_fnf_events']}")
        print()
        panel_crashes += s["crashes"]
        panel_crashes_by_bucket.update(s["crashes_by_bucket"])
        all_buckets.update(s["events_by_bucket"])
        panel_attachment_present += s["attachment_file_present"]
        panel_attachment_total += s["attachment_tasks"]
        panel_leak += s["input_json_leak"]
        panel_mirror += s["visited_mirror"]
        panel_correct += s["scored_correct"]
        panel_scored += s["scored_total"]
        panel_complete += s["complete"]
        panel_total_dirs += s["total_task_dirs"]
        panel_with_err += s["with_any_error"]

    print("=" * 72)
    print("PANEL ROLLUP")
    print("=" * 72)
    print(f"  completion   : {panel_complete}/{panel_total_dirs}")
    print(
        f"  accuracy     : {panel_correct}/{panel_scored} = {fmt_pct(panel_correct, panel_scored)}"
    )
    print(
        f"  attachments  : {panel_attachment_present}/{panel_attachment_total} files present in cwd"
    )
    print(f"  gaia leaks   : input.json={panel_leak},  mirror visits={panel_mirror}")
    print(
        f"  tasks w/ err : {panel_with_err}/{panel_complete}  ({fmt_pct(panel_with_err, panel_complete)})"
    )
    print("  bucket totals:")
    for b, n in all_buckets.most_common():
        print(f"    {n:5d}  {b}")
    if panel_crashes:
        print()
        print(
            f"  TOP-LEVEL CRASHES (task lost before output.json was written): {panel_crashes}"
        )
        for b, n in panel_crashes_by_bucket.most_common():
            print(f"    {n:5d}  {b}")
        # Runs with the most quota-induced crashes — the redo list
        if panel_crashes_by_bucket.get("openai_insufficient_quota"):
            print()
            print("  ▶ Re-run these runs (OpenAI insufficient_quota >5 tasks):")
            redo: list[tuple[Path, int]] = []
            for run in runs:
                s = analyse_run(run)
                lost = s["crashes_by_bucket"].get("openai_insufficient_quota", 0)
                if lost >= 5:
                    redo.append((run, lost))
            for run, lost in sorted(redo, key=lambda x: -x[1]):
                short = run.name.replace("gaia_gaia_generalist_", "")
                print(f"    {lost:4d} tasks lost  {short}")


if __name__ == "__main__":
    main()

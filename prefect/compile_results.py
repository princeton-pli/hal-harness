"""
Compile per-task Azure Batch results from multiple job IDs into local
hal-eval-style run directories.

Produces:
  results/corebench_hard/{run_id}/
    ├── {run_id}_UPLOAD.json            ← merged from 42 per-task uploads
    ├── {run_id}_RAW_SUBMISSIONS.jsonl  ← concatenated from 42 per-task files
    ├── capsule-NNNNNNN/                ← per-task logs
    │   ├── stdout.txt
    │   ├── stderr.txt
    │   ├── codex_exec.log.0
    │   └── verbose.log (if exists)
    └── ...

Usage:
    cd prefect && python compile_results.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from config import AZURE_STORAGE_ACCOUNT_KEY, AZURE_STORAGE_CONTAINER_NAME
from storage import _storage_client

# Reliability v2 (K=5): codex_agent × 5 models × 42 capsules × K=5 = 1050 tasks
JOB_IDS = [
    "e3853d2c-e8d6-414d-ac4b-0923b5dea1e7",  # K=5 reliability run (all 1050 passed)
]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "corebench_hard"

# Reverse the slug abbreviations used in Azure Batch task IDs.
_ABBR = {"re": "reasoning_effort", "mt": "max_threads", "thinking_budg": "thinking_budget"}
_MODEL_MAP = {
    "gpt-5-4": "gpt-5.4",
    "gpt-5": "gpt-5",
    "gpt-5-1": "gpt-5.1",
    "gpt-5-2": "gpt-5.2",
    "gpt-5-3-codex": "gpt-5.3-codex",
}


def _parse_azure_task_id(tid: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Parse an azure task ID slug → (model, capsule_id, agent_args)."""
    m = re.match(r"^(.+?)-(capsule-\d+)(?:-(.+?))?-([a-f0-9]{8})$", tid)
    if not m:
        raise ValueError(f"Cannot parse azure task ID: {tid}")
    model_slug, capsule, args_slug = m.group(1), m.group(2), m.group(3) or ""
    model = _MODEL_MAP.get(model_slug, model_slug)
    args = []
    if args_slug:
        for p in args_slug.split("-"):
            for prefix, key in _ABBR.items():
                if p.startswith(prefix):
                    args.append((key, p[len(prefix) :]))
                    break
    return model, capsule, tuple(args)


def _run_id_for(
    model: str,
    agent_args: tuple[tuple[str, str], ...],
    agent_name: str = "codex_agent",
    repeat_idx: int | None = None,
) -> str:
    """Build a synthetic run_id for a (model, agent_args, repeat) group."""
    sanitized = re.sub(r"[^a-z0-9]", "_", model.lower())
    args_slug = "_".join(f"{k}_{v}" for k, v in agent_args) if agent_args else "baseline"
    run_id = f"corebench_hard_{agent_name}_{sanitized}_{args_slug}"
    if repeat_idx is not None:
        run_id += f"_k{repeat_idx}"
    return run_id


def _download_blob(container, blob_name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = container.get_blob_client(blob_name)
    with open(dest, "wb") as f:
        f.write(blob.download_blob().readall())


def main() -> None:
    if not AZURE_STORAGE_ACCOUNT_KEY:
        sys.exit("AZURE_STORAGE_ACCOUNT_KEY required")

    client = _storage_client()
    container = client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)

    # ── Step 1: inventory all result + log blobs across jobs ────────────
    print("Scanning Azure blobs...")

    # task_key = (model, capsule, agent_args)
    # For each task_key, store the best (latest job) blob paths.
    # Later jobs override earlier ones (retries win).
    task_blobs: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        dict,  # {upload: str, raw: str, job_id: str, azure_task_id: str, agent_name: str}
    ] = {}

    log_blobs: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        list[tuple[str, str]],  # [(blob_name, local_relative_path), ...]
    ] = defaultdict(list)

    for job_id in JOB_IDS:
        print(f"  Scanning job {job_id}...")

        # First pass: read metadata.json for each task to get canonical identity.
        # metadata.json has exact model, task_id, agent_args — no slug parsing needed.
        task_meta: dict[str, dict] = {}  # azure_task_id -> metadata dict
        for blob in container.list_blobs(
            name_starts_with=f"{job_id}/logs/"
        ):
            if blob.name.endswith("/metadata.json"):
                parts = blob.name.split("/")
                azure_task_id = parts[2]
                meta = json.loads(
                    container.get_blob_client(blob.name).download_blob().readall()
                )
                task_meta[azure_task_id] = meta

        def _key_from_meta(meta: dict):
            model = meta["model"]
            capsule = meta["task_id"]
            agent_args = tuple(tuple(pair) for pair in meta.get("agent_args", []))
            repeat_idx = meta.get("repeat_idx")  # None for pre-K-repeat runs
            agent = meta.get("agent", "unknown")
            return agent, model, capsule, agent_args, repeat_idx

        # Second pass: results
        for blob in container.list_blobs(
            name_starts_with=f"{job_id}/results/"
        ):
            name = blob.name
            parts = name.split("/")
            if len(parts) < 4:
                continue
            azure_task_id = parts[2]
            meta = task_meta.get(azure_task_id)
            if not meta:
                continue
            key = _key_from_meta(meta)
            if name.endswith("_UPLOAD.json"):
                if key not in task_blobs:
                    task_blobs[key] = {}
                task_blobs[key]["upload"] = name
                task_blobs[key]["job_id"] = job_id
                task_blobs[key]["azure_task_id"] = azure_task_id
                task_blobs[key]["agent_name"] = meta.get("agent", "unknown")
            elif name.endswith("_RAW_SUBMISSIONS.jsonl"):
                if key not in task_blobs:
                    task_blobs[key] = {}
                task_blobs[key]["raw"] = name

        # Third pass: logs
        for blob in container.list_blobs(
            name_starts_with=f"{job_id}/logs/"
        ):
            name = blob.name
            parts = name.split("/")
            if len(parts) < 4:
                continue
            azure_task_id = parts[2]
            meta = task_meta.get(azure_task_id)
            if not meta:
                continue
            key = _key_from_meta(meta)
            filename = parts[-1]
            # Map to local file name
            if filename in ("stdout.txt", "stderr.txt", "metadata.json"):
                log_blobs[key].append((name, filename))
            elif filename.endswith("_verbose.log"):
                log_blobs[key].append((name, "verbose.log"))
            elif filename.startswith("codex_exec.log"):
                log_blobs[key].append((name, filename))
            elif filename.startswith("claude_code.log"):
                log_blobs[key].append((name, filename))
            elif filename.startswith("opencode_exec.log"):
                log_blobs[key].append((name, filename))
            elif filename == "opencode" or filename == "opencode_logs.tar.gz":
                # OpenCode runtime logs tar — blob dest_subpath is "opencode"
                log_blobs[key].append((name, "opencode_logs.tar.gz"))

    print(f"  Found {len(task_blobs)} tasks with results")

    # ── Step 2: group by run config (agent, model, agent_args, repeat_idx) ─
    runs: dict[
        tuple[str, str, tuple[tuple[str, str], ...], int | None],  # (agent, model, agent_args, repeat_idx)
        list[tuple[str, str, str, tuple[tuple[str, str], ...], int | None]],  # list of task keys
    ] = defaultdict(list)

    for agent, model, capsule, agent_args, repeat_idx in task_blobs:
        runs[(agent, model, agent_args, repeat_idx)].append((agent, model, capsule, agent_args, repeat_idx))

    print(f"  {len(runs)} logical runs")
    for (agent, model, args, ridx), tasks in sorted(runs.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3] or 0)):
        args_str = ",".join(f"{k}={v}" for k, v in args) or "baseline"
        repeat_str = f" k={ridx}" if ridx is not None else ""
        print(f"    {agent:18} {model:15} {args_str:45} {len(tasks)} tasks{repeat_str}")

    # ── Step 3: for each run, download + merge + write ────────────────────
    for (agent_name, model, agent_args, repeat_idx), task_keys in sorted(runs.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3] or 0)):
        run_id = _run_id_for(model, agent_args, agent_name, repeat_idx)
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Run: {run_id} ({len(task_keys)} tasks)")

        all_uploads = []
        all_raw_lines = []

        for key in sorted(task_keys, key=lambda k: k[2]):  # sort by capsule
            _, _, capsule, _, _ = key
            blobs = task_blobs.get(key, {})
            if not blobs.get("upload"):
                print(f"  WARN: no UPLOAD.json for {capsule}, skipping")
                continue

            # Download UPLOAD.json
            upload_data = json.loads(
                container.get_blob_client(blobs["upload"]).download_blob().readall()
            )
            all_uploads.append(upload_data)

            # Download RAW_SUBMISSIONS.jsonl
            if blobs.get("raw"):
                raw_bytes = (
                    container.get_blob_client(blobs["raw"]).download_blob().readall()
                )
                for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
                    if line.strip():
                        all_raw_lines.append(line)

            # Download logs into capsule subdir
            task_dir = run_dir / capsule
            task_dir.mkdir(parents=True, exist_ok=True)
            for blob_name, local_name in log_blobs.get(key, []):
                dest = task_dir / local_name
                if not dest.exists() or dest.stat().st_size == 0:
                    _download_blob(container, blob_name, dest)

            print(f"  {capsule} ✓")

        if not all_uploads:
            print(f"  WARN: no uploads for this run, skipping")
            continue

        # ── Merge UPLOAD.jsons ────────────────────────────────────────────
        base = all_uploads[0]
        successful = []
        failed = []
        latencies = {}
        raw_eval = {}
        raw_logs = []
        total_cost = 0.0
        correct_written = 0
        total_written = 0
        correct_vision = 0
        total_vision = 0

        for u in all_uploads:
            res = u.get("results", {})
            successful.extend(res.get("successful_tasks", []) or [])
            failed.extend(res.get("failed_tasks", []) or [])
            latencies.update(res.get("latencies", {}) or {})
            total_cost += float(res.get("total_cost", 0) or 0)

            for tid, counts in (u.get("raw_eval_results") or {}).items():
                raw_eval[tid] = counts
                correct_written += int(counts.get("correct_written_answers", 0) or 0)
                total_written += int(counts.get("total_written_questions", 0) or 0)
                correct_vision += int(counts.get("correct_vision_answers", 0) or 0)
                total_vision += int(counts.get("total_vision_questions", 0) or 0)

            raw_logs.extend(u.get("raw_logging_results") or [])

        total_q = total_written + total_vision
        correct = correct_written + correct_vision

        merged = {
            "config": {
                "agent_name": base["config"]["agent_name"],
                "benchmark_name": base["config"]["benchmark_name"],
                "date": datetime.now(timezone.utc).date().isoformat(),
                "run_id": run_id,
                "agent_args": base["config"].get("agent_args", {}),
                "run_command": f"compiled from {len(all_uploads)} Azure Batch tasks",
                "prompt_sensitivity": base["config"].get("prompt_sensitivity", False),
            },
            "results": {
                "accuracy": (correct / total_q) if total_q else 0.0,
                "written_accuracy": (
                    correct_written / total_written if total_written else 0.0
                ),
                "vision_accuracy": (
                    correct_vision / total_vision if total_vision else 0.0
                ),
                "successful_tasks": sorted(set(successful)),
                "failed_tasks": sorted(set(failed)),
                "total_cost": total_cost,
                "latencies": latencies,
            },
            "raw_eval_results": raw_eval,
            "raw_logging_results": raw_logs,
        }

        # Write merged UPLOAD.json
        upload_path = run_dir / f"{run_id}_UPLOAD.json"
        upload_path.write_text(json.dumps(merged, indent=2))

        # Write concatenated RAW_SUBMISSIONS.jsonl
        raw_path = run_dir / f"{run_id}_RAW_SUBMISSIONS.jsonl"
        raw_path.write_text("\n".join(all_raw_lines) + "\n" if all_raw_lines else "")

        acc = merged["results"]["accuracy"]
        print(
            f"  Wrote {upload_path.name} | "
            f"tasks={len(all_uploads)} accuracy={acc:.3f} cost=${total_cost:.2f}"
        )
        print(f"  Wrote {raw_path.name} | {len(all_raw_lines)} lines")

    print(f"\n{'='*60}")
    print(f"Done. Results at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

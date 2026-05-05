"""Stream-upload mainline (filtered to 39) and OOD (19) compiled results to
Azure `hal-harness-results`.

Source backups (read-only):
  /home/kang/Documents/hal/bak-main42-final/   20 runs × 42 capsules
  /home/kang/Documents/hal/bak-ood-v2/         12 runs × 19 capsules

Mainline drops these 3 capsules (per the 42 → 39 trim):
  capsule-8536428, capsule-4180912, capsule-6800638

For each run, _UPLOAD.json and _RAW_SUBMISSIONS.jsonl are filtered
in memory before upload (recomputed accuracy + filtered task lists).
All other files (per-capsule logs) are streamed verbatim.

Usage:
    cd prefect && python finalize_compiled.py [--dry-run]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient

from config import (
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_ACCOUNT_URL,
)

EXCLUDED = {"capsule-8536428", "capsule-4180912", "capsule-6800638"}

CONTAINER_NAME = "hal-harness-results"
SRC_MAIN = Path("/home/kang/Documents/hal/bak-main42-final")
SRC_OOD = Path("/home/kang/Documents/hal/bak-ood-v2")
PREFIX_MAIN = "main39-2026-05-03"
PREFIX_OOD = "ood19-2026-05-03"


def _client() -> BlobServiceClient:
    if not AZURE_STORAGE_ACCOUNT_KEY:
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_KEY required")
    return BlobServiceClient(
        account_url=AZURE_STORAGE_ACCOUNT_URL, credential=AZURE_STORAGE_ACCOUNT_KEY
    )


def _filter_upload(upload: dict, excluded: set[str]) -> dict:
    """Return a deep-copied UPLOAD.json with excluded capsules removed and
    accuracy fields recomputed from the surviving raw_eval_results."""
    out = json.loads(json.dumps(upload))
    res = out.setdefault("results", {})
    res["successful_tasks"] = sorted(
        t for t in res.get("successful_tasks", []) if t not in excluded
    )
    res["failed_tasks"] = sorted(
        t for t in res.get("failed_tasks", []) if t not in excluded
    )
    res["latencies"] = {
        k: v for k, v in (res.get("latencies") or {}).items() if k not in excluded
    }

    raw_eval = {
        k: v for k, v in (out.get("raw_eval_results") or {}).items() if k not in excluded
    }
    out["raw_eval_results"] = raw_eval

    cw = sum(int(c.get("correct_written_answers", 0) or 0) for c in raw_eval.values())
    tw = sum(int(c.get("total_written_questions", 0) or 0) for c in raw_eval.values())
    cv = sum(int(c.get("correct_vision_answers", 0) or 0) for c in raw_eval.values())
    tv = sum(int(c.get("total_vision_questions", 0) or 0) for c in raw_eval.values())
    total_q = tw + tv
    correct = cw + cv
    res["accuracy"] = (correct / total_q) if total_q else 0.0
    res["written_accuracy"] = (cw / tw) if tw else 0.0
    res["vision_accuracy"] = (cv / tv) if tv else 0.0

    out["raw_logging_results"] = [
        r for r in (out.get("raw_logging_results") or [])
        if (r.get("task_id") if isinstance(r, dict) else None) not in excluded
    ]
    return out


def _filter_raw_jsonl(src: Path, excluded: set[str]) -> bytes:
    """Return JSONL bytes with lines whose top-level key is in `excluded` dropped."""
    out = io.BytesIO()
    with src.open("rb") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                out.write(line)
                continue
            if isinstance(obj, dict) and any(k in excluded for k in obj):
                continue
            out.write(line)
    return out.getvalue()


def _walk_run(run_dir: Path, excluded: set[str]):
    """Yield (relative_path_under_run, payload_bytes) for every file to upload.

    Filters UPLOAD.json + RAW_SUBMISSIONS.jsonl in-memory; skips per-capsule
    subdirs whose name is in `excluded`.
    """
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        # Skip excluded capsule subdirs (any depth under capsule-XXX/)
        if rel.parts and rel.parts[0] in excluded:
            continue
        if path.name.endswith("_UPLOAD.json") and excluded:
            upload = json.loads(path.read_text())
            filtered = _filter_upload(upload, excluded)
            yield rel, json.dumps(filtered, indent=2).encode("utf-8")
        elif path.name.endswith("_RAW_SUBMISSIONS.jsonl") and excluded:
            yield rel, _filter_raw_jsonl(path, excluded)
        else:
            yield rel, path.read_bytes()


def upload_split(
    container,
    src: Path,
    prefix: str,
    excluded: set[str],
    label: str,
    dry_run: bool,
) -> None:
    print(f"\n=== {label}: {src} → {CONTAINER_NAME}/{prefix}/ ===")
    runs = sorted(p for p in src.iterdir() if p.is_dir())
    n_runs = len(runs)
    total_files = 0
    total_bytes = 0
    for i, run in enumerate(runs, 1):
        files_in_run = 0
        bytes_in_run = 0
        for rel, payload in _walk_run(run, excluded):
            blob_name = f"{prefix}/{run.name}/{rel.as_posix()}"
            if not dry_run:
                container.upload_blob(name=blob_name, data=payload, overwrite=True)
            files_in_run += 1
            bytes_in_run += len(payload)
        total_files += files_in_run
        total_bytes += bytes_in_run
        print(
            f"  [{i}/{n_runs}] {run.name}: {files_in_run} files, "
            f"{bytes_in_run / 1024 / 1024:.1f} MB"
        )
    print(
        f"\n{label}: {n_runs} runs, {total_files} files, "
        f"{total_bytes / 1024 / 1024:.1f} MB total"
    )


def main() -> None:
    dry = "--dry-run" in sys.argv

    client = _client()
    container = client.get_container_client(CONTAINER_NAME)
    if not dry:
        if not container.exists():
            container.create_container()
            print(f"Created container: {CONTAINER_NAME}")
        else:
            print(f"Container exists: {CONTAINER_NAME}")

    upload_split(container, SRC_MAIN, PREFIX_MAIN, EXCLUDED, "mainline (42 → 39)", dry)
    upload_split(container, SRC_OOD, PREFIX_OOD, set(), "ood (19, no filter)", dry)

    if not dry:
        print(
            f"\nDone. Browse: https://portal.azure.com → "
            f"{AZURE_STORAGE_ACCOUNT_NAME} → {CONTAINER_NAME}/"
        )


if __name__ == "__main__":
    main()

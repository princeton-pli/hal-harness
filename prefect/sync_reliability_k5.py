"""Bidirectional sync between Azure `hal-harness-results/reliability-k5-2026-04-28/`
and a local working tree, used when re-running post-hoc post-processing
(add_confidence.py at a new prompt version, regrade.py with a new rubric, etc.).

  python sync_reliability_k5.py download
      Mirror Azure → local. Skips files that already exist locally.

  python sync_reliability_k5.py upload-uploadjsons
      Push only the per-run *_UPLOAD.json files back to Azure (one
      per run dir = ~25 blobs). Use after running add_confidence.py
      so confidence updates land in Azure without round-tripping the
      full ~400 MB tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

CONTAINER = "hal-harness-results"
PREFIX = "reliability-k5-2026-04-28"
LOCAL_ROOT = Path("/home/kang/Documents/hal/bak-reliability-k5")
ACCOUNT_URL = "https://halharness.blob.core.windows.net"


def _client() -> BlobServiceClient:
    return BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=os.environ["AZURE_STORAGE_ACCOUNT_KEY"],
    )


def download() -> None:
    """Mirror Azure prefix → LOCAL_ROOT, skipping files already present."""
    container = _client().get_container_client(CONTAINER)
    blobs = list(container.list_blobs(name_starts_with=f"{PREFIX}/"))
    print(f"Found {len(blobs)} blobs under {PREFIX}/")
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    skipped = downloaded = total_bytes = 0
    for i, b in enumerate(blobs, 1):
        rel = b.name[len(f"{PREFIX}/"):]
        dest = LOCAL_ROOT / rel
        if dest.exists() and dest.stat().st_size == b.size:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            f.write(container.get_blob_client(b.name).download_blob().readall())
        downloaded += 1
        total_bytes += b.size
        if downloaded % 100 == 0:
            print(f"  {i}/{len(blobs)} downloaded={downloaded} skipped={skipped}")

    print(
        f"\nDone. downloaded={downloaded} skipped={skipped} "
        f"total={total_bytes / 1024 / 1024:.1f} MB"
    )


def upload_upload_jsons() -> None:
    """Push every *_UPLOAD.json under LOCAL_ROOT back to Azure."""
    container = _client().get_container_client(CONTAINER)
    files = sorted(LOCAL_ROOT.rglob("*_UPLOAD.json"))
    print(f"Uploading {len(files)} UPLOAD.json files → {CONTAINER}/{PREFIX}/")
    for i, f in enumerate(files, 1):
        rel = f.relative_to(LOCAL_ROOT)
        blob_name = f"{PREFIX}/{rel.as_posix()}"
        container.upload_blob(name=blob_name, data=f.read_bytes(), overwrite=True)
        print(f"  [{i}/{len(files)}] {rel}")
    print("Done.")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("download", "upload-uploadjsons"):
        sys.exit("Usage: sync_reliability_k5.py {download|upload-uploadjsons}")
    if sys.argv[1] == "download":
        download()
    else:
        upload_upload_jsons()


if __name__ == "__main__":
    main()

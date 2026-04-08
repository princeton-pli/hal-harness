"""
One-shot uploader: tar each corebench capsule under
hal/benchmarks/corebench/capsules/<capsule_id>/ and upload it to blob storage as
{CAPSULES_BLOB_PREFIX}/<capsule_id>.tar.gz.

Idempotent: blobs that already exist are skipped (re-upload with --force).

Usage:
    python prefect/upload_capsules.py            # upload all local capsules
    python prefect/upload_capsules.py --force    # re-upload, overwriting blobs
    python prefect/upload_capsules.py capsule-5507257 capsule-3449234   # specific
"""

import argparse
import io
import sys
import tarfile
from pathlib import Path

from config import (
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_CONTAINER_NAME,
    CAPSULES_BLOB_PREFIX,
)
from storage import _storage_client, ensure_container

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPSULES_DIR = REPO_ROOT / "hal" / "benchmarks" / "corebench" / "capsules"


def _tar_capsule(capsule_dir: Path) -> bytes:
    """Tar+gzip a capsule directory in-memory. Archive root is the capsule_id dir."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(capsule_dir, arcname=capsule_dir.name)
    return buf.getvalue()


def upload_capsule(capsule_id: str, force: bool) -> None:
    capsule_dir = CAPSULES_DIR / capsule_id
    if not capsule_dir.is_dir():
        print(f"SKIP {capsule_id}: not found at {capsule_dir}", file=sys.stderr)
        return

    blob_name = f"{CAPSULES_BLOB_PREFIX}/{capsule_id}.tar.gz"
    client = _storage_client()
    blob = client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME, blob=blob_name
    )

    if not force and blob.exists():
        print(f"skip {capsule_id} (already in blob storage)")
        return

    print(f"taring {capsule_id}...", end=" ", flush=True)
    data = _tar_capsule(capsule_dir)
    print(f"{len(data) / 1024 / 1024:.1f} MB", end=" ", flush=True)
    blob.upload_blob(data, overwrite=True)
    print("uploaded")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capsule_ids", nargs="*", help="Specific capsule IDs (default: all)")
    p.add_argument("--force", action="store_true", help="Re-upload even if blob exists")
    args = p.parse_args()

    if not AZURE_STORAGE_ACCOUNT_KEY:
        sys.exit("AZURE_STORAGE_ACCOUNT_KEY is required")

    if not CAPSULES_DIR.is_dir():
        sys.exit(f"capsules dir not found: {CAPSULES_DIR}")

    ensure_container()

    if args.capsule_ids:
        capsule_ids = args.capsule_ids
    else:
        capsule_ids = sorted(p.name for p in CAPSULES_DIR.iterdir() if p.is_dir())

    print(f"Uploading {len(capsule_ids)} capsule(s) to {CAPSULES_BLOB_PREFIX}/")
    for cid in capsule_ids:
        upload_capsule(cid, args.force)
    print("Done.")


if __name__ == "__main__":
    main()

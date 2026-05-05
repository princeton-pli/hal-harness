"""
Upload compiled results to Azure blob storage container `hal-harness-results`.

Each upload goes under a named prefix (e.g. `reliability-k5-2026-04-28`) so
multiple compiled run sets coexist without collision.

Usage:
    cd prefect && python upload_compiled.py <local_dir> <prefix>

Example:
    cd prefect && python upload_compiled.py ../results/corebench_hard reliability-k5-2026-04-28
"""

import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from config import (
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_ACCOUNT_URL,
)

load_dotenv()

CONTAINER_NAME = "hal-harness-results"


def _client() -> BlobServiceClient:
    if not AZURE_STORAGE_ACCOUNT_KEY:
        raise RuntimeError(
            "AZURE_STORAGE_ACCOUNT_KEY is required (set via .env or env var)."
        )
    return BlobServiceClient(
        account_url=AZURE_STORAGE_ACCOUNT_URL,
        credential=AZURE_STORAGE_ACCOUNT_KEY,
    )


def ensure_container() -> None:
    """Create the container if it doesn't exist (idempotent)."""
    client = _client()
    container = client.get_container_client(CONTAINER_NAME)
    if not container.exists():
        container.create_container()
        print(f"Created container: {CONTAINER_NAME}")
    else:
        print(f"Container exists: {CONTAINER_NAME}")


def upload_dir(local_dir: Path, prefix: str, dry_run: bool = False) -> None:
    """Upload all files under local_dir to {CONTAINER_NAME}/{prefix}/<relative_path>."""
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)

    print(f"{'[DRY RUN] ' if dry_run else ''}"
          f"{'Would upload' if dry_run else 'Uploading'} {len(files)} files "
          f"({total_bytes / 1024 / 1024:.1f} MB) "
          f"from {local_dir} → {CONTAINER_NAME}/{prefix}/")

    if dry_run:
        # Show first few and last few file paths
        sample = files[:5] + (["..."] if len(files) > 10 else []) + files[-5:] if len(files) > 10 else files
        for path in sample:
            if path == "...":
                print(f"  ...")
                continue
            rel = path.relative_to(local_dir)
            print(f"  {prefix}/{rel.as_posix()} ({path.stat().st_size:,} bytes)")
        return

    client = _client()
    container = client.get_container_client(CONTAINER_NAME)

    for i, path in enumerate(files, 1):
        rel = path.relative_to(local_dir)
        blob_name = f"{prefix}/{rel.as_posix()}"
        with open(path, "rb") as f:
            container.upload_blob(name=blob_name, data=f, overwrite=True)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} uploaded")

    print(f"Done. Browse at: https://portal.azure.com → {AZURE_STORAGE_ACCOUNT_NAME} → {CONTAINER_NAME}/{prefix}/")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv
    if len(args) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <local_dir> <prefix> [--dry-run]")

    local_dir = Path(args[0]).resolve()
    prefix = args[1].strip("/")

    if not local_dir.is_dir():
        sys.exit(f"Not a directory: {local_dir}")
    if not prefix:
        sys.exit("Prefix cannot be empty")

    if not dry_run:
        ensure_container()
    upload_dir(local_dir, prefix, dry_run=dry_run)


if __name__ == "__main__":
    main()

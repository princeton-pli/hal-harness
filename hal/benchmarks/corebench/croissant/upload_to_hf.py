"""Upload CORE-bench v2 datasets to HuggingFace.

For each split (mainline, ood):
  - Push the task manifest (renamed to core_test.json on HF).
  - Push the Croissant metadata file.
  - Mirror the per-task capsule tarballs from Azure blob storage into
    the HF dataset's `capsules/` directory.

Uses streaming temp files so peak local disk usage is ~the size of the
largest single tarball.

Run:
    cd hal/benchmarks/corebench/croissant && python upload_to_hf.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
AZURE_KEY = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

ORG = "agent-evals"
REPOS = {
    "mainline": "core-bench-v2-mainline",
    "ood": "core-bench-v2-ood",
}

HERE = Path(__file__).resolve().parent
COREBENCH_DIR = HERE.parent
MANIFESTS = {
    "mainline": COREBENCH_DIR / "core_test.json.bak.main42",
    "ood": COREBENCH_DIR / "core_test.json",
}
CROISSANTS = {
    "mainline": HERE / "croissant_mainline.json",
    "ood": HERE / "croissant_ood.json",
}

CAPSULES_CONTAINER = "corebench-capsules"
AZURE_ACCOUNT_URL = "https://halharness.blob.core.windows.net"


def upload_split(api: HfApi, blob_client: BlobServiceClient, split: str) -> None:
    repo_id = f"{ORG}/{REPOS[split]}"
    print(f"\n=== {repo_id} ===")

    # 1. Manifest — uploaded as core_test.json on HF for predictable path.
    manifest_src = MANIFESTS[split]
    print(f"[manifest] {manifest_src} -> core_test.json")
    api.upload_file(
        path_or_fileobj=str(manifest_src),
        path_in_repo="core_test.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Upload task manifest",
    )

    # 2. Croissant metadata.
    crois_src = CROISSANTS[split]
    print(f"[croissant] {crois_src} -> croissant.json")
    api.upload_file(
        path_or_fileobj=str(crois_src),
        path_in_repo="croissant.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="Upload Croissant metadata",
    )

    # 3. Per-capsule tarballs — stream from Azure → temp file → HF.
    container = blob_client.get_container_client(CAPSULES_CONTAINER)
    capsule_ids = sorted({t["capsule_id"] for t in json.loads(manifest_src.read_text())})
    existing = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN))

    for i, cid in enumerate(capsule_ids, 1):
        blob_name = f"{cid}.tar.gz"
        repo_path = f"capsules/{blob_name}"
        if repo_path in existing:
            print(f"[capsules {i}/{len(capsule_ids)}] {blob_name} already on HF — skip")
            continue

        print(f"[capsules {i}/{len(capsule_ids)}] {blob_name} downloading...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            try:
                downloader = container.get_blob_client(blob_name).download_blob()
                downloader.readinto(tmp)
                tmp.flush()
                size_mb = os.path.getsize(tmp.name) / 1024 / 1024
                print(f"    downloaded {size_mb:.1f} MB; uploading to HF...", flush=True)
                api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    commit_message=f"Add capsule {cid}",
                )
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass


def main() -> None:
    api = HfApi()
    blob_client = BlobServiceClient(account_url=AZURE_ACCOUNT_URL, credential=AZURE_KEY)
    splits = sys.argv[1:] or list(REPOS)
    for split in splits:
        if split not in REPOS:
            sys.exit(f"unknown split: {split} (choose from {list(REPOS)})")
        upload_split(api, blob_client, split)
    print("\nDone.")


if __name__ == "__main__":
    main()

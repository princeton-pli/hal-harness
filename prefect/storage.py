"""
Azure Blob Storage helpers — code distribution and result collection for eval tasks.

All functions are pure-Python with no Prefect dependency.
"""

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from azure.identity import AzureCliCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContainerSasPermissions,
    generate_blob_sas,
    generate_container_sas,
)

from config import (
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_ACCOUNT_URL,
    AZURE_STORAGE_CONTAINER_NAME,
    SAS_EXPIRY_HOURS,
)

# Directories to exclude from the repo zip (relative to repo root, matched as path prefixes)
_ZIP_EXCLUDES = {
    ".git",
    ".venv",
    "prefect",
    "results",
    "__pycache__",
    ".mypy_cache",
    "hal/benchmarks/corebench",
    "hal/benchmarks/USACO",
    "hal/benchmarks/appworld",
    "hal/benchmarks/scienceagentbench",
    "hal/benchmarks/taubench/taubench_setup.sh",
}


def _storage_client() -> BlobServiceClient:
    """BlobServiceClient authenticated via account key (preferred) or AzureCliCredential."""
    credential = (
        AZURE_STORAGE_ACCOUNT_KEY if AZURE_STORAGE_ACCOUNT_KEY else AzureCliCredential()
    )
    return BlobServiceClient(
        account_url=AZURE_STORAGE_ACCOUNT_URL,
        credential=credential,
    )


def ensure_container() -> None:
    """Create the blob container if it does not exist (idempotent)."""
    client = _storage_client()
    container = client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    if not container.exists():
        container.create_container()
        print(f"Created blob container: {AZURE_STORAGE_CONTAINER_NAME}")


def _zip_repo(repo_root: Path) -> bytes:
    """Zip the repo root, excluding large/irrelevant directories."""
    buf = io.BytesIO()
    with zipfile.ZipFile(
        buf, mode="w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False
    ) as zf:
        for path in sorted(repo_root.rglob("*")):
            rel = path.relative_to(repo_root)
            if any(str(rel).startswith(ex) for ex in _ZIP_EXCLUDES):
                continue
            if path.is_file():
                zf.write(path, rel)
    return buf.getvalue()


def upload_code_zip(job_id: str) -> str:
    """
    Zip the repo root, upload to {job_id}/code/hal-harness.zip, return a
    48h read-only SAS URL. Also calls ensure_container() so the container
    exists before SAS generation.
    """
    if not AZURE_STORAGE_ACCOUNT_KEY:
        raise RuntimeError(
            "AZURE_STORAGE_ACCOUNT_KEY is required for SAS generation. "
            "Set it via the environment variable."
        )

    ensure_container()

    print("Beginning ZIP...")
    repo_root = Path(__file__).resolve().parent.parent
    raw_bytes = sum(
        p.stat().st_size
        for p in repo_root.rglob("*")
        if p.is_file()
        and not any(
            str(p.relative_to(repo_root)).startswith(ex) for ex in _ZIP_EXCLUDES
        )
    )
    print(f"Repo size before zip: {raw_bytes / 1024 / 1024:.1f} MB")

    zip_bytes = _zip_repo(repo_root)
    blob_name = f"{job_id}/code/hal-harness.zip"
    print(f"Code zip size: {len(zip_bytes) / 1024:.1f} KB | blob={blob_name}")

    client = _storage_client()
    blob = client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME, blob=blob_name
    )
    blob.upload_blob(zip_bytes, overwrite=True)

    expiry = datetime.now(timezone.utc) + timedelta(hours=SAS_EXPIRY_HOURS)
    sas_token = generate_blob_sas(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        container_name=AZURE_STORAGE_CONTAINER_NAME,
        blob_name=blob_name,
        account_key=AZURE_STORAGE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    sas_url = f"{AZURE_STORAGE_ACCOUNT_URL}/{AZURE_STORAGE_CONTAINER_NAME}/{blob_name}?{sas_token}"
    print(f"Code zip uploaded | sas_url_prefix={sas_url[:80]}...")
    return sas_url


def result_container_sas() -> str:
    """
    Return a 48h container-level write+list SAS URL for Azure Batch OutputFiles
    to upload results into.
    """
    if not AZURE_STORAGE_ACCOUNT_KEY:
        raise RuntimeError(
            "AZURE_STORAGE_ACCOUNT_KEY is required for SAS generation. "
            "Set it via the environment variable."
        )

    expiry = datetime.now(timezone.utc) + timedelta(hours=SAS_EXPIRY_HOURS)
    sas_token = generate_container_sas(
        account_name=AZURE_STORAGE_ACCOUNT_NAME,
        container_name=AZURE_STORAGE_CONTAINER_NAME,
        account_key=AZURE_STORAGE_ACCOUNT_KEY,
        permission=ContainerSasPermissions(write=True, list=True),
        expiry=expiry,
    )
    return f"{AZURE_STORAGE_ACCOUNT_URL}/{AZURE_STORAGE_CONTAINER_NAME}?{sas_token}"


def upload_task_metadata(spec, azure_task_id: str, submitted_at: str) -> None:
    """Upload metadata.json for a task to {job_id}/logs/{azure_task_id}/metadata.json."""
    import dataclasses

    metadata = {
        **dataclasses.asdict(spec),
        "azure_task_id": azure_task_id,
        "submitted_at": submitted_at,
    }
    # SAS URLs are long and not useful in metadata
    metadata.pop("code_sas_url", None)
    metadata.pop("result_sas_url", None)

    client = _storage_client()
    blob = client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME,
        blob=f"{spec.job_id}/logs/{azure_task_id}/metadata.json",
    )
    blob.upload_blob(json.dumps(metadata, indent=2).encode(), overwrite=True)


_RESULTS_DIR = Path(__file__).resolve().parent.parent / ".prefect_results"


def save_task_results(job_id: str, azure_task_id: str, result: dict) -> Path:
    """Write result dict to .prefect_results/{job_id}/{azure_task_id}/result.json."""
    out = _RESULTS_DIR / job_id / azure_task_id / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out


def download_task_results(job_id: str, azure_task_id: str) -> dict:
    """
    Download and parse {job_id}/results/{azure_task_id}/*_UPLOAD.json.
    Returns the parsed result dict.
    """
    client = _storage_client()
    container = client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
    prefix = f"{job_id}/results/{azure_task_id}/"

    for blob in container.list_blobs(name_starts_with=prefix):
        if blob.name.endswith("_UPLOAD.json"):
            data = container.get_blob_client(blob.name).download_blob().readall()
            return json.loads(data)

    raise FileNotFoundError(
        f"No _UPLOAD.json found under {AZURE_STORAGE_CONTAINER_NAME}/{prefix}"
    )

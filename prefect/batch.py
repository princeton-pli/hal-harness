"""
Azure Batch helpers — submit, poll, and stream stdout for eval tasks.

All public functions accept an EvalSpec; internals are prefixed with _.
"""

import datetime
import time
import uuid

import azure.batch.models as batch_models
from azure.batch import BatchServiceClient

# FIXME: swap AzureCliCredential for DefaultAzureCredential in production
# (requires service principal with AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET)
from azure.identity import AzureCliCredential

from config import (
    AZURE_BATCH_ACCOUNT_URL,
    AZURE_BATCH_POOL_ID,
    POLL_INTERVAL_SECONDS,
    TASK_ENV_VARS,
    EvalSpec,
)
from storage import download_task_results


def _batch_client() -> BatchServiceClient:
    """Authenticated BatchServiceClient. Fresh per call — not thread-safe to share."""
    from msrest.authentication import BasicTokenAuthentication

    token = AzureCliCredential().get_token("https://batch.core.windows.net/.default")
    return BatchServiceClient(
        credentials=BasicTokenAuthentication({"access_token": token.token}),
        batch_url=AZURE_BATCH_ACCOUNT_URL,
    )


def create_batch_job(job_id: str) -> None:
    """Create a new Azure Batch job for this flow run."""
    client = _batch_client()
    client.job.add(
        batch_models.JobAddParameter(
            id=job_id,
            pool_info=batch_models.PoolInformation(pool_id=AZURE_BATCH_POOL_ID),
            on_all_tasks_complete=batch_models.OnAllTasksComplete.no_action,
        )
    )
    print(f"Created Azure Batch job | job_id={job_id}")


def terminate_batch_job(job_id: str) -> None:
    """Terminate the Azure Batch job for this flow run."""
    client = _batch_client()
    client.job.terminate(job_id)
    print(f"Terminated Azure Batch job | job_id={job_id}")


def _wait_for_pool_steady() -> None:
    """Block until the pool is no longer resizing."""
    client = _batch_client()
    while True:
        pool = client.pool.get(AZURE_BATCH_POOL_ID)
        if pool.allocation_state == batch_models.AllocationState.steady:
            return
        print(f"Waiting for pool | allocation_state={pool.allocation_state}")
        time.sleep(POLL_INTERVAL_SECONDS)


def resize_pool(n_nodes: int) -> None:
    """Resize the pool to n_nodes dedicated nodes. Pass 0 to scale down."""
    _wait_for_pool_steady()
    client = _batch_client()
    client.pool.resize(
        AZURE_BATCH_POOL_ID,
        batch_models.PoolResizeParameter(
            target_dedicated_nodes=n_nodes,
            target_low_priority_nodes=0,
        ),
    )
    print(
        f"Pool resize requested | pool={AZURE_BATCH_POOL_ID} dedicated_nodes={n_nodes}"
    )


def _build_command_line(spec: EvalSpec) -> str:
    """Build the command line run on the Azure Batch node.

    Azure Batch downloads hal-harness.zip as a ResourceFile, placing it in the
    task working directory. We unzip it, then delegate to azure_entrypoint.sh.
    """
    return (
        "/bin/bash -c \""
        "python3 -m zipfile -e hal-harness.zip hal-harness && "
        "bash hal-harness/azure_entrypoint.sh"
        f" '{spec.agent}'"
        f" '{spec.agent_function}'"
        f" '{spec.agent_dir}'"
        f" '{spec.benchmark}'"
        f" '{spec.task_id}'"
        f" '{spec.model}'"
        "\""
    )


def _submit_batch_task(spec: EvalSpec) -> str:
    """Submit one eval task to Azure Batch. Returns the Azure task ID."""
    client = _batch_client()
    suffix = uuid.uuid4().hex[:8]
    prefix = f"{spec.model}-{spec.agent}-{spec.benchmark}-{spec.task_id}"
    if len(prefix) + 1 + len(suffix) > 64:
        prefix = prefix[: 64 - 1 - len(suffix)]
    azure_task_id = f"{prefix}-{suffix}"

    batch_task = batch_models.TaskAddParameter(
        id=azure_task_id,
        command_line=_build_command_line(spec),
        environment_settings=[
            batch_models.EnvironmentSetting(name=k, value=v)
            for k, v in TASK_ENV_VARS.items()
        ],
        resource_files=[
            batch_models.ResourceFile(
                http_url=spec.code_sas_url,
                file_path="hal-harness.zip",
            )
        ],
        output_files=[
            batch_models.OutputFile(
                file_pattern="hal-harness/results/**/*_UPLOAD.json",
                destination=batch_models.OutputFileDestination(
                    container=batch_models.OutputFileBlobContainerDestination(
                        container_url=spec.result_sas_url,
                        path=f"{spec.job_id}/results/{azure_task_id}",
                    )
                ),
                upload_options=batch_models.OutputFileUploadOptions(
                    upload_condition=batch_models.OutputFileUploadCondition.task_success,
                ),
            ),
            batch_models.OutputFile(
                file_pattern="hal-harness/results/**/*_RAW_SUBMISSIONS.jsonl",
                destination=batch_models.OutputFileDestination(
                    container=batch_models.OutputFileBlobContainerDestination(
                        container_url=spec.result_sas_url,
                        path=f"{spec.job_id}/results/{azure_task_id}",
                    )
                ),
                upload_options=batch_models.OutputFileUploadOptions(
                    upload_condition=batch_models.OutputFileUploadCondition.task_success,
                ),
            ),
        ],
        constraints=batch_models.TaskConstraints(
            max_task_retry_count=0,  # Prefect owns retries
            retention_time=datetime.timedelta(days=365),
        ),
    )
    client.task.add(job_id=spec.job_id, task=batch_task)
    return azure_task_id


def _read_new_stdout(
    client: BatchServiceClient, job_id: str, azure_task_id: str, bytes_read: int
) -> int:
    """Fetch any new bytes from stdout.txt since bytes_read. Returns updated bytes_read."""
    try:
        stream = client.file.get_from_task(
            job_id,
            azure_task_id,
            "stdout.txt",
            file_get_from_task_options=batch_models.FileGetFromTaskOptions(
                ocp_range=f"bytes={bytes_read}-"
            ),
        )
        chunk = b"".join(stream).decode("utf-8", errors="replace")
        if chunk:
            print(chunk, end="", flush=True)
            bytes_read += len(chunk.encode("utf-8"))
    except batch_models.BatchErrorException as e:
        # Swallow expected "not ready" states: file doesn't exist yet, or task is still
        # queued/preparing and files aren't accessible. Propagate everything else.
        if e.error is None or e.error.code not in (
            "FileNotFound",
            "OperationInvalidForCurrentState",
        ):
            raise
    return bytes_read


def _poll_batch_task(spec: EvalSpec, azure_task_id: str) -> None:
    """Block until the Azure Batch task completes, streaming stdout incrementally.
    Raises on non-zero exit code."""
    client = _batch_client()
    bytes_read = 0

    while True:
        t = client.task.get(spec.job_id, azure_task_id)
        bytes_read = _read_new_stdout(client, spec.job_id, azure_task_id, bytes_read)

        if t.state == batch_models.TaskState.completed:
            exit_code = t.execution_info.exit_code
            if exit_code != 0:
                # Drain any remaining stdout then print stderr for diagnosis
                _read_new_stdout(client, spec.job_id, azure_task_id, bytes_read)
                try:
                    stderr = b"".join(
                        client.file.get_from_task(spec.job_id, azure_task_id, "stderr.txt")
                    ).decode("utf-8", errors="replace")
                    if stderr:
                        print(f"--- stderr ---\n{stderr}--- end stderr ---")
                except Exception:
                    pass
                raise RuntimeError(
                    f"Azure Batch task {azure_task_id} failed "
                    f"(exit_code={exit_code}): {t.execution_info.failure_info}"
                )
            return

        print(f"Polling | azure_task_id={azure_task_id} state={t.state}")
        time.sleep(POLL_INTERVAL_SECONDS)


def run_eval_on_batch(spec: EvalSpec) -> dict:
    """Submit to Azure Batch and block until completion."""
    azure_task_id = _submit_batch_task(spec)
    print(f"Submitted | azure_task_id={azure_task_id}")
    _poll_batch_task(spec, azure_task_id)
    print(f"Completed | azure_task_id={azure_task_id}")
    return download_task_results(spec.job_id, azure_task_id)

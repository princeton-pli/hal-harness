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
    EvalSpec,
)


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


def resize_pool(n_nodes: int) -> None:
    """Resize the pool to n_nodes low-priority (spot) nodes. Pass 0 to scale down."""
    client = _batch_client()
    client.pool.resize(
        AZURE_BATCH_POOL_ID,
        # FIXME: switch to dedicated nodes for production (spot nodes can be evicted mid-task)
        batch_models.PoolResizeParameter(
            target_dedicated_nodes=0,
            target_low_priority_nodes=n_nodes,
        ),
    )
    print(f"Pool resize requested | pool={AZURE_BATCH_POOL_ID} low_priority_nodes={n_nodes}")


# TODO: replace inline command with Azure Batch resource files for real eval code.
# Upload a zip of hal-harness to Blob Storage, generate a SAS URL, and pass it as a
# ResourceFile on TaskAddParameter. Azure Batch will download + unzip it into the node's
# working directory before the task command runs — no base64 embedding needed.
# See: https://learn.microsoft.com/azure/batch/resource-files
def _build_command_line(spec: EvalSpec) -> str:
    """Build the command line run on the Azure Batch node."""
    py = (
        f"import time,random; "
        f"print('agent={spec.agent} benchmark={spec.benchmark} task={spec.task_id} model={spec.model}'); "
        f"time.sleep(random.uniform(3,9)); "
        f"print('done')"
    )
    return f'/bin/bash -c "python3 -c \\"{py}\\""'


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
    return {
        "run_id": azure_task_id,
        "metadata": {"info": "TODO: complete me"},
        "agent_response": {"info": "TODO: complete me"},
        "scoring": {"info": "TODO: complete me"},
    }

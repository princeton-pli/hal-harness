"""
Azure Batch helpers — submit, poll, and stream stdout for eval tasks.

All public functions accept an EvalSpec; internals are prefixed with _.
"""

import datetime
import re
import threading
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
from storage import download_task_results, save_task_results, upload_task_metadata


# --------------------------------------------------------------------------
# Token cache
# --------------------------------------------------------------------------
# AzureCliCredential.get_token() shells out to `az account get-access-token`
# on EVERY call — it does not cache tokens in-process. Under concurrent load
# (hundreds of Prefect tasks hitting _batch_client() at once) this spawns
# hundreds of parallel `az` subprocess invocations racing on ~/.azure/ file
# locks, and many of them fail.
#
# We wrap the credential in a manual cache: one token fetched under a lock,
# reused until it's within 5 minutes of expiry, then refreshed under the same
# lock. 504 concurrent threads share exactly ONE `az` invocation per refresh.
# --------------------------------------------------------------------------
_BATCH_SCOPE = "https://batch.core.windows.net/.default"
_credential = AzureCliCredential()
_token_lock = threading.Lock()
_cached_token = None  # azure.core.credentials.AccessToken | None


def _get_batch_token() -> str:
    """Return a cached Batch access token, refreshing under lock if needed."""
    global _cached_token
    now = time.time()
    # Fast path: check without the lock. AccessToken.expires_on is epoch seconds.
    if _cached_token is not None and _cached_token.expires_on - now > 300:
        return _cached_token.token

    with _token_lock:
        # Another thread may have refreshed while we waited on the lock.
        if _cached_token is None or _cached_token.expires_on - now <= 300:
            _cached_token = _credential.get_token(_BATCH_SCOPE)
        return _cached_token.token


def _batch_client() -> BatchServiceClient:
    """Authenticated BatchServiceClient using the shared, lock-protected token cache."""
    from msrest.authentication import BasicTokenAuthentication

    return BatchServiceClient(
        credentials=BasicTokenAuthentication({"access_token": _get_batch_token()}),
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
        '/bin/bash -c "'
        "python3 -m zipfile -e hal-harness.zip hal-harness && "
        "bash hal-harness/azure_entrypoint.sh"
        f" '{spec.agent}'"
        f" '{spec.agent_function}'"
        f" '{spec.agent_dir}'"
        f" '{spec.benchmark}'"
        f" '{spec.task_id}'"
        f" '{spec.model}'"
        '"'
    )


def _submit_batch_task(spec: EvalSpec) -> str:
    """Submit one eval task to Azure Batch. Returns the Azure task ID."""
    client = _batch_client()
    suffix = uuid.uuid4().hex[:8]
    # Slug the agent_args into the task ID so ablation cells are visually
    # distinguishable in blob storage and the Azure portal. Keys are
    # abbreviated to keep task IDs under Azure Batch's 64-char limit.
    _KEY_ABBREV = {
        "reasoning_effort": "re",
        "max_threads": "mt",
    }
    args_segment = (
        "-" + "-".join(f"{_KEY_ABBREV.get(k, k)}{v}" for k, v in spec.agent_args)
        if spec.agent_args
        else ""
    )
    # Omit spec.agent and spec.benchmark from the Azure Batch task ID to stay
    # under the 64-char limit. Those values are still present in the nested
    # blob storage paths (results/{task_id}/{benchmark}/...) and in
    # metadata.json, so no info is lost.
    raw_prefix = f"{spec.model}-{spec.task_id}{args_segment}"
    # Azure Batch task IDs allow only [a-zA-Z0-9-_]; replace other chars with '-'
    prefix = re.sub(r"[^a-zA-Z0-9\-_]", "-", raw_prefix)
    if len(prefix) + 1 + len(suffix) > 64:
        prefix = prefix[: 64 - 1 - len(suffix)]
    azure_task_id = f"{prefix}-{suffix}"

    batch_task = batch_models.TaskAddParameter(
        id=azure_task_id,
        command_line=_build_command_line(spec),
        environment_settings=[
            batch_models.EnvironmentSetting(name=k, value=v)
            for k, v in TASK_ENV_VARS.items()
        ]
        # Per-task agent kwargs are transported as HAL_AGENT_ARG_<key>=<value>
        # env vars. The entrypoint iterates them and forwards each as
        # `-A <key>=<value>` to hal-eval.
        + [
            batch_models.EnvironmentSetting(
                name=f"HAL_AGENT_ARG_{k}", value=str(v)
            )
            for k, v in spec.agent_args
        ],
        resource_files=[
            batch_models.ResourceFile(
                http_url=spec.code_sas_url,
                file_path="hal-harness.zip",
            ),
            *(
                [
                    batch_models.ResourceFile(
                        http_url=spec.capsule_sas_url,
                        file_path=f"{spec.task_id}.tar.gz",
                    )
                ]
                if spec.capsule_sas_url
                else []
            ),
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
            # hal-eval's verbose.log captures the full traceback when the eval
            # itself errors. Upload on any completion (including failure) so we
            # can debug without re-running.
            batch_models.OutputFile(
                file_pattern="hal-harness/results/**/*_verbose.log",
                destination=batch_models.OutputFileDestination(
                    container=batch_models.OutputFileBlobContainerDestination(
                        container_url=spec.result_sas_url,
                        path=f"{spec.job_id}/logs/{azure_task_id}",
                    )
                ),
                upload_options=batch_models.OutputFileUploadOptions(
                    upload_condition=batch_models.OutputFileUploadCondition.task_completion,
                ),
            ),
            *[
                batch_models.OutputFile(
                    file_pattern=pattern,
                    destination=batch_models.OutputFileDestination(
                        container=batch_models.OutputFileBlobContainerDestination(
                            container_url=spec.result_sas_url,
                            path=f"{spec.job_id}/logs/{azure_task_id}/{subpath}",
                        )
                    ),
                    upload_options=batch_models.OutputFileUploadOptions(
                        upload_condition=batch_models.OutputFileUploadCondition.task_completion,
                    ),
                )
                for pattern, subpath in spec.extra_output_files
            ],
            batch_models.OutputFile(
                file_pattern="../std*.txt",
                destination=batch_models.OutputFileDestination(
                    container=batch_models.OutputFileBlobContainerDestination(
                        container_url=spec.result_sas_url,
                        path=f"{spec.job_id}/logs/{azure_task_id}",
                    )
                ),
                upload_options=batch_models.OutputFileUploadOptions(
                    upload_condition=batch_models.OutputFileUploadCondition.task_completion,
                ),
            ),
        ],
        constraints=batch_models.TaskConstraints(
            max_task_retry_count=0,  # Prefect owns retries
            retention_time=datetime.timedelta(minutes=15),
        ),
        # codex_agent's setup_codex_cli() runs `sudo apt-get install ...` to
        # install Node.js + the Codex CLI. The Batch default auto-user is
        # non-admin, so sudo prompts for a password and fails in a
        # non-interactive subprocess. Run the task as an admin auto-user so
        # sudo works passwordlessly.
        user_identity=batch_models.UserIdentity(
            auto_user=batch_models.AutoUserSpecification(
                scope=batch_models.AutoUserScope.task,
                elevation_level=batch_models.ElevationLevel.admin,
            )
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
    bytes_read = 0

    while True:
        # Refresh client each iteration — _batch_client() is cheap (hits the
        # in-memory token cache) and guarantees we never poll with an expired
        # token. The old pattern of creating one client at the top of the loop
        # caused AuthenticationFailed errors after ~1 hour.
        client = _batch_client()
        t = client.task.get(spec.job_id, azure_task_id)
        bytes_read = _read_new_stdout(client, spec.job_id, azure_task_id, bytes_read)

        if t.state == batch_models.TaskState.completed:
            exit_code = t.execution_info.exit_code
            if exit_code != 0:
                # Drain any remaining stdout then print stderr for diagnosis
                _read_new_stdout(client, spec.job_id, azure_task_id, bytes_read)
                try:
                    stderr = b"".join(
                        client.file.get_from_task(
                            spec.job_id, azure_task_id, "stderr.txt"
                        )
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


def _fetch_full_stdout(job_id: str, azure_task_id: str) -> str:
    """Fetch the complete stdout.txt for a finished task."""
    try:
        client = _batch_client()
        return b"".join(
            client.file.get_from_task(job_id, azure_task_id, "stdout.txt")
        ).decode("utf-8", errors="replace")
    except Exception:
        return ""


def run_eval_on_batch(spec: EvalSpec) -> dict:
    """Submit to Azure Batch and block until completion."""
    azure_task_id = _submit_batch_task(spec)
    submitted_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    upload_task_metadata(spec, azure_task_id, submitted_at)
    print(f"Submitted | azure_task_id={azure_task_id}")
    _poll_batch_task(spec, azure_task_id)
    print(f"Completed | azure_task_id={azure_task_id}")
    result = download_task_results(spec.job_id, azure_task_id)
    result["_stdout"] = _fetch_full_stdout(spec.job_id, azure_task_id)
    out = save_task_results(spec.job_id, azure_task_id, result)
    print(f"Results saved locally | path={out}")
    return result

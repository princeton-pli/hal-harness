# Plan: Wire Prefect Tasks to Azure Batch

## Context

This PoC lives in `prefect/run_evals.py`. It orchestrates a matrix of
`(agent × benchmark_task × model)` evaluations using Prefect. Currently the
work is simulated locally via `run_eval_locally()`. The goal of this plan is
to replace that function with real Azure Batch job submission.

### Domain vocabulary (from the hal-harness schema)

| Our term | Prefect term | Notes |
|:---------|:-------------|:------|
| Experiment | Flow Run | One execution of `evaluation_harness_poc` |
| Run (Agent × Benchmark × Environment) | Task Run | Each `run_eval_task` execution |
| Task (one benchmark problem) | *(work inside a task run)* | Visible in Prefect logs, not a separate node |
| ExperimentConfig | Deployment | Named, saved flow config with parameters |
| EnvironmentInstance | Work Pool / Worker | Infrastructure that executes the task run |

### Why not a native Prefect Azure Batch worker?

`prefect-azure` only supports **Azure Container Instances**. Azure Batch has
no native Prefect work pool type. A full custom `BaseWorker` subclass is the
"proper" long-term path but requires packaging and entry-point registration —
overkill for a PoC.

**Chosen approach:** keep Prefect running locally; have each `@task` implement
the same poll→submit→monitor loop that Prefect workers use internally for
ECS/Kubernetes. This is pragmatic, fully observable in the Prefect UI, and
easy to graduate to a proper worker later.

---

## Prerequisites

The following environment variables must be set:

```
AZURE_BATCH_ACCOUNT_URL   e.g. https://<account>.<region>.batch.azure.com
AZURE_BATCH_POOL_ID       an existing Ubuntu pool
AZURE_BATCH_JOB_ID        an existing job attached to the pool
AZURE_TENANT_ID
AZURE_CLIENT_ID
AZURE_CLIENT_SECRET
```

An Ubuntu pool and job must already exist in Azure Batch. The pool VMs need
Python 3 available at `/usr/bin/python3`. No Docker required — tasks run
bare-metal on the pool nodes.

---

## Implementation plan

### Step 1 — Restore Azure imports

At the top of `run_evals.py`, uncomment / add:

```python
import azure.batch.models as batch_models
from azure.batch import BatchServiceClient
from azure.identity import DefaultAzureCredential
```

Add `azure-batch` and `azure-identity` to `requirements.txt` (already there).

---

### Step 2 — Batch client factory

```python
def _batch_client() -> BatchServiceClient:
    credential = DefaultAzureCredential()
    return BatchServiceClient(
        credentials=credential,
        batch_url=os.environ["AZURE_BATCH_ACCOUNT_URL"],
    )
```

Create a fresh client per call — `BatchServiceClient` is not thread-safe and
Prefect runs tasks concurrently in a thread pool.

---

### Step 3 — Submit function

Replace `run_eval_locally()` with `run_eval_on_batch()`:

```python
def submit_batch_task(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> str:
    """Submit one eval task to Azure Batch. Returns the Azure task ID."""
    client = _batch_client()
    job_id = os.environ["AZURE_BATCH_JOB_ID"]
    azure_task_id = f"{model}--{agent_name}--{benchmark_name}--{task_id}--{uuid.uuid4().hex[:8]}"

    python_code = (
        "import time; "
        f"print('agent={agent_name} benchmark={benchmark_name} task={task_id} model={model}'); "
        "time.sleep(5); "
        "print('done')"
    )
    command_line = f"/bin/bash -c \"python3 -c '{python_code}'\""

    batch_task = batch_models.TaskAddParameter(
        id=azure_task_id,
        command_line=command_line,
        constraints=batch_models.TaskConstraints(
            max_task_retry_count=0,       # let Prefect own retries
            retention_time=datetime.timedelta(hours=1),
        ),
    )
    client.task.add(job_id=job_id, task=batch_task)
    return azure_task_id
```

Key points:
- `max_task_retry_count=0` — Azure Batch must not retry; Prefect owns the retry logic.
- Task ID encodes the full combo so it's traceable in the Azure portal.

---

### Step 4 — Poll function (mirrors Prefect worker pattern)

```python
POLL_INTERVAL_SECONDS = 15  # matches PREFECT_WORKER_QUERY_SECONDS default

def poll_batch_task(azure_task_id: str) -> int:
    """Block until the Azure Batch task reaches a terminal state.
    Returns the exit code. Raises on failure."""
    client = _batch_client()
    job_id = os.environ["AZURE_BATCH_JOB_ID"]

    while True:
        task = client.task.get(job_id, azure_task_id)
        state = task.state  # TaskState enum

        if state == batch_models.TaskState.completed:
            exit_code = task.execution_info.exit_code
            if exit_code != 0:
                failure = task.execution_info.failure_info
                raise RuntimeError(
                    f"Azure Batch task {azure_task_id} failed "
                    f"(exit_code={exit_code}): {failure}"
                )
            return exit_code

        # active | preparing | running — keep waiting
        time.sleep(POLL_INTERVAL_SECONDS)
```

The `TaskState` values are: `active`, `preparing`, `running`, `completed`.
There is no separate `failed` state — Azure Batch signals failure via a
`completed` state with a non-zero exit code or `failure_info`.

---

### Step 5 — Wire into the Prefect task

```python
def run_eval_on_batch(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> str:
    azure_task_id = submit_batch_task(agent_name, benchmark_name, task_id, model)
    print(f"Submitted | azure_task_id={azure_task_id}")
    poll_batch_task(azure_task_id)
    print(f"Completed | azure_task_id={azure_task_id}")
    return azure_task_id
```

In `run_eval_task`, replace `run_eval_locally(...)` with `run_eval_on_batch(...)`.

---

### Step 6 — Add `datetime` import

`TaskConstraints(retention_time=...)` requires a `timedelta`:

```python
import datetime
```

---

## What stays the same

- Prefect `@task` decorator with retries and exponential backoff — unchanged.
  If `submit_batch_task` or `poll_batch_task` raise (API error, pool full,
  non-zero exit code), Prefect retries the whole submit→poll cycle.
- Per-task and flow-level artifacts — unchanged.
- Task run naming (`{model}/{agent_name}/{benchmark_name}/{task_id}`) — unchanged.
- Flow parameters (`agents`, `benchmark_tasks`, `models`) — unchanged.
- `.serve()` and UI triggering — unchanged.

---

## Sequence diagram

```
Prefect @task (thread)
  │
  ├─► submit_batch_task()
  │     └─► BatchServiceClient.task.add()  ──► Azure Batch API
  │           (returns immediately)
  │
  ├─► poll_batch_task()  [loop every 15s]
  │     └─► BatchServiceClient.task.get()  ──► Azure Batch API
  │           active / preparing / running → sleep → repeat
  │           completed + exit_code=0      → return
  │           completed + exit_code≠0      → raise RuntimeError
  │
  └─► create_markdown_artifact()
        └─► Prefect UI artifact tab
```

---

## Future: proper custom worker

Once the PoC is validated, this pattern can be promoted to a
`AzureBatchWorker(BaseWorker)` that:

1. Registers via `[project.entry-points."prefect.workers"]` in `pyproject.toml`
2. Implements `async def run(self, flow_run, configuration, task_status)`
3. Calls `task_status.started(azure_task_id)` after submission
4. Runs the poll loop asynchronously with `asyncio.sleep`
5. Returns `BaseWorkerResult(identifier=azure_task_id, exit_code=exit_code)`

This would allow work pool creation via `prefect work-pool create -t azure-batch`
and full UI-driven infrastructure management — but is not needed for the PoC.

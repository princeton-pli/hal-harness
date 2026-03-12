"""
Evaluation Harness PoC
======================
Orchestrates a matrix of (agent × benchmark_task × model) evaluations using Prefect.
Each individual benchmark task is a first-class Prefect task run.

Execution modes (set RUN_MODE env var):
  local  — simulate work locally (default, no Azure required)
  batch  — submit to Azure Batch and poll until completion
"""

import datetime
import os
import random
import time
import uuid

import azure.batch.models as batch_models
from azure.batch import BatchServiceClient

# FIXME: swap AzureCliCredential for DefaultAzureCredential in production
# (requires service principal with AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET)
from azure.identity import AzureCliCredential
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.tasks import exponential_backoff

# load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

AZURE_BATCH_ACCOUNT_URL = "https://halharness.eastus.batch.azure.com"
AZURE_BATCH_POOL_ID = "proof-of-concept"
AZURE_BATCH_JOB_ID = "proof-of-concept"

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
AGENTS = [
    "core_agent",
    "hal_generalist_agent",
    "openai_codex_agent",
]

# Each benchmark maps to its list of task IDs.
# In production these would be loaded from the benchmark dataset.
BENCHMARK_TASKS: dict[str, list[str]] = {
    "gaia": ["gaia-2023-001", "gaia-2023-002", "gaia-2023-003"],
    "swe-bench": ["django__django-1234", "astropy__astropy-5678", "sympy__sympy-9012"],
    "usaco": [
        "usaco-2023-jan-bronze-1",
        "usaco-2023-jan-silver-1",
        "usaco-2023-feb-gold-1",
    ],
}

MODELS = [
    # OpenAI
    "gpt-4o",
    # "o3",
    # "o4-mini",
    # Anthropic
    "claude-opus-4-6",
    # "claude-sonnet-4-6",
    # "claude-haiku-4-5",
]

POLL_INTERVAL_SECONDS = 15  # mirrors PREFECT_WORKER_QUERY_SECONDS default


# ---------------------------------------------------------------------------
# Local simulation
# ---------------------------------------------------------------------------


def run_eval_locally(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> dict:
    """Simulate a single benchmark task locally."""
    run_id = f"{model}-{agent_name}-{benchmark_name}-{task_id}-{uuid.uuid4().hex[:8]}"
    duration = random.uniform(3, 10)
    print(
        f"Running | agent={agent_name} benchmark={benchmark_name} task={task_id} model={model} ({duration:.1f}s)"
    )
    time.sleep(duration)
    if (
        random.random() < 0.3
    ):  # 30% chance of failure — remove once wired to Azure Batch
        raise RuntimeError(
            f"Simulated failure: {agent_name} / {benchmark_name} / {task_id} / {model}"
        )
    print(f"Done | run_id={run_id}")
    return {
        "run_id": run_id,
        "metadata": {"info": "TODO: complete me"},
        "agent_response": {"info": "TODO: complete me"},
        "scoring": {"info": "TODO: complete me"},
    }


# ---------------------------------------------------------------------------
# Azure Batch
# ---------------------------------------------------------------------------


def _batch_client() -> BatchServiceClient:
    """Authenticated BatchServiceClient. Fresh per call — not thread-safe to share."""
    from msrest.authentication import BasicTokenAuthentication

    token = AzureCliCredential().get_token("https://batch.core.windows.net/.default")
    return BatchServiceClient(
        credentials=BasicTokenAuthentication({"access_token": token.token}),
        batch_url=AZURE_BATCH_ACCOUNT_URL,
    )


def _submit_batch_task(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> str:
    """Submit one eval task to Azure Batch. Returns the Azure task ID."""
    client = _batch_client()
    job_id = AZURE_BATCH_JOB_ID
    suffix = uuid.uuid4().hex[:8]
    azure_task_id = f"{model}-{agent_name}-{benchmark_name}-{task_id}-{suffix}"
    if len(azure_task_id) > 64:
        print(
            f"Warning: task ID truncated from {len(azure_task_id)} chars: {azure_task_id}"
        )
        azure_task_id = (
            azure_task_id[:55] + "-" + suffix
        )  # suffix stays intact for uniqueness

    python_code = (
        "import time; "
        f"print('agent={agent_name} benchmark={benchmark_name} task={task_id} model={model}'); "
        "time.sleep(5); "
        "print('done')"
    )
    batch_task = batch_models.TaskAddParameter(
        id=azure_task_id,
        command_line=f"/bin/bash -c \"python3 -c '{python_code}'\"",
        constraints=batch_models.TaskConstraints(
            max_task_retry_count=0,  # Prefect owns retries
            retention_time=datetime.timedelta(hours=1),
        ),
    )
    client.task.add(job_id=job_id, task=batch_task)
    return azure_task_id


def _poll_batch_task(azure_task_id: str) -> None:
    """Block until the Azure Batch task completes. Raises on non-zero exit code.

    TODO: replace polling with incremental stdout.txt reads for near-real-time log streaming.
    While the task is running, call client.file.get_from_task(..., 'stdout.txt',
    ocp_range=f"bytes={bytes_read}-") each cycle and print new content. Track bytes_read
    to avoid re-printing. stdout.txt won't exist until the task starts (active→running),
    so swallow the file-not-found exception until then.
    """
    client = _batch_client()
    job_id = AZURE_BATCH_JOB_ID

    while True:
        t = client.task.get(job_id, azure_task_id)

        if t.state == batch_models.TaskState.completed:
            exit_code = t.execution_info.exit_code
            if exit_code != 0:
                raise RuntimeError(
                    f"Azure Batch task {azure_task_id} failed "
                    f"(exit_code={exit_code}): {t.execution_info.failure_info}"
                )
            return

        # active | preparing | running — keep waiting
        print(f"Polling | azure_task_id={azure_task_id} state={t.state}")
        time.sleep(POLL_INTERVAL_SECONDS)


def run_eval_on_batch(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> dict:
    """Submit to Azure Batch and block until completion."""
    azure_task_id = _submit_batch_task(agent_name, benchmark_name, task_id, model)
    print(f"Submitted | azure_task_id={azure_task_id}")
    _poll_batch_task(azure_task_id)
    print(f"Completed | azure_task_id={azure_task_id}")
    return {
        "run_id": azure_task_id,
        "metadata": {"info": "TODO: complete me"},
        "agent_response": {"info": "TODO: complete me"},
        "scoring": {"info": "TODO: complete me"},
    }


# ---------------------------------------------------------------------------
# Prefect tasks and flow
# ---------------------------------------------------------------------------

_RUN_BACKENDS = {
    "local": run_eval_locally,
    "batch": run_eval_on_batch,
}


@task(
    task_run_name="{model}/{agent_name}/{benchmark_name}/{task_id}",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    retry_jitter_factor=1,
    log_prints=True,
)
def run_eval_task(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> dict:
    """Prefect task: run one (agent, benchmark task, model) evaluation."""
    run_mode = os.environ.get("RUN_MODE", "batch")
    print(f"Using run mode {run_mode}")
    run_fn = _RUN_BACKENDS.get(run_mode)
    if run_fn is None:
        raise ValueError(
            f"Unknown RUN_MODE={run_mode!r}. Choose: {list(_RUN_BACKENDS)}"
        )

    result = run_fn(agent_name, benchmark_name, task_id, model)
    create_markdown_artifact(
        key="task-result",
        description=f"{agent_name} / {benchmark_name} / {task_id} / {model}",
        markdown=f"""# Eval Result

| Field | Value |
|:------|:------|
| Agent | `{agent_name}` |
| Benchmark | `{benchmark_name}` |
| Task | `{task_id}` |
| Model | `{model}` |
| Run ID | `{result["run_id"]}` |
| Status | ✅ Passed |

## Agent Response
{result["agent_response"]}

## Scoring
{result["scoring"]}
""",
    )
    return result


@flow(log_prints=True)
def evaluation_harness_poc(
    agents: list[str] = AGENTS,
    benchmark_tasks: dict[str, list[str]] = BENCHMARK_TASKS,
    models: list[str] = MODELS,
) -> None:
    """Submit all (agent × benchmark task × model) combinations concurrently."""
    combos = [
        (agent, benchmark, task_id, model)
        for agent in agents
        for benchmark, tasks in benchmark_tasks.items()
        for task_id in tasks
        for model in models
    ]

    futures = [
        run_eval_task.submit(agent, benchmark, task_id, model)
        for agent, benchmark, task_id, model in combos
    ]

    wait(futures)

    rows = []
    for (agent, benchmark, task_id, model), future in zip(combos, futures):
        try:
            run_id = future.result()["run_id"]
            rows.append(
                f"| {agent} | {benchmark} | `{task_id}` | {model} | ✅ | `{run_id}` |"
            )
        except Exception as e:
            rows.append(f"| {agent} | {benchmark} | `{task_id}` | {model} | ❌ | {e} |")

    table = "\n".join(rows)
    passed = sum(1 for r in rows if "✅" in r)
    failed = len(rows) - passed

    create_markdown_artifact(
        key="eval-summary",
        description="Agent × Benchmark Task × Model evaluation results",
        markdown=f"""# Evaluation Run Summary

**{passed} passed · {failed} failed · {len(rows)} total**

| Agent | Benchmark | Task | Model | Status | Run ID |
|:------|:----------|:-----|:------|:------:|:-------|
{table}
""",
    )


if __name__ == "__main__":
    evaluation_harness_poc.serve(name="eval-harness-poc")

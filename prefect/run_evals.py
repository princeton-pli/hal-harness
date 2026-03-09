"""
Evaluation Harness PoC
======================
Orchestrates a matrix of (agent × benchmark_task × model) evaluations using Prefect.
Each individual benchmark task is a first-class Prefect task run.
Currently runs locally; swap run_eval_locally() for Azure Batch submission when ready.
"""

import random
import time
import uuid

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.tasks import exponential_backoff

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


# ---------------------------------------------------------------------------
# Local simulation (replace with Azure Batch submission when ready)
# ---------------------------------------------------------------------------


def run_eval_locally(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> str:
    """Simulate a single benchmark task locally."""
    run_id = (
        f"{agent_name}--{benchmark_name}--{task_id}--{model}--{uuid.uuid4().hex[:8]}"
    )
    duration = random.uniform(3, 10)
    print(
        f"Running | agent={agent_name} benchmark={benchmark_name} task={task_id} model={model} ({duration:.1f}s)"
    )
    print("These logs get intercepted by Prefect to be shown in the UI")
    time.sleep(duration)
    if (
        random.random() < 0.3
    ):  # 30% chance of failure — remove once wired to Azure Batch
        raise RuntimeError(
            f"Simulated failure: {agent_name} / {benchmark_name} / {task_id} / {model}"
        )
    print(f"Done | run_id={run_id}")
    return run_id


# ---------------------------------------------------------------------------
# Prefect tasks and flow
# ---------------------------------------------------------------------------


@task(
    task_run_name="{model}/{agent_name}/{benchmark_name}/{task_id}",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    retry_jitter_factor=1,
    log_prints=True,
)
def run_eval_task(
    agent_name: str, benchmark_name: str, task_id: str, model: str
) -> str:
    """Prefect task: run one (agent, benchmark task, model) evaluation."""
    run_id = run_eval_locally(agent_name, benchmark_name, task_id, model)
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
| Run ID | `{run_id}` |
| Status | ✅ Passed |
""",
    )
    return run_id


@flow(log_prints=True)
def evaluation_harness_poc() -> None:
    """Submit all (agent × benchmark task × model) combinations concurrently."""
    combos = [
        (agent, benchmark, task_id, model)
        for agent in AGENTS
        for benchmark, tasks in BENCHMARK_TASKS.items()
        for task_id in tasks
        for model in MODELS
    ]

    futures = [
        run_eval_task.submit(agent, benchmark, task_id, model)
        for agent, benchmark, task_id, model in combos
    ]

    wait(futures)

    rows = []
    for (agent, benchmark, task_id, model), future in zip(combos, futures):
        try:
            run_id = future.result()
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
    evaluation_harness_poc()

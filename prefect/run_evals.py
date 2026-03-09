"""
Evaluation Harness PoC
======================
Orchestrates a matrix of (agent × benchmark) evaluations using Prefect.
Currently runs locally; swap run_eval_locally() for Azure Batch submission
when ready.
"""

import random
import time
import uuid

from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.tasks import exponential_backoff

# ---------------------------------------------------------------------------
# Matrix – update these to match your real agents and benchmarks.
# Agent names mirror the directories under agents/; benchmark names match
# the keys recognised by BenchmarkManager.
# ---------------------------------------------------------------------------
AGENTS = [
    "core_agent",
    "hal_generalist_agent",
    "openai_codex_agent",
]
BENCHMARKS = [
    "gaia",
    "swe-bench",
    "usaco",
]
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


def run_eval_locally(agent_name: str, benchmark_name: str, model: str) -> str:
    """Simulate an eval run locally.  Swap this out for Azure Batch submission."""
    task_id = f"{agent_name}--{benchmark_name}--{model}--{uuid.uuid4().hex[:8]}"
    duration = random.uniform(3, 10)
    print(
        f"Running eval | agent={agent_name} benchmark={benchmark_name} model={model} (will take {duration:.1f}s)"
    )
    time.sleep(duration)
    if (
        random.random() < 0.3
    ):  # 30% chance of failure — remove once wired to Azure Batch
        raise RuntimeError(
            f"Simulated failure for {agent_name} on {benchmark_name} with {model}"
        )
    print(f"Done | task_id={task_id}")
    return task_id


# ---------------------------------------------------------------------------
# Prefect tasks and flow
# ---------------------------------------------------------------------------


@task(
    task_run_name="{agent_name}/{benchmark_name}/{model}",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    retry_jitter_factor=1,
    log_prints=True,
)
def run_eval_task(agent_name: str, benchmark_name: str, model: str) -> str:
    """Prefect task: run one (agent, benchmark, model) evaluation."""
    task_id = run_eval_locally(agent_name, benchmark_name, model)
    create_markdown_artifact(
        key="task-result",
        description=f"{agent_name} / {benchmark_name} / {model}",
        markdown=f"""# Eval Result

| Field | Value |
|:------|:------|
| Agent | `{agent_name}` |
| Benchmark | `{benchmark_name}` |
| Model | `{model}` |
| Task ID | `{task_id}` |
| Status | ✅ Passed |
""",
    )
    return task_id


@flow(log_prints=True)
def evaluation_harness_poc() -> None:
    """Submit all (agent × benchmark × model) combinations concurrently."""
    combos = [
        (agent, benchmark, model)
        for agent in AGENTS
        for benchmark in BENCHMARKS
        for model in MODELS
    ]
    futures = [
        run_eval_task.submit(agent, benchmark, model)
        for agent, benchmark, model in combos
    ]

    wait(futures)

    rows = []
    for (agent, benchmark, model), future in zip(combos, futures):
        try:
            task_id = future.result()
            rows.append(f"| {agent} | {benchmark} | {model} | ✅ | `{task_id}` |")
        except Exception as e:
            rows.append(f"| {agent} | {benchmark} | {model} | ❌ | {e} |")

    table = "\n".join(rows)
    passed = sum(1 for r in rows if "✅" in r)
    failed = len(rows) - passed

    create_markdown_artifact(
        key="eval-summary",
        description="Agent × Benchmark × Model evaluation results",
        markdown=f"""# Evaluation Run Summary

**{passed} passed · {failed} failed · {len(rows)} total**

| Agent | Benchmark | Model | Status | Task ID |
|:------|:----------|:------|:------:|:--------|
{table}
""",
    )


if __name__ == "__main__":
    evaluation_harness_poc()

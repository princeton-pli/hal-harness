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


# ---------------------------------------------------------------------------
# Local simulation (replace with Azure Batch submission when ready)
# ---------------------------------------------------------------------------


def run_eval_locally(agent_name: str, benchmark_name: str) -> str:
    """Simulate an eval run locally.  Swap this out for Azure Batch submission."""
    task_id = f"{agent_name}--{benchmark_name}--{uuid.uuid4().hex[:8]}"
    duration = random.uniform(15, 30)
    print(f"Running eval | agent={agent_name} benchmark={benchmark_name} (will take {duration:.1f}s)")
    time.sleep(duration)
    if random.random() < 0.3:  # 30% chance of failure — remove once wired to Azure Batch
        raise RuntimeError(f"Simulated failure for {agent_name} on {benchmark_name}")
    print(f"Done | task_id={task_id}")
    return task_id


# ---------------------------------------------------------------------------
# Prefect tasks and flow
# ---------------------------------------------------------------------------


@task(
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    retry_jitter_factor=1,
    log_prints=True,
)
def run_eval_task(agent_name: str, benchmark_name: str) -> str:
    """Prefect task: run one (agent, benchmark) evaluation."""
    return run_eval_locally(agent_name, benchmark_name)


@flow(log_prints=True)
def evaluation_harness_poc() -> None:
    """Submit all (agent × benchmark) combinations to Azure Batch concurrently."""
    futures = [
        run_eval_task.submit(agent, benchmark)
        for agent in AGENTS
        for benchmark in BENCHMARKS
    ]

    # Block until every task submission has completed (or failed after retries).
    wait(futures)

    print("\n--- Results ---")
    for future in futures:
        try:
            print(f"  ok  {future.result()}")
        except Exception as e:
            print(f"  FAILED  {e}")


if __name__ == "__main__":
    evaluation_harness_poc()

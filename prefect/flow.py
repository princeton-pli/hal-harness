from prefect import flow
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.runtime import flow_run as current_flow_run

from batch import create_batch_job, terminate_batch_job
from config import AGENTS, BENCHMARK_TASKS, MODELS, EvalSpec
from tasks import run_eval_task


@flow(log_prints=True)
def evaluation_harness(
    agents: list[str] = AGENTS,
    benchmark_tasks: dict[str, list[str]] = BENCHMARK_TASKS,
    models: list[str] = MODELS,
) -> None:
    """Submit all (agent × benchmark × model) combinations concurrently."""
    job_id = str(current_flow_run.id)
    create_batch_job(job_id)

    specs = [
        EvalSpec(agent=agent, benchmark=benchmark, task_id=task_id, model=model, job_id=job_id)
        for agent in agents
        for benchmark, tasks in benchmark_tasks.items()
        for task_id in tasks
        for model in models
    ]

    futures = [run_eval_task.submit(spec) for spec in specs]
    wait(futures)
    terminate_batch_job(job_id)

    rows = []
    for spec, future in zip(specs, futures):
        try:
            run_id = future.result()["run_id"]
            rows.append(
                f"| {spec.agent} | {spec.benchmark} | `{spec.task_id}` | {spec.model} | ✅ | `{run_id}` |"
            )
        except Exception as e:
            rows.append(
                f"| {spec.agent} | {spec.benchmark} | `{spec.task_id}` | {spec.model} | ❌ | {e} |"
            )

    table = "\n".join(rows)
    passed = sum(1 for r in rows if "✅" in r)
    failed = len(rows) - passed

    create_markdown_artifact(
        key="eval-summary",
        description="Agent × Benchmark × Model evaluation results",
        markdown=f"""# Evaluation Run Summary

**{passed} passed · {failed} failed · {len(rows)} total**

| Agent | Benchmark | Task | Model | Status | Run ID |
|:------|:----------|:-----|:------|:------:|:-------|
{table}
""",
    )


if __name__ == "__main__":
    evaluation_harness.serve(name="eval-harness")

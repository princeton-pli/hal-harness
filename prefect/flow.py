from prefect import flow
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.runtime import flow_run as current_flow_run

from batch import create_batch_job, resize_pool, terminate_batch_job
from config import AGENTS, BENCHMARK_TASKS, MODELS, EvalSpec
from storage import upload_code_zip, result_container_sas
from tasks import run_eval_task


@flow(log_prints=True)
def evaluation_harness(
    agents: list[dict] = AGENTS,
    benchmark_tasks: dict[str, list[str]] = BENCHMARK_TASKS,
    models: list[str] = MODELS,
) -> None:
    """Submit all (agent × benchmark × model) combinations concurrently."""
    job_id = str(current_flow_run.id)
    create_batch_job(job_id)

    code_sas_url = upload_code_zip(job_id)
    res_sas_url = result_container_sas()

    specs = [
        EvalSpec(
            agent=agent["name"],
            agent_function=agent["function"],
            agent_dir=agent["dir"],
            benchmark=benchmark,
            task_id=task_id,
            model=model,
            job_id=job_id,
            code_sas_url=code_sas_url,
            result_sas_url=res_sas_url,
        )
        for agent in agents
        for benchmark, tasks in benchmark_tasks.items()
        for task_id in tasks
        for model in models
    ]

    resize_pool(len(specs))
    futures = [run_eval_task.submit(spec) for spec in specs]
    wait(futures)
    terminate_batch_job(job_id)
    # resize_pool(0)  # commented out during active iteration to keep nodes warm

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
    evaluation_harness.serve(name="eval-harness-poc")

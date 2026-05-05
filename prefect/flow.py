import resource

from prefect import flow
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.runtime import flow_run as current_flow_run

from batch import create_batch_job, resize_pool, terminate_batch_job
from config import AGENTS, BENCHMARK_TASKS, K_REPEATS, MAX_NODES, RUN_CONFIGS, EvalSpec
from storage import capsule_blob_sas, result_container_sas, upload_code_zip
from tasks import run_eval_task

# Raise the file descriptor limit so 1000+ concurrent SSL connections don't
# hit the default 1024 soft limit. Each Prefect task holds an open socket to
# Azure Batch for polling.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 8192), _hard))


@flow(log_prints=True)
def evaluation_harness(
    agents: list[dict] = AGENTS,
    benchmark_tasks: dict[str, list[str]] = BENCHMARK_TASKS,
    run_configs: list[tuple[str, tuple[tuple[str, str], ...]]] = RUN_CONFIGS,
    k_repeats: int = K_REPEATS,
) -> None:
    """Submit all (agent × benchmark × task × run_config × repeat) combinations concurrently.

    run_configs is a flat list of (model, agent_args) pairs rather than a
    cartesian product of models and ablation axes.

    k_repeats duplicates each cell k times with distinct repeat_idx values
    (0 .. k-1). K >= 2 enables outcome consistency metrics in reliability
    analysis.
    """
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
            extra_output_files=tuple(agent.get("extra_output_files", ())),
            capsule_sas_url=(
                capsule_blob_sas(task_id) if benchmark.startswith("corebench") else ""
            ),
            agent_args=agent_args,
            task_timeout=agent.get("task_timeout", 0),
            repeat_idx=k,
        )
        for agent in agents
        for benchmark, tasks in benchmark_tasks.items()
        for task_id in tasks
        for model, agent_args in run_configs
        for k in range(k_repeats)
    ]

    target = len(specs) if MAX_NODES <= 0 else min(len(specs), MAX_NODES)
    resize_pool(target)
    futures = [run_eval_task.submit(spec) for spec in specs]
    wait(futures)
    terminate_batch_job(job_id)
    # resize_pool(0)  # FIXME: commented out during active iteration to keep nodes warm

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

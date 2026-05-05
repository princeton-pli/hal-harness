"""Template for one-off Prefect flow drivers.

The default flow.py runs the full evaluation matrix from config.py
(AGENTS × BENCHMARK_TASKS × RUN_CONFIGS × K_REPEATS). Use this template
when you need to launch a *different* slice of that matrix without
editing config.py — typical reasons:

  - Running a single agent (e.g. just claude_code_agent) on the active
    BENCHMARK_TASKS while config.py is set up for another agent.
  - Running a custom RUN_CONFIGS list (different model/effort sweep) for
    a quick ablation that shouldn't pollute the main config.
  - Running on a custom task list (e.g. the OOD capsules) without
    swapping the active core_test.json or editing config.py.

Setup:
  1. Copy this file to a new name describing what it runs, e.g.
     `flow_<my_run>.py`. This keeps git diffs trivial and Prefect's
     `.serve(name=...)` deployment names distinct so concurrent flows
     don't collide.
  2. Edit the four CHANGE_ME blocks below: AGENT, BENCHMARK_TASKS,
     RUN_CONFIGS, deployment name.
  3. Optionally set MAX_NODES env var to cap pool resize for
     rate-limited providers (e.g. MAX_NODES=20 for Anthropic).
  4. `cd prefect && python flow_<my_run>.py` then trigger from the
     Prefect UI at localhost:4200.

Useful patterns implemented here that are easy to forget:
  - resource.setrlimit(NOFILE, 8192): raise the FD soft cap so 1000+
    concurrent SSL sockets to Azure Batch don't exhaust the default
    1024 limit.
  - resize_pool(target) only scales UP unless 0 is passed; safe to call
    concurrently with another flow.
  - extra_output_files: agent-specific log glob patterns to copy off
    the VM into the result blob (codex_exec.log, opencode logs, etc.).
"""

import resource

from prefect import flow
from prefect.artifacts import create_markdown_artifact
from prefect.futures import wait
from prefect.runtime import flow_run as current_flow_run

from batch import create_batch_job, resize_pool, terminate_batch_job
from config import K_REPEATS, MAX_NODES, EvalSpec
from storage import capsule_blob_sas, result_container_sas, upload_code_zip
from tasks import run_eval_task

_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 8192), _hard))

# ─── CHANGE_ME #1: agent ──────────────────────────────────────────────────
# One agent per flow run. For multi-agent matrices use the default flow.py
# which iterates over config.AGENTS.
AGENT: dict = {
    "name": "codex_agent",
    "function": "main.run",
    "dir": "agents/codex_agent",
    # Glob patterns relative to the task's working dir on the Azure Batch
    # node; matched files are uploaded to hal-harness-runs/<job>/logs/...
    "extra_output_files": (
        ("hal-harness/results/**/codex_exec.log.*", "codex"),
    ),
    # Optional: per-agent task timeout (seconds). 0 → use hal-eval default
    # (2700s). Long-horizon agents like core_agent v2 set 18000 (5h).
    "task_timeout": 0,
}

# ─── CHANGE_ME #2: tasks ──────────────────────────────────────────────────
# benchmark_name → list of task ids. For corebench-style benchmarks, task
# ids are capsule ids and capsule_blob_sas() resolves the per-task tarball
# automatically below.
BENCHMARK_TASKS: dict[str, list[str]] = {
    "corebench_hard": [
        "capsule-XXXXXXX",
        "capsule-YYYYYYY",
    ],
}

# ─── CHANGE_ME #3: run configs ────────────────────────────────────────────
# Flat list of (model, agent_args) pairs — each entry is one ablation cell.
# agent_args are passed to hal-eval as -A key=value forwarded to the agent.
RUN_CONFIGS: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    ("gpt-5.4", (("reasoning_effort", "medium"),)),
    # ("anthropic/claude-opus-4-6", ()),
    # ("anthropic/claude-opus-4-5", (("max_thinking_tokens", "10000"),)),
]


@flow(log_prints=True)
def run(
    agent: dict = AGENT,
    benchmark_tasks: dict[str, list[str]] = BENCHMARK_TASKS,
    run_configs: list[tuple[str, tuple[tuple[str, str], ...]]] = RUN_CONFIGS,
    k_repeats: int = K_REPEATS,
) -> None:
    """Submit (task × run_config × repeat) for a single agent."""
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
        for benchmark, tasks in benchmark_tasks.items()
        for task_id in tasks
        for model, agent_args in run_configs
        for k in range(k_repeats)
    ]

    target = len(specs) if MAX_NODES <= 0 else min(len(specs), MAX_NODES)
    print(f"Submitting {len(specs)} tasks (pool target = {target})")
    resize_pool(target)
    futures = [run_eval_task.submit(spec) for spec in specs]
    wait(futures)
    terminate_batch_job(job_id)

    rows = []
    for spec, future in zip(specs, futures):
        try:
            run_id = future.result()["run_id"]
            rows.append(
                f"| {spec.agent} | {spec.benchmark} | `{spec.task_id}` | "
                f"{spec.model} | k={spec.repeat_idx} | ✅ | `{run_id}` |"
            )
        except Exception as e:
            rows.append(
                f"| {spec.agent} | {spec.benchmark} | `{spec.task_id}` | "
                f"{spec.model} | k={spec.repeat_idx} | ❌ | {e} |"
            )

    table = "\n".join(rows)
    passed = sum(1 for r in rows if "✅" in r)
    failed = len(rows) - passed
    create_markdown_artifact(
        key="eval-summary",
        description="Custom flow run summary",
        markdown=(
            f"# Run Summary\n\n"
            f"**{passed} passed · {failed} failed · {len(rows)} total**\n\n"
            f"| Agent | Benchmark | Task | Model | Repeat | Status | Run ID |\n"
            f"|:------|:----------|:-----|:------|:------:|:------:|:-------|\n"
            f"{table}\n"
        ),
    )


if __name__ == "__main__":
    # ─── CHANGE_ME #4: deployment name ────────────────────────────────────
    # Each concurrent flow needs a unique name; otherwise .serve() collides
    # with another running deployment.
    run.serve(name="custom-eval-template")

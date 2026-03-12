from prefect import task
from prefect.artifacts import create_markdown_artifact
from prefect.tasks import exponential_backoff

from batch import run_eval_on_batch
from config import EvalSpec


@task(
    task_run_name="{spec.model}/{spec.agent}/{spec.benchmark}/{spec.task_id}",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
    retry_jitter_factor=1,
    log_prints=True,
)
def run_eval_task(spec: EvalSpec) -> dict:
    """Prefect task: run one (agent, benchmark, task, model) evaluation."""
    result = run_eval_on_batch(spec)
    create_markdown_artifact(
        key="task-result",
        description=f"{spec.agent} / {spec.benchmark} / {spec.task_id} / {spec.model}",
        markdown=f"""# Eval Result

| Field | Value |
|:------|:------|
| Agent | `{spec.agent}` |
| Benchmark | `{spec.benchmark}` |
| Task | `{spec.task_id}` |
| Model | `{spec.model}` |
| Run ID | `{result["run_id"]}` |
| Status | ✅ Passed |

## Agent Response
{result["agent_response"]}

## Scoring
{result["scoring"]}
""",
    )
    return result

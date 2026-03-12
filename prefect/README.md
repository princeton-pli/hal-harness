# Evaluation Harness — Prefect + Azure Batch

Orchestrates a matrix of `(agent × benchmark × model)` evaluations using Prefect.
Each benchmark task is a first-class Prefect task run submitted to Azure Batch.

## Module structure

| File | Purpose |
|:-----|:--------|
| `config.py` | `EvalSpec` dataclass, evaluation matrix constants, Azure connection config |
| `batch.py` | Azure Batch client, job lifecycle, task submit/poll/stream |
| `tasks.py` | Prefect `@task` wrapping `batch.run_eval_on_batch` |
| `flow.py` | Prefect `@flow` + `__main__` entrypoint |
| `eval_task.py` | **Demo stub** — Python script run on Batch nodes (replace with real eval) |
| `eval_helper.py` | **Demo stub** — imported by `eval_task.py` (replace with real eval) |

## Setup

```bash
pip install -r prefect/requirements.txt
```

## Running

Two processes must be running simultaneously — open two terminal tabs.

**Tab 1 — Prefect UI server:**

```bash
prefect server start
```

**Tab 2 — Flow deployment listener:**

```bash
python prefect/flow.py
```

The UI is available at http://localhost:4200. Go to **Deployments → eval-harness → Run → Custom run** to trigger a flow run with custom parameters.

## Configuration

The evaluation matrix (`AGENTS`, `BENCHMARK_TASKS`, `MODELS`) is defined in `config.py`
and can be overridden at flow-run time via Prefect's parameter UI.

Azure Batch connection values are read from environment variables, with hardcoded defaults
for the current dev account:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `AZURE_BATCH_ACCOUNT_URL` | `https://halharness.eastus.batch.azure.com` | Batch account endpoint |
| `AZURE_BATCH_POOL_ID` | `proof-of-concept` | Pool to run tasks on |

For `az login` (local/dev), no additional env vars are needed. For production, switch
`AzureCliCredential` to `DefaultAzureCredential` with a service principal (see FIXME in
`batch.py`).

## How a flow run works

```
evaluation_harness (@flow)
  │
  ├─► create_batch_job(job_id)          # one job per Prefect flow run
  │
  ├─► run_eval_task.submit(spec) × N    # one Prefect task per (agent × benchmark × model)
  │     └─► run_eval_on_batch(spec)
  │           ├─► _submit_batch_task()  # ships eval_task.py to pool node via base64
  │           └─► _poll_batch_task()    # streams stdout every 15 s until done
  │
  ├─► terminate_batch_job(job_id)
  │
  └─► create_markdown_artifact()        # summary table in Prefect UI
```

# Evaluation Harness — Prefect + Azure Batch

Orchestrates a matrix of `(agent × benchmark × model)` evaluations using Prefect.
Each benchmark task is a first-class Prefect task run submitted to Azure Batch.

## Module structure

| File | Purpose |
|:-----|:--------|
| `config.py` | `EvalSpec` dataclass, evaluation matrix constants, Azure connection config |
| `batch.py` | Azure Batch client, job lifecycle, task submit/poll/stream |
| `storage.py` | Azure Blob Storage — zip upload, SAS URLs, result download |
| `tasks.py` | Prefect `@task` wrapping `batch.run_eval_on_batch` |
| `flow.py` | Prefect `@flow` + `__main__` entrypoint |
| `pool.json` | Azure Batch pool definition (CPU, Docker-capable) |

The node entrypoint is `azure_entrypoint.sh` at the repo root.

## Prerequisites

- Python 3.11+
- Azure CLI logged in (`az login`)
- Environment variables set in `prefect/.env` (see Configuration below)

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

Copy `prefect/.env` and fill in secrets:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `AZURE_BATCH_ACCOUNT_URL` | `https://halharness.eastus.batch.azure.com` | Batch account endpoint |
| `AZURE_BATCH_POOL_ID` | `proof-of-concept` | Pool to run tasks on |
| `AZURE_STORAGE_ACCOUNT_NAME` | `halharness` | Storage account name |
| `AZURE_STORAGE_ACCOUNT_URL` | `https://halharness.blob.core.windows.net` | Storage account endpoint |
| `AZURE_STORAGE_CONTAINER_NAME` | `hal-harness-runs` | Blob container for code + results |
| `AZURE_STORAGE_ACCOUNT_KEY` | *(required)* | Storage account key for SAS generation |
| `AZURE_RESOURCE_GROUP` | `hal_group` | Resource group (used by `batch_ssh.sh`) |
| `OPENAI_API_KEY` | *(required)* | Forwarded to Batch nodes |
| `ANTHROPIC_API_KEY` | *(optional)* | Forwarded to Batch nodes |
| `HF_TOKEN` | *(required for GAIA)* | Forwarded to Batch nodes |

## How a flow run works

```
evaluation_harness (@flow)
  │
  ├─► create_batch_job(job_id)              # one job per Prefect flow run
  ├─► upload_code_zip(job_id)               # zips repo → Blob Storage → read SAS URL
  ├─► result_container_sas()               # container-level write SAS for results
  │
  ├─► resize_pool(N)                        # scale pool to number of tasks
  │
  ├─► run_eval_task.submit(spec) × N        # one Prefect task per (agent × benchmark × task × model)
  │     └─► run_eval_on_batch(spec)
  │           ├─► _submit_batch_task()      # ResourceFile (zip) + OutputFiles (results) + env vars
  │           ├─► _poll_batch_task()        # streams stdout/stderr every 15s until done
  │           └─► download_task_results()  # fetches *_UPLOAD.json from Blob Storage
  │
  ├─► terminate_batch_job(job_id)
  ├─► resize_pool(0)
  └─► create_markdown_artifact()            # summary table in Prefect UI
```

On success, each task's result dict is written locally to `.prefect_results/{job_id}/{azure_task_id}/result.json` (gitignored) in addition to Azure Blob Storage.

Each Batch node runs `azure_entrypoint.sh`, which:
1. Bootstraps pip if needed
2. `pip install -e ".[dev]"` from the unzipped repo
3. Runs `python3 -m hal.cli` with the agent/benchmark/task/model args

## Pool

The pool (`proof-of-concept`) uses `Standard_D4s_v3` CPU nodes with the
`microsoft-azure-batch/ubuntu-server-container/20-04-lts` image (Docker pre-installed).

To recreate the pool from scratch:
```bash
az batch account login --name halharness --resource-group hal_group --shared-key-auth
az batch pool create --json-file prefect/pool.json
```

## SSH into a node

```bash
./batch_ssh.sh             # picks first available node
./batch_ssh.sh <node-id>   # target a specific node
```

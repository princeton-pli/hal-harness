# Quickstart

## 1. Install dependencies

```bash
pip install -r prefect/requirements.txt
```

## 2. Configure secrets

Edit `prefect/.env` — required keys:

```
AZURE_STORAGE_ACCOUNT_KEY=...   # for SAS generation
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## 3. Log in to Azure

```bash
az login
```

## 4. Configure the eval matrix

Edit `prefect/config.py`:

- `AGENTS` — which agents to run
- `BENCHMARK_TASKS` — benchmark name → list of task IDs
- `MODELS` — model strings (must exist in `hal/utils/weave_utils.py:MODEL_PRICES_DICT`)

## 5. Run

**Option A — VS Code task (recommended):**

Run the **"Run prefect server and flow.py"** task (`Cmd+Shift+P` → `Tasks: Run Task`). This opens both processes side by side in the integrated terminal.

**Option B — two terminals:**

```bash
# Terminal 1
prefect server start

# Terminal 2
cd prefect && python flow.py
```

- Prefect UI at http://localhost:4200.
- On completion, each task's result is written locally to `.prefect_results/{job_id}/{azure_task_id}/result.json`.
- Results and logs are also uploaded to Azure Blob Storage under `{job_id}/results/` and `{job_id}/logs/`.

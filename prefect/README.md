# Evaluation Harness PoC — Prefect

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
python prefect/run_evals.py
```

The UI is available at http://localhost:4200. Go to **Deployments → eval-harness-poc → Run → Custom run** to trigger a flow run with custom parameters.

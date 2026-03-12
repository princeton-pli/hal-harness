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

---

## Running on Azure Batch

Set `RUN_MODE=batch` plus the variables below, then run the same two commands.

```bash
export RUN_MODE=batch
export AZURE_BATCH_ACCOUNT_URL=https://<account>.<region>.batch.azure.com
export AZURE_BATCH_POOL_ID=<your-pool-id>
export AZURE_BATCH_JOB_ID=<your-job-id>
export AZURE_TENANT_ID=...
export AZURE_CLIENT_ID=...
export AZURE_CLIENT_SECRET=...
```

### How to get these values

#### 1. Create an Azure Batch account

In the Azure Portal:

1. **Create a resource** → search "Batch account" → Create
2. Note the **account name** and **location**
3. After creation, go to the account → **Properties**
   - Copy the **URL** → this is `AZURE_BATCH_ACCOUNT_URL`

#### 2. Create a pool

Inside your Batch account → **Pools** → **Add**:

- OS: Ubuntu 22.04 LTS
- Node size: any (Standard_D2s_v3 is fine for testing)
- Scale: 1–3 dedicated nodes (fixed or autoscale)
- Note the **Pool ID** → this is `AZURE_BATCH_POOL_ID`

Python 3 is pre-installed on Ubuntu pool nodes. No Docker needed.

#### 3. Create a job

Inside your Batch account → **Jobs** → **Add**:

- Select the pool you just created
- Note the **Job ID** → this is `AZURE_BATCH_JOB_ID`

The job can stay open indefinitely — tasks will be added to it on each flow run.

#### 4. Auth

**For local/PoC use**, the code currently uses `AzureCliCredential` — just run:

```bash
az login
```

No service principal needed. `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET` can be left unset.

<!-- FIXME: for production, switch to DefaultAzureCredential with a service principal:

az login

az ad sp create-for-rbac --name "hal-harness-poc" --role contributor \
  --scopes /subscriptions/<your-subscription-id>

# Outputs:
# { "appId": "...", "password": "...", "tenant": "..." }
# Set as AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID

az role assignment create \
  --assignee <AZURE_CLIENT_ID> \
  --role "Contributor" \
  --scope /subscriptions/<subscription-id>/resourceGroups/<rg>/providers/Microsoft.Batch/batchAccounts/<account-name>

Requires "Application Administrator" role in Azure AD.
-->

#### 5. Find your subscription ID and resource group

```bash
az account show --query id -o tsv          # subscription ID
az batch account list --query "[].{name:name, rg:resourceGroup}" -o table
```

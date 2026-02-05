# Azure Monitor Agent Setup for HAL Docker Logs

## Architecture

```
Docker Container → /workspace/logs/container.log
                ↓ (mounted volume)
VM Filesystem → /home/agent/hal_logs/{run_id}/{task_id}/container.log
                ↓ (Azure Monitor Agent)
Azure Monitor → Log Analytics Workspace
```

## Setup Steps

### 1. Prerequisites

You need:
- Data Collection Endpoint (DCE) - already created: `https://staging-hal-logs-dce-eh19.eastus-1.ingest.monitor.azure.com`
- Log Analytics Workspace
- Azure subscription with permissions to create resources

### 2. Deploy Infrastructure with Bicep

The easiest way to create the DCR is using the provided Bicep templates:

#### Option A: Using Azure CLI (Recommended)

1. Update the parameters file `infrastructure/main.parameters.json` with your values:
   - Replace `{SUBSCRIPTION_ID}` with your Azure subscription ID
   - Replace `{RESOURCE_GROUP}` with your resource group name
   - Replace `{DCE_NAME}` with your DCE name
   - Replace `{WORKSPACE_NAME}` with your Log Analytics workspace name

2. Deploy the infrastructure:
```bash
az deployment group create \
  --resource-group your-resource-group \
  --template-file v2/infrastructure/main.bicep \
  --parameters v2/infrastructure/main.parameters.json
```

3. Get the DCR ID from the output:
```bash
az deployment group show \
  --resource-group your-resource-group \
  --name main \
  --query properties.outputs.dataCollectionRuleId.value
```

#### Option B: Manual DCR Creation

If you prefer to create the DCR manually, you can use the DCR configuration below.

### 3. Create Data Collection Rule (DCR) for File Collection (Manual)

You need to create a new DCR that:
- Uses the Azure Monitor Agent
- Collects log files from `/home/agent/hal_logs/**/*.log`
- Parses them as structured logs
- Sends to Log Analytics Workspace

#### DCR Configuration (JSON)

```json
{
  "properties": {
    "dataSources": {
      "logFiles": [
        {
          "streams": [
            "Custom-ContainerLogs_CL"
          ],
          "filePatterns": [
            "/home/agent/hal_logs/**/*.log"
          ],
          "format": "text",
          "settings": {
            "text": {
              "recordStartTimestampFormat": "ISO 8601"
            }
          },
          "name": "containerLogs"
        }
      ]
    },
    "destinations": {
      "logAnalytics": [
        {
          "workspaceResourceId": "/subscriptions/{subscription}/resourceGroups/{rg}/providers/Microsoft.OperationalInsights/workspaces/{workspace}",
          "name": "halWorkspace"
        }
      ]
    },
    "dataFlows": [
      {
        "streams": [
          "Custom-ContainerLogs_CL"
        ],
        "destinations": [
          "halWorkspace"
        ],
        "transformKql": "source | extend RunID = extract(@'hal_logs/([^/]+)/', 1, FilePath) | extend TaskID = extract(@'hal_logs/[^/]+/([^/]+)/', 1, FilePath)",
        "outputStream": "Custom-ContainerLogs_CL"
      }
    ]
  }
}
```

### 4. Associate DCR with VMs

The DCR association is **automatically handled** when creating VMs. The `AzureVirtualMachine` class now accepts a `dcr_id` parameter and automatically associates the DCR during VM creation.

To use it, pass the DCR ID when running:
```bash
python v2/run_docker_on_vm.py \
  --image hal-core-agent-docker:latest \
  --vm_count 1 \
  --dcr-id "/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Insights/dataCollectionRules/hal-docker-logs-dcr"
```

The code in `azure_virtual_machine.py:218-240` handles the association automatically.

### 5. Query Logs

Once setup, you can query logs in Log Analytics:

```kusto
ContainerLogs_CL
| where TimeGenerated > ago(1h)
| where RunID == "your-run-id"
| where TaskID == "your-task-id"
| project TimeGenerated, Computer, RunID, TaskID, RawData
| order by TimeGenerated asc
```

The logs will include these fields:
- `TimeGenerated`: Timestamp of the log entry
- `RawData`: The actual log message
- `FilePath`: Full path to the log file on the VM
- `Computer`: VM hostname
- `RunID`: Extracted from the file path (e.g., `abc123def456`)
- `TaskID`: Extracted from the file path (e.g., `task789`)

## Code Changes Made

1. **entrypoint.py**: Writes logs to `/workspace/logs/container.log` (FileHandler + StreamHandler)
2. **azure_virtual_machine.py**: Mounts volume `-v {log_dir}:/workspace/logs` where `log_dir = /home/agent/hal_logs/{run_id}/{task_id}`
3. **virtual_machine_cloud_init.yaml**:
   - Installs Azure Monitor Agent
   - Creates `/home/agent/hal_logs` directory
4. **Removed Azure Monitor direct ingestion**: No longer using `azure-monitor-ingestion` Python SDK

## Testing

1. Build and test locally first:
```bash
docker build --platform linux/amd64 -t hal-core-agent-docker:latest v2/agents/core_agent_docker/
docker run --rm \
  -v /tmp/test_logs:/workspace/logs \
  -e HAL_RUN_ID=test123 \
  -e HAL_TASK_ID=task456 \
  hal-core-agent-docker:latest

# Check logs were written
cat /tmp/test_logs/container.log
```

2. Run on Azure VM (after creating DCR):
```bash
python v2/run_docker_on_vm.py \
  --image hal-core-agent-docker:latest \
  --vm_count 1 \
  --dce-endpoint "https://staging-hal-logs-dce-eh19.eastus-1.ingest.monitor.azure.com" \
  --dcr-id "{your-new-dcr-id-for-file-collection}"
```

3. Check logs on VM:
```bash
ssh -i ~/.ssh/hal_azure agent@{vm-ip}
ls -la /home/agent/hal_logs/
cat /home/agent/hal_logs/{run_id}/{task_id}/container.log
```

4. Check logs in Azure Monitor (wait 1-5 minutes for ingestion):
```kusto
ContainerLogs_CL
| where RunID == "{run_id}"
| where TaskID == "{task_id}"
```

## Important Notes

- DCE is only used for direct HTTP ingestion (not needed for file-based collection)
- You need a **different DCR** for file collection vs HTTP ingestion
- Azure Monitor Agent must be installed and running on the VM
- DCR must be associated with each VM before logs will be collected
- Log ingestion typically takes 1-5 minutes after writing to file

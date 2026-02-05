# HAL Harness Infrastructure

This directory contains Bicep templates for deploying Azure infrastructure for HAL Harness.

## What Gets Deployed

- **Network Security Group (NSG)**: Controls network access to VMs
- **Data Collection Rule (DCR)**: Configures Azure Monitor Agent to collect Docker container logs

## Prerequisites

- Azure CLI installed and authenticated
- Data Collection Endpoint (DCE) already created
- Log Analytics Workspace already created
- Azure subscription with permissions to create resources

## Deployment Steps

### 1. Update Parameters

Edit `main.parameters.json` and replace the placeholder values:

```json
{
  "networkSecurityGroupName": {
    "value": "hal-harness-nsg"
  },
  "dataCollectionEndpointId": {
    "value": "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_RG/providers/Microsoft.Insights/dataCollectionEndpoints/YOUR_DCE_NAME"
  },
  "logAnalyticsWorkspaceId": {
    "value": "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_RG/providers/Microsoft.OperationalInsights/workspaces/YOUR_WORKSPACE_NAME"
  }
}
```

### 2. Deploy

```bash
# Set your resource group
RESOURCE_GROUP="your-resource-group"

# Deploy infrastructure
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file main.bicep \
  --parameters main.parameters.json
```

### 3. Get Outputs

After deployment, retrieve the DCR ID:

```bash
# Get DCR ID
az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query properties.outputs.dataCollectionRuleId.value \
  --output tsv
```

Save this DCR ID - you'll need it when running `run_docker_on_vm.py`.

### 4. Verify Deployment

```bash
# List all resources in the resource group
az resource list --resource-group $RESOURCE_GROUP --output table

# Verify DCR was created
az monitor data-collection rule show \
  --name hal-docker-logs-dcr \
  --resource-group $RESOURCE_GROUP
```

## Using the DCR

Once deployed, pass the DCR ID when running Docker on VMs:

```bash
python v2/run_docker_on_vm.py \
  --image hal-core-agent-docker:latest \
  --vm_count 1 \
  --dcr-id "/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Insights/dataCollectionRules/hal-docker-logs-dcr"
```

The DCR will be automatically associated with each VM during creation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Docker Container                                            │
│   └─ Logs to: /workspace/logs/container.log                │
└─────────────────────────────────────────────────────────────┘
                        ↓ (mounted volume)
┌─────────────────────────────────────────────────────────────┐
│ VM Filesystem                                               │
│   └─ /home/agent/hal_logs/{run_id}/{task_id}/container.log │
└─────────────────────────────────────────────────────────────┘
                        ↓ (Azure Monitor Agent watches)
┌─────────────────────────────────────────────────────────────┐
│ Data Collection Rule (DCR)                                  │
│   └─ Parses logs, extracts RunID/TaskID from path          │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Log Analytics Workspace                                     │
│   └─ ContainerLogs_CL table                                │
└─────────────────────────────────────────────────────────────┘
```

## Modules

- **main.bicep**: Main template that orchestrates deployment
- **modules/monitoring.bicep**: Data Collection Rule for log collection
- **modules/networking.bicep**: (Placeholder for future networking resources)

## Cleanup

To remove all resources created by this deployment:

```bash
# Delete DCR
az monitor data-collection rule delete \
  --name hal-docker-logs-dcr \
  --resource-group $RESOURCE_GROUP

# Delete NSG
az network nsg delete \
  --name hal-harness-nsg \
  --resource-group $RESOURCE_GROUP
```

Or delete the entire resource group (WARNING: deletes everything):

```bash
az group delete --name $RESOURCE_GROUP
```

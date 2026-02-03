# Quick Start - Deploy Infrastructure

## Prerequisites

1. Install Azure CLI: `brew install azure-cli` (macOS) or see [docs](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
2. Login: `az login`
3. Configure your `.env` file in the parent directory with Azure settings

## Deploy Base Infrastructure (NSG)

### Using the deploy script (Recommended)

```bash
cd v2/infrastructure
./deploy.sh
```

This automatically:
- Loads your `.env` configuration
- Creates the resource group if needed
- Deploys the NSG infrastructure

## Deploy VMs

### Option 1: Using the deploy-vm script (Recommended)

```bash
# Deploy a standard VM
./deploy-vm.sh my-test-vm

# Deploy a GPU VM
./deploy-vm.sh my-gpu-vm gpu
```

This automatically:
- Loads your `.env` configuration
- Deploys the VM with all networking
- Shows the public IP for SSH access

### Option 2: Manual deployment

```bash
# Load .env variables
export $(grep -v '^#' ../.env | xargs)

# Set subscription
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# Create resource group
az group create \
    --name "$AZURE_RESOURCE_GROUP_NAME" \
    --location "$AZURE_LOCATION"

# Deploy infrastructure
export SSH_PUBLIC_KEY=$(cat "$SSH_PUBLIC_KEY_PATH")
az deployment group create \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --template-file main.bicep \
    --parameters main.bicepparam
```

### Option 3: Using Python

```python
from v2.bicep_manager import BicepManager
import os

manager = BicepManager(
    resource_group=os.getenv("AZURE_RESOURCE_GROUP_NAME"),
    location=os.getenv("AZURE_LOCATION"),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    nsg_name=os.getenv("NETWORK_SECURITY_GROUP_NAME")
)

# Deploy NSG infrastructure
manager.deploy(
    template_path="main.bicep",
    deployment_name="hal-infrastructure",
    parameters={
        "networkSecurityGroupName": os.getenv("NETWORK_SECURITY_GROUP_NAME"),
    }
)

# Deploy a standard VM
ssh_key = open(os.getenv("SSH_PUBLIC_KEY_PATH")).read()
result = manager.deploy_vm(vm_name="my-test-vm", ssh_public_key=ssh_key)
print(f"VM IP: {result['public_ip']}")

# Deploy a GPU VM
result = manager.deploy_vm(vm_name="my-gpu-vm", ssh_public_key=ssh_key, gpu=True)
print(f"GPU VM IP: {result['public_ip']}")
```

## Verify Deployment

```bash
# List resources in the group
az resource list \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --output table
```

## Clean Up

### Delete a single VM

```bash
./delete-vm.sh my-test-vm
```

Or using Python:
```python
manager.delete_vm("my-test-vm")
```

### Delete everything

```bash
# Delete entire resource group
az group delete \
    --name "$AZURE_RESOURCE_GROUP_NAME" \
    --yes \
    --no-wait
```

Or using Python:
```python
manager.delete_resource_group()
```

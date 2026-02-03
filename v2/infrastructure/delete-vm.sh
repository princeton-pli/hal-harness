#!/bin/bash
set -e

# Delete a VM and all its resources
# Usage: ./delete-vm.sh <vm-name>

if [ -z "$1" ]; then
    echo "Usage: ./delete-vm.sh <vm-name>"
    exit 1
fi

VM_NAME=$1

# Load environment variables from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
else
    echo "Error: .env file not found in parent directory"
    exit 1
fi

if [ -z "$AZURE_RESOURCE_GROUP_NAME" ]; then
    echo "Error: AZURE_RESOURCE_GROUP_NAME not set in .env"
    exit 1
fi

echo "Deleting VM: $VM_NAME and all associated resources..."

# Set subscription
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# Delete VM
az vm delete \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --name "$VM_NAME" \
    --yes

# Delete NIC
az network nic delete \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --name "${VM_NAME}-nic" || true

# Delete public IP
az network public-ip delete \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --name "${VM_NAME}-public-ip" || true

# Delete VNet
az network vnet delete \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --name "${VM_NAME}-vnet" || true

# Delete OS disk
DISK_NAME=$(az disk list \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --query "[?contains(name, '$VM_NAME')].name" \
    --output tsv)

if [ -n "$DISK_NAME" ]; then
    az disk delete \
        --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
        --name "$DISK_NAME" \
        --yes || true
fi

echo "✅ VM $VM_NAME and all resources deleted"

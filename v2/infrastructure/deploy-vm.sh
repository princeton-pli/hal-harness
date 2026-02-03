#!/bin/bash
set -e

# Deploy a VM (standard or GPU)
# Usage: ./deploy-vm.sh <vm-name> [gpu]

if [ -z "$1" ]; then
    echo "Usage: ./deploy-vm.sh <vm-name> [gpu]"
    echo "Examples:"
    echo "  ./deploy-vm.sh my-test-vm       # Deploy standard VM"
    echo "  ./deploy-vm.sh my-gpu-vm gpu    # Deploy GPU VM"
    exit 1
fi

VM_NAME=$1
IS_GPU=${2:-""}

# Load environment variables from .env file
if [ -f "../.env" ]; then
    export $(grep -v '^#' ../.env | xargs)
else
    echo "Error: .env file not found in parent directory"
    exit 1
fi

# Validate required variables
if [ -z "$AZURE_SUBSCRIPTION_ID" ]; then
    echo "Error: AZURE_SUBSCRIPTION_ID not set in .env"
    exit 1
fi

if [ -z "$AZURE_RESOURCE_GROUP_NAME" ]; then
    echo "Error: AZURE_RESOURCE_GROUP_NAME not set in .env"
    exit 1
fi

if [ -z "$NETWORK_SECURITY_GROUP_NAME" ]; then
    echo "Error: NETWORK_SECURITY_GROUP_NAME not set in .env"
    exit 1
fi

if [ -z "$SSH_PUBLIC_KEY_PATH" ]; then
    echo "Error: SSH_PUBLIC_KEY_PATH not set in .env"
    exit 1
fi

# Read SSH public key
SSH_PUBLIC_KEY=$(cat "$SSH_PUBLIC_KEY_PATH")

# Get NSG ID
NSG_ID="/subscriptions/$AZURE_SUBSCRIPTION_ID/resourceGroups/$AZURE_RESOURCE_GROUP_NAME/providers/Microsoft.Network/networkSecurityGroups/$NETWORK_SECURITY_GROUP_NAME"

# Set subscription
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# Choose template based on GPU flag
if [ "$IS_GPU" = "gpu" ]; then
    echo "Deploying GPU VM: $VM_NAME"
    TEMPLATE="modules/gpu-vm-v2.bicep"
else
    echo "Deploying standard VM: $VM_NAME"
    TEMPLATE="modules/standard-vm-v2.bicep"
fi

# Deploy VM
az deployment group create \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --name "$VM_NAME-deployment" \
    --template-file "$TEMPLATE" \
    --parameters vmName="$VM_NAME" \
    --parameters networkSecurityGroupId="$NSG_ID" \
    --parameters sshPublicKey="$SSH_PUBLIC_KEY"

# Get the public IP
PUBLIC_IP=$(az deployment group show \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --name "$VM_NAME-deployment" \
    --query properties.outputs.publicIpAddress.value \
    --output tsv)

echo ""
echo "✅ VM deployed successfully!"
echo "VM Name: $VM_NAME"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Connect via SSH:"
echo "  ssh agent@$PUBLIC_IP"

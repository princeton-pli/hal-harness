#!/bin/bash
set -e

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

if [ -z "$AZURE_LOCATION" ]; then
    echo "Error: AZURE_LOCATION not set in .env"
    exit 1
fi

if [ -z "$SSH_PUBLIC_KEY_PATH" ]; then
    echo "Error: SSH_PUBLIC_KEY_PATH not set in .env"
    exit 1
fi

# Read SSH public key
export SSH_PUBLIC_KEY=$(cat "$SSH_PUBLIC_KEY_PATH")

echo "Deploying HAL Harness infrastructure..."
echo "Subscription: $AZURE_SUBSCRIPTION_ID"
echo "Resource Group: $AZURE_RESOURCE_GROUP_NAME"
echo "Location: $AZURE_LOCATION"

# Set subscription
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# Create resource group if it doesn't exist
az group create \
    --name "$AZURE_RESOURCE_GROUP_NAME" \
    --location "$AZURE_LOCATION"

# Deploy main infrastructure
az deployment group create \
    --resource-group "$AZURE_RESOURCE_GROUP_NAME" \
    --template-file main.bicep \
    --parameters main.bicepparam

echo "Deployment complete!"

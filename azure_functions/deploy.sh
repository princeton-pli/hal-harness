#!/usr/bin/env bash
set -euo pipefail

# Deploy the cost report Azure Function.
#
# Prerequisites:
#   - Azure CLI (az) logged in
#   - Azure Functions Core Tools (func) installed
#
# Usage:
#   ./deploy.sh
#
# Required env vars:
#   AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP_NAME, SLACK_WEBHOOK_URL

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP_NAME:?Set AZURE_RESOURCE_GROUP_NAME}"
LOCATION="${AZURE_LOCATION:-eastus}"
FUNC_APP_NAME="hal-cost-report"
STORAGE_ACCOUNT="halcostreport"

# Create storage account if it doesn't exist
if ! az storage account show --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "Creating storage account $STORAGE_ACCOUNT..."
    az storage account create \
        --name "$STORAGE_ACCOUNT" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Standard_LRS
fi

# Create function app if it doesn't exist
if ! az functionapp show --name "$FUNC_APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "Creating function app $FUNC_APP_NAME..."
    az functionapp create \
        --resource-group "$RESOURCE_GROUP" \
        --consumption-plan-location "$LOCATION" \
        --runtime python \
        --runtime-version 3.12 \
        --functions-version 4 \
        --name "$FUNC_APP_NAME" \
        --storage-account "$STORAGE_ACCOUNT" \
        --os-type Linux
fi

az functionapp config appsettings set \
    --name "$FUNC_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --settings \
        AZURE_SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}" \
        AZURE_RESOURCE_GROUP_NAME="${AZURE_RESOURCE_GROUP_NAME}" \
        SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}"

# Deploy
cd "$SCRIPT_DIR"
func azure functionapp publish "$FUNC_APP_NAME"

#!/bin/bash
# batch_ssh.sh — SSH into a running Azure Batch pool node.
#
# Usage: bash batch_ssh.sh [node-id]
#
# If node-id is omitted, picks the first idle/running node in the pool.

set -euo pipefail

POOL_ID="${AZURE_BATCH_POOL_ID:-proof-of-concept}"
BATCH_ACCOUNT="${AZURE_BATCH_ACCOUNT_NAME:-halharness}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-hal_group}"
SSH_USER="debuguser"
SSH_PASS="Debug$(openssl rand -hex 8)!"
EXPIRY=$(date -u -v+2H "+%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u -d "+2 hours" "+%Y-%m-%dT%H:%M:%SZ")

# Log in to Batch account
az batch account login \
  --name "$BATCH_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --shared-key-auth

# Pick node
if [[ $# -ge 1 ]]; then
  NODE_ID="$1"
else
  echo "No node-id provided — picking first available node..."
  NODE_ID=$(az batch node list \
    --pool-id "$POOL_ID" \
    --query "[?state=='idle' || state=='running'].id | [0]" \
    -o tsv)
  if [[ -z "$NODE_ID" ]]; then
    echo "No idle/running nodes found in pool '$POOL_ID'. Is the pool scaled up?"
    exit 1
  fi
fi

echo "Node: $NODE_ID"

# Create temporary user (delete first if already exists)
az batch node user delete --pool-id "$POOL_ID" --node-id "$NODE_ID" --name "$SSH_USER" --yes 2>/dev/null || true
az batch node user create \
  --pool-id "$POOL_ID" \
  --node-id "$NODE_ID" \
  --name "$SSH_USER" \
  --password "$SSH_PASS" \
  --expiry-time "$EXPIRY"

# Get connection details
IP=$(az batch node remote-login-settings show \
  --pool-id "$POOL_ID" \
  --node-id "$NODE_ID" \
  --query "remoteLoginIPAddress" -o tsv)
PORT=$(az batch node remote-login-settings show \
  --pool-id "$POOL_ID" \
  --node-id "$NODE_ID" \
  --query "remoteLoginPort" -o tsv)

echo ""
echo "Connecting to $IP:$PORT as $SSH_USER"
echo "Password: $SSH_PASS"
echo ""

ssh -o StrictHostKeyChecking=no -p "$PORT" "$SSH_USER@$IP"

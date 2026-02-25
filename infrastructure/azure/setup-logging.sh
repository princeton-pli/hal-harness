#!/bin/bash
# Azure Monitor Logging Infrastructure Setup Script
# This script creates all required Azure resources for HAL harness logging

# FIXME: this should be replaced by a terraform or similar script to ensure
# reusability

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP_NAME:-HAL_GROUP}"
LOCATION="${AZURE_LOCATION:-eastus}"
# FIXME: remove staging from names
WORKSPACE_NAME="staging-hal-logs-workspace"
DCE_NAME="staging-hal-logs-dce"
DCR_NAME="staging-hal-logs-dcr"
TABLE_NAME="BenchmarkRuns_CL"
STREAM_NAME="Custom-BenchmarkRuns_CL"

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ ${1}${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ ${1}${NC}"
}

print_error() {
    echo -e "${RED}✗ ${1}${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}${1}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first:"
    echo "  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
print_info "Checking Azure login status..."
if ! az account show &> /dev/null; then
    print_error "Not logged in to Azure. Running 'az login'..."
    az login
fi

print_success "Azure CLI is authenticated"

# Configure Azure CLI to auto-install extensions without prompting
print_info "Configuring Azure CLI extensions..."
az config set extension.use_dynamic_install=yes_without_prompt &> /dev/null || true
az config set extension.dynamic_install_allow_preview=true &> /dev/null || true

# Install required extension if not present
if ! az extension show --name monitor-control-service &> /dev/null; then
    print_info "Installing monitor-control-service extension..."
    az extension add --name monitor-control-service --yes &> /dev/null
fi

print_success "Azure CLI configured"

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
print_info "Using subscription: ${SUBSCRIPTION_ID}"

print_header "Step 1: Create Log Analytics Workspace"

# Check if workspace already exists
if az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" &> /dev/null; then
    print_warning "Log Analytics Workspace '${WORKSPACE_NAME}' already exists"
else
    print_info "Creating Log Analytics Workspace '${WORKSPACE_NAME}'..."
    az monitor log-analytics workspace create \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --location "$LOCATION" \
        --query id -o tsv
    print_success "Log Analytics Workspace created"
fi

WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --query id -o tsv)

print_success "Workspace ID: ${WORKSPACE_ID}"

print_header "Step 2: Create Custom Table"

print_info "Creating custom table '${TABLE_NAME}'..."

# Create table schema JSON
TABLE_SCHEMA=$(cat <<EOF
{
  "properties": {
    "schema": {
      "name": "${TABLE_NAME}",
      "columns": [
        {"name": "TimeGenerated", "type": "datetime"},
        {"name": "Level", "type": "string"},
        {"name": "Message", "type": "string"},
        {"name": "LoggerName", "type": "string"},
        {"name": "RunID", "type": "string"},
        {"name": "Benchmark", "type": "string"},
        {"name": "AgentName", "type": "string"},
        {"name": "TaskID", "type": "string"},
        {"name": "LogType", "type": "string"},
        {"name": "VMName", "type": "string"},
        {"name": "ExecutionMode", "type": "string"},
        {"name": "ExecutedBy", "type": "string"},
        {"name": "Properties", "type": "dynamic"}
      ]
    }
  }
}
EOF
)

# Check if table exists
if az monitor log-analytics workspace table show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --name "${TABLE_NAME}" &> /dev/null; then
    print_warning "Table '${TABLE_NAME}' already exists"
else
    # Create the custom table
    echo "$TABLE_SCHEMA" > /tmp/table-schema.json
    az monitor log-analytics workspace table create \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --name "${TABLE_NAME}" \
        --columns TimeGenerated=datetime Level=string Message=string LoggerName=string \
                  RunID=string Benchmark=string AgentName=string TaskID=string \
                  LogType=string VMName=string ExecutionMode=string ExecutedBy=string \
                  Properties=dynamic
    rm /tmp/table-schema.json
    print_success "Custom table created"
fi

print_header "Step 3: Create Data Collection Endpoint (DCE)"

# Check if DCE already exists
if az monitor data-collection endpoint show \
    --name "$DCE_NAME" \
    --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    print_warning "Data Collection Endpoint '${DCE_NAME}' already exists"
else
    print_info "Creating Data Collection Endpoint '${DCE_NAME}'..."
    az monitor data-collection endpoint create \
        --name "$DCE_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --public-network-access Enabled
    print_success "Data Collection Endpoint created"
fi

DCE_ID=$(az monitor data-collection endpoint show \
    --name "$DCE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query id -o tsv)

DCE_ENDPOINT=$(az monitor data-collection endpoint show \
    --name "$DCE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query logsIngestion.endpoint -o tsv)

print_success "DCE Endpoint: ${DCE_ENDPOINT}"

print_header "Step 4: Create Data Collection Rule (DCR)"

# Check if DCR already exists
if az monitor data-collection rule show \
    --name "$DCR_NAME" \
    --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    print_warning "Data Collection Rule '${DCR_NAME}' already exists"
else
    print_info "Creating Data Collection Rule '${DCR_NAME}'..."

    # Create DCR configuration
    DCR_CONFIG=$(cat <<EOF
{
  "location": "${LOCATION}",
  "properties": {
    "dataCollectionEndpointId": "${DCE_ID}",
    "streamDeclarations": {
      "${STREAM_NAME}": {
        "columns": [
          {"name": "TimeGenerated", "type": "datetime"},
          {"name": "Level", "type": "string"},
          {"name": "Message", "type": "string"},
          {"name": "LoggerName", "type": "string"},
          {"name": "RunID", "type": "string"},
          {"name": "Benchmark", "type": "string"},
          {"name": "AgentName", "type": "string"},
          {"name": "TaskID", "type": "string"},
          {"name": "LogType", "type": "string"},
          {"name": "VMName", "type": "string"},
          {"name": "ExecutionMode", "type": "string"},
          {"name": "ExecutedBy", "type": "string"},
          {"name": "Properties", "type": "dynamic"}
        ]
      }
    },
    "destinations": {
      "logAnalytics": [
        {
          "workspaceResourceId": "${WORKSPACE_ID}",
          "name": "hal-workspace"
        }
      ]
    },
    "dataFlows": [
      {
        "streams": ["${STREAM_NAME}"],
        "destinations": ["hal-workspace"],
        "transformKql": "source",
        "outputStream": "${STREAM_NAME}"
      }
    ]
  }
}
EOF
)

    echo "$DCR_CONFIG" > /tmp/dcr-config.json
    az monitor data-collection rule create \
        --name "$DCR_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --rule-file /tmp/dcr-config.json
    rm /tmp/dcr-config.json
    print_success "Data Collection Rule created"
fi

DCR_ID=$(az monitor data-collection rule show \
    --name "$DCR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query immutableId -o tsv)

# Get the full resource ID (needed for role assignment)
DCR_RESOURCE_ID=$(az monitor data-collection rule show \
    --name "$DCR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query id -o tsv)

print_success "DCR ID: ${DCR_ID}"

print_header "Step 5: Assign Permissions"

# Get current user/service principal
CURRENT_USER=$(az account show --query user.name -o tsv)
print_info "Assigning 'Monitoring Metrics Publisher' role to ${CURRENT_USER}..."

# Get the object ID (works for both user and service principal)
OBJECT_ID=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || az account show --query user.name -o tsv)

# FIXME: this role assignment doesn't seem to work, we probably need
# to set up a Service Principal
# Assign role to DCR using the full resource ID
az role assignment create \
    --role "Monitoring Metrics Publisher" \
    --assignee "$OBJECT_ID" \
    --scope "$DCR_RESOURCE_ID" \
    --output none 2>/dev/null || print_warning "Role assignment may already exist"

print_success "Permissions assigned"

print_header "Setup Complete!"

print_success "All Azure resources have been created successfully!"
echo ""
print_info "Add these values to your .env file:"
echo ""
echo "AZURE_MONITOR_DATA_COLLECTION_ENDPOINT=${DCE_ENDPOINT}"
echo "AZURE_MONITOR_DATA_COLLECTION_RULE_ID=${DCR_ID}"
echo "AZURE_MONITOR_STREAM_NAME=${STREAM_NAME}"
echo "EXECUTED_BY=YourName"
echo ""
print_info "Query your logs in Azure Portal:"
echo "https://portal.azure.com/#@/resource${WORKSPACE_ID}/logs"
echo ""
print_warning "Note: It may take 5-10 minutes for the table to be ready for ingestion."

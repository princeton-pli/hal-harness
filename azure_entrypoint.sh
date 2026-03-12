#!/bin/bash
# azure_entrypoint.sh — Entry point for Azure Batch eval tasks.
#
# Azure Batch downloads hal-harness.zip as a ResourceFile, then the task command
# unzips it and runs this script from within the repo root.
#
# Usage: bash azure_entrypoint.sh <agent_name> <benchmark> <task_id> <model>
#
# Inspired by hal/utils/setup_vm.sh

set -euo pipefail

AGENT_NAME="${1:?Usage: azure_entrypoint.sh <agent_name> <benchmark> <task_id> <model>}"
BENCHMARK="${2:?}"
TASK_ID="${3:?}"
MODEL="${4:?}"

echo "=== hal-harness Azure Batch entrypoint ==="
echo "agent=$AGENT_NAME benchmark=$BENCHMARK task_id=$TASK_ID model=$MODEL"
echo "python=$(python3 --version)"

echo "Installing pip..."
# Bootstrap pip if not available, then install the harness
python3 -m pip --version 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | python3
echo "Installing packages..."
python3 -m pip install --quiet -e ".[dev]"

echo "Running hal.cli..."
python3 -m hal.cli \
  --agent_name "$AGENT_NAME" \
  --benchmark "$BENCHMARK" \
  --task_ids "$TASK_ID" \
  -A "model_name=$MODEL"

#!/bin/bash
# azure_entrypoint.sh — Entry point for Azure Batch eval tasks.
#
# Azure Batch downloads hal-harness.zip as a ResourceFile, the task command
# unzips it and runs this script. The script cds into the repo root so all
# relative paths (agent_dir, results/) resolve correctly.
#
# Usage: bash azure_entrypoint.sh <agent_name> <agent_function> <agent_dir> <benchmark> <task_id> <model>

set -euo pipefail

AGENT_NAME="${1:?Usage: azure_entrypoint.sh <agent_name> <agent_function> <agent_dir> <benchmark> <task_id> <model>}"
AGENT_FUNCTION="${2:?}"
AGENT_DIR="${3:?}"
BENCHMARK="${4:?}"
TASK_ID="${5:?}"
MODEL="${6:?}"

# Run from inside the repo so pip install -e . and relative agent_dir work
cd "$(dirname "$0")"

echo "=== hal-harness Azure Batch entrypoint ==="
echo "agent=$AGENT_NAME function=$AGENT_FUNCTION dir=$AGENT_DIR"
echo "benchmark=$BENCHMARK task_id=$TASK_ID model=$MODEL"
echo "python=$(python3 --version)"

echo "Installing packages..."
python3 -m pip install --quiet --break-system-packages -e ".[dev]"

if [ -f "$AGENT_DIR/requirements.txt" ]; then
  echo "Installing agent requirements from $AGENT_DIR/requirements.txt..."
  python3 -m pip install --quiet --break-system-packages -r "$AGENT_DIR/requirements.txt"
fi

# Ensure `python` resolves to python3 (not present by default on Ubuntu 24.04)
mkdir -p ~/.local/bin
ln -sf "$(which python3)" ~/.local/bin/python
export PATH="$HOME/.local/bin:$PATH"

echo "Running hal.cli..."
python3 -m hal.cli \
  --agent_name "$AGENT_NAME" \
  --agent_function "$AGENT_FUNCTION" \
  --agent_dir "$AGENT_DIR" \
  --benchmark "$BENCHMARK" \
  --task_ids "$TASK_ID" \
  -A "model_name=$MODEL"

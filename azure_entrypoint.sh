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

# For corebench, the per-task capsule was downloaded as a ResourceFile next to
# hal-harness.zip (one level up from this repo). Extract it into the capsules dir
# so corebench.py finds it on disk and skips its (broken, partial) auto-download.
CAPSULE_TARBALL="../${TASK_ID}.tar.gz"
if [ -f "$CAPSULE_TARBALL" ]; then
  echo "Extracting capsule $TASK_ID from $CAPSULE_TARBALL..."
  mkdir -p hal/benchmarks/corebench/capsules
  tar xzf "$CAPSULE_TARBALL" -C hal/benchmarks/corebench/capsules
fi

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
# Build a unique run_id so concurrent tasks don't collide in Weave.
# Sanitize model name: lowercase, replace non-alphanumeric with underscores.
SANITIZED_MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
RUN_ID="${BENCHMARK}_${AGENT_NAME}_${SANITIZED_MODEL}_task${TASK_ID}_$(date +%s)"

python3 -m hal.cli \
  --agent_name "$AGENT_NAME" \
  --agent_function "$AGENT_FUNCTION" \
  --agent_dir "$AGENT_DIR" \
  --benchmark "$BENCHMARK" \
  --task_ids "$TASK_ID" \
  --run_id "$RUN_ID" \
  -A "model_name=$MODEL"

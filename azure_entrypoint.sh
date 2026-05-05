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

# Extra agent kwargs are passed via HAL_AGENT_ARG_<key>=<value> env vars,
# set per-task by prefect/batch.py from spec.agent_args. The loop below
# collects them into -A flags for hal-eval.

# Pin HOME to the Azure Batch task working directory (one level up from this
# script, which lives inside the unzipped repo). The Batch auto-user's default
# $HOME varies with elevation scope; forcing it here gives pip, weave, and
# anything else using ~ a consistent, writable, per-task location.
export HOME="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HOME/.cache"

# Pin TMPDIR and the Weave disk-cache dir to paths under $HOME. Weave's default
# is tempfile.mkdtemp() which fails to open a sqlite DB under certain Azure
# Batch auto-user configurations ("unable to open database file"). Being
# explicit here sidesteps that entire class of issue.
export TMPDIR="$HOME/tmp"
export WEAVE_SERVER_CACHE_DIR="$HOME/.cache/weave"
mkdir -p "$TMPDIR" "$WEAVE_SERVER_CACHE_DIR"

# OpenCode's google provider expects GOOGLE_GENERATIVE_AI_API_KEY.
# Map from GEMINI_API_KEY which is what we forward via TASK_ENV_VARS.
if [ -n "${GEMINI_API_KEY:-}" ]; then
  export GOOGLE_GENERATIVE_AI_API_KEY="$GEMINI_API_KEY"
  export GOOGLE_API_KEY="$GEMINI_API_KEY"
  echo "Mapped GEMINI_API_KEY → GOOGLE_GENERATIVE_AI_API_KEY + GOOGLE_API_KEY"
fi

# Run from inside the repo so pip install -e . and relative agent_dir work
cd "$(dirname "$0")"

echo "=== hal-harness Azure Batch entrypoint ==="
echo "HOME=$HOME"
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
  # Remove the tarball so the agent can't bypass the file filter by reading
  # it directly. The agent has full filesystem access (codex runs with
  # --dangerously-bypass-approvals-and-sandbox) and will find and untar it
  # to read pre-computed results/ if left on disk.
  rm -f "$CAPSULE_TARBALL"
  # Also strip results/ directories from the extracted capsule. CoreBenchHard
  # filters these out of the agent's ./environment/ via _get_capsule_files_dict,
  # but the capsule source dir remains on disk and agents can browse to it
  # directly (data leakage — see capsule-6800638 incident).
  find "hal/benchmarks/corebench/capsules/${TASK_ID}" -type d -name "results" -exec rm -rf {} + 2>/dev/null || true
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
# Weave project names have a 128-char limit. Use a short prefix + the uuid
# suffix from $AZ_BATCH_TASK_ID for uniqueness. The full model/capsule info
# is already in the UPLOAD.json config block and in blob storage paths.
SANITIZED_MODEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
# Extract just the 8-char uuid suffix from the Azure task ID (last segment after final -)
AZ_SUFFIX=$(echo "${AZ_BATCH_TASK_ID:-$(date +%s%N)}" | grep -oE '[a-f0-9]{8}$' || date +%s%N)
RUN_ID="${BENCHMARK}_${AGENT_NAME}_${SANITIZED_MODEL}_${TASK_ID}_${AZ_SUFFIX}"
# Enforce 128-char cap (truncate from the middle if needed, preserving suffix for uniqueness)
if [ ${#RUN_ID} -gt 128 ]; then
  RUN_ID="${RUN_ID:0:119}_${AZ_SUFFIX}"
fi

HAL_CLI_ARGS=(
  --agent_name "$AGENT_NAME"
  --agent_function "$AGENT_FUNCTION"
  --agent_dir "$AGENT_DIR"
  --benchmark "$BENCHMARK"
  --task_ids "$TASK_ID"
  --run_id "$RUN_ID"
  -A "model_name=$MODEL"
)

# If HAL_TASK_TIMEOUT is set (via env var from batch.py), override the default timeout.
if [ -n "${HAL_TASK_TIMEOUT:-}" ]; then
  HAL_CLI_ARGS+=(--task_timeout "$HAL_TASK_TIMEOUT")
  echo "Task timeout: ${HAL_TASK_TIMEOUT}s"
fi

# Forward any HAL_AGENT_ARG_<key>=<value> env vars as -A args to hal-eval.
while IFS= read -r var; do
  key="${var#HAL_AGENT_ARG_}"
  val="${!var}"
  HAL_CLI_ARGS+=(-A "$key=$val")
  echo "Forwarding agent arg: $key=$val"
done < <(compgen -e | grep '^HAL_AGENT_ARG_' || true)

python3 -m hal.cli "${HAL_CLI_ARGS[@]}"

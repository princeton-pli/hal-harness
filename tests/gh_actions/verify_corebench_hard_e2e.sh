#!/usr/bin/env bash
# Validate CoreBench hard VM e2e run directory and *_UPLOAD.json (schema alongside this script).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_DIR="${ROOT}/results/corebench_hard"
SCHEMA="${SCRIPT_DIR}/corebench_hard_upload.schema.json"

if [[ ! -d "$BENCH_DIR" ]]; then
  echo "::error::Expected benchmark results directory missing: $BENCH_DIR" >&2
  exit 1
fi

# Newest run directory (default hal-eval run_id: corebench_hard_<sanitized_agent>_<unix_time>).
shopt -s nullglob
dirs=("$BENCH_DIR"/corebench_hard_*)
shopt -u nullglob
run_dir=""
if ((${#dirs[@]})); then
  run_dir="$(ls -td "${dirs[@]}" | head -n1)"
fi

if [[ -z "${run_dir:-}" || ! -d "$run_dir" ]]; then
  echo "::error::No run directory matching results/corebench_hard/corebench_hard_*" >&2
  exit 1
fi

run_id="$(basename "$run_dir")"
upload="${run_dir}/${run_id}_UPLOAD.json"

echo "Verifying e2e artifacts under: $run_id"

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "::error::Missing required file: $f" >&2
    exit 1
  fi
}

require_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "::error::Missing required directory: $d" >&2
    exit 1
  fi
}

require_file "$upload"
require_file "${run_dir}/${run_id}.json"
require_file "${run_dir}/${run_id}.log"
require_file "${run_dir}/${run_id}_RAW_SUBMISSIONS.jsonl"

require_dir "${run_dir}/agent_logs"
if ! compgen -G "${run_dir}/agent_logs/*_log.log" >/dev/null; then
  echo "::error::Expected at least one agent_logs/*_log.log under $run_dir" >&2
  exit 1
fi

if ! find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'capsule-*' -print -quit | grep -q .; then
  echo "::error::Expected a per-task capsule-* directory under $run_dir (VM result copy)" >&2
  exit 1
fi

if ! find "$run_dir" -mindepth 1 -maxdepth 1 -type f -name 'setup_vm_log_*.log' -print -quit | grep -q .; then
  echo "::error::Expected setup_vm_log_*.log under $run_dir" >&2
  exit 1
fi

python3 - "$SCHEMA" "$upload" <<'PY'
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

schema_path, upload_path = Path(sys.argv[1]), Path(sys.argv[2])
schema = json.loads(schema_path.read_text(encoding="utf-8"))
instance = json.loads(upload_path.read_text(encoding="utf-8"))
Draft202012Validator(schema).validate(instance)
PY

echo "CoreBench hard e2e artifact checks passed."

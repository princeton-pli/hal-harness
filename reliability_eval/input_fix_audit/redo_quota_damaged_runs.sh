#!/usr/bin/env bash
# Re-run the 4 GAIA runs that lost tasks to OpenAI `insufficient_quota`.
#
# Each invocation uses `--continue_run --run_id <existing>` so completed tasks
# are skipped and only the tasks that crashed before producing output.json are
# re-executed. Arguments are reconstructed from each run's `agent_args.json`
# and the matching phase in `reliability_eval/phases/runner.py`.
#
# BEFORE RUNNING:
#   1. Top up your OpenAI account so `insufficient_quota` won't fire again.
#   2. From the repo root, activate the venv:  source .venv/bin/activate
#   3. Run individually (preferred — easier to monitor) or in sequence.
#
# To kick off all four in one shot, run this script:
#   bash reliability_eval/input_fix_audit/redo_quota_damaged_runs.sh

set -uo pipefail

cd "$(dirname "$0")/../.."

# Shared args used by every run
COMMON_BASELINE_ARGS=(
    -A compute_confidence=true
    -A store_confidence_details=true
    -A store_conversation_history=true
    -A enable_compliance_monitoring=true
    -A "compliance_constraints=pii_handling_gaia,no_destructive_ops,safe_code_execution,no_unauthorized_access"
)

run_hal_eval() {
    local run_id="$1"; shift
    echo
    echo "=========================================================="
    echo "Continuing run: $run_id"
    echo "=========================================================="
    hal-eval \
        --benchmark gaia \
        --agent_dir agents/hal_generalist_agent \
        --agent_function main.run \
        --max_concurrent 20 \
        --task_timeout 1800 \
        --results_dir results_fixed \
        --run_id "$run_id" \
        --continue_run \
        "$@"
}

# 1) gpt_o1 baseline rep1 — 100 / 165 lost
run_hal_eval gaia_gaia_generalist_gpt_o1_rep1_1779175297 \
    --agent_name gaia_generalist_gpt_o1 \
    -A model_name=o1-2024-12-17 \
    -A provider=openai \
    -A benchmark_name=gaia \
    -A temperature=0.0 \
    "${COMMON_BASELINE_ARGS[@]}"

# 2) gpt_o1 baseline rep2 — 40 / 165 lost
run_hal_eval gaia_gaia_generalist_gpt_o1_rep2_1779143377 \
    --agent_name gaia_generalist_gpt_o1 \
    -A model_name=o1-2024-12-17 \
    -A provider=openai \
    -A benchmark_name=gaia \
    -A temperature=0.0 \
    "${COMMON_BASELINE_ARGS[@]}"

# 3) gpt_5_2 fault-injection rep1 — 60 / 165 lost
run_hal_eval gaia_gaia_generalist_gpt_5_2_fault_20pct_rep1_1779195701 \
    --agent_name gaia_generalist_gpt_5_2_fault_20pct \
    -A model_name=gpt-5.2-2025-12-11 \
    -A provider=openai \
    -A benchmark_name=gaia \
    -A temperature=0.0 \
    -A enable_fault_injection=true \
    -A fault_rate=0.2 \
    -A track_recovery=true

# 4) gpt_4_turbo structural-perturbation medium — 55 / 146 lost
#    Note: this run only seeded 146 task dirs before crashing. --continue_run
#    will still finish the missing ones if hal-eval is run with the same
#    --run_id; the harness rebuilds the task list from the (sanitised)
#    dataset on resume.
run_hal_eval gaia_gaia_generalist_gpt_4_turbo_struct_medium_1779240897 \
    --agent_name gaia_generalist_gpt_4_turbo_struct_medium \
    -A model_name=gpt-4-turbo-2024-04-09 \
    -A provider=openai \
    -A benchmark_name=gaia \
    -A temperature=0.0 \
    -A enable_structural_perturbations=true \
    -A perturbation_strength=medium \
    -A perturbation_type=all

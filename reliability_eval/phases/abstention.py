"""Abstention phase runner for reliability_eval."""

import json
from datetime import datetime
from pathlib import Path

from reliability_eval.metrics.abstention import detect_abstention
from reliability_eval.types import EvaluationLog, RunResult


# =============================================================================
# CONFIGURATION - Edit config.py to customize your evaluation
# =============================================================================


# =============================================================================
# DATA CLASSES
# =============================================================================


def run_abstention_phase(
    combinations: list[tuple],
    results_dir: Path,
    log: EvaluationLog,
    log_path: Path,
) -> int:
    """
    Run abstention detection on existing traces.

    This phase:
    1. Finds all existing result files for the configured agents/benchmarks
    2. For each task, extracts conversation_history and taken_actions
    3. Runs regex-based abstention detection
    4. Writes results back into the JSON files under 'abstention' key

    Computes: Abstention rate, type distribution, correlation with success/failure
    """
    print("\n" + "=" * 80)
    print("🛑 PHASE: ABSTENTION DETECTION")
    print("=" * 80)
    print(f"   Results dir: {results_dir}")

    total_tasks_analyzed = 0
    total_files_updated = 0
    abstention_summary = {
        "total_abstained": 0,
        "by_type": {
            "inability": 0,
            "uncertainty": 0,
            "clarification": 0,
            "refusal": 0,
            "none": 0,
        },
        "abstained_and_failed": 0,
        "abstained_and_succeeded": 0,
        "not_abstained_and_failed": 0,
        "not_abstained_and_succeeded": 0,
    }

    for agent_config, benchmark_config, bench_name in combinations:
        agent_name = agent_config["name"]

        print(f"\n{'─' * 60}")
        print(f"🔍 Analyzing: {agent_name} on {bench_name}")
        print(f"{'─' * 60}")

        # Find all result directories for this agent/benchmark
        benchmark_dir = results_dir / bench_name
        if not benchmark_dir.exists():
            print(f"   ⚠️  No results directory found: {benchmark_dir}")
            continue

        # Find matching run directories
        for run_dir in sorted(benchmark_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Check if this run matches our agent
            if agent_name not in run_dir.name:
                continue

            # Skip non-baseline results (fault, structural, prompt_sensitivity)
            run_dir_name = run_dir.name.lower()
            if any(
                phase in run_dir_name
                for phase in [
                    "fault",
                    "struct",
                    "structural",
                    "prompt_sensitivity",
                    "prompt_mild",
                    "prompt_medium",
                    "prompt_strong",
                    "prompt_naturalistic",
                ]
            ):
                continue

            # Find the UPLOAD.json file
            upload_files = list(run_dir.glob("*_UPLOAD.json"))
            if not upload_files:
                continue

            upload_file = upload_files[0]
            print(f"\n   📄 Processing: {upload_file.name}")

            # Load the results
            try:
                with open(upload_file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"      ❌ Failed to load: {e}")
                continue

            raw_eval = data.get("raw_eval_results", {})
            if not raw_eval:
                print("      ⚠️  No raw_eval_results found")
                continue

            tasks_in_file = 0
            modified = False

            for task_id, task_eval in raw_eval.items():
                if not isinstance(task_eval, dict):
                    continue

                # Always recompute abstention (replace existing data if present)
                # Get conversation history and actions
                conversation_history = task_eval.get("conversation_history", [])
                taken_actions = task_eval.get("taken_actions", [])

                if not conversation_history:
                    # Try to get from other possible locations
                    if "messages" in task_eval:
                        conversation_history = task_eval["messages"]

                if not conversation_history and not taken_actions:
                    continue

                success = float(task_eval.get("reward", 0.0)) > 0

                print(f"      🔬 Task {task_id}: Analyzing...", end=" ", flush=True)

                try:
                    # Run abstention detection
                    abstention_result = detect_abstention(
                        conversation_history=conversation_history,
                        actions_taken=taken_actions,
                    )

                    # Store back in task
                    task_eval["abstention"] = {
                        "abstained": abstention_result["abstained"],
                        "abstention_type": abstention_result["abstention_type"],
                        "abstention_strength": abstention_result["abstention_strength"],
                        "early_termination": abstention_result["early_termination"],
                        "evidence": abstention_result["evidence"],
                        "scores_by_type": abstention_result["scores_by_type"],
                    }

                    modified = True
                    tasks_in_file += 1
                    total_tasks_analyzed += 1

                    # Update summary
                    if abstention_result["abstained"]:
                        abstention_summary["total_abstained"] += 1
                        abstention_summary["by_type"][
                            abstention_result["abstention_type"]
                        ] += 1
                        if success:
                            abstention_summary["abstained_and_succeeded"] += 1
                        else:
                            abstention_summary["abstained_and_failed"] += 1
                        print(
                            f"🛑 {abstention_result['abstention_type']} (strength={abstention_result['abstention_strength']:.2f})"
                        )
                    else:
                        abstention_summary["by_type"]["none"] += 1
                        if success:
                            abstention_summary["not_abstained_and_succeeded"] += 1
                        else:
                            abstention_summary["not_abstained_and_failed"] += 1
                        print("✅ no abstention")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    task_eval["abstention"] = {
                        "abstained": None,
                        "error": str(e),
                    }
                    modified = True

            # Save back to file if modified
            if modified:
                try:
                    with open(upload_file, "w") as f:
                        json.dump(data, f, indent=2)
                    print(
                        f"   💾 Saved {tasks_in_file} task analyses to {upload_file.name}"
                    )
                    total_files_updated += 1
                except Exception as e:
                    print(f"   ❌ Failed to save: {e}")

        # Log result for this agent
        result = RunResult(
            agent=agent_name,
            benchmark=bench_name,
            phase="abstention",
            repetition=1,
            success=True,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
            error_message=None,
        )
        log.add_result(result)
        log.save(log_path)

    # Print summary
    print("\n✨ Abstention phase complete:")
    print(f"   📊 Tasks analyzed: {total_tasks_analyzed}")
    print(f"   📁 Files updated: {total_files_updated}")
    print("\n   📈 Abstention Summary:")
    print(f"      Total abstained: {abstention_summary['total_abstained']}")
    print(f"      By type: {abstention_summary['by_type']}")
    print("\n   🎯 Correlation with success:")
    print(f"      Abstained + Failed:     {abstention_summary['abstained_and_failed']}")
    print(
        f"      Abstained + Succeeded:  {abstention_summary['abstained_and_succeeded']}"
    )
    print(
        f"      No abstention + Failed: {abstention_summary['not_abstained_and_failed']}"
    )
    print(
        f"      No abstention + Succeeded: {abstention_summary['not_abstained_and_succeeded']}"
    )

    # Compute calibration metrics if we have data
    total = (
        abstention_summary["abstained_and_failed"]
        + abstention_summary["abstained_and_succeeded"]
        + abstention_summary["not_abstained_and_failed"]
        + abstention_summary["not_abstained_and_succeeded"]
    )
    if total > 0 and abstention_summary["total_abstained"] > 0:
        # Precision: P(fail | abstain)
        precision = (
            abstention_summary["abstained_and_failed"]
            / abstention_summary["total_abstained"]
            if abstention_summary["total_abstained"] > 0
            else 0
        )
        # Recall: P(abstain | fail)
        total_failed = (
            abstention_summary["abstained_and_failed"]
            + abstention_summary["not_abstained_and_failed"]
        )
        recall = (
            abstention_summary["abstained_and_failed"] / total_failed
            if total_failed > 0
            else 0
        )
        print("\n   📊 Abstention Calibration:")
        print(f"      Precision (P(fail|abstain)): {precision:.2%}")
        print(f"      Recall (P(abstain|fail)):    {recall:.2%}")

    return total_tasks_analyzed


# =============================================================================
# MAIN
# =============================================================================

"""Safety analysis phase runner."""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reliability_eval.types import EvaluationLog, RunResult


def run_safety_phase(
    combinations: List[tuple],
    results_dir: Path,
    safety_model: str,
    default_constraints: List[str],
    log: EvaluationLog,
    log_path: Path,
    max_reps: Optional[int] = None,
    max_concurrent: int = 1,
) -> int:
    """
    Run LLM-based safety analysis on existing traces.

    This phase:
    1. Finds all existing result files for the configured agents/benchmarks
    2. For each task, extracts conversation_history and taken_actions
    3. Calls LLM to analyze for harm severity and compliance violations
    4. Writes results back into the JSON files under 'llm_safety' key

    Uses benchmark-specific compliance_constraints when available,
    otherwise falls back to default_constraints.

    Computes: S_harm (error severity), S_comp (compliance)
    """
    print("\n" + "=" * 80)
    print("🛡️  PHASE: SAFETY ANALYSIS (S_harm, S_comp)")
    print("=" * 80)
    print(f"   Model: {safety_model}")
    print(f"   Default constraints: {', '.join(default_constraints)}")
    print(f"   Results dir: {results_dir}")
    print(f"   Max concurrent: {max_concurrent}")
    if max_reps is not None:
        print(
            f"   Filter: Analyzing first {max_reps} baseline run(s) (rep1-rep{max_reps})"
        )

    try:
        from hal.utils.llm_log_analyzer import LLMLogAnalyzer
    except ImportError as e:
        print(f"\n❌ Failed to import LLMLogAnalyzer: {e}")
        print("   Make sure hal.utils.llm_log_analyzer is available")
        return 0

    # Initialize analyzer
    analyzer = LLMLogAnalyzer(model=safety_model, cache_responses=True)

    total_tasks_analyzed = 0
    total_files_updated = 0

    for agent_config, benchmark_config, bench_name in combinations:
        agent_name = agent_config["name"]

        # Use benchmark-specific constraints if available, otherwise default
        constraints = benchmark_config.get(
            "compliance_constraints", default_constraints
        )

        print(f"\n{'─' * 60}")
        print(f"🔍 Analyzing: {agent_name} on {bench_name}")
        print(f"   Constraints: {', '.join(constraints)}")
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

            # If max_reps is set, only analyze rep1 through repN
            if max_reps is not None:
                # Check if this run matches _rep1_ through _rep{max_reps}_
                match = re.search(r"_rep(\d+)_", run_dir.name)
                if not match or int(match.group(1)) > max_reps:
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

            # Collect tasks that need analysis
            tasks_to_analyze = []
            for task_id, task_eval in raw_eval.items():
                if not isinstance(task_eval, dict):
                    continue

                # Get conversation history and actions
                conversation_history = task_eval.get("conversation_history", [])
                taken_actions = task_eval.get("taken_actions", [])

                if not conversation_history and not taken_actions:
                    print(f"      ⚠️  Task {task_id}: No trace data")
                    continue

                success = int(task_eval.get("reward", 0.0))
                tasks_to_analyze.append(
                    {
                        "task_id": task_id,
                        "conversation_history": conversation_history,
                        "taken_actions": taken_actions,
                        "success": success,
                    }
                )

            if not tasks_to_analyze:
                print("      ℹ️  No tasks to analyze")
                continue

            print(
                f"      🔬 Analyzing {len(tasks_to_analyze)} tasks (max_concurrent={max_concurrent})..."
            )

            # Helper function to analyze a single task
            def analyze_task(task_info):
                task_id = task_info["task_id"]
                try:
                    # Analyze compliance (for all tasks)
                    compliance_result = analyzer.analyze_compliance(
                        conversation_history=task_info["conversation_history"],
                        actions_taken=task_info["taken_actions"],
                        constraints=constraints,
                    )

                    # Analyze error severity (only for failed tasks)
                    severity_result = None
                    if task_info["success"] == 0:
                        severity_result = analyzer.analyze_error_severity(
                            conversation_history=task_info["conversation_history"],
                            actions_taken=task_info["taken_actions"],
                            task_result={"success": False, "task_id": task_id},
                        )

                    # Build the llm_safety result
                    llm_safety = {
                        "analyzed": True,
                        "model": safety_model,
                        "timestamp": datetime.now().isoformat(),
                        # Compliance results
                        "S_comp": compliance_result.S_comp,
                        "compliance_violations": [
                            {
                                "constraint": v.constraint,
                                "severity": v.severity,
                                "evidence": v.evidence,
                                "explanation": v.explanation,
                            }
                            for v in compliance_result.violations
                        ],
                        "num_violations": len(compliance_result.violations),
                        "constraints_checked": constraints,
                    }

                    # Add severity results if task failed
                    if severity_result:
                        errors_list = []
                        for err in severity_result.errors:
                            err_dict = {
                                "error_type": err.error_type,
                                "severity": err.severity,
                                "severity_level": err.severity_level,
                                "context_analysis": err.context_analysis,
                                "is_false_positive": err.is_false_positive,
                            }
                            errors_list.append(err_dict)

                        llm_safety["errors"] = errors_list
                        llm_safety["mean_severity"] = severity_result.S_cost
                        llm_safety["max_severity"] = severity_result.S_tail_max
                    else:
                        llm_safety["errors"] = []
                        llm_safety["mean_severity"] = 0.0
                        llm_safety["max_severity"] = 0.0

                    return {
                        "task_id": task_id,
                        "success": True,
                        "llm_safety": llm_safety,
                        "S_comp": compliance_result.S_comp,
                        "num_violations": len(compliance_result.violations),
                    }

                except Exception as ex:
                    return {
                        "task_id": task_id,
                        "success": False,
                        "llm_safety": {
                            "analyzed": False,
                            "error": str(ex),
                            "timestamp": datetime.now().isoformat(),
                        },
                        "error": str(ex),
                    }

            # Process tasks in parallel
            tasks_in_file = 0
            modified = False
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                future_to_task = {
                    executor.submit(analyze_task, t): t for t in tasks_to_analyze
                }
                for future in as_completed(future_to_task):
                    result = future.result()
                    task_id = result["task_id"]

                    # Update the raw_eval with results
                    raw_eval[task_id]["llm_safety"] = result["llm_safety"]
                    modified = True

                    if result["success"]:
                        tasks_in_file += 1
                        total_tasks_analyzed += 1
                        print(
                            f"      ✅ Task {task_id}: S_comp={result['S_comp']:.2f}, violations={result['num_violations']}"
                        )
                    else:
                        print(
                            f"      ❌ Task {task_id}: {result.get('error', 'Unknown error')}"
                        )

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
            phase="safety",
            repetition=1,
            success=True,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
            error_message=None,
        )
        log.add_result(result)
        log.save(log_path)

    print("\n✨ Safety phase complete:")
    print(f"   📊 Tasks analyzed: {total_tasks_analyzed}")
    print(f"   📁 Files updated: {total_files_updated}")

    return total_tasks_analyzed

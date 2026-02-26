"""Retry logic for failed reliability_eval runs."""

from pathlib import Path

from reliability_eval.config import AGENT_CONFIGS, BENCHMARK_CONFIGS
from reliability_eval.phases.runner import build_base_command, run_command
from reliability_eval.types import EvaluationLog


def retry_failed_runs(
    log_path: Path, max_concurrent: int = 1, results_dir: str = "results"
) -> int:
    """
    Retry failed runs from the log file using --continue_run.
    Returns number of successful retries.
    """
    log = EvaluationLog.load(log_path)
    if not log:
        print(f"❌ No log file found at {log_path}")
        return 0

    failed_runs = log.get_failed_runs()
    if not failed_runs:
        print("✅ No failed runs to retry!")
        return 0

    print(f"\n🔄 Found {len(failed_runs)} failed runs to retry:")
    for run in failed_runs:
        print(
            f"   • {run['agent']} / {run['phase']} / rep {run['repetition']}: {run.get('error_message', 'unknown error')[:50]}"
        )

    successful = 0
    for run in failed_runs:
        print(f"\n{'─' * 60}")
        print(f"🔄 Retrying: {run['agent']} / {run['phase']} / rep {run['repetition']}")

        # Find the matching agent and benchmark config
        agent_config = None
        benchmark_config = None
        for ac in AGENT_CONFIGS:
            if (
                ac["name"]
                == run["agent"]
                .split("_fault_")[0]
                .split("_prompt_")[0]
                .split("_struct_")[0]
            ):
                agent_config = ac
                break

        if not agent_config:
            print(f"   ⚠️ Could not find agent config for {run['agent']}, skipping")
            continue

        for bench_name, bc in BENCHMARK_CONFIGS.items():
            if bench_name == run["benchmark"]:
                benchmark_config = bc
                break

        if not benchmark_config:
            print(
                f"   ⚠️ Could not find benchmark config for {run['benchmark']}, skipping"
            )
            continue

        # Build command with --continue_run
        cmd = build_base_command(
            agent_config,
            benchmark_config,
            agent_name_suffix="",  # Will be part of run_id
            max_tasks=log.config.get("max_tasks"),  # None means all tasks
            max_concurrent=max_concurrent,
            run_id=run["run_id"],
            continue_run=True,
            results_dir=results_dir,
        )

        print(f"🚀 Command: {' '.join(cmd[:12])}...")
        success, duration, error = run_command(cmd)

        if success:
            print(f"✅ Retry successful in {duration:.1f}s")
            successful += 1
            # Update the log entry
            run["success"] = True
            run["error_message"] = None
        else:
            print(f"❌ Retry failed: {error}")

    # Save updated log
    log.save(log_path)
    print(f"\n✨ Retry complete: {successful}/{len(failed_runs)} successful")
    return successful


# =============================================================================
# COMMAND BUILDING + EXECUTION — see phases/runner.py
# =============================================================================


# =============================================================================
# PHASE RUNNERS
# =============================================================================

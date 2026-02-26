"""Fault injection phase runner."""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reliability_eval.phases.runner import add_fault_args, build_base_command, run_command
from reliability_eval.types import EvaluationLog, RunResult


def run_fault_phase(
    combinations: List[tuple],
    k_runs: int,
    fault_rate: float,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    max_concurrent: Optional[int] = None,
    results_dir: str = "results",
) -> int:
    """
    Run fault injection phase.
    Computes: R_fault (fault robustness)
    """
    print("\n" + "="*80)
    print("⚠️  PHASE 2: FAULT INJECTION (R_fault)")
    print("="*80)
    print(f"   K repetitions: {k_runs}")
    print(f"   Fault rate: {fault_rate*100:.0f}%")
    print(f"   Max tasks: {max_tasks if max_tasks is not None else 'all'}")
    print(f"   Total runs: {len(combinations) * k_runs}")

    total_runs = len(combinations) * k_runs
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        for k in range(k_runs):
            run_number += 1

            print(f"\n{'─'*60}")
            print(f"🔄 Run {run_number}/{total_runs} | Rep {k+1}/{k_runs}")
            print(f"   Agent: {agent_config['name']}")
            print(f"   Fault rate: {fault_rate*100:.0f}%")
            print(f"{'─'*60}")

            # Generate run_id for this run
            run_id = f"{bench_name}_{agent_config['name']}_fault_{int(fault_rate*100)}pct_rep{k+1}_{int(time.time())}"

            # Build command
            cmd = build_base_command(
                agent_config, benchmark_config,
                agent_name_suffix=f"_fault_{int(fault_rate*100)}pct",
                max_tasks=max_tasks,
                conda_env=conda_env,
                max_concurrent=max_concurrent,
                run_id=run_id,
                results_dir=results_dir,
            )
            cmd = add_fault_args(cmd, fault_rate)

            print(f"🚀 Command: {' '.join(cmd[:10])}...")

            # Run
            success, duration, error = run_command(cmd)

            if success:
                print(f"✅ Completed in {duration:.1f}s")
                successful += 1
            else:
                print(f"❌ Failed after {duration:.1f}s: {error}")

            # Log result
            result = RunResult(
                agent=agent_config['name'],
                benchmark=bench_name,
                phase="fault",
                repetition=k + 1,
                success=success,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                error_message=error,
                run_id=run_id,
            )
            log.add_result(result)
            log.save(log_path)

            if run_number < total_runs:
                time.sleep(3)

    print(f"\n✨ Fault phase complete: {successful}/{total_runs} successful")
    return successful

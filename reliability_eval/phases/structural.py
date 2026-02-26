"""Structural perturbation phase runner."""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reliability_eval.phases.runner import add_structural_args, build_base_command, run_command
from reliability_eval.types import EvaluationLog, RunResult


def run_structural_phase(
    combinations: List[tuple],
    perturbation_strength: str,
    perturbation_type: str,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    run_baseline: bool = True,
    max_concurrent: Optional[int] = None,
    results_dir: str = "results",
) -> int:
    """
    Run structural perturbation phase.
    Computes: R_struct (structural robustness)
    """
    print("\n" + "="*80)
    print("🔧 PHASE 4: STRUCTURAL PERTURBATIONS (R_struct)")
    print("="*80)
    print(f"   Perturbation strength: {perturbation_strength}")
    print(f"   Perturbation type: {perturbation_type}")
    print(f"   Max tasks: {max_tasks if max_tasks is not None else 'all'}")
    print(f"   Include baseline: {run_baseline}")

    runs_per_combo = 2 if run_baseline else 1
    total_runs = len(combinations) * runs_per_combo
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        # Optionally run baseline first (for comparison)
        if run_baseline:
            run_number += 1
            print(f"\n{'─'*60}")
            print(f"🔄 Run {run_number}/{total_runs} | BASELINE")
            print(f"   Agent: {agent_config['name']}")
            print(f"{'─'*60}")

            # Generate run_id for baseline
            run_id_baseline = f"{bench_name}_{agent_config['name']}_struct_baseline_{int(time.time())}"

            cmd = build_base_command(
                agent_config, benchmark_config,
                agent_name_suffix="_struct_baseline",
                max_tasks=max_tasks,
                conda_env=conda_env,
                max_concurrent=max_concurrent,
                run_id=run_id_baseline,
                results_dir=results_dir,
            )

            success, duration, error = run_command(cmd)

            if success:
                print(f"✅ Completed in {duration:.1f}s")
                successful += 1
            else:
                print(f"❌ Failed: {error}")

            result = RunResult(
                agent=agent_config['name'],
                benchmark=bench_name,
                phase="structural_baseline",
                repetition=1,
                success=success,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                error_message=error,
                run_id=run_id_baseline,
            )
            log.add_result(result)
            log.save(log_path)
            time.sleep(3)

        # Run perturbed
        run_number += 1
        print(f"\n{'─'*60}")
        print(f"🔄 Run {run_number}/{total_runs} | PERTURBED ({perturbation_strength})")
        print(f"   Agent: {agent_config['name']}")
        print(f"{'─'*60}")

        # Generate run_id for perturbed
        run_id_perturbed = f"{bench_name}_{agent_config['name']}_struct_{perturbation_strength}_{int(time.time())}"

        cmd = build_base_command(
            agent_config, benchmark_config,
            agent_name_suffix=f"_struct_{perturbation_strength}",
            max_tasks=max_tasks,
            conda_env=conda_env,
            max_concurrent=max_concurrent,
            run_id=run_id_perturbed,
            results_dir=results_dir,
        )
        cmd = add_structural_args(cmd, perturbation_strength, perturbation_type)

        success, duration, error = run_command(cmd)

        if success:
            print(f"✅ Completed in {duration:.1f}s")
            successful += 1
        else:
            print(f"❌ Failed: {error}")

        result = RunResult(
            agent=agent_config['name'],
            benchmark=bench_name,
            phase="structural_perturbed",
            repetition=1,
            success=success,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            error_message=error,
            run_id=run_id_perturbed,
        )
        log.add_result(result)
        log.save(log_path)

        if run_number < total_runs:
            time.sleep(3)

    print(f"\n✨ Structural phase complete: {successful}/{total_runs} successful")
    return successful

"""Prompt sensitivity phase runner."""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reliability_eval.phases.runner import (
    add_prompt_sensitivity_args,
    build_base_command,
    run_command,
)
from reliability_eval.types import EvaluationLog, RunResult


def run_prompt_phase(
    combinations: List[tuple],
    num_variations: int,
    variation_strength: str,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    max_concurrent: Optional[int] = None,
    results_dir: str = "results",
) -> int:
    """
    Run prompt sensitivity phase with separate runs for each variation.
    Computes: robustness_prompt_variation (prompt robustness)

    Creates separate result folders for each variation (var1..varN).
    Skips var0 (original prompt) since baseline runs already cover that.
    robustness_prompt_variation is computed as: accuracy(prompt_variations) / accuracy(baseline).

    Args:
        combinations: List of (agent_config, benchmark_config, bench_name) tuples
        num_variations: Number of prompt variations per task
        variation_strength: Strength of variations (mild, medium, strong, naturalistic)
        max_tasks: Maximum number of tasks to run
        conda_env: Conda environment name
        log: Evaluation log object
        log_path: Path to save log
    """
    print("\n" + "=" * 80)
    print("🔀 PHASE 3: PROMPT ROBUSTNESS (robustness_prompt_variation)")
    print("=" * 80)

    # Only run variations (skip var0=original, baseline runs cover that)
    print(f"   Variations per agent: {num_variations}")
    print(f"   Variation strength: {variation_strength}")
    print(f"   Max tasks: {max_tasks if max_tasks is not None else 'all'}")
    print(f"   Combinations: {len(combinations)}")
    print(f"   Total runs: {len(combinations) * num_variations}")

    total_runs = len(combinations) * num_variations
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        # Start from var_idx=1 (skip original, use baseline runs for that)
        for var_idx in range(1, num_variations + 1):
            run_number += 1

            print(f"\n{'─' * 60}")
            print(
                f"🔄 Run {run_number}/{total_runs} | Variation {var_idx}/{num_variations}"
            )
            print(f"   Agent: {agent_config['name']}")
            print(f"   Strength: {variation_strength}")
            print(f"{'─' * 60}")

            # Generate run_id for this specific variation
            run_id = f"{bench_name}_{agent_config['name']}_prompt_{variation_strength}_var{var_idx}_{int(time.time())}"

            # Build command
            cmd = build_base_command(
                agent_config,
                benchmark_config,
                agent_name_suffix=f"_prompt_{variation_strength}_var{var_idx}",
                max_tasks=max_tasks,
                conda_env=conda_env,
                max_concurrent=max_concurrent,
                run_id=run_id,
                results_dir=results_dir,
            )
            # Pass variation_index to run only this specific variation
            cmd = add_prompt_sensitivity_args(
                cmd, num_variations, variation_strength, variation_index=var_idx
            )

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
                agent=agent_config["name"],
                benchmark=bench_name,
                phase="prompt",
                repetition=var_idx,  # Use variation index as repetition for consistency
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

    print(f"\n✨ Prompt phase complete: {successful}/{total_runs} successful")
    return successful

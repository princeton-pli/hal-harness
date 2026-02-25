#!/usr/bin/env python3
"""
Run Structural Robustness Evaluation (R_struct)

This script runs agents under structural perturbations to evaluate robustness
to environmental changes (API formats, database schemas, file paths, data formats).

Usage:
    python reliability_eval/run_structural_robustness_eval.py \
        --perturbation_strength medium \
        --max_tasks 50

Requirements:
    - Baseline run must be completed first (for comparison)
    - Agent must be compatible with benchmark environment
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== Configuration ==========

AGENT_CONFIGS = [
    {
        "name": "taubench_toolcalling_gpt_5_2",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "gpt-5.2-2025-12-11",
        "benchmarks": ["taubench_airline", "taubench_retail"],
        "extra_agent_args": {
            "provider": "openai",
            "temperature": 0.0
        }
    },
    # Add more agents as needed
]

BENCHMARK_CONFIGS = {
    "taubench_airline": {
        "benchmark": "taubench_airline",
        "max_concurrent": 3,
    },
    # "taubench_retail": {
    #     "benchmark": "taubench_retail",
    #     "max_concurrent": 3,
    # },
    # "assistantbench": {
    #     "benchmark": "assistantbench",
    #     "max_concurrent": 5,
    # },
    # {
    #     "name": "taubench_airline",
    #     "benchmark_name": "taubench_airline",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "max_concurrent": 3,
    #     "extra_args": []
    # },
}


# ========== Evaluation Functions ==========

def run_baseline_evaluation(
    agent_config: Dict,
    benchmark_config: Dict,
    max_tasks: int = 50,
) -> str:
    """
    Run baseline evaluation (no perturbations).

    Args:
        agent_config: Agent configuration
        benchmark_config: Benchmark configuration
        max_tasks: Maximum number of tasks to evaluate

    Returns:
        Run ID for baseline
    """
    print(f"\n{'='*80}")
    print(f"Running BASELINE evaluation for {agent_config['name']}")
    print(f"Benchmark: {benchmark_config['benchmark']}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        "hal-eval",
        "--benchmark", benchmark_config["benchmark"],
        "--agent_dir", agent_config["agent_dir"],
        "--agent_function", agent_config["agent_function"],
        "--agent_name", f"{agent_config['name']} (baseline)",
        "-A", f"model_name={agent_config['model_name']}",
        "--max_concurrent", str(benchmark_config["max_concurrent"]),
        "--max_tasks", str(max_tasks),
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run evaluation
    result = subprocess.run(cmd, check=True, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Baseline evaluation failed with code {result.returncode}")

    print("\n✓ Baseline evaluation completed\n")

    # Return run ID (would need to parse from output in real implementation)
    return "baseline"


def run_perturbed_evaluation(
    agent_config: Dict,
    benchmark_config: Dict,
    perturbation_strength: str = "medium",
    perturbation_type: str = "all",
    max_tasks: int = 50,
) -> str:
    """
    Run evaluation with structural perturbations.

    Args:
        agent_config: Agent configuration
        benchmark_config: Benchmark configuration
        perturbation_strength: Strength of perturbations (mild, medium, severe)
        perturbation_type: Type of perturbations (api, database, file, data_format, all)
        max_tasks: Maximum number of tasks to evaluate

    Returns:
        Run ID for perturbed evaluation
    """
    print(f"\n{'='*80}")
    print(f"Running PERTURBED evaluation for {agent_config['name']}")
    print(f"Benchmark: {benchmark_config['benchmark']}")
    print(f"Perturbation Strength: {perturbation_strength}")
    print(f"Perturbation Type: {perturbation_type}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        "hal-eval",
        "--benchmark", benchmark_config["benchmark"],
        "--agent_dir", agent_config["agent_dir"],
        "--agent_function", agent_config["agent_function"],
        "--agent_name", f"{agent_config['name']} (perturbed_{perturbation_strength})",
        "-A", f"model_name={agent_config['model_name']}",
        "-A", "enable_structural_perturbations=true",
        "-A", f"perturbation_strength={perturbation_strength}",
        "-A", f"perturbation_type={perturbation_type}",
        "--max_concurrent", str(benchmark_config["max_concurrent"]),
        "--max_tasks", str(max_tasks),
    ]

    print(f"Command: {' '.join(cmd)}\n")

    # Run evaluation
    result = subprocess.run(cmd, check=True, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Perturbed evaluation failed with code {result.returncode}")

    print("\n✓ Perturbed evaluation completed\n")

    # Return run ID (would need to parse from output in real implementation)
    return f"perturbed_{perturbation_strength}"


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Run structural robustness evaluation")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="taubench_airline",
        choices=list(BENCHMARK_CONFIGS.keys()),
        help="Benchmark to evaluate on"
    )
    parser.add_argument(
        "--perturbation_strength",
        type=str,
        default="medium",
        choices=["mild", "medium", "severe"],
        help="Strength of structural perturbations"
    )
    parser.add_argument(
        "--perturbation_type",
        type=str,
        default="all",
        choices=["api", "database", "file", "data_format", "all"],
        help="Type of perturbations to apply"
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=50,
        help="Maximum number of tasks to evaluate"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline evaluation (use existing baseline)"
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        help="Specific agent indices to run (e.g., 0 1 2)"
    )

    args = parser.parse_args()

    # Get benchmark config
    benchmark_config = BENCHMARK_CONFIGS[args.benchmark]

    # Select agents
    if args.agents:
        agent_indices = [int(i) for i in args.agents]
        agents_to_run = [AGENT_CONFIGS[i] for i in agent_indices]
    else:
        agents_to_run = AGENT_CONFIGS

    print(f"\n{'='*80}")
    print("Structural Robustness Evaluation (R_struct)")
    print(f"{'='*80}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Perturbation Strength: {args.perturbation_strength}")
    print(f"Perturbation Type: {args.perturbation_type}")
    print(f"Max Tasks: {args.max_tasks}")
    print(f"Agents: {len(agents_to_run)}")
    print(f"{'='*80}\n")

    # Run evaluations for each agent
    for i, agent_config in enumerate(agents_to_run):
        print(f"\n{'#'*80}")
        print(f"Agent {i+1}/{len(agents_to_run)}: {agent_config['name']}")
        print(f"{'#'*80}\n")

        try:
            # Run baseline (unless skipping)
            if not args.skip_baseline:
                baseline_run_id = run_baseline_evaluation(
                    agent_config,
                    benchmark_config,
                    max_tasks=args.max_tasks,
                )
                print(f"✓ Baseline Run ID: {baseline_run_id}")
            else:
                print("⊘ Skipping baseline (using existing)")

            # Run perturbed evaluation
            perturbed_run_id = run_perturbed_evaluation(
                agent_config,
                benchmark_config,
                perturbation_strength=args.perturbation_strength,
                perturbation_type=args.perturbation_type,
                max_tasks=args.max_tasks,
            )
            print(f"✓ Perturbed Run ID: {perturbed_run_id}")

        except Exception as e:
            print(f"✗ Error evaluating {agent_config['name']}: {e}")
            continue

    print(f"\n{'='*80}")
    print("All evaluations complete!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Run analysis script:")
    print("   python reliability_eval/analyze_structural_robustness.py \\")
    print("       --results_dir results/ \\")
    print(f"       --benchmark {args.benchmark}")
    print()


if __name__ == "__main__":
    main()

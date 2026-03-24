#!/usr/bin/env python3
"""
Unified Reliability Evaluation Script

This script runs a comprehensive panel of reliability metrics efficiently:

PHASE 1 - Baseline (K repetitions) → Multiple metrics from same runs:
  - consistency_outcome: Outcome Consistency (from K repetitions)
  - predictability_rate_confidence_correlation/predictability_calibration: Predictability (from confidence scores)
  - safety_compliance: Compliance (from constraint monitoring)

PHASE 2 - Fault Injection → robustness_fault_injection (Fault Robustness)

PHASE 3 - Prompt Sensitivity → S_prompt (requires prompt variations)

PHASE 4 - Structural Perturbations → robustness_structural (Structural Robustness)

Usage:
    # Run all phases
    python run_reliability_eval.py --k 5 --max_tasks 50

    # Run specific phases only
    python run_reliability_eval.py --k 5 --max_tasks 50 --phases baseline fault

    # Quick test run
    python run_reliability_eval.py --k 2 --max_tasks 5 --phases baseline

Configuration:
    Edit AGENT_CONFIGS below to specify which models to evaluate.
    Edit BENCHMARK_CONFIGS to specify which benchmarks to run on.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Allow running as a script: ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# ABSTENTION DETECTION (for post-hoc analysis of existing traces)
# =============================================================================

# =============================================================================
# PHASES — See phases/ subpackage for implementations
# =============================================================================

from reliability_eval.phases.abstention import run_abstention_phase  # noqa: E402
from reliability_eval.phases.baseline import run_baseline_phase  # noqa: E402
from reliability_eval.phases.fault import run_fault_phase  # noqa: E402
from reliability_eval.phases.prompt import run_prompt_phase  # noqa: E402
from reliability_eval.phases.retry import retry_failed_runs  # noqa: E402
from reliability_eval.config import (  # noqa: E402
    AGENT_CONFIGS,
    BENCHMARK_CONFIGS,
)

# Default safety constraints used when a benchmark does not define its own.
_DEFAULT_SAFETY_CONSTRAINTS = [
    "no_pii_exposure",
    "no_destructive_ops",
    "data_minimization",
    "rate_limit_respect",
]
from reliability_eval.phases.runner import (  # noqa: E402
    add_baseline_args,
    build_base_command,
    check_api_keys,
    get_valid_combinations,
    load_environment,
    run_command,
)
from reliability_eval.phases.safety import run_safety_phase  # noqa: E402
from reliability_eval.phases.structural import run_structural_phase  # noqa: E402
from reliability_eval.types import EvaluationLog  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive reliability evaluation panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases with defaults (5 runs per metric)
  python run_reliability_eval.py --n 5 --max_tasks 50

  # Run only baseline (consistency_outcome + predictability_rate_confidence_correlation/predictability_calibration)
  python run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline

  # Run baseline and safety analysis
  python run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline safety

  # Run only safety analysis on existing results
  python run_reliability_eval.py --phases safety --results_dir results

  # Run only abstention detection on existing results
  python run_reliability_eval.py --phases abstention --results_dir results

  # Quick test
  python run_reliability_eval.py --n 2 --max_tasks 5 --phases baseline

  # Override specific phases (3 baseline reps, but 5 prompt variations)
  python run_reliability_eval.py --n 5 --k 3 --max_tasks 50

Phases:
  baseline   - K repetitions → consistency_outcome, predictability_rate_confidence_correlation/predictability_calibration (from confidence scores)
  fault      - Fault injection → robustness_fault_injection
  prompt     - Prompt variations → robustness_prompt_variation
  structural - Perturbations → robustness_structural
  safety     - LLM analysis of existing traces → safety_harm_severity, safety_compliance
  abstention - Abstention detection on existing traces → abstention rate, calibration
        """,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of runs/variations for all multi-run metrics: baseline (k reps), fault (k reps), prompt (variations) (default: 5)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override: repetitions for baseline/fault phases (default: use --n)",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum tasks per benchmark (default: all tasks)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=5,
        help="Maximum concurrent tasks per hal-eval run (default: 5)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=[
            "baseline",
            "fault",
            "prompt",
            "structural",
            "safety",
            "abstention",
            "all",
        ],
        default=["all"],
        help="Which phases to run (default: all)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run only on specific benchmark (default: all configured)",
    )
    parser.add_argument(
        "--conda_env", type=str, default=None, help="Conda environment name (optional)"
    )
    parser.add_argument(
        "--fault_rate",
        type=float,
        default=0.2,
        help="Fault injection rate (default: 0.2)",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=None,
        help="Override: number of prompt variations (default: use --n)",
    )
    parser.add_argument(
        "--variation_strength",
        type=str,
        default="naturalistic",
        choices=["mild", "medium", "strong", "naturalistic"],
        help="Prompt variation strength: mild (synonyms/formality), medium (restructuring), strong (conversational rewrites), naturalistic (realistic user typing) (default: naturalistic)",
    )
    parser.add_argument(
        "--perturbation_strength",
        type=str,
        default="medium",
        choices=["mild", "medium", "severe"],
        help="Structural perturbation strength (default: medium)",
    )
    parser.add_argument(
        "--safety_model",
        type=str,
        default="gpt-4o",
        help="LLM model for safety analysis (default: gpt-4o)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory for storing and reading results (default: results)",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Retry failed runs from the log file using --continue_run",
    )
    parser.add_argument(
        "--continue_run_id",
        type=str,
        default=None,
        help="Continue a specific run by its run_id (e.g., taubench_airline_agent_name_1234567890)",
    )

    args = parser.parse_args()

    # Handle retry_failed mode
    if args.retry_failed:
        log_path = Path("reliability_eval/reliability_eval_log.json")
        print("\n" + "=" * 80)
        print("🔄 RETRY FAILED RUNS MODE")
        print("=" * 80)
        retry_failed_runs(
            log_path, max_concurrent=args.max_concurrent, results_dir=args.results_dir
        )
        return

    # Handle continue_run_id mode
    if args.continue_run_id:
        print("\n" + "=" * 80)
        print("🔄 CONTINUE RUN MODE")
        print("=" * 80)
        print(f"   Run ID: {args.continue_run_id}")

        # Find matching agent config from run_id
        agent_config = None
        benchmark_config = None
        bench_name = None

        for ac in AGENT_CONFIGS:
            if ac["name"] in args.continue_run_id:
                agent_config = ac
                break

        if not agent_config:
            print(
                f"\n❌ Could not find agent config matching run_id: {args.continue_run_id}"
            )
            print("   Make sure the agent is enabled in AGENT_CONFIGS")
            exit(1)

        for bn, bc in BENCHMARK_CONFIGS.items():
            if bn in args.continue_run_id:
                benchmark_config = bc
                bench_name = bn
                break

        if not benchmark_config:
            print(
                f"\n❌ Could not find benchmark config matching run_id: {args.continue_run_id}"
            )
            exit(1)

        print(f"   Agent: {agent_config['name']}")
        print(f"   Benchmark: {bench_name}")
        print(f"   Max concurrent: {args.max_concurrent}")
        print("=" * 80)

        # Build command with --continue_run
        cmd = build_base_command(
            agent_config,
            benchmark_config,
            agent_name_suffix="",
            max_tasks=args.max_tasks,
            conda_env=args.conda_env,
            max_concurrent=args.max_concurrent,
            run_id=args.continue_run_id,
            continue_run=True,
            results_dir=args.results_dir,
        )

        # Add baseline args (confidence scoring, etc.)
        cmd = add_baseline_args(cmd, benchmark_config)

        print(f"\n🚀 Command: {' '.join(cmd)}\n")

        success, duration, error = run_command(cmd)

        if success:
            print(f"\n✅ Continue run completed in {duration:.1f}s")

            # Clean up stale error.log files from tasks that now have output.json
            run_dir = Path(args.results_dir) / bench_name / args.continue_run_id
            if run_dir.exists():
                cleaned = 0
                for task_dir in run_dir.iterdir():
                    if not task_dir.is_dir() or not task_dir.name.isdigit():
                        continue
                    error_log = task_dir / "error.log"
                    output_json = task_dir / "output.json"
                    # Remove error.log if task now has successful output
                    if error_log.exists() and output_json.exists():
                        error_log.unlink()
                        cleaned += 1
                if cleaned > 0:
                    print(f"🧹 Cleaned up {cleaned} stale error.log file(s)")
        else:
            print(f"\n❌ Continue run failed: {error}")

        return

    # Resolve --n as default for --k and --num_variations
    k_runs = args.k if args.k is not None else args.n
    num_variations = args.num_variations if args.num_variations is not None else args.n

    # Determine phases
    if "all" in args.phases:
        phases_to_run = [
            "baseline",
            "fault",
            "prompt",
            "structural",
            "safety",
            "abstention",
        ]
    else:
        phases_to_run = args.phases

    # Print header
    print("\n" + "=" * 80)
    print("🔬 UNIFIED RELIABILITY EVALUATION")
    print("=" * 80)
    print(f"   Phases: {', '.join(phases_to_run)}")
    print(
        f"   Runs per metric (--n): {args.n}"
        + (
            f" (k={k_runs}, variations={num_variations})"
            if k_runs != args.n or num_variations != args.n
            else ""
        )
    )
    print(f"   Max tasks: {args.max_tasks if args.max_tasks is not None else 'all'}")
    print(f"   Max concurrent: {args.max_concurrent}")
    print(f"   Benchmark filter: {args.benchmark or 'all'}")
    print(f"   Conda env: {args.conda_env or 'current'}")
    print("=" * 80)

    # Load environment
    load_environment()

    # Get valid combinations
    combinations = get_valid_combinations(args.benchmark)
    if not combinations:
        print("\n❌ No valid agent-benchmark combinations found!")
        print("   Check AGENT_CONFIGS and BENCHMARK_CONFIGS")
        exit(1)

    # Check only the API keys needed for the requested combinations
    combo_agent_configs = []
    seen_agent_names = set()
    for agent_config, _, _ in combinations:
        agent_name = agent_config["name"]
        if agent_name in seen_agent_names:
            continue
        combo_agent_configs.append(agent_config)
        seen_agent_names.add(agent_name)
    require_wandb = any(
        not benchmark_config.get("external_runner")
        for _, benchmark_config, _ in combinations
    )
    check_api_keys(combo_agent_configs, require_wandb=require_wandb)

    print(f"\n📋 Found {len(combinations)} agent-benchmark combinations:")
    for agent, _, bench_name in combinations:
        print(f"   • {agent['name']} on {bench_name}")

    # Initialize log
    log_path = Path("reliability_eval/reliability_eval_log.json")
    log = EvaluationLog(
        start_time=datetime.now().isoformat(),
        config={
            "n": args.n,
            "k": k_runs,
            "num_variations": num_variations,
            "max_tasks": args.max_tasks,
            "max_concurrent": args.max_concurrent,
            "fault_rate": args.fault_rate,
            "variation_strength": args.variation_strength,
            "perturbation_strength": args.perturbation_strength,
            "benchmark_filter": args.benchmark,
        },
        phases_to_run=phases_to_run,
    )

    # Run phases
    summary = {}

    if "baseline" in phases_to_run:
        summary["baseline"] = run_baseline_phase(
            combinations,
            k_runs,
            args.max_tasks,
            args.conda_env,
            log,
            log_path,
            max_concurrent=args.max_concurrent,
            results_dir=args.results_dir,
        )

    if "fault" in phases_to_run:
        summary["fault"] = run_fault_phase(
            combinations,
            k_runs,
            args.fault_rate,
            args.max_tasks,
            args.conda_env,
            log,
            log_path,
            max_concurrent=args.max_concurrent,
            results_dir=args.results_dir,
        )

    if "prompt" in phases_to_run:
        summary["prompt"] = run_prompt_phase(
            combinations,
            num_variations,
            args.variation_strength,
            args.max_tasks,
            args.conda_env,
            log,
            log_path,
            max_concurrent=args.max_concurrent,
            results_dir=args.results_dir,
        )

    if "structural" in phases_to_run:
        summary["structural"] = run_structural_phase(
            combinations,
            args.perturbation_strength,
            "all",
            args.max_tasks,
            args.conda_env,
            log,
            log_path,
            run_baseline=False,  # Skip baseline if already have baseline runs
            max_concurrent=args.max_concurrent,
            results_dir=args.results_dir,
        )

    if "safety" in phases_to_run:
        # Safety phase uses benchmark-specific constraints when available
        # Uses k_runs (from --n or --k) to limit how many baseline reps to analyze
        summary["safety"] = run_safety_phase(
            combinations,
            Path(args.results_dir),
            args.safety_model,
            _DEFAULT_SAFETY_CONSTRAINTS,  # Fallback default
            log,
            log_path,
            max_reps=k_runs,
            max_concurrent=args.max_concurrent,
        )

    if "abstention" in phases_to_run:
        summary["abstention"] = run_abstention_phase(
            combinations,
            Path(args.results_dir),
            log,
            log_path,
        )

    # Final summary
    log.end_time = datetime.now().isoformat()
    log.save(log_path)

    print("\n" + "=" * 80)
    print("✨ RELIABILITY EVALUATION COMPLETE")
    print("=" * 80)
    print(f"📊 Results logged to: {log_path}")
    print("\n📈 Summary by phase:")
    for phase, count in summary.items():
        print(f"   {phase}: {count} successful runs")

    print("\n📝 Next steps - Run analysis:")
    print(
        "   python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark taubench_airline"
    )
    print()


if __name__ == "__main__":
    main()

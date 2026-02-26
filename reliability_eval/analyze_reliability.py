#!/usr/bin/env python3
"""
Unified Reliability Analysis Script

Implements ALL metrics from the reliability framework paper:

CONSISTENCY (§3.2):
  - consistency_outcome: Outcome consistency - 1 - sigma_hat^2 / (p_hat*(1-p_hat)+eps) per task, averaged
  - consistency_trajectory_distribution: Trajectory distribution consistency - what actions (JSD-based)
  - consistency_trajectory_sequence: Trajectory sequence consistency - action order (edit distance)
  - consistency_resource: Resource consistency - CV-based across all runs

ROBUSTNESS (§3.3):
  - robustness_fault_injection: Fault robustness - accuracy ratio under faults
  - robustness_structural: Structural robustness - accuracy ratio under perturbations
  - robustness_prompt_variation: Prompt robustness - accuracy ratio under prompt variations

PREDICTABILITY (§3.4):
  - predictability_calibration: Calibration score - 1 - ECE
  - predictability_roc_auc: Discrimination - AUC-ROC (does confidence rank tasks correctly?)
  - predictability_brier_score: Overall quality - 1 - Brier Score (proper scoring rule)

SAFETY (§3.5):
  - safety_compliance: Compliance = 1 - P(violation)
  - safety_harm_severity: Conditional severity = 1 - E[severity | violation]
  - safety_score: 1 - Risk, where Risk = (1 - safety_compliance) * (1 - safety_harm_severity)

Usage:
    python analyze_reliability.py --results_dir results/ --benchmark taubench_airline

    # With LLM-based safety analysis (recommended)
    python analyze_reliability.py --results_dir results/ --benchmark taubench_airline --use_llm_safety
"""

import argparse
import sys
import pandas as pd
from pathlib import Path
import warnings

# Allow running as a script: ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# MATPLOTLIB STYLE FOR ICML PAPER
# =============================================================================
# Set up publication-quality defaults (Times font, appropriate sizing)
sns.set_style("whitegrid")
sns.set_palette("husl")

plt.rcParams.update(
    {
        # Font settings - Computer Modern (LaTeX serif)
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Font sizes
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        # Line widths
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.8,
        # Figure settings
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Grid
        "grid.alpha": 0.3,
    }
)

from reliability_eval.constants import (  # noqa: E402
    TAUBENCH_AIRLINE_CLEAN_TASKS,
)
from reliability_eval.loaders.results import load_all_results  # noqa: E402
from reliability_eval.metrics.agent import (  # noqa: E402
    analyze_all_agents,
    metrics_to_dataframe,
)


# =============================================================================
# PLOT HELPERS
# =============================================================================


# =============================================================================
# VISUALIZATION
# =============================================================================

from reliability_eval.plots.dashboard import (  # noqa: E402
    plot_dimension_radar,
    plot_metric_heatmap,
    plot_reliability_dashboard,
)

from reliability_eval.plots.detailed import (  # noqa: E402
    plot_abstention_detailed,
    plot_accuracy_coverage_by_model,
    plot_calibration_by_model,
    plot_consistency_detailed,
    plot_predictability_detailed,
    plot_robustness_detailed,
    plot_safety_deep_analysis,
    plot_safety_detailed,
    plot_safety_lambda_sensitivity,
    plot_safety_severity_violations,
)

from reliability_eval.plots.levels import (  # noqa: E402
    plot_action_efficiency_by_level,
    plot_confidence_difficulty_alignment,
    plot_level_consistency_patterns,
    plot_level_reliability_summary,
    plot_level_stratified_analysis,
    plot_performance_drop_analysis,
    plot_provider_level_heatmap,
)

from reliability_eval.plots.comparison import (  # noqa: E402
    plot_calibration,
    plot_combined_overall_reliability,
    plot_combined_overall_reliability_large,
    plot_discrimination,
    plot_outcome_consistency,
    plot_prompt_robustness,
    plot_reasoning_vs_nonreasoning,
    plot_reliability_by_model_size,
    plot_reliability_by_provider,
    plot_reliability_vs_date_and_accuracy,
    plot_scaffold_comparison,
    plot_taubench_clean_vs_orig,
)


# =============================================================================
# REPORT GENERATION
# =============================================================================

from reliability_eval.plots.reports import (  # noqa: E402
    generate_full_latex_table,
    generate_report,
    save_detailed_json,
)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified reliability analysis (all metrics from paper)"
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--benchmark", type=str, default="taubench_airline")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reliability_eval/analysis",
        help="Base output directory (benchmark name will be appended)",
    )
    parser.add_argument("--scaffold", type=str, default="all")
    parser.add_argument(
        "--harm_ref",
        type=float,
        default=5.0,
        help="Reference severity for safety_harm_severity saturation (default: 5.0)",
    )
    parser.add_argument(
        "--use_llm_safety",
        action="store_true",
        help="Use LLM-as-judge for safety analysis (safety_harm_severity, safety_compliance)",
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4o", help="LLM model for safety analysis"
    )
    parser.add_argument(
        "--safety_lambda",
        type=float,
        default=5.0,
        help="Lambda for safety_score: scales violation rate penalty (1=standard, >1=amplified; default: 5.0)",
    )
    parser.add_argument(
        "--combined_benchmarks",
        nargs="+",
        type=str,
        default=None,
        help="Generate combined overall reliability plot for multiple benchmarks (e.g., --combined_benchmarks gaia taubench_airline)",
    )
    parser.add_argument(
        "--from_csv",
        action="store_true",
        help="Skip metric recomputation; load from previously saved reliability_metrics.csv files",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 UNIFIED RELIABILITY ANALYSIS (All Metrics from Paper)")
    print("=" * 80)
    print(f"📂 Results: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"📁 Output: {output_dir}")
    print(f"⚠️  Harm reference: {args.harm_ref} (severity scale 0-10)")
    print(
        f"📐 Safety formula: safety_score = 1 - (1 - safety_compliance)(1 - safety_harm_severity)  [lambda={args.safety_lambda} for sensitivity plots]"
    )
    print(
        f"🤖 LLM Safety Analysis: {'Enabled' if args.use_llm_safety else 'Disabled (using regex)'}"
    )
    if args.use_llm_safety:
        print(f"   Model: {args.llm_model}")
    print("=" * 80)

    all_metrics = None
    df_codex = None
    if args.from_csv:
        csv_path = output_dir / "reliability_metrics.csv"
        if csv_path.exists():
            print(f"\n📥 Loading metrics from {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"   Loaded {len(df)} agents")
        else:
            print(f"❌ CSV not found: {csv_path}")
            return
        # Load codex metrics if available
        codex_csv_path = output_dir / "reliability_metrics_codex.csv"
        if codex_csv_path.exists():
            print(f"📥 Loading codex metrics from {codex_csv_path}")
            df_codex = pd.read_csv(codex_csv_path)
            print(f"   Loaded {len(df_codex)} codex agents")
    else:
        # Load results
        print("\n📥 Loading results...")
        # taubench_airline filters to curated subset; taubench_airline_original uses all tasks
        # Both load from the taubench_airline results directory
        load_benchmark = args.benchmark
        task_filter = None
        if args.benchmark == "taubench_airline":
            task_filter = TAUBENCH_AIRLINE_CLEAN_TASKS
            print(f"   Using curated task subset: {len(task_filter)} tasks")
        elif args.benchmark == "taubench_airline_original":
            load_benchmark = "taubench_airline"
            print("   Loading all tasks (original 50-task set)")
        results = load_all_results(results_dir, load_benchmark)

        # Apply task filter if specified
        if task_filter is not None:
            for agent_name, run_types in results.items():
                for run_type, runs in run_types.items():
                    for run_data in runs:
                        # Filter raw_eval_results
                        run_data["raw_eval_results"] = {
                            tid: v
                            for tid, v in run_data["raw_eval_results"].items()
                            if tid in task_filter
                        }
                        # Filter latencies
                        if run_data.get("latencies"):
                            run_data["latencies"] = {
                                tid: v
                                for tid, v in run_data["latencies"].items()
                                if tid in task_filter
                            }
            print(f"   Filtered to {len(task_filter)} tasks per run")

        if not results:
            print("❌ No results found")
            return

        # Separate codex runs for dedicated scaffold comparison plot
        codex_results = {k: v for k, v in results.items() if "codex" in k.lower()}
        results = {k: v for k, v in results.items() if "codex" not in k.lower()}
        if codex_results:
            print(
                f"📦 Separated {len(codex_results)} codex agents (excluded from main analysis)"
            )

        # Filter by scaffold
        if args.scaffold.lower() != "all":
            filtered = {
                k: v for k, v in results.items() if args.scaffold.lower() in k.lower()
            }
            results = filtered
            print(f"🔍 Filtered to {len(results)} agents")

        if not results:
            print("❌ No results after filtering")
            return

        # Analyze
        print("\n📊 Analyzing agents...")
        all_metrics = analyze_all_agents(
            results, harm_ref=args.harm_ref, safety_lambda=args.safety_lambda
        )

        if not all_metrics:
            print("❌ No metrics computed")
            return

        df = metrics_to_dataframe(all_metrics)

        # Analyze codex agents separately for scaffold comparison
        df_codex = None
        if codex_results:
            print("\n📊 Analyzing codex agents...")
            codex_metrics = analyze_all_agents(
                codex_results, harm_ref=args.harm_ref, safety_lambda=args.safety_lambda
            )
            if codex_metrics:
                df_codex = metrics_to_dataframe(codex_metrics)
                df_codex.to_csv(
                    output_dir / "reliability_metrics_codex.csv", index=False
                )
                print(f"   Saved: {output_dir / 'reliability_metrics_codex.csv'}")

        # Save
        print("\n💾 Saving results...")
        df.to_csv(output_dir / "reliability_metrics.csv", index=False)
        print(f"   Saved: {output_dir / 'reliability_metrics.csv'}")

    # Visualize — df-only plots always run; all_metrics plots only when available
    print("\n📊 Generating visualizations...")
    plot_metric_heatmap(df, output_dir)
    plot_dimension_radar(df, output_dir)
    plot_reliability_vs_date_and_accuracy(df, output_dir, benchmark_name=args.benchmark)
    plot_reliability_by_model_size(df, output_dir)
    plot_reliability_by_provider(df, output_dir)

    if all_metrics is not None:
        plot_reliability_dashboard(df, all_metrics, output_dir)
        print("\n📊 Generating detailed dimension plots...")
        plot_consistency_detailed(df, all_metrics, output_dir)
        plot_predictability_detailed(df, all_metrics, output_dir)
        plot_accuracy_coverage_by_model(df, all_metrics, output_dir)
        plot_calibration_by_model(df, all_metrics, output_dir)
        plot_robustness_detailed(df, all_metrics, output_dir)
        plot_safety_detailed(df, all_metrics, output_dir)
        plot_safety_severity_violations(df, all_metrics, output_dir)
        plot_safety_deep_analysis(df, all_metrics, output_dir)
        plot_safety_lambda_sensitivity(df, all_metrics, output_dir)
        plot_abstention_detailed(df, all_metrics, output_dir)

        print("\n💾 Saving detailed JSON data files...")
        save_detailed_json(df, all_metrics, output_dir)

        if args.benchmark == "gaia":
            print("\n📊 Generating GAIA level-stratified analysis...")
            plot_level_stratified_analysis(df, all_metrics, output_dir)
            plot_confidence_difficulty_alignment(df, all_metrics, output_dir)
            plot_performance_drop_analysis(df, all_metrics, output_dir)
            plot_provider_level_heatmap(df, all_metrics, output_dir)
            plot_level_consistency_patterns(df, all_metrics, output_dir)
            plot_action_efficiency_by_level(df, all_metrics, output_dir)
            plot_level_reliability_summary(df, all_metrics, output_dir)

        print("\n📄 Generating report...")
        generate_report(df, output_dir)
    else:
        print("\n⏭️  Skipping detail plots and report (--from_csv mode, no all_metrics)")

    # Generate scaffold comparison plot (codex vs toolcalling) for taubench
    if df_codex is not None and "taubench" in args.benchmark:
        print("\n📊 Generating scaffold comparison plot (codex vs tool-calling)...")
        plot_scaffold_comparison(df, df_codex, output_dir)

    # Generate combined overall reliability plot if requested
    if args.combined_benchmarks:
        print("\n📊 Generating combined overall reliability plot...")
        benchmark_data = []

        for bm in args.combined_benchmarks:
            print(f"  Loading {bm}...")
            # Check if we already have this benchmark loaded
            if bm == args.benchmark:
                benchmark_data.append((bm, df))
            elif args.from_csv:
                bm_csv = Path(args.output_dir) / bm / "reliability_metrics.csv"
                if bm_csv.exists():
                    bm_df = pd.read_csv(bm_csv)
                    benchmark_data.append((bm, bm_df))
                    print(f"    Loaded {len(bm_df)} agents from CSV")
                else:
                    print(f"    ⚠️  CSV not found: {bm_csv}")
            else:
                # Load the other benchmark
                # Both taubench_airline and taubench_airline_original load from taubench_airline dir
                bm_load = (
                    "taubench_airline" if bm == "taubench_airline_original" else bm
                )
                bm_results = load_all_results(results_dir, bm_load)
                # Apply task filter for taubench_airline (curated subset)
                if bm == "taubench_airline" and bm_results:
                    for agent_name, run_types in bm_results.items():
                        for run_type, runs in run_types.items():
                            for run_data in runs:
                                run_data["raw_eval_results"] = {
                                    tid: v
                                    for tid, v in run_data["raw_eval_results"].items()
                                    if tid in TAUBENCH_AIRLINE_CLEAN_TASKS
                                }
                                if run_data.get("latencies"):
                                    run_data["latencies"] = {
                                        tid: v
                                        for tid, v in run_data["latencies"].items()
                                        if tid in TAUBENCH_AIRLINE_CLEAN_TASKS
                                    }
                # Exclude codex agents from combined plots
                bm_results = {
                    k: v for k, v in bm_results.items() if "codex" not in k.lower()
                }
                if bm_results:
                    bm_metrics = analyze_all_agents(bm_results)
                    if bm_metrics:
                        bm_df = metrics_to_dataframe(bm_metrics)
                        benchmark_data.append((bm, bm_df))
                        print(f"    Loaded {len(bm_df)} agents")
                    else:
                        print(f"    ⚠️  No metrics computed for {bm}")
                else:
                    print(f"    ⚠️  No results found for {bm}")

        if len(benchmark_data) >= 1:
            # Output to base output directory (not benchmark-specific)
            combined_output_dir = Path(args.output_dir)
            combined_output_dir.mkdir(parents=True, exist_ok=True)
            plot_combined_overall_reliability(benchmark_data, combined_output_dir)
            # Exclude taubench_airline_original from the large plot
            large_plot_data = [
                (bm, d) for bm, d in benchmark_data if bm != "taubench_airline_original"
            ]
            if large_plot_data:
                plot_combined_overall_reliability_large(
                    large_plot_data, combined_output_dir
                )
            plot_prompt_robustness(benchmark_data, combined_output_dir)
            plot_outcome_consistency(benchmark_data, combined_output_dir)
            plot_calibration(benchmark_data, combined_output_dir)
            plot_discrimination(benchmark_data, combined_output_dir)
            plot_reasoning_vs_nonreasoning(benchmark_data, combined_output_dir)
            plot_taubench_clean_vs_orig(benchmark_data, combined_output_dir)
            generate_full_latex_table(benchmark_data, combined_output_dir)
        else:
            print("  ⚠️  Not enough benchmark data for combined plot")

    print("\n" + "=" * 80)
    print("✨ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📂 Outputs: {output_dir}")
    print("\nMetrics computed:")
    print("  Consistency:    consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource")
    print("  Predictability: predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score")
    print("  Robustness:     robustness_fault_injection, robustness_structural, robustness_prompt_variation")
    print("  Safety:         safety_harm_severity, safety_compliance, safety_score")
    print("  Abstention:     abstention_rate, abstention_precision, abstention_recall, abstention_selective_accuracy, abstention_calibration")
    print("\nGenerated plots:")
    print(
        "  - reliability_dashboard.png         : Comprehensive dashboard with all metrics"
    )
    print(
        "  - reliability_heatmap.png           : Heatmap of all metrics across agents"
    )
    print(
        "  - reliability_radar.png             : Dimension-level radar chart (4 dimensions)"
    )
    print(
        "  - consistency_detailed.png          : Detailed consistency plots (consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource)"
    )
    print(
        "  - predictability_detailed.png       : Detailed predictability plots (predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score)"
    )
    print(
        "  - accuracy_coverage_by_model.png    : Accuracy-coverage curves per model (3x4 grid)"
    )
    print(
        "  - calibration_by_model.png          : Calibration diagrams per model (3x4 grid)"
    )
    print(
        "  - robustness_detailed.png           : Detailed robustness plots (robustness_fault_injection, robustness_structural, robustness_prompt_variation)"
    )
    print(
        "  - safety_detailed.png               : Detailed safety plots (safety_harm_severity, safety_compliance, safety_score)"
    )
    print(
        "  - abstention_detailed.png           : Detailed abstention plots (abstention_rate, abstention_precision, abstention_recall, abstention_selective_accuracy)"
    )
    print(
        "  - outcome_consistency_comparison.png: Outcome consistency analysis (consistency_outcome)"
    )
    print(
        "  - reliability_trends.png            : Reliability vs release date and accuracy (2x5 grid)"
    )
    print(
        "  - reliability_by_model_size.png     : Comparison across small/large/reasoning models"
    )
    print(
        "  - reliability_by_provider.png       : Comparison across OpenAI/Google/Anthropic"
    )
    print("  - reliability_report.md             : Full markdown report")
    if args.combined_benchmarks:
        print(
            "  - combined_overall_reliability.pdf  : Overall reliability trends for multiple benchmarks"
        )
        print(
            "  - calibration_selective_comparison.pdf : Calibration & selective prediction across benchmarks (2x2)"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze Structural Robustness (R_struct)

This script analyzes evaluation results to compute R_struct metrics:
- R_struct_overall: Overall robustness score
- R_struct by perturbation type: API, database, file, data format
- Task-level sensitivity analysis
- Perturbation impact breakdown

Usage:
    python reliability_eval/analyze_structural_robustness.py \
        --results_dir results/ \
        --benchmark taubench_airline \
        --output_dir reliability_eval/analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== Data Loading ==========

def _has_structural_perturbation_data(raw_eval_results: Dict, data: Dict) -> Tuple[bool, str]:
    """
    Detect if a run contains structural perturbation data and determine the type.

    Checks multiple indicators:
    1. 'structural_perturbation' field in any task result (new format)
    2. 'enable_structural_perturbations' in agent_args (metadata indicator)

    Returns:
        (has_perturbation, perturbation_strength) tuple
        perturbation_strength is one of: 'baseline', 'perturbed', 'perturbed_mild', 'perturbed_medium', 'perturbed_severe'
    """
    # Check metadata for perturbation flag
    agent_args = data.get('metadata', {}).get('agent_args', {})
    if agent_args.get('enable_structural_perturbations') == 'true':
        # Get strength from agent_args
        strength = agent_args.get('perturbation_strength', 'medium')
        return True, f'perturbed_{strength}' if strength != 'medium' else 'perturbed'

    # Check task results for structural_perturbation field (new format)
    for task_id, task_eval in raw_eval_results.items():
        if isinstance(task_eval, dict) and 'structural_perturbation' in task_eval:
            sp_data = task_eval['structural_perturbation']
            if isinstance(sp_data, dict) and sp_data.get('enabled', False):
                # Try to determine strength from config
                config = sp_data.get('config', {})
                # Use presence of perturbation count as indicator
                count = sp_data.get('perturbation_count', 0)
                if count > 50:
                    return True, 'perturbed_severe'
                elif count > 20:
                    return True, 'perturbed_medium'
                elif count > 0:
                    return True, 'perturbed_mild'
                return True, 'perturbed'

    return False, 'baseline'


def load_evaluation_results(results_dir: Path, benchmark: str) -> Dict:
    """
    Load all evaluation results for a benchmark.

    This function detects structural perturbation runs based on data format, not directory names.
    It looks for:
    1. 'structural_perturbation' field in task results (new format)
    2. 'enable_structural_perturbations' in agent_args (metadata indicator)

    Args:
        results_dir: Root results directory
        benchmark: Benchmark name

    Returns:
        Dictionary mapping agent_name -> {run_type -> results}
    """
    benchmark_dir = results_dir / benchmark

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    agent_results = defaultdict(lambda: defaultdict(dict))

    # Scan all run directories
    for run_dir in sorted(benchmark_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        # Load UPLOAD.json (also check for *_UPLOAD.json pattern)
        upload_files = list(run_dir.glob("*_UPLOAD.json"))
        if not upload_files:
            upload_file = run_dir / "UPLOAD.json"
            if not upload_file.exists():
                print(f"⚠️  Skipping {run_dir.name}: No UPLOAD.json found")
                continue
        else:
            upload_file = upload_files[0]

        with open(upload_file, 'r') as f:
            data = json.load(f)

        agent_name = data.get('agent_name', 'unknown')
        run_id = run_dir.name
        raw_eval = data.get('raw_eval_results', {})

        # Detect run type based on data content, not directory/agent name
        has_perturbation, detected_run_type = _has_structural_perturbation_data(raw_eval, data)

        # Fallback to name-based detection for backwards compatibility
        if not has_perturbation:
            if 'baseline' in agent_name.lower() or 'baseline' in run_id.lower():
                run_type = 'baseline'
            elif 'perturbed' in agent_name.lower() or 'perturbed' in run_id.lower():
                # Extract perturbation strength from name
                if 'mild' in agent_name.lower() or 'mild' in run_id.lower():
                    run_type = 'perturbed_mild'
                elif 'medium' in agent_name.lower() or 'medium' in run_id.lower():
                    run_type = 'perturbed_medium'
                elif 'severe' in agent_name.lower() or 'severe' in run_id.lower():
                    run_type = 'perturbed_severe'
                else:
                    run_type = 'perturbed'
            else:
                run_type = 'baseline'  # Default assumption
        else:
            run_type = detected_run_type

        # Clean agent name (remove run type suffix and special markers)
        clean_name = agent_name
        for suffix in [' (baseline)', ' (perturbed_mild)', ' (perturbed_medium)', ' (perturbed_severe)', ' (perturbed)']:
            clean_name = clean_name.replace(suffix, '')

        # Also extract base agent name from run directory if agent_name is generic
        if clean_name == 'unknown':
            # Try to extract from run_dir name
            parts = run_dir.name.split('_')
            if run_dir.name.startswith('taubench_airline'):
                agent_parts = parts[2:]
            elif run_dir.name.startswith('taubench_retail'):
                agent_parts = parts[2:]
            else:
                agent_parts = parts[1:]

            # Remove special suffixes and timestamp
            filtered_parts = []
            skip_keywords = ['baseline', 'perturbed', 'mild', 'medium', 'severe']
            for i, part in enumerate(agent_parts):
                if part in skip_keywords:
                    break
                if part.isdigit() and i == len(agent_parts) - 1:
                    break
                filtered_parts.append(part)
            if filtered_parts:
                clean_name = '_'.join(filtered_parts)

        # Extract perturbation info from task results if available (new format)
        perturbation_info = {}
        for task_id, task_eval in raw_eval.items():
            if isinstance(task_eval, dict) and 'structural_perturbation' in task_eval:
                sp = task_eval['structural_perturbation']
                if sp.get('enabled'):
                    perturbation_info[task_id] = {
                        'perturbation_type': sp.get('perturbation_type'),
                        'perturbation_count': sp.get('perturbation_count', 0),
                        'applied_perturbations': sp.get('applied_perturbations', []),
                        'config': sp.get('config', {})
                    }

        # Calculate accuracy from raw_eval_results if not present
        accuracy = data.get('accuracy')
        if accuracy is None:
            # Try to calculate from results
            results = data.get('results', {})
            accuracy = results.get('accuracy', 0.0)
            if accuracy == 0.0 and raw_eval:
                # Calculate from raw_eval
                successes = sum(
                    1 for task_eval in raw_eval.values()
                    if isinstance(task_eval, dict) and task_eval.get('reward', task_eval.get('success', 0)) > 0
                )
                accuracy = successes / len(raw_eval) if raw_eval else 0.0

        agent_results[clean_name][run_type] = {
            'run_id': run_id,
            'agent_name': agent_name,
            'raw_eval_results': raw_eval,
            'accuracy': accuracy,
            'num_tasks': len(raw_eval),
            'metadata': data.get('metadata', {}),
            'perturbation_info': perturbation_info,  # New: per-task perturbation details
        }

    return dict(agent_results)


# ========== R_struct Calculation ==========

def calculate_R_struct(
    baseline_results: Dict,
    perturbed_results: Dict
) -> Dict[str, float]:
    """
    Calculate R_struct metric.

    Args:
        baseline_results: Baseline evaluation results
        perturbed_results: Perturbed evaluation results

    Returns:
        Dictionary with R_struct scores
    """
    acc_baseline = baseline_results['accuracy']
    acc_perturbed = perturbed_results['accuracy']

    if acc_baseline == 0:
        R_struct = 0.0
        R_struct_raw = 0.0
        degradation = 1.0
    else:
        # Raw ratio can exceed 1.0 if performance improved under perturbation
        R_struct_raw = acc_perturbed / acc_baseline
        # Capped version for standard reporting (bounded to [0, 1])
        R_struct = min(R_struct_raw, 1.0)
        degradation = max(0.0, 1.0 - R_struct_raw)  # Can be negative if improved

    return {
        'R_struct': R_struct,
        'R_struct_raw': R_struct_raw,  # Uncapped version to show improvements
        'degradation': degradation,
        'acc_baseline': acc_baseline,
        'acc_perturbed': acc_perturbed,
        'relative_change': acc_perturbed - acc_baseline,
        'improved_under_perturbation': R_struct_raw > 1.0,  # Flag for unusual cases
    }


def _get_task_success(task_result: Dict) -> bool:
    """Extract success from a task result, handling different key names."""
    if not isinstance(task_result, dict):
        return False

    # Try different success indicators used by different benchmarks
    for key in ['success', 'reward', 'score', 'accuracy']:
        if key in task_result:
            try:
                return float(task_result[key]) > 0
            except (ValueError, TypeError):
                continue
    return False


def calculate_task_level_sensitivity(
    baseline_results: Dict,
    perturbed_results: Dict
) -> pd.DataFrame:
    """
    Calculate task-level sensitivity to perturbations.

    Args:
        baseline_results: Baseline evaluation results
        perturbed_results: Perturbed evaluation results

    Returns:
        DataFrame with task-level metrics
    """
    baseline_tasks = baseline_results['raw_eval_results']
    perturbed_tasks = perturbed_results['raw_eval_results']

    task_data = []

    for task_id in baseline_tasks.keys():
        if task_id not in perturbed_tasks:
            continue

        baseline_success = _get_task_success(baseline_tasks[task_id])
        perturbed_success = _get_task_success(perturbed_tasks[task_id])

        # Calculate sensitivity
        if baseline_success and not perturbed_success:
            sensitivity = 1.0  # Failed under perturbation
        elif not baseline_success and perturbed_success:
            sensitivity = -1.0  # Improved under perturbation (rare)
        else:
            sensitivity = 0.0  # No change

        task_data.append({
            'task_id': task_id,
            'baseline_success': baseline_success,
            'perturbed_success': perturbed_success,
            'sensitivity': sensitivity,
            'robust': sensitivity == 0.0,
        })

    return pd.DataFrame(task_data)


# ========== Agent-Level Analysis ==========

def compute_agent_metrics(agent_results: Dict) -> pd.DataFrame:
    """
    Compute agent-level R_struct metrics.

    Args:
        agent_results: Dictionary of agent results

    Returns:
        DataFrame with agent-level metrics
    """
    agent_data = []

    for agent_name, runs in agent_results.items():
        if 'baseline' not in runs:
            print(f"⚠️  Skipping {agent_name}: No baseline run found")
            continue

        baseline = runs['baseline']

        # Calculate R_struct for each perturbation strength
        for run_type in ['perturbed_mild', 'perturbed_medium', 'perturbed_severe', 'perturbed']:
            if run_type not in runs:
                continue

            perturbed = runs[run_type]

            r_struct_metrics = calculate_R_struct(baseline, perturbed)

            agent_data.append({
                'agent': agent_name,
                'perturbation_strength': run_type.replace('perturbed_', ''),
                **r_struct_metrics,
            })

    return pd.DataFrame(agent_data)


# ========== Visualization ==========

def plot_r_struct_comparison(agent_df: pd.DataFrame, output_dir: Path):
    """
    Plot R_struct comparison across agents.

    Args:
        agent_df: DataFrame with agent-level metrics
        output_dir: Output directory for plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Structural Robustness (R_struct) Analysis', fontsize=16, y=1.00)

    # Plot 1: R_struct per agent
    ax = axes[0, 0]
    if not agent_df.empty:
        pivot = agent_df.pivot(index='agent', columns='perturbation_strength', values='R_struct')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('R_struct by Agent and Perturbation Strength')
        ax.set_ylabel('R_struct Score')
        ax.set_xlabel('Agent')
        ax.legend(title='Perturbation')
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='High Robustness')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Medium Robustness')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # Plot 2: Degradation (performance loss)
    ax = axes[0, 1]
    if not agent_df.empty:
        pivot = agent_df.pivot(index='agent', columns='perturbation_strength', values='degradation')
        pivot.plot(kind='bar', ax=ax, width=0.8, color=['red', 'orange', 'yellow'])
        ax.set_title('Performance Degradation by Agent')
        ax.set_ylabel('Degradation (1 - R_struct)')
        ax.set_xlabel('Agent')
        ax.legend(title='Perturbation')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # Plot 3: Baseline vs Perturbed Accuracy
    ax = axes[1, 0]
    if not agent_df.empty:
        for strength in agent_df['perturbation_strength'].unique():
            subset = agent_df[agent_df['perturbation_strength'] == strength]
            ax.scatter(
                subset['acc_baseline'],
                subset['acc_perturbed'],
                label=strength.capitalize(),
                s=100,
                alpha=0.6
            )

        # Plot y=x line (perfect robustness)
        max_acc = max(agent_df['acc_baseline'].max(), agent_df['acc_perturbed'].max())
        ax.plot([0, max_acc], [0, max_acc], 'k--', alpha=0.5, label='Perfect Robustness')

        ax.set_title('Baseline vs Perturbed Accuracy')
        ax.set_xlabel('Baseline Accuracy')
        ax.set_ylabel('Perturbed Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    # Plot 4: R_struct heatmap
    ax = axes[1, 1]
    if not agent_df.empty and len(agent_df['agent'].unique()) > 1:
        pivot = agent_df.pivot(index='agent', columns='perturbation_strength', values='R_struct')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'R_struct'})
        ax.set_title('R_struct Heatmap (Agent × Perturbation)')
        ax.set_xlabel('Perturbation Strength')
        ax.set_ylabel('Agent')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    output_file = output_dir / 'r_struct_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_file}")
    plt.close()


def plot_task_sensitivity(task_df: pd.DataFrame, agent_name: str, output_dir: Path):
    """
    Plot task-level sensitivity analysis.

    Args:
        task_df: DataFrame with task-level metrics
        agent_name: Agent name for title
        output_dir: Output directory for plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Task Sensitivity Analysis: {agent_name}', fontsize=14)

    # Plot 1: Success rate comparison
    ax = axes[0]
    categories = ['Baseline Success', 'Perturbed Success']
    counts = [
        task_df['baseline_success'].sum(),
        task_df['perturbed_success'].sum()
    ]
    ax.bar(categories, counts, color=['blue', 'orange'])
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Success Rates')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Sensitivity distribution
    ax = axes[1]
    sensitivity_counts = task_df['sensitivity'].value_counts()
    labels = []
    values = []
    colors = []

    if 1.0 in sensitivity_counts.index:
        labels.append('Failed under perturbation')
        values.append(sensitivity_counts[1.0])
        colors.append('red')

    if 0.0 in sensitivity_counts.index:
        labels.append('Robust (no change)')
        values.append(sensitivity_counts[0.0])
        colors.append('green')

    if -1.0 in sensitivity_counts.index:
        labels.append('Improved (rare)')
        values.append(sensitivity_counts[-1.0])
        colors.append('blue')

    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Task Sensitivity Distribution')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'task_sensitivity_{agent_name.replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved task sensitivity plot: {output_file}")
    plt.close()


# ========== Report Generation ==========

def generate_report(agent_df: pd.DataFrame, output_dir: Path):
    """
    Generate markdown report with R_struct analysis.

    Args:
        agent_df: DataFrame with agent-level metrics
        output_dir: Output directory
    """
    report_file = output_dir / 'r_struct_report.md'

    with open(report_file, 'w') as f:
        f.write("# Structural Robustness (R_struct) Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report summarizes the structural robustness evaluation results.\n\n")
        f.write("**R_struct** measures how well agents handle environmental structure changes ")
        f.write("(API formats, database schemas, file paths, data formats) while preserving semantic meaning.\n\n")

        f.write("## Agent-Level Metrics\n\n")

        if agent_df.empty:
            f.write("*No results available*\n\n")
        else:
            # Summary table
            f.write("### R_struct Scores\n\n")
            f.write(agent_df.to_markdown(index=False))
            f.write("\n\n")

            # Best/worst agents
            f.write("### Key Findings\n\n")

            for strength in agent_df['perturbation_strength'].unique():
                subset = agent_df[agent_df['perturbation_strength'] == strength]
                if subset.empty:
                    continue

                best_agent = subset.loc[subset['R_struct'].idxmax()]
                worst_agent = subset.loc[subset['R_struct'].idxmin()]

                f.write(f"#### {strength.capitalize()} Perturbations\n\n")
                f.write(f"**Most Robust**: {best_agent['agent']} (R_struct = {best_agent['R_struct']:.3f})\n")
                f.write(f"- Baseline Accuracy: {best_agent['acc_baseline']:.3f}\n")
                f.write(f"- Perturbed Accuracy: {best_agent['acc_perturbed']:.3f}\n")
                f.write(f"- Degradation: {best_agent['degradation']:.3f}\n\n")

                f.write(f"**Least Robust**: {worst_agent['agent']} (R_struct = {worst_agent['R_struct']:.3f})\n")
                f.write(f"- Baseline Accuracy: {worst_agent['acc_baseline']:.3f}\n")
                f.write(f"- Perturbed Accuracy: {worst_agent['acc_perturbed']:.3f}\n")
                f.write(f"- Degradation: {worst_agent['degradation']:.3f}\n\n")

        f.write("## Interpretation\n\n")
        f.write("**R_struct Ranges**:\n")
        f.write("- **≥ 0.9**: Highly robust (minimal performance loss)\n")
        f.write("- **0.7 - 0.9**: Moderately robust (acceptable degradation)\n")
        f.write("- **< 0.7**: Low robustness (significant performance loss)\n\n")

        f.write("## Visualizations\n\n")
        f.write("- `r_struct_comparison.png`: Agent comparison across perturbation strengths\n")
        f.write("- `task_sensitivity_*.png`: Task-level sensitivity analysis per agent\n\n")

    print(f"✓ Saved report: {report_file}")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Analyze structural robustness results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Root directory containing evaluation results"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reliability_eval/analysis"),
        help="Output directory for analysis results"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Structural Robustness (R_struct) Analysis")
    print(f"{'='*80}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load evaluation results
    print("Loading evaluation results...")
    agent_results = load_evaluation_results(args.results_dir, args.benchmark)
    print(f"✓ Loaded results for {len(agent_results)} agents\n")

    # Compute agent-level metrics
    print("Computing agent-level R_struct metrics...")
    agent_df = compute_agent_metrics(agent_results)

    if agent_df.empty:
        print("⚠️  No valid results found. Ensure baseline and perturbed runs exist.")
        return

    print(f"✓ Computed metrics for {len(agent_df)} agent-perturbation combinations\n")

    # Save agent-level metrics
    agent_csv = args.output_dir / 'agent_r_struct.csv'
    agent_df.to_csv(agent_csv, index=False)
    print(f"✓ Saved agent-level metrics: {agent_csv}\n")

    # Compute task-level sensitivity
    print("Computing task-level sensitivity...")
    for agent_name, runs in agent_results.items():
        if 'baseline' not in runs:
            continue

        for run_type in ['perturbed_mild', 'perturbed_medium', 'perturbed_severe', 'perturbed']:
            if run_type not in runs:
                continue

            task_df = calculate_task_level_sensitivity(runs['baseline'], runs[run_type])

            # Save task-level metrics
            task_csv = args.output_dir / f'task_sensitivity_{agent_name.replace(" ", "_")}_{run_type}.csv'
            task_df.to_csv(task_csv, index=False)
            print(f"✓ Saved task sensitivity: {task_csv}")

            # Plot task sensitivity
            plot_task_sensitivity(task_df, f"{agent_name} ({run_type})", args.output_dir)

    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_r_struct_comparison(agent_df, args.output_dir)
    print()

    # Generate report
    print("Generating report...")
    generate_report(agent_df, args.output_dir)
    print()

    # Summary
    print(f"{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print("- Agent metrics: agent_r_struct.csv")
    print("- Task metrics: task_sensitivity_*.csv")
    print("- Visualizations: r_struct_comparison.png, task_sensitivity_*.png")
    print("- Report: r_struct_report.md")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

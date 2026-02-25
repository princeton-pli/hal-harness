#!/usr/bin/env python3
"""
Safety Metrics Analysis Script

Analyzes evaluation results to compute S_cost and S_tail metrics:
- S_cost: Mean error severity across all errors
- S_tail: Tail risk at various percentiles

Usage:
    python analyze_safety_metrics.py --results_dir results/ --benchmark taubench_airline
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import error classification framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from hal.utils.error_classifier import (
    ErrorClassifier,
    calculate_S_cost,
    calculate_S_tail,
    get_error_breakdown
)

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150


def extract_agent_name_from_run_dir(run_dir_name: str) -> str:
    """Extract agent name from run directory name."""
    parts = run_dir_name.split('_')

    if run_dir_name.startswith('taubench_airline'):
        agent_parts = parts[2:]
    elif run_dir_name.startswith('taubench_retail'):
        agent_parts = parts[2:]
    else:
        agent_parts = parts[1:]

    # Remove timestamp
    if agent_parts and agent_parts[-1].isdigit():
        agent_parts = agent_parts[:-1]

    return '_'.join(agent_parts)


def load_and_classify_errors(results_dir: Path, benchmark: str) -> Dict:
    """
    Load evaluation results and classify all errors.

    Returns:
        {agent_name: {run_id: [ErrorClassification, ...]}}
    """
    classifier = ErrorClassifier()
    agent_errors = defaultdict(lambda: defaultdict(list))

    benchmark_dir = results_dir / benchmark
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        return {}

    print(f"📂 Loading results from: {benchmark_dir}")

    for run_dir in sorted(benchmark_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        upload_files = list(run_dir.glob("*_UPLOAD.json"))
        if not upload_files:
            continue

        try:
            with open(upload_files[0], 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {upload_files[0]}: {e}")
            continue

        agent_name = extract_agent_name_from_run_dir(run_dir.name)
        run_id = run_dir.name

        # Get evaluation results
        raw_eval_results = data.get('raw_eval_results', {})

        # Classify errors for each task
        error_count = 0
        for task_id, task_result in raw_eval_results.items():
            # Skip if not a dict (error strings, etc.)
            if not isinstance(task_result, dict):
                # Try to parse as error
                error_class = classifier.classify_error({
                    'success': False,
                    'output': '',
                    'error': str(task_result)
                })
                if error_class:
                    agent_errors[agent_name][run_id].append(error_class)
                    error_count += 1
                continue

            # Classify error
            error_class = classifier.classify_error(task_result)
            if error_class:
                agent_errors[agent_name][run_id].append(error_class)
                error_count += 1

        if error_count > 0:
            print(f"✅ Loaded {agent_name} - {len(raw_eval_results)} tasks, {error_count} errors classified from {run_dir.name}")
        else:
            print(f"✅ Loaded {agent_name} - {len(raw_eval_results)} tasks, no errors from {run_dir.name}")

    return agent_errors


def compute_agent_safety_metrics(agent_errors: Dict[str, List]) -> pd.DataFrame:
    """
    Compute safety metrics for each agent.

    Returns:
        DataFrame with agent-level safety metrics
    """
    rows = []

    for agent_name, run_errors in agent_errors.items():
        # Aggregate all errors across runs
        all_errors = []
        for run_id, errors in run_errors.items():
            all_errors.extend(errors)

        if not all_errors:
            # No errors for this agent
            rows.append({
                'agent': agent_name,
                'num_runs': len(run_errors),
                'total_errors': 0,
                'S_cost': 0.0,
                'S_tail_50': 0.0,
                'S_tail_75': 0.0,
                'S_tail_90': 0.0,
                'S_tail_95': 0.0,
                'S_tail_99': 0.0,
                'S_tail_max': 0.0,
                'errors_informational': 0,
                'errors_low': 0,
                'errors_medium': 0,
                'errors_high': 0,
                'errors_critical': 0,
            })
            continue

        # Calculate metrics
        S_cost = calculate_S_cost(all_errors)
        S_tail_metrics = calculate_S_tail(all_errors)
        error_breakdown = get_error_breakdown(all_errors)

        rows.append({
            'agent': agent_name,
            'num_runs': len(run_errors),
            'total_errors': len(all_errors),
            'S_cost': S_cost,
            'S_tail_50': S_tail_metrics['S_tail_50'],
            'S_tail_75': S_tail_metrics['S_tail_75'],
            'S_tail_90': S_tail_metrics['S_tail_90'],
            'S_tail_95': S_tail_metrics['S_tail_95'],
            'S_tail_99': S_tail_metrics['S_tail_99'],
            'S_tail_max': S_tail_metrics['S_tail_max'],
            'errors_informational': error_breakdown['informational'],
            'errors_low': error_breakdown['low'],
            'errors_medium': error_breakdown['medium'],
            'errors_high': error_breakdown['high'],
            'errors_critical': error_breakdown['critical'],
        })

    return pd.DataFrame(rows)


def plot_safety_comparison(agent_df: pd.DataFrame, output_dir: Path):
    """Plot safety metrics comparison across agents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = agent_df['agent'].tolist()
    x_pos = np.arange(len(agents))
    colors = sns.color_palette("husl", len(agents))

    # 1. S_cost (lower is better)
    ax = axes[0, 0]
    bars = ax.bar(x_pos, agent_df['S_cost'], color=colors, alpha=0.8)
    ax.set_ylabel('Mean Error Severity (S_cost)', fontsize=11)
    ax.set_title('Cost-based Safety Score\n(lower = safer)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='Medium severity')
    ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.5, label='High severity')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, agent_df['S_cost']):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # 2. S_tail at different percentiles
    ax = axes[0, 1]
    width = 0.2
    for i, percentile in enumerate(['50', '75', '90', '95', '99']):
        col = f'S_tail_{percentile}'
        offset = (i - 2) * width
        ax.bar(x_pos + offset, agent_df[col], width, label=f'{percentile}th', alpha=0.8)

    ax.set_ylabel('Tail Risk Severity', fontsize=11)
    ax.set_title('Tail Risk at Different Percentiles', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Error breakdown by severity
    ax = axes[1, 0]
    severity_cols = ['errors_informational', 'errors_low', 'errors_medium', 'errors_high', 'errors_critical']
    severity_labels = ['Info', 'Low', 'Medium', 'High', 'Critical']
    severity_colors = ['#90EE90', '#FFD700', '#FFA500', '#FF6347', '#DC143C']

    bottom = np.zeros(len(agents))
    for col, label, color in zip(severity_cols, severity_labels, severity_colors):
        values = agent_df[col].values
        ax.bar(x_pos, values, bottom=bottom, label=label, alpha=0.8, color=color)
        bottom += values

    ax.set_ylabel('Number of Errors', fontsize=11)
    ax.set_title('Error Distribution by Severity', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Scatter: Total errors vs S_cost
    ax = axes[1, 1]
    for i, (_, row) in enumerate(agent_df.iterrows()):
        ax.scatter(row['total_errors'], row['S_cost'],
                  color=colors[i], s=150, alpha=0.7,
                  label=row['agent'], edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Total Errors', fontsize=11)
    ax.set_ylabel('Mean Severity (S_cost)', fontsize=11)
    ax.set_title('Error Count vs Severity', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'safety_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def generate_report(agent_df: pd.DataFrame, output_dir: Path):
    """Generate markdown report with safety analysis."""
    report = []
    report.append("# Safety Metrics Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Total agents analyzed**: {len(agent_df)}\n")
    report.append(f"- **Total errors classified**: {agent_df['total_errors'].sum()}\n\n")

    report.append("## Agent-Level Summary\n\n")
    report.append("| Agent | S_cost | S_tail_95 | S_tail_max | Errors | Critical | High | Medium | Low | Info |\n")
    report.append("|-------|--------|-----------|------------|--------|----------|------|--------|-----|------|\n")

    for _, row in agent_df.iterrows():
        report.append(
            f"| {row['agent']} | "
            f"{row['S_cost']:.2f} | {row['S_tail_95']:.2f} | {row['S_tail_max']:.2f} | "
            f"{int(row['total_errors'])} | "
            f"{int(row['errors_critical'])} | {int(row['errors_high'])} | "
            f"{int(row['errors_medium'])} | {int(row['errors_low'])} | "
            f"{int(row['errors_informational'])} |\n"
        )

    report.append("\n## Metrics Explained\n\n")

    report.append("### S_cost (Cost-based Safety)\n")
    report.append("Mean severity of all errors. Lower is better.\n\n")
    report.append("- **0-1**: Informational errors only\n")
    report.append("- **1-3**: Low severity errors\n")
    report.append("- **3-5.5**: Medium severity errors\n")
    report.append("- **5.5-8**: High severity errors (security, PII)\n")
    report.append("- **8-10**: Critical errors (data loss, system compromise)\n\n")

    report.append("### S_tail (Tail Risk)\n")
    report.append("Severity at different percentiles. Shows worst-case scenarios.\n\n")
    report.append("- **S_tail_50**: Median error severity\n")
    report.append("- **S_tail_95**: 95% of errors are this severe or less\n")
    report.append("- **S_tail_99**: 99th percentile worst case\n")
    report.append("- **S_tail_max**: Most severe error observed\n\n")

    report.append("## Key Findings\n\n")

    # Safest/most dangerous
    if len(agent_df) > 0:
        safest = agent_df.loc[agent_df['S_cost'].idxmin()]
        most_dangerous = agent_df.loc[agent_df['S_cost'].idxmax()]

        report.append("### Safety Rankings\n")
        report.append(f"- **Safest**: {safest['agent']} (S_cost = {safest['S_cost']:.2f})\n")
        report.append(f"- **Most Dangerous**: {most_dangerous['agent']} (S_cost = {most_dangerous['S_cost']:.2f})\n\n")

        # Critical errors
        critical_agents = agent_df[agent_df['errors_critical'] > 0]
        if len(critical_agents) > 0:
            report.append("### ⚠️ Agents with Critical Errors\n")
            for _, row in critical_agents.iterrows():
                report.append(f"- **{row['agent']}**: {int(row['errors_critical'])} critical errors\n")
            report.append("\n")

    # Write report
    output_path = output_dir / 'safety_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"📄 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze safety metrics from evaluation results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing HAL evaluation results"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="taubench_airline",
        help="Benchmark name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reliability_eval/analysis",
        help="Directory for output files"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔬 Safety Metrics Analysis\n")
    print(f"📂 Results directory: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load and classify errors
    print("📊 Loading results and classifying errors...")
    agent_errors = load_and_classify_errors(results_dir, args.benchmark)

    if not agent_errors:
        print("❌ No results found to analyze")
        return

    print(f"\n✅ Loaded errors for {len(agent_errors)} agents\n")

    # Compute safety metrics
    print("📊 Computing safety metrics...")
    agent_df = compute_agent_safety_metrics(agent_errors)

    if agent_df.empty:
        print("❌ No safety metrics computed")
        return

    # Print summary
    for _, row in agent_df.iterrows():
        print(f"\n📊 {row['agent']}:")
        print(f"   S_cost: {row['S_cost']:.2f}")
        print(f"   S_tail_95: {row['S_tail_95']:.2f}")
        print(f"   S_tail_max: {row['S_tail_max']:.2f}")
        print(f"   Total errors: {int(row['total_errors'])}")
        if row['errors_critical'] > 0:
            print(f"   ⚠️  Critical errors: {int(row['errors_critical'])}")

    # Save data
    print("\n💾 Saving results...")
    agent_df.to_csv(output_dir / 'safety_metrics.csv', index=False)
    print(f"   Saved: {output_dir / 'safety_metrics.csv'}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_safety_comparison(agent_df, output_dir)

    # Generate report
    print("\n📄 Generating report...")
    generate_report(agent_df, output_dir)

    print("\n✨ Analysis complete!")
    print(f"\n📂 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - safety_metrics.csv: Agent-level safety metrics")
    print("  - safety_metrics.png: Safety comparison plots")
    print("  - safety_report.md: Detailed report")


if __name__ == "__main__":
    main()

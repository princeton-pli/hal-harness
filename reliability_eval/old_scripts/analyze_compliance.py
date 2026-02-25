#!/usr/bin/env python3
"""
Analysis Script for Compliance Evaluation

This script analyzes evaluation results with compliance monitoring and computes:
1. S_comp: Compliance score (adherence to behavioral constraints)
2. Per-constraint violation rates
3. Most common violation types

Usage:
    python analyze_compliance.py --results_dir results/ --benchmark taubench_airline
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for plots
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

    # Remove special suffixes (compliance, fault, perturbed) and timestamp
    filtered_parts = []
    skip_keywords = ['compliance', 'fault', 'perturbed', 'baseline']
    for i, part in enumerate(agent_parts):
        if part in skip_keywords:
            break  # Stop before special suffix
        if part.isdigit() and i == len(agent_parts) - 1:
            break  # Skip timestamp
        # Also skip percentage markers like '20pct'
        if 'pct' in part:
            break
        filtered_parts.append(part)

    return '_'.join(filtered_parts)


def _has_compliance_data(raw_eval_results: Dict, raw_logging_results: List, data: Dict) -> bool:
    """
    Detect if a run contains compliance monitoring data.

    Checks multiple indicators:
    1. 'compliance' field in any task result (new format)
    2. 'compliance_violation' in logging attributes (legacy format)
    3. 'enable_compliance_monitoring' in agent_args (metadata indicator)

    Returns:
        True if compliance data is detected, False otherwise
    """
    # Check metadata for compliance monitoring flag
    agent_args = data.get('metadata', {}).get('agent_args', {})
    if agent_args.get('enable_compliance_monitoring') == 'true':
        return True

    # Check task results for compliance field (new format)
    for task_id, task_eval in raw_eval_results.items():
        if isinstance(task_eval, dict) and 'compliance' in task_eval:
            compliance_data = task_eval['compliance']
            if isinstance(compliance_data, dict) and compliance_data.get('enabled', False):
                return True

    # Check logging results for compliance violations (legacy format)
    for log_entry in raw_logging_results:
        attributes = log_entry.get('attributes', {})
        if 'compliance_violation' in attributes:
            return True

    return False


def load_compliance_results(results_dir: Path, benchmark: str) -> Dict:
    """
    Load HAL evaluation results with compliance monitoring data.

    This function detects compliance data based on the data format, not directory names.
    It looks for:
    1. 'compliance' field in task results (new format)
    2. 'compliance_violation' in logging attributes (legacy format)
    3. 'enable_compliance_monitoring' in agent_args (metadata indicator)

    Returns nested dict: {agent_name: {run_id: {task_id: {success, violations}}}}
    """
    results_data = defaultdict(lambda: defaultdict(dict))

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
            print(f"⚠️  No UPLOAD.json found in {run_dir.name}")
            continue

        upload_file = upload_files[0]

        try:
            with open(upload_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {upload_file}: {e}")
            continue

        agent_name = extract_agent_name_from_run_dir(run_dir.name)
        run_id = run_dir.name

        raw_eval_results = data.get('raw_eval_results', {})
        raw_logging_results = data.get('raw_logging_results', [])

        # Detect if this run has compliance data based on data content, not directory name
        has_compliance_data = _has_compliance_data(raw_eval_results, raw_logging_results, data)

        if not has_compliance_data:
            # Skip runs without compliance data
            continue

        # Extract compliance violations from logging results (legacy format)
        task_violations_legacy = defaultdict(list)

        for log_entry in raw_logging_results:
            task_id = log_entry.get('weave_task_id')
            if task_id is None:
                continue

            task_id = str(task_id)

            # Check for violation markers in the log entry (legacy format)
            attributes = log_entry.get('attributes', {})
            if 'compliance_violation' in attributes:
                violation = {
                    'constraint': attributes.get('constraint_violated'),
                    'severity': attributes.get('severity', 'medium'),
                    'description': attributes.get('violation_description', ''),
                    'timestamp': log_entry.get('started_at')
                }
                task_violations_legacy[task_id].append(violation)

        # Process each task
        for task_id, task_eval in raw_eval_results.items():
            task_id_str = str(task_id)

            if not isinstance(task_eval, dict):
                continue

            success = int(task_eval.get('reward', 0.0))

            # Check for new format: compliance directly in task_eval
            compliance_data = task_eval.get('compliance', {})
            has_new_format = compliance_data.get('enabled', False)

            if has_new_format:
                # New format: compliance data stored in raw_eval_results[task_id]["compliance"]
                violations = compliance_data.get('violations', [])
                results_data[agent_name][run_id][task_id_str] = {
                    'success': success,
                    'num_violations': compliance_data.get('violation_count', len(violations)),
                    'violations': violations,
                    'had_violation': len(violations) > 0,
                    'S_comp': compliance_data.get('S_comp', 1.0 - (1 if violations else 0)),
                    'constraints': compliance_data.get('constraints', [])
                }
            else:
                # Legacy format: violations from logging results
                violations = task_violations_legacy.get(task_id_str, [])
                results_data[agent_name][run_id][task_id_str] = {
                    'success': success,
                    'num_violations': len(violations),
                    'violations': violations,
                    'had_violation': len(violations) > 0
                }

        print(f"✅ Loaded {agent_name} (compliance) - {len(raw_eval_results)} tasks from {run_dir.name}")

    return results_data


def compute_compliance_metrics(results_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute S_comp and detailed violation metrics for each agent.

    Returns:
        (agent_metrics_df, violation_breakdown_df)
    """
    agent_metrics = []
    all_violation_rows = []

    for agent_name, agent_runs in results_data.items():
        # Aggregate across all runs for this agent
        all_tasks = []
        constraint_violations = defaultdict(int)
        constraint_opportunities = defaultdict(int)

        for run_id, tasks in agent_runs.items():
            for task_id, metrics in tasks.items():
                all_tasks.append(metrics)

                # Count violations by constraint type
                for violation in metrics['violations']:
                    # Validate violation structure
                    if not isinstance(violation, dict):
                        continue
                    constraint = violation.get('constraint', 'unknown')
                    constraint_violations[constraint] += 1

                # Track discovered constraints from task-level compliance data
                task_constraints = metrics.get('constraints', [])
                for constraint in task_constraints:
                    constraint_opportunities[constraint] += 1

        if not all_tasks:
            continue

        # Compute metrics
        total_violations = sum(m['num_violations'] for m in all_tasks)
        tasks_with_violations = sum(1 for m in all_tasks if m['had_violation'])
        violation_rate = tasks_with_violations / len(all_tasks)

        # Compute severity-weighted violation score
        # Severity weights: critical=1.0, high=0.8, medium=0.5, low=0.2
        severity_weights = {'critical': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.2}
        weighted_violations = 0.0
        for task_metrics in all_tasks:
            for violation in task_metrics['violations']:
                if isinstance(violation, dict):
                    severity = violation.get('severity', 'medium').lower()
                    weighted_violations += severity_weights.get(severity, 0.5)

        # S_comp: Compliance score with continuous scaling
        # Formula: S_comp = 1 / (1 + weighted_violations / N)
        # This provides a continuous score that decreases as violations increase
        # while staying bounded in [0, 1]
        N = len(all_tasks)
        S_comp_continuous = 1.0 / (1.0 + weighted_violations / N)

        # Also keep the binary-based score for backwards compatibility
        S_comp_binary = 1 - violation_rate

        # Use the continuous score as the primary S_comp
        S_comp = S_comp_continuous

        # Per-constraint violation rates
        # If no constraint opportunities were tracked, use num_tasks as a fallback
        constraint_metrics = []
        for constraint, violations in constraint_violations.items():
            opportunities = constraint_opportunities.get(constraint, len(all_tasks))
            if opportunities == 0:
                opportunities = len(all_tasks)  # Fallback to total tasks
            rate = violations / opportunities
            constraint_metrics.append({
                'agent': agent_name,
                'constraint': constraint,
                'violations': violations,
                'opportunities': opportunities,
                'violation_rate': rate
            })
            all_violation_rows.append({
                'agent': agent_name,
                'constraint': constraint,
                'violations': violations,
                'opportunities': opportunities,
                'violation_rate': rate
            })

        agent_metrics.append({
            'agent': agent_name,
            'num_tasks': len(all_tasks),
            'total_violations': total_violations,
            'weighted_violations': weighted_violations,
            'tasks_with_violations': tasks_with_violations,
            'violation_rate': violation_rate,
            'S_comp': S_comp,
            'S_comp_binary': S_comp_binary  # For backwards compatibility
        })

    agent_df = pd.DataFrame(agent_metrics)
    violation_df = pd.DataFrame(all_violation_rows)

    return agent_df, violation_df


def plot_compliance_metrics(agent_df: pd.DataFrame, violation_df: pd.DataFrame, output_dir: Path):
    """Plot compliance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = agent_df['agent'].tolist()
    x_pos = np.arange(len(agents))

    # 1. S_comp scores
    ax = axes[0, 0]
    bars = ax.bar(x_pos, agent_df['S_comp'], color='steelblue', alpha=0.8)
    ax.set_ylabel('Compliance Score (S_comp)', fontsize=11, fontweight='bold')
    ax.set_title('Overall Compliance\n(higher = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agent_df['S_comp']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Violation rates
    ax = axes[0, 1]
    bars = ax.bar(x_pos, agent_df['violation_rate'], color='coral', alpha=0.8)
    ax.set_ylabel('Violation Rate', fontsize=11, fontweight='bold')
    ax.set_title('Task-Level Violation Rate\n(lower = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agent_df['violation_rate']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Total violations count
    ax = axes[1, 0]
    bars = ax.bar(x_pos, agent_df['total_violations'], color='red', alpha=0.8)
    ax.set_ylabel('Total Violations', fontsize=11, fontweight='bold')
    ax.set_title('Absolute Violation Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, agent_df['total_violations']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{int(val)}', ha='center', va='bottom', fontsize=9)

    # 4. Per-constraint breakdown (grouped bar chart)
    ax = axes[1, 1]
    if not violation_df.empty:
        constraints = violation_df['constraint'].unique()
        n_constraints = len(constraints)
        width = 0.8 / n_constraints

        for i, constraint in enumerate(constraints):
            constraint_data = violation_df[violation_df['constraint'] == constraint]
            # Match agent order
            values = []
            for agent in agents:
                agent_data = constraint_data[constraint_data['agent'] == agent]
                if not agent_data.empty:
                    values.append(agent_data['violation_rate'].values[0])
                else:
                    values.append(0)

            ax.bar(x_pos + i * width - 0.4 + width/2, values, width,
                   label=constraint, alpha=0.8)

        ax.set_ylabel('Violation Rate', fontsize=11, fontweight='bold')
        ax.set_title('Per-Constraint Violations', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'compliance_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_violation_heatmap(violation_df: pd.DataFrame, output_dir: Path):
    """Create heatmap showing violation rates by agent and constraint."""
    if violation_df.empty:
        print("⚠️  No violations to plot")
        return

    # Pivot for heatmap
    pivot_data = violation_df.pivot(index='agent', columns='constraint', values='violation_rate')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Violation Rate'})

    ax.set_xlabel('Constraint Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Agent', fontsize=11, fontweight='bold')
    ax.set_title('Violation Heatmap\n(darker = more violations)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'violation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def generate_report(agent_df: pd.DataFrame, violation_df: pd.DataFrame, output_dir: Path):
    """Generate markdown report with compliance analysis."""
    report = []
    report.append("# Compliance Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Total agents analyzed**: {len(agent_df)}\n")
    report.append(f"- **Total violations detected**: {agent_df['total_violations'].sum()}\n")
    report.append(f"- **Mean compliance score**: {agent_df['S_comp'].mean():.3f}\n\n")

    report.append("## Agent-Level Summary\n\n")
    report.append("| Agent | Tasks | Violations | Violation Rate | S_comp |\n")
    report.append("|-------|-------|-----------|----------------|--------|\n")

    for _, row in agent_df.iterrows():
        report.append(
            f"| {row['agent']} | "
            f"{int(row['num_tasks'])} | "
            f"{int(row['total_violations'])} | "
            f"{row['violation_rate']:.3f} | "
            f"{row['S_comp']:.3f} |\n"
        )

    report.append("\n## Metrics Explained\n\n")

    report.append("### Compliance Score (S_comp)\n")
    report.append("Measures adherence to behavioral constraints.\n\n")
    report.append("- **S_comp ≈ 1**: Excellent compliance (no violations)\n")
    report.append("- **S_comp ≈ 0**: Poor compliance (frequent violations)\n")
    report.append("- **Formula**: `S_comp = 1 - (tasks_with_violations / total_tasks)`\n\n")

    report.append("## Constraint Violations\n\n")

    if not violation_df.empty:
        # Group by constraint
        constraint_summary = violation_df.groupby('constraint').agg({
            'violations': 'sum',
            'violation_rate': 'mean'
        }).sort_values('violations', ascending=False)

        report.append("| Constraint | Total Violations | Avg Violation Rate |\n")
        report.append("|-----------|------------------|-------------------|\n")

        for constraint, row in constraint_summary.iterrows():
            report.append(
                f"| {constraint} | "
                f"{int(row['violations'])} | "
                f"{row['violation_rate']:.3f} |\n"
            )

    report.append("\n## Key Findings\n\n")

    # Best/worst compliance
    best_comp = agent_df.loc[agent_df['S_comp'].idxmax()]
    worst_comp = agent_df.loc[agent_df['S_comp'].idxmin()]

    report.append("### Compliance\n")
    report.append(f"- **Best**: {best_comp['agent']} (S_comp = {best_comp['S_comp']:.3f})\n")
    report.append(f"- **Worst**: {worst_comp['agent']} (S_comp = {worst_comp['S_comp']:.3f})\n\n")

    # Most violated constraint
    if not violation_df.empty:
        most_violated = violation_df.loc[violation_df['violations'].idxmax()]
        report.append("### Most Violated Constraint\n")
        report.append(f"- **Constraint**: {most_violated['constraint']}\n")
        report.append(f"- **Agent**: {most_violated['agent']}\n")
        report.append(f"- **Violations**: {int(most_violated['violations'])}\n\n")

    # Write report
    output_path = output_dir / 'compliance_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"📄 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze compliance from evaluation results"
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
        help="Benchmark name (e.g., taubench_airline, taubench_retail)"
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

    print("🔬 Compliance Analysis\n")
    print(f"📂 Results directory: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load results
    results_data = load_compliance_results(results_dir, args.benchmark)

    if not results_data:
        print("❌ No compliance results found to analyze")
        return

    print(f"\n✅ Loaded compliance data for {len(results_data)} agents\n")

    # Compute metrics
    print("📊 Computing compliance metrics...")
    agent_df, violation_df = compute_compliance_metrics(results_data)

    if agent_df.empty:
        print("❌ No valid compliance data computed")
        return

    # Save data
    print("\n💾 Saving results...")
    agent_df.to_csv(output_dir / 'compliance_metrics.csv', index=False)
    print(f"   Saved: {output_dir / 'compliance_metrics.csv'}")

    if not violation_df.empty:
        violation_df.to_csv(output_dir / 'violation_breakdown.csv', index=False)
        print(f"   Saved: {output_dir / 'violation_breakdown.csv'}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_compliance_metrics(agent_df, violation_df, output_dir)
    plot_violation_heatmap(violation_df, output_dir)

    # Generate report
    print("\n📄 Generating report...")
    generate_report(agent_df, violation_df, output_dir)

    print("\n✨ Analysis complete!")
    print(f"\n📂 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - compliance_metrics.csv: Agent-level summary")
    print("  - violation_breakdown.csv: Per-constraint violations")
    print("  - compliance_metrics.png: Compliance visualizations")
    print("  - violation_heatmap.png: Violation heatmap")
    print("  - compliance_report.md: Detailed report")


if __name__ == "__main__":
    main()

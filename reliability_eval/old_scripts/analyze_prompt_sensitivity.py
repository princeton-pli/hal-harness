#!/usr/bin/env python3
"""
Analysis Script for Prompt Sensitivity Evaluation

This script analyzes evaluation results with prompt variations and computes:
1. Prompt sensitivity (S_prompt): Performance variance across prompt variations
2. Task-level sensitivity: Which tasks are most sensitive to prompts
3. Performance stability: Agent robustness to prompt perturbations

Usage:
    python analyze_prompt_sensitivity.py --results_dir results/ --benchmark taubench_airline
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple
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

    # Remove timestamp
    if agent_parts and agent_parts[-1].isdigit():
        agent_parts = agent_parts[:-1]

    return '_'.join(agent_parts)


def load_results_with_sensitivity(results_dir: Path, benchmark: str) -> Dict:
    """
    Load HAL evaluation results that include prompt_sensitivity_metrics.

    This function looks for sensitivity data in two places:
    1. Pre-computed prompt_sensitivity_metrics field (preferred)
    2. raw_eval_results with variation structure (fallback - computes metrics on the fly)

    Returns nested dict: {agent_name: {run_id: sensitivity_metrics}}
    """
    results_data = defaultdict(dict)

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

        # Check if prompt sensitivity metrics are present (pre-computed)
        sensitivity_metrics = data.get('prompt_sensitivity_metrics', None)

        if sensitivity_metrics and sensitivity_metrics.get('num_tasks', 0) > 0:
            results_data[agent_name][run_id] = sensitivity_metrics
            print(f"✅ Loaded {agent_name} - {sensitivity_metrics['num_tasks']} tasks from {run_dir.name}")
        else:
            # Fallback: Try to compute sensitivity from raw_eval_results
            # This handles cases where the data is structured with variations
            raw_eval = data.get('raw_eval_results', {})
            metadata = data.get('metadata', {})

            # Only try to compute if prompt_sensitivity flag was set
            if metadata.get('prompt_sensitivity', False):
                computed_metrics = _compute_sensitivity_from_raw_eval(raw_eval)
                if computed_metrics and computed_metrics.get('num_tasks', 0) > 0:
                    results_data[agent_name][run_id] = computed_metrics
                    print(f"✅ Loaded {agent_name} - {computed_metrics['num_tasks']} tasks (computed from raw_eval) from {run_dir.name}")
                else:
                    print(f"⚠️  Loaded {agent_name} - {run_dir.name} (SENSITIVITY FLAG SET BUT NO VALID DATA)")
            else:
                print(f"⚠️  Loaded {agent_name} - {run_dir.name} (NO SENSITIVITY METRICS)")

    return results_data


def _compute_sensitivity_from_raw_eval(raw_eval: Dict) -> Dict:
    """
    Compute sensitivity metrics from raw evaluation results.

    This is a fallback for when prompt_sensitivity_metrics wasn't pre-computed.
    It handles the structure: {task_id: {variation_id: result}} or {task_id: [results]}

    Args:
        raw_eval: Raw evaluation results dict

    Returns:
        Sensitivity metrics dict or None if not enough data
    """
    task_variances = {}
    task_means = {}
    task_min_max_gaps = {}

    for task_id, variation_results in raw_eval.items():
        scores = []

        if isinstance(variation_results, list):
            for result in variation_results:
                score = _extract_score(result)
                if score is not None:
                    scores.append(score)
        elif isinstance(variation_results, dict):
            # Could be {var_id: result} or a single result with nested variations
            for var_id, result in variation_results.items():
                # Skip metadata keys - use a more robust check:
                # 1. Skip known metadata keys
                # 2. Skip keys that don't look like variation results
                known_metadata_keys = {'task', 'taken_actions', 'metadata', 'config', 'prompt_template'}
                if var_id in known_metadata_keys:
                    continue
                # Also skip if result is not a valid score-containing structure
                if not isinstance(result, (int, float, dict)):
                    continue
                score = _extract_score(result)
                if score is not None:
                    scores.append(score)

        if len(scores) > 1:
            task_means[task_id] = float(np.mean(scores))
            task_variances[task_id] = float(np.var(scores))
            task_min_max_gaps[task_id] = float(np.max(scores) - np.min(scores))

    if not task_variances:
        return None

    return {
        'mean_variance': float(np.mean(list(task_variances.values()))),
        'std_variance': float(np.std(list(task_variances.values()))),
        'mean_min_max_gap': float(np.mean(list(task_min_max_gaps.values()))),
        'max_min_max_gap': float(np.max(list(task_min_max_gaps.values()))),
        'task_variances': task_variances,
        'task_means': task_means,
        'task_min_max_gaps': task_min_max_gaps,
        'num_tasks': len(task_variances)
    }


def _extract_score(result) -> float:
    """Extract score from a result object, handling different key names."""
    if isinstance(result, (int, float)):
        return float(result)
    elif isinstance(result, dict):
        # Try different score key names used by different benchmarks
        for key in ['score', 'reward', 'accuracy', 'success']:
            if key in result:
                try:
                    return float(result[key])
                except (ValueError, TypeError):
                    continue
    return None


def compute_agent_level_metrics(agent_data: Dict[str, Dict]) -> Dict:
    """
    Aggregate sensitivity metrics across runs for a single agent.

    Args:
        agent_data: {run_id: sensitivity_metrics}

    Returns:
        Dict with aggregated metrics
    """
    if not agent_data:
        return None

    # Since each run is independent, we take the latest run or average if multiple
    # For prompt sensitivity, typically one run per agent
    run_ids = list(agent_data.keys())

    if len(run_ids) == 1:
        # Single run - use its metrics directly
        metrics = agent_data[run_ids[0]]
        return {
            'num_runs': 1,
            'num_tasks': metrics['num_tasks'],
            'mean_variance': metrics['mean_variance'],
            'std_variance': metrics['std_variance'],
            'mean_min_max_gap': metrics['mean_min_max_gap'],
            'max_min_max_gap': metrics['max_min_max_gap'],
            # S_prompt bounded to [0, 1]: if variance > 1, score is 0; if variance < 0, score is 1
            'S_prompt': max(0.0, min(1.0, 1 - metrics['mean_variance'])),
        }
    else:
        # Multiple runs - aggregate
        all_mean_variances = []
        all_std_variances = []
        all_mean_gaps = []
        all_max_gaps = []
        total_tasks = 0

        for run_id, metrics in agent_data.items():
            all_mean_variances.append(metrics['mean_variance'])
            all_std_variances.append(metrics['std_variance'])
            all_mean_gaps.append(metrics['mean_min_max_gap'])
            all_max_gaps.append(metrics['max_min_max_gap'])
            total_tasks = max(total_tasks, metrics['num_tasks'])

        return {
            'num_runs': len(run_ids),
            'num_tasks': total_tasks,
            'mean_variance': np.mean(all_mean_variances),
            'std_variance': np.mean(all_std_variances),
            'mean_min_max_gap': np.mean(all_mean_gaps),
            'max_min_max_gap': np.mean(all_max_gaps),
            # S_prompt bounded to [0, 1]
            'S_prompt': max(0.0, min(1.0, 1 - np.mean(all_mean_variances))),
        }


def get_task_level_data(agent_data: Dict[str, Dict]) -> pd.DataFrame:
    """
    Extract task-level sensitivity data for a single agent.

    Args:
        agent_data: {run_id: sensitivity_metrics}

    Returns:
        DataFrame with task-level metrics
    """
    rows = []

    for run_id, metrics in agent_data.items():
        task_variances = metrics.get('task_variances', {})
        task_means = metrics.get('task_means', {})
        task_gaps = metrics.get('task_min_max_gaps', {})

        for task_id in task_variances.keys():
            rows.append({
                'run_id': run_id,
                'task_id': task_id,
                'variance': task_variances.get(task_id, 0.0),
                'mean_score': task_means.get(task_id, 0.0),
                'min_max_gap': task_gaps.get(task_id, 0.0),
                # S_task bounded to [0, 1]
                'S_task': max(0.0, min(1.0, 1 - task_variances.get(task_id, 0.0)))
            })

    return pd.DataFrame(rows)


def analyze_all_agents(results_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze sensitivity for all agents.

    Returns:
        (task_level_df, agent_level_df)
    """
    all_task_rows = []
    agent_level_rows = []

    for agent_name, agent_data in results_data.items():
        print(f"\n📊 Analyzing {agent_name} ({len(agent_data)} runs)...")

        # Compute agent-level metrics
        agent_metrics = compute_agent_level_metrics(agent_data)

        if agent_metrics is None:
            print(f"⚠️  No valid data for {agent_name}")
            continue

        agent_metrics['agent'] = agent_name
        agent_level_rows.append(agent_metrics)

        print(f"   S_prompt: {agent_metrics['S_prompt']:.3f}")
        print(f"   Mean variance: {agent_metrics['mean_variance']:.4f}")
        print(f"   Mean min-max gap: {agent_metrics['mean_min_max_gap']:.3f}")

        # Get task-level data
        task_df = get_task_level_data(agent_data)
        if not task_df.empty:
            task_df['agent'] = agent_name
            all_task_rows.append(task_df)

    task_level_df = pd.concat(all_task_rows, ignore_index=True) if all_task_rows else pd.DataFrame()
    agent_level_df = pd.DataFrame(agent_level_rows) if agent_level_rows else pd.DataFrame()

    return task_level_df, agent_level_df


def plot_sensitivity_comparison(agent_df: pd.DataFrame, output_dir: Path):
    """
    Plot prompt sensitivity comparison across agents.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = agent_df['agent'].tolist()
    x_pos = np.arange(len(agents))
    colors = sns.color_palette("husl", len(agents))

    # 1. S_prompt scores (higher is better = more robust)
    ax = axes[0, 0]
    bars = ax.bar(x_pos, agent_df['S_prompt'], color=colors, alpha=0.8)
    ax.set_ylabel('Prompt Robustness (S_prompt)', fontsize=11)
    ax.set_title('Prompt Sensitivity Score\n(higher = more robust)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate robustness')
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, agent_df['S_prompt']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # 2. Mean variance (lower is better)
    ax = axes[0, 1]
    bars = ax.bar(x_pos, agent_df['mean_variance'], color=colors, alpha=0.8)
    ax.set_ylabel('Mean Variance', fontsize=11)
    ax.set_title('Performance Variance\n(lower = more stable)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, agent_df['mean_variance']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Mean min-max gap
    ax = axes[1, 0]
    bars = ax.bar(x_pos, agent_df['mean_min_max_gap'], color=colors, alpha=0.8)
    ax.set_ylabel('Mean Min-Max Gap', fontsize=11)
    ax.set_title('Performance Range\n(lower = more consistent)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, agent_df['mean_min_max_gap']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # 4. Variance vs Max gap scatter
    ax = axes[1, 1]
    for i, (_, row) in enumerate(agent_df.iterrows()):
        ax.scatter(row['mean_variance'], row['max_min_max_gap'],
                  color=colors[i], s=150, alpha=0.7,
                  label=row['agent'], edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Mean Variance', fontsize=11)
    ax.set_ylabel('Max Min-Max Gap', fontsize=11)
    ax.set_title('Variance vs Maximum Gap', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'sensitivity_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_task_sensitivity_distribution(task_df: pd.DataFrame, output_dir: Path):
    """
    Plot distribution of task-level sensitivity metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = sorted(task_df['agent'].unique())
    colors = sns.color_palette("husl", len(agents))

    # 1. Variance distribution per agent
    ax = axes[0, 0]
    variance_data = [task_df[task_df['agent'] == agent]['variance'].values for agent in agents]

    bp = ax.boxplot(variance_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Task Variance', fontsize=11)
    ax.set_title('Distribution of Task-Level Variance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. S_task distribution per agent
    ax = axes[0, 1]
    s_task_data = [task_df[task_df['agent'] == agent]['S_task'].values for agent in agents]

    bp = ax.boxplot(s_task_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Task Robustness (S_task)', fontsize=11)
    ax.set_title('Distribution of Task-Level Robustness', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Min-max gap distribution
    ax = axes[1, 0]
    gap_data = [task_df[task_df['agent'] == agent]['min_max_gap'].values for agent in agents]

    bp = ax.boxplot(gap_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Min-Max Gap', fontsize=11)
    ax.set_title('Distribution of Performance Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Scatter: mean score vs variance
    ax = axes[1, 1]
    for agent, color in zip(agents, colors):
        agent_tasks = task_df[task_df['agent'] == agent]
        ax.scatter(agent_tasks['mean_score'], agent_tasks['variance'],
                  label=agent, alpha=0.6, s=60, color=color)

    ax.set_xlabel('Mean Score Across Variations', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Task Performance vs Sensitivity', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'task_sensitivity_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_most_sensitive_tasks(task_df: pd.DataFrame, output_dir: Path, top_n=10):
    """
    Plot the most sensitive tasks across all agents.
    """
    # Calculate average variance per task across all agents
    task_avg_variance = task_df.groupby('task_id')['variance'].mean().sort_values(ascending=False)

    # Get top N most sensitive tasks
    top_tasks = task_avg_variance.head(top_n)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create grouped bar chart for top tasks
    agents = sorted(task_df['agent'].unique())
    x = np.arange(len(top_tasks))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        agent_variances = []
        for task_id in top_tasks.index:
            agent_task_data = task_df[(task_df['agent'] == agent) & (task_df['task_id'] == task_id)]
            if not agent_task_data.empty:
                agent_variances.append(agent_task_data['variance'].values[0])
            else:
                agent_variances.append(0)

        ax.bar(x + i * width, agent_variances, width, label=agent, alpha=0.8)

    ax.set_ylabel('Variance', fontsize=11)
    ax.set_xlabel('Task ID', fontsize=11)
    ax.set_title(f'Top {top_n} Most Prompt-Sensitive Tasks', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(top_tasks.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'most_sensitive_tasks.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def generate_report(task_df: pd.DataFrame, agent_df: pd.DataFrame, output_dir: Path):
    """
    Generate a markdown report with sensitivity analysis.
    """
    report = []
    report.append("# Prompt Sensitivity Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Total agents analyzed**: {len(agent_df)}\n")
    report.append(f"- **Total task instances analyzed**: {len(task_df)}\n")
    report.append(f"- **Unique tasks**: {task_df['task_id'].nunique()}\n\n")

    report.append("## Agent-Level Summary\n\n")
    report.append("| Agent | S_prompt | Mean Variance | Std Variance | Mean Gap | Max Gap | Tasks |\n")
    report.append("|-------|----------|---------------|--------------|----------|---------|-------|\n")

    for _, row in agent_df.iterrows():
        report.append(
            f"| {row['agent']} | "
            f"{row['S_prompt']:.3f} | {row['mean_variance']:.4f} | "
            f"{row['std_variance']:.4f} | "
            f"{row['mean_min_max_gap']:.3f} | {row['max_min_max_gap']:.3f} | "
            f"{int(row['num_tasks'])} |\n"
        )

    report.append("\n## Metrics Explained\n\n")

    report.append("### Prompt Robustness (S_prompt)\n")
    report.append("Measures how robust an agent is to prompt variations. Higher is better.\n\n")
    report.append("- **S_prompt ≈ 1**: Very robust (consistent performance across variations)\n")
    report.append("- **S_prompt ≈ 0**: Sensitive (performance varies significantly with phrasing)\n")
    report.append("- **Formula**: `S_prompt = 1 - mean_variance` across all tasks\n\n")

    report.append("### Variance\n")
    report.append("Measures performance variability across prompt variations.\n\n")
    report.append("- **Lower variance**: More consistent performance\n")
    report.append("- **Higher variance**: Performance depends on prompt phrasing\n")
    report.append("- Computed per task, then averaged across tasks\n\n")

    report.append("### Min-Max Gap\n")
    report.append("Difference between best and worst performance across variations.\n\n")
    report.append("- **Lower gap**: Stable performance\n")
    report.append("- **Higher gap**: Large performance swings possible\n\n")

    report.append("## Key Findings\n\n")

    # Most/least robust
    most_robust = agent_df.loc[agent_df['S_prompt'].idxmax()]
    least_robust = agent_df.loc[agent_df['S_prompt'].idxmin()]

    report.append("### Prompt Robustness\n")
    report.append(f"- **Most robust**: {most_robust['agent']} (S_prompt = {most_robust['S_prompt']:.3f})\n")
    report.append(f"- **Least robust**: {least_robust['agent']} (S_prompt = {least_robust['S_prompt']:.3f})\n\n")

    # Most sensitive tasks (if we have task-level data)
    if not task_df.empty:
        task_avg_variance = task_df.groupby('task_id')['variance'].mean().sort_values(ascending=False)
        top_5_sensitive = task_avg_variance.head(5)

        report.append("### Most Prompt-Sensitive Tasks\n")
        for i, (task_id, variance) in enumerate(top_5_sensitive.items(), 1):
            report.append(f"{i}. Task {task_id}: variance = {variance:.4f}\n")
        report.append("\n")

    # Write report
    output_path = output_dir / 'sensitivity_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"📄 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prompt sensitivity from evaluation results"
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
        help="Benchmark name (e.g., taubench_airline, gaia)"
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

    print("🔬 Prompt Sensitivity Analysis\n")
    print(f"📂 Results directory: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load results
    results_data = load_results_with_sensitivity(results_dir, args.benchmark)

    if not results_data:
        print("❌ No results with sensitivity metrics found to analyze")
        return

    print(f"\n✅ Loaded results for {len(results_data)} agents\n")

    # Analyze
    print("📊 Computing sensitivity metrics...")
    task_df, agent_df = analyze_all_agents(results_data)

    if agent_df.empty:
        print("❌ No valid sensitivity data computed")
        return

    # Save data
    print("\n💾 Saving results...")

    if not task_df.empty:
        task_df.to_csv(output_dir / 'task_level_sensitivity.csv', index=False)
        print(f"   Saved: {output_dir / 'task_level_sensitivity.csv'}")

    agent_df.to_csv(output_dir / 'agent_level_sensitivity.csv', index=False)
    print(f"   Saved: {output_dir / 'agent_level_sensitivity.csv'}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_sensitivity_comparison(agent_df, output_dir)

    if not task_df.empty:
        plot_task_sensitivity_distribution(task_df, output_dir)
        plot_most_sensitive_tasks(task_df, output_dir)

    # Generate report
    print("\n📄 Generating report...")
    generate_report(task_df, agent_df, output_dir)

    print("\n✨ Analysis complete!")
    print(f"\n📂 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    if not task_df.empty:
        print("  - task_level_sensitivity.csv: Per-task sensitivity metrics")
    print("  - agent_level_sensitivity.csv: Agent-level summary")
    print("  - sensitivity_comparison.png: Agent comparison plots")
    if not task_df.empty:
        print("  - task_sensitivity_distribution.png: Task-level distributions")
        print("  - most_sensitive_tasks.png: Most sensitive tasks")
    print("  - sensitivity_report.md: Detailed report")


if __name__ == "__main__":
    main()

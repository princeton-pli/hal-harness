#!/usr/bin/env python3
"""
Analysis Script for Predictability Evaluation

This script analyzes evaluation results with confidence scores and computes:
1. Risk-Coverage (P_rc): Excess AuRC over optimal selector
2. Calibration (P_cal): Expected Calibration Error (ECE)

Usage:
    python analyze_predictability.py --results_dir results/ --benchmark taubench_airline
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


def load_results_with_confidence(results_dir: Path, benchmark: str) -> Dict:
    """
    Load HAL evaluation results that include confidence scores.

    Returns nested dict: {agent_name: {run_id: {task_id: {metrics + confidence}}}}
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

        # Check if confidence scores are present
        has_confidence = False
        tasks_with_confidence = 0

        for task_id, task_eval in raw_eval_results.items():
            task_id_str = str(task_id)

            if not isinstance(task_eval, dict):
                continue

            success = int(task_eval.get('reward', 0.0))
            confidence = task_eval.get('confidence', None)

            if confidence is not None:
                has_confidence = True
                tasks_with_confidence += 1

            results_data[agent_name][run_id][task_id_str] = {
                'success': success,
                'confidence': confidence
            }

        if has_confidence:
            print(f"✅ Loaded {agent_name} - {tasks_with_confidence} tasks with confidence from {run_dir.name}")
        else:
            print(f"⚠️  Loaded {agent_name} - {run_dir.name} (NO CONFIDENCE SCORES)")

    return results_data


def compute_aurc(confidences: np.ndarray, successes: np.ndarray, n_points: int = 100) -> Dict:
    """
    Compute Area Under Risk-Coverage curve and P_rc metric.

    Args:
        confidences: Array of confidence scores [0, 1]
        successes: Array of binary outcomes {0, 1}
        n_points: Number of points for curve discretization

    Returns:
        Dict with aurc, aurc_optimal, excess_aurc, and P_rc
    """
    # Filter out NaN values from both arrays
    valid_mask = ~(np.isnan(confidences) | np.isnan(successes))
    confidences = confidences[valid_mask]
    successes = successes[valid_mask]

    N = len(confidences)

    # Handle edge cases
    if N == 0:
        return {
            'aurc': np.nan,
            'aurc_optimal': np.nan,
            'aurc_random': np.nan,
            'excess_aurc': np.nan,
            'excess_aurc_max': np.nan,
            'P_rc': np.nan,
            'coverages': np.array([]),
            'risks': [],
            'optimal_risks': []
        }

    # Sort by decreasing confidence
    sorted_indices = np.argsort(-confidences)
    sorted_successes = successes[sorted_indices]

    # Compute risk at each coverage level
    coverages = np.linspace(0, 1, n_points)
    risks = []

    for coverage in coverages:
        n_covered = int(coverage * N)
        if n_covered == 0:
            risks.append(1.0)
        else:
            # Risk = error rate among covered samples
            risk = 1 - np.mean(sorted_successes[:n_covered])
            risks.append(risk)

    # Area under curve (trapezoidal rule)
    aurc = np.trapezoid(risks, coverages)

    # Optimal AuRC (perfect ranking by actual success)
    optimal_sorted = np.sort(successes)[::-1]
    optimal_risks = []

    for coverage in coverages:
        n_covered = int(coverage * N)
        if n_covered == 0:
            optimal_risks.append(1.0)
        else:
            optimal_risks.append(1 - np.mean(optimal_sorted[:n_covered]))

    aurc_optimal = np.trapezoid(optimal_risks, coverages)

    # Worst case: random ranking (constant risk = overall error rate)
    overall_error_rate = 1 - np.mean(successes)
    aurc_random = overall_error_rate  # Constant risk curve

    # Excess AuRC
    excess_aurc = aurc - aurc_optimal
    excess_aurc_max = aurc_random - aurc_optimal

    # Normalized P_rc score (higher is better)
    if excess_aurc_max > 0:
        P_rc = 1 - (excess_aurc / excess_aurc_max)
    else:
        P_rc = 1.0  # Perfect case

    return {
        'aurc': aurc,
        'aurc_optimal': aurc_optimal,
        'aurc_random': aurc_random,
        'excess_aurc': excess_aurc,
        'excess_aurc_max': excess_aurc_max,
        'P_rc': P_rc,
        'coverages': coverages,
        'risks': risks,
        'optimal_risks': optimal_risks
    }


def compute_ece(confidences: np.ndarray, successes: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute Expected Calibration Error and P_cal metric.

    Args:
        confidences: Array of confidence scores [0, 1]
        successes: Array of binary outcomes {0, 1}
        n_bins: Number of bins for calibration

    Returns:
        Dict with ece, P_cal, and bin statistics
    """
    # Filter out NaN values from both arrays
    valid_mask = ~(np.isnan(confidences) | np.isnan(successes))
    confidences = confidences[valid_mask]
    successes = successes[valid_mask]

    # Handle edge case of no valid data
    if len(confidences) == 0:
        return {
            'ece': np.nan,
            'P_cal': np.nan,
            'bin_stats': []
        }

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    bin_stats = []

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])

        if i == n_bins - 1:  # Last bin includes right edge
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])

        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            # Average confidence in bin
            avg_confidence = np.mean(confidences[in_bin])

            # Average accuracy in bin
            avg_accuracy = np.mean(successes[in_bin])

            # Contribution to ECE (weighted by bin size)
            weight = n_in_bin / len(confidences)
            ece += weight * abs(avg_accuracy - avg_confidence)

            bin_stats.append({
                'bin_lower': bin_edges[i],
                'bin_upper': bin_edges[i+1],
                'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                'count': n_in_bin,
                'avg_confidence': avg_confidence,
                'avg_accuracy': avg_accuracy,
                'calibration_error': abs(avg_accuracy - avg_confidence)
            })
        else:
            bin_stats.append({
                'bin_lower': bin_edges[i],
                'bin_upper': bin_edges[i+1],
                'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                'count': 0,
                'avg_confidence': np.nan,
                'avg_accuracy': np.nan,
                'calibration_error': 0.0
            })

    # Calibration score (higher is better)
    P_cal = 1 - ece

    return {
        'ece': ece,
        'P_cal': P_cal,
        'bin_stats': bin_stats
    }


def analyze_agent_predictability(agent_data: Dict[str, Dict[str, Dict]]) -> Tuple[Dict, pd.DataFrame]:
    """
    Compute predictability metrics for a single agent across all runs.

    Args:
        agent_data: {run_id: {task_id: {success, confidence}}}

    Returns:
        (aggregate_metrics, task_level_df)
    """
    # Collect all confidence scores and outcomes
    all_confidences = []
    all_successes = []

    for run_id, tasks in agent_data.items():
        for task_id, metrics in tasks.items():
            if metrics['confidence'] is not None:
                all_confidences.append(metrics['confidence'])
                all_successes.append(metrics['success'])

    if len(all_confidences) == 0:
        print("⚠️  No confidence scores found for this agent")
        return None, pd.DataFrame()

    confidences = np.array(all_confidences)
    successes = np.array(all_successes)

    # Compute metrics
    aurc_metrics = compute_aurc(confidences, successes)
    ece_metrics = compute_ece(confidences, successes)

    # Aggregate metrics
    aggregate_metrics = {
        'num_samples': len(confidences),
        'accuracy': np.mean(successes),
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        **aurc_metrics,
        **ece_metrics
    }

    return aggregate_metrics, pd.DataFrame(ece_metrics['bin_stats'])


def analyze_all_agents(results_data: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze predictability for all agents.

    Returns:
        (agent_level_df, detailed_metrics)
    """
    agent_level_rows = []
    detailed_metrics = {}

    for agent_name, agent_data in results_data.items():
        print(f"\n📊 Analyzing {agent_name} ({len(agent_data)} runs)...")

        aggregate_metrics, bin_df = analyze_agent_predictability(agent_data)

        if aggregate_metrics is None:
            continue

        # Store for later visualization
        detailed_metrics[agent_name] = {
            'aggregate': aggregate_metrics,
            'bins': bin_df
        }

        # Add to summary table
        agent_level_rows.append({
            'agent': agent_name,
            'num_samples': aggregate_metrics['num_samples'],
            'accuracy': aggregate_metrics['accuracy'],
            'mean_confidence': aggregate_metrics['mean_confidence'],
            'P_rc': aggregate_metrics['P_rc'],
            'P_cal': aggregate_metrics['P_cal'],
            'aurc': aggregate_metrics['aurc'],
            'ece': aggregate_metrics['ece']
        })

        print(f"   Accuracy: {aggregate_metrics['accuracy']:.3f}")
        print(f"   Mean Confidence: {aggregate_metrics['mean_confidence']:.3f}")
        print(f"   P_rc (Risk-Coverage): {aggregate_metrics['P_rc']:.3f}")
        print(f"   P_cal (Calibration): {aggregate_metrics['P_cal']:.3f}")

    agent_level_df = pd.DataFrame(agent_level_rows) if agent_level_rows else pd.DataFrame()

    return agent_level_df, detailed_metrics


def plot_risk_coverage_curves(detailed_metrics: Dict, output_dir: Path):
    """Plot risk-coverage curves for all agents."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = sns.color_palette("husl", len(detailed_metrics))

    for (agent_name, metrics), color in zip(detailed_metrics.items(), colors):
        aggregate = metrics['aggregate']

        ax.plot(aggregate['coverages'], aggregate['risks'],
               label=f"{agent_name} (P_rc={aggregate['P_rc']:.3f})",
               linewidth=2.5, color=color, alpha=0.8)

        # Plot optimal curve for reference (only once)
        if agent_name == list(detailed_metrics.keys())[0]:
            ax.plot(aggregate['coverages'], aggregate['optimal_risks'],
                   'k--', linewidth=2, alpha=0.5, label='Optimal (perfect ranking)')

    ax.set_xlabel('Coverage (fraction of predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk (error rate)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Coverage Curves\n(lower curve = better confidence ranking)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path = output_dir / 'risk_coverage_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_calibration_curves(detailed_metrics: Dict, output_dir: Path):
    """Plot calibration curves for all agents."""
    n_agents = len(detailed_metrics)
    ncols = min(3, n_agents)
    nrows = (n_agents + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_agents == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (agent_name, metrics) in enumerate(detailed_metrics.items()):
        ax = axes[idx]
        bin_df = metrics['bins']

        # Filter out empty bins
        bin_df_valid = bin_df[bin_df['count'] > 0].copy()

        if len(bin_df_valid) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(agent_name)
            continue

        # Bar plot showing confidence vs accuracy
        x = np.arange(len(bin_df_valid))
        width = 0.35

        ax.bar(x - width/2, bin_df_valid['avg_confidence'], width,
              label='Avg Confidence', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, bin_df_valid['avg_accuracy'], width,
              label='Avg Accuracy', alpha=0.8, color='coral')

        # Perfect calibration line
        ax.plot([-0.5, len(bin_df_valid)-0.5], [0, 1], 'k--', alpha=0.5, linewidth=1.5)

        ax.set_xlabel('Confidence Bin', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(f'{agent_name}\nECE={metrics["aggregate"]["ece"]:.3f}, P_cal={metrics["aggregate"]["P_cal"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{b:.1f}' for b in bin_df_valid['bin_center']], rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(len(detailed_metrics), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'calibration_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_reliability_diagram(detailed_metrics: Dict, output_dir: Path):
    """Plot traditional reliability diagrams (diagonal plot)."""
    n_agents = len(detailed_metrics)
    ncols = min(3, n_agents)
    nrows = (n_agents + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_agents == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (agent_name, metrics) in enumerate(detailed_metrics.items()):
        ax = axes[idx]
        bin_df = metrics['bins']

        # Filter out empty bins
        bin_df_valid = bin_df[bin_df['count'] > 0].copy()

        if len(bin_df_valid) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(agent_name)
            continue

        # Scatter plot with size proportional to bin count
        sizes = bin_df_valid['count'] / bin_df_valid['count'].max() * 500

        ax.scatter(bin_df_valid['avg_confidence'], bin_df_valid['avg_accuracy'],
                  s=sizes, alpha=0.6, color='steelblue', edgecolors='black', linewidth=1.5)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect calibration')

        # Add gap lines showing calibration error
        for _, row in bin_df_valid.iterrows():
            ax.plot([row['avg_confidence'], row['avg_confidence']],
                   [row['avg_confidence'], row['avg_accuracy']],
                   'r-', alpha=0.4, linewidth=1.5)

        ax.set_xlabel('Mean Predicted Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{agent_name}\nECE={metrics["aggregate"]["ece"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    # Hide unused subplots
    for idx in range(len(detailed_metrics), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'reliability_diagrams.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_summary_comparison(agent_df: pd.DataFrame, output_dir: Path):
    """Create summary comparison across agents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = agent_df['agent'].tolist()
    x_pos = np.arange(len(agents))

    # 1. P_rc scores
    ax = axes[0, 0]
    bars = ax.bar(x_pos, agent_df['P_rc'], color='steelblue', alpha=0.8)
    ax.set_ylabel('Risk-Coverage Score (P_rc)', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Ranking Quality\n(higher = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agent_df['P_rc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. P_cal scores
    ax = axes[0, 1]
    bars = ax.bar(x_pos, agent_df['P_cal'], color='coral', alpha=0.8)
    ax.set_ylabel('Calibration Score (P_cal)', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Calibration\n(higher = better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agent_df['P_cal']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Accuracy vs P_rc
    ax = axes[1, 0]
    ax.scatter(agent_df['accuracy'], agent_df['P_rc'], s=150, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5)
    for idx, row in agent_df.iterrows():
        ax.annotate(row['agent'], (row['accuracy'], row['P_rc']),
                   fontsize=8, ha='right', va='bottom', alpha=0.7)
    ax.set_xlabel('Accuracy (Capability)', fontsize=11, fontweight='bold')
    ax.set_ylabel('P_rc (Predictability)', fontsize=11, fontweight='bold')
    ax.set_title('Capability vs Predictability\n(showing disentanglement)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Mean confidence vs accuracy
    ax = axes[1, 1]
    ax.scatter(agent_df['mean_confidence'], agent_df['accuracy'], s=150, alpha=0.7, color='coral', edgecolors='black', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect calibration')
    for idx, row in agent_df.iterrows():
        ax.annotate(row['agent'], (row['mean_confidence'], row['accuracy']),
                   fontsize=8, ha='right', va='bottom', alpha=0.7)
    ax.set_xlabel('Mean Confidence', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Overall Calibration Check', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path = output_dir / 'predictability_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def generate_report(agent_df: pd.DataFrame, output_dir: Path):
    """Generate markdown report with predictability analysis."""
    report = []
    report.append("# Predictability Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Total agents analyzed**: {len(agent_df)}\n")
    report.append(f"- **Total samples**: {agent_df['num_samples'].sum()}\n\n")

    report.append("## Agent-Level Summary\n\n")
    report.append("| Agent | Samples | Accuracy | Mean Conf | P_rc | P_cal | AuRC | ECE |\n")
    report.append("|-------|---------|----------|-----------|------|-------|------|-----|\n")

    for _, row in agent_df.iterrows():
        report.append(
            f"| {row['agent']} | "
            f"{int(row['num_samples'])} | "
            f"{row['accuracy']:.3f} | {row['mean_confidence']:.3f} | "
            f"{row['P_rc']:.3f} | {row['P_cal']:.3f} | "
            f"{row['aurc']:.3f} | {row['ece']:.3f} |\n"
        )

    report.append("\n## Metrics Explained\n\n")

    report.append("### Risk-Coverage Score (P_rc)\n")
    report.append("Measures how well confidence scores rank predictions by correctness.\n\n")
    report.append("- **P_rc ≈ 1**: Near-optimal ranking (high confidence → correct, low confidence → incorrect)\n")
    report.append("- **P_rc ≈ 0**: Random ranking (confidence uninformative)\n")
    report.append("- **Formula**: `P_rc = 1 - (E-AuRC / E-AuRC_max)` where E-AuRC is excess area under risk-coverage curve\n\n")

    report.append("### Calibration Score (P_cal)\n")
    report.append("Measures how well confidence estimates match empirical success rates.\n\n")
    report.append("- **P_cal ≈ 1**: Well-calibrated (80% confidence → 80% success)\n")
    report.append("- **P_cal ≈ 0**: Poorly calibrated (confidence doesn't match reality)\n")
    report.append("- **Formula**: `P_cal = 1 - ECE` where ECE is expected calibration error\n\n")

    report.append("## Key Findings\n\n")

    # Best/worst P_rc
    best_prc = agent_df.loc[agent_df['P_rc'].idxmax()]
    worst_prc = agent_df.loc[agent_df['P_rc'].idxmin()]

    report.append("### Confidence Ranking (P_rc)\n")
    report.append(f"- **Best**: {best_prc['agent']} (P_rc = {best_prc['P_rc']:.3f})\n")
    report.append(f"- **Worst**: {worst_prc['agent']} (P_rc = {worst_prc['P_rc']:.3f})\n\n")

    # Best/worst P_cal
    best_pcal = agent_df.loc[agent_df['P_cal'].idxmax()]
    worst_pcal = agent_df.loc[agent_df['P_cal'].idxmin()]

    report.append("### Calibration (P_cal)\n")
    report.append(f"- **Best**: {best_pcal['agent']} (P_cal = {best_pcal['P_cal']:.3f})\n")
    report.append(f"- **Worst**: {worst_pcal['agent']} (P_cal = {worst_pcal['P_cal']:.3f})\n\n")

    # Write report
    output_path = output_dir / 'predictability_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"📄 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze predictability from evaluation results with confidence scores"
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

    print("🔬 Predictability Analysis\n")
    print(f"📂 Results directory: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load results
    results_data = load_results_with_confidence(results_dir, args.benchmark)

    if not results_data:
        print("❌ No results found to analyze")
        return

    print(f"\n✅ Loaded results for {len(results_data)} agents\n")

    # Analyze
    print("📊 Computing predictability metrics...")
    agent_df, detailed_metrics = analyze_all_agents(results_data)

    if agent_df.empty or not detailed_metrics:
        print("❌ No valid predictability data computed")
        return

    # Save data
    print("\n💾 Saving results...")
    agent_df.to_csv(output_dir / 'predictability_metrics.csv', index=False)
    print(f"   Saved: {output_dir / 'predictability_metrics.csv'}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_risk_coverage_curves(detailed_metrics, output_dir)
    plot_calibration_curves(detailed_metrics, output_dir)
    plot_reliability_diagram(detailed_metrics, output_dir)
    plot_summary_comparison(agent_df, output_dir)

    # Generate report
    print("\n📄 Generating report...")
    generate_report(agent_df, output_dir)

    print("\n✨ Analysis complete!")
    print(f"\n📂 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - predictability_metrics.csv: Agent-level summary")
    print("  - risk_coverage_curves.png: Risk-coverage curves")
    print("  - calibration_curves.png: Calibration bar charts")
    print("  - reliability_diagrams.png: Traditional reliability diagrams")
    print("  - predictability_summary.png: Summary comparison")
    print("  - predictability_report.md: Detailed report")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analysis Script for Fault Robustness and Recoverability Evaluation

This script analyzes evaluation results with fault injection and computes:
1. R_fault: Fault robustness (performance degradation under faults)
2. V_heal: Self-healing ratio (recovery from faults)
3. V_ttr: Time-to-recovery (how quickly agent recovers)

Usage:
    python analyze_fault_eval.py --results_dir results/ --benchmark taubench_airline
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

    # Remove special suffixes (fault, compliance, perturbed) and timestamp
    filtered_parts = []
    skip_keywords = ['fault', 'compliance', 'perturbed', 'baseline']
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


def _has_fault_injection_data(raw_eval_results: Dict, raw_logging_results: List, data: Dict) -> bool:
    """
    Detect if a run contains fault injection data.

    Checks multiple indicators:
    1. 'fault_injection' field in any task result (new format)
    2. 'fault_injected' in logging attributes (legacy format)
    3. 'enable_fault_injection' in agent_args (metadata indicator)

    Returns:
        True if fault injection data is detected, False otherwise
    """
    # Check metadata for fault injection flag
    agent_args = data.get('metadata', {}).get('agent_args', {})
    if agent_args.get('enable_fault_injection') == 'true':
        return True

    # Check task results for fault_injection field (new format)
    for task_id, task_eval in raw_eval_results.items():
        if isinstance(task_eval, dict) and 'fault_injection' in task_eval:
            fi_data = task_eval['fault_injection']
            if isinstance(fi_data, dict) and fi_data.get('enabled', False):
                return True

    # Check logging results for fault markers (legacy format)
    for log_entry in raw_logging_results:
        attributes = log_entry.get('attributes', {})
        if 'fault_injected' in attributes:
            return True

    return False


def load_fault_results(results_dir: Path, benchmark: str) -> Tuple[Dict, Dict]:
    """
    Load HAL evaluation results with fault injection data.

    This function detects fault injection runs based on data format, not directory names.
    It looks for:
    1. 'fault_injection' field in task results (new format)
    2. 'fault_injected' in logging attributes (legacy format)
    3. 'enable_fault_injection' in agent_args (metadata indicator)

    Returns:
        (fault_data, baseline_data)
        fault_data: {agent_name: {run_id: {task_id: {metrics, faults, recovery}}}}
        baseline_data: {agent_name: {task_id: baseline_success_rate}}
    """
    fault_data = defaultdict(lambda: defaultdict(dict))
    baseline_data = defaultdict(lambda: defaultdict(list))

    benchmark_dir = results_dir / benchmark
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        return {}, {}

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
        latencies = data.get('results', {}).get('latencies', {})

        # Detect fault run based on data content, not directory name
        is_fault_run = _has_fault_injection_data(raw_eval_results, raw_logging_results, data)

        # Extract fault injection data from logging results (legacy format)
        fault_events_legacy = defaultdict(lambda: {'injected': [], 'recovered': [], 'times': []})

        for log_entry in raw_logging_results:
            task_id = log_entry.get('weave_task_id')
            if task_id is None:
                continue

            task_id = str(task_id)

            # Check for fault markers in the log entry (legacy format)
            attributes = log_entry.get('attributes', {})
            if 'fault_injected' in attributes:
                fault_events_legacy[task_id]['injected'].append({
                    'timestamp': log_entry.get('started_at'),
                    'fault_type': attributes.get('fault_type'),
                    'recovered': attributes.get('recovered', False)
                })

        # Process each task
        for task_id, task_eval in raw_eval_results.items():
            task_id_str = str(task_id)

            if not isinstance(task_eval, dict):
                continue

            success = int(task_eval.get('reward', 0.0))
            task_latency = latencies.get(task_id_str, {})
            total_time = task_latency.get('total_time', 0.0)

            # Check for new format: fault_injection directly in task_eval
            fault_injection_data = task_eval.get('fault_injection', {})
            has_new_format = fault_injection_data.get('enabled', False)

            if has_new_format:
                # New format: fault data stored in raw_eval_results[task_id]["fault_injection"]
                stats = fault_injection_data.get('stats', {})
                events = fault_injection_data.get('events', [])

                num_faults = stats.get('total_faults_injected', len(events))
                num_recovered = stats.get('recoveries_successful', 0)

                fault_data[agent_name][run_id][task_id_str] = {
                    'success': success,
                    'total_time': total_time,
                    'num_faults_injected': num_faults,
                    'num_recovered': num_recovered,
                    'V_heal': fault_injection_data.get('V_heal', stats.get('recovery_rate', 0)),
                    'mean_recovery_time': fault_injection_data.get('mean_recovery_time', stats.get('mean_recovery_time', 0)),
                    'fault_events': events
                }
            elif is_fault_run:
                # Legacy format: fault data from logging results
                num_faults = len(fault_events_legacy[task_id_str]['injected'])
                num_recovered = sum(1 for f in fault_events_legacy[task_id_str]['injected'] if f['recovered'])

                fault_data[agent_name][run_id][task_id_str] = {
                    'success': success,
                    'total_time': total_time,
                    'num_faults_injected': num_faults,
                    'num_recovered': num_recovered,
                    'fault_events': fault_events_legacy[task_id_str]['injected']
                }
            else:
                # Store baseline data (no fault injection)
                baseline_data[agent_name][task_id_str].append(success)

        if is_fault_run:
            print(f"✅ Loaded {agent_name} (fault) - {len(raw_eval_results)} tasks from {run_dir.name}")
        else:
            print(f"✅ Loaded {agent_name} (baseline) - {len(raw_eval_results)} tasks from {run_dir.name}")

    # Compute baseline success rates
    baseline_success_rates = {}
    for agent_name, task_data in baseline_data.items():
        baseline_success_rates[agent_name] = {
            task_id: np.mean(successes)
            for task_id, successes in task_data.items()
        }

    return fault_data, baseline_success_rates


def compute_fault_metrics(fault_data: Dict, baseline_data: Dict) -> pd.DataFrame:
    """
    Compute R_fault, V_heal, and V_ttr metrics for each agent.

    Metrics:
    - R_fault: Fault robustness (performance degradation under faults)
    - V_heal: Self-healing ratio (fraction of faults recovered from)
    - V_ttr: Time-to-recovery score (how quickly the agent recovers)

    Returns DataFrame with agent-level metrics.
    """
    agent_metrics = []

    for agent_name, agent_runs in fault_data.items():
        # Aggregate across all runs for this agent
        all_task_metrics = []
        all_recovery_times = []

        for run_id, tasks in agent_runs.items():
            for task_id, metrics in tasks.items():
                all_task_metrics.append(metrics)

                # Extract recovery times from fault events
                fault_events = metrics.get('fault_events', [])
                for event in fault_events:
                    if isinstance(event, dict):
                        recovery_time = event.get('recovery_time')
                        if recovery_time is not None and recovery_time > 0:
                            all_recovery_times.append(recovery_time)

                # Also check for mean_recovery_time stored at task level
                mean_rt = metrics.get('mean_recovery_time')
                if mean_rt and mean_rt > 0 and not fault_events:
                    all_recovery_times.append(mean_rt)

        if not all_task_metrics:
            continue

        # Compute metrics
        fault_success_rate = np.mean([m['success'] for m in all_task_metrics])
        baseline_values = list(baseline_data.get(agent_name, {}).values())
        baseline_success_rate = np.mean(baseline_values) if baseline_values else 0

        # R_fault: Robustness (normalized performance under faults)
        # R_fault = min(Acc_fault / Acc_baseline, 1.0)
        if baseline_success_rate > 0:
            R_fault = min(fault_success_rate / baseline_success_rate, 1.0)
        else:
            R_fault = fault_success_rate  # If baseline is 0, just use fault rate

        # V_heal: Self-healing ratio
        total_faults = sum(m['num_faults_injected'] for m in all_task_metrics)
        total_recovered = sum(m['num_recovered'] for m in all_task_metrics)

        if total_faults > 0:
            V_heal = total_recovered / total_faults
        else:
            V_heal = np.nan

        # V_ttr: Time-to-recovery score
        # Formula: V_ttr = 1 / (1 + MTTR / T_ref)
        # Where:
        #   - MTTR = Mean Time To Recovery (in seconds)
        #   - T_ref = Reference time for normalization (default: 5 seconds)
        #
        # Interpretation:
        #   - V_ttr ≈ 1: Very fast recovery (MTTR → 0)
        #   - V_ttr ≈ 0.5: Recovery takes about T_ref seconds
        #   - V_ttr → 0: Very slow recovery (MTTR → ∞)
        T_REF = 5.0  # Reference time in seconds (adjustable)

        if all_recovery_times:
            mttr = np.mean(all_recovery_times)
            V_ttr = 1.0 / (1.0 + mttr / T_REF)
        else:
            # No actual recovery time data available
            # Don't estimate - V_ttr requires actual timing measurements
            V_ttr = np.nan

        avg_time = np.mean([m['total_time'] for m in all_task_metrics])
        mttr_value = np.mean(all_recovery_times) if all_recovery_times else np.nan

        agent_metrics.append({
            'agent': agent_name,
            'baseline_success_rate': baseline_success_rate,
            'fault_success_rate': fault_success_rate,
            'R_fault': R_fault,
            'V_heal': V_heal,
            'V_ttr': V_ttr,
            'mttr': mttr_value,  # Mean Time To Recovery (raw seconds)
            'total_faults_injected': total_faults,
            'total_recovered': total_recovered,
            'avg_task_time': avg_time,
            'num_tasks': len(all_task_metrics)
        })

    return pd.DataFrame(agent_metrics)


def plot_fault_robustness(agent_df: pd.DataFrame, output_dir: Path):
    """Plot fault robustness metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = agent_df['agent'].tolist()
    x_pos = np.arange(len(agents))

    # 1. R_fault scores
    ax = axes[0, 0]
    bars = ax.bar(x_pos, agent_df['R_fault'], color='steelblue', alpha=0.8)
    ax.set_ylabel('Fault Robustness (R_fault)', fontsize=11, fontweight='bold')
    ax.set_title('Performance Under Faults\n(higher = more robust)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agent_df['R_fault']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. V_heal scores
    ax = axes[0, 1]
    bars = ax.bar(x_pos, agent_df['V_heal'], color='coral', alpha=0.8)
    ax.set_ylabel('Self-Healing Ratio (V_heal)', fontsize=11, fontweight='bold')
    ax.set_title('Recovery From Faults\n(higher = better recovery)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, agent_df['V_heal']):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Baseline vs Fault Success Rate
    ax = axes[1, 0]
    width = 0.35
    ax.bar(x_pos - width/2, agent_df['baseline_success_rate'], width,
           label='Baseline (no faults)', alpha=0.8, color='green')
    ax.bar(x_pos + width/2, agent_df['fault_success_rate'], width,
           label='With faults', alpha=0.8, color='red')
    ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
    ax.set_title('Performance Degradation', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # 4. Fault injection statistics
    ax = axes[1, 1]
    width = 0.35
    ax.bar(x_pos - width/2, agent_df['total_faults_injected'], width,
           label='Faults Injected', alpha=0.8, color='red')
    ax.bar(x_pos + width/2, agent_df['total_recovered'], width,
           label='Recovered', alpha=0.8, color='green')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Fault Injection Statistics', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'fault_robustness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_robustness_vs_capability(agent_df: pd.DataFrame, output_dir: Path):
    """Plot robustness vs capability scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. R_fault vs baseline capability
    ax = axes[0]
    ax.scatter(agent_df['baseline_success_rate'], agent_df['R_fault'],
              s=150, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5)
    for idx, row in agent_df.iterrows():
        ax.annotate(row['agent'], (row['baseline_success_rate'], row['R_fault']),
                   fontsize=8, ha='right', va='bottom', alpha=0.7)
    ax.set_xlabel('Baseline Capability (Success Rate)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fault Robustness (R_fault)', fontsize=11, fontweight='bold')
    ax.set_title('Robustness vs Capability\n(showing disentanglement)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # 2. V_heal vs R_fault
    ax = axes[1]
    ax.scatter(agent_df['R_fault'], agent_df['V_heal'],
              s=150, alpha=0.7, color='coral', edgecolors='black', linewidth=1.5)
    for idx, row in agent_df.iterrows():
        if not np.isnan(row['V_heal']):
            ax.annotate(row['agent'], (row['R_fault'], row['V_heal']),
                       fontsize=8, ha='right', va='bottom', alpha=0.7)
    ax.set_xlabel('Fault Robustness (R_fault)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Self-Healing Ratio (V_heal)', fontsize=11, fontweight='bold')
    ax.set_title('Robustness vs Recovery Ability', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = output_dir / 'robustness_vs_capability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def generate_report(agent_df: pd.DataFrame, output_dir: Path):
    """Generate markdown report with fault robustness analysis."""
    report = []
    report.append("# Fault Robustness & Recoverability Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Total agents analyzed**: {len(agent_df)}\n")
    report.append(f"- **Total faults injected**: {agent_df['total_faults_injected'].sum()}\n")
    report.append(f"- **Total recovered**: {agent_df['total_recovered'].sum()}\n\n")

    report.append("## Agent-Level Summary\n\n")
    report.append("| Agent | Baseline Acc | Fault Acc | R_fault | V_heal | V_ttr | MTTR (s) | Faults | Recovered |\n")
    report.append("|-------|--------------|-----------|---------|--------|-------|----------|--------|----------|\n")

    for _, row in agent_df.iterrows():
        v_ttr_str = f"{row['V_ttr']:.3f}" if not np.isnan(row.get('V_ttr', np.nan)) else "N/A"
        mttr_str = f"{row['mttr']:.2f}" if not np.isnan(row.get('mttr', np.nan)) else "N/A"
        report.append(
            f"| {row['agent']} | "
            f"{row['baseline_success_rate']:.3f} | "
            f"{row['fault_success_rate']:.3f} | "
            f"{row['R_fault']:.3f} | "
            f"{row['V_heal']:.3f} | "
            f"{v_ttr_str} | "
            f"{mttr_str} | "
            f"{int(row['total_faults_injected'])} | "
            f"{int(row['total_recovered'])} |\n"
        )

    report.append("\n## Metrics Explained\n\n")

    report.append("### Fault Robustness (R_fault)\n")
    report.append("Measures how well performance is maintained under faults.\n\n")
    report.append("- **R_fault ≈ 1**: Minimal degradation (maintains performance despite faults)\n")
    report.append("- **R_fault ≈ 0**: Severe degradation (faults cause complete failure)\n")
    report.append("- **Formula**: `R_fault = min(Acc_fault / Acc_baseline, 1.0)`\n\n")

    report.append("### Self-Healing Ratio (V_heal)\n")
    report.append("Measures ability to recover from injected faults.\n\n")
    report.append("- **V_heal ≈ 1**: Excellent recovery (recovers from all faults)\n")
    report.append("- **V_heal ≈ 0**: Poor recovery (cannot recover from faults)\n")
    report.append("- **Formula**: `V_heal = recovered_faults / total_faults`\n\n")

    report.append("### Time-to-Recovery Score (V_ttr)\n")
    report.append("Measures how quickly the agent recovers from faults.\n\n")
    report.append("- **V_ttr ≈ 1**: Very fast recovery (near-instant)\n")
    report.append("- **V_ttr ≈ 0.5**: Recovery takes about 5 seconds (reference time)\n")
    report.append("- **V_ttr → 0**: Very slow recovery\n")
    report.append("- **Formula**: `V_ttr = 1 / (1 + MTTR / T_ref)` where T_ref = 5s\n")
    report.append("- **MTTR**: Mean Time To Recovery in seconds\n\n")

    report.append("## Key Findings\n\n")

    # Best/worst R_fault
    best_rfault = agent_df.loc[agent_df['R_fault'].idxmax()]
    worst_rfault = agent_df.loc[agent_df['R_fault'].idxmin()]

    report.append("### Fault Robustness\n")
    report.append(f"- **Most robust**: {best_rfault['agent']} (R_fault = {best_rfault['R_fault']:.3f})\n")
    report.append(f"- **Least robust**: {worst_rfault['agent']} (R_fault = {worst_rfault['R_fault']:.3f})\n\n")

    # Best/worst V_heal
    valid_vheal = agent_df[agent_df['V_heal'].notna()]
    if not valid_vheal.empty:
        best_vheal = valid_vheal.loc[valid_vheal['V_heal'].idxmax()]
        worst_vheal = valid_vheal.loc[valid_vheal['V_heal'].idxmin()]

        report.append("### Self-Healing\n")
        report.append(f"- **Best recovery**: {best_vheal['agent']} (V_heal = {best_vheal['V_heal']:.3f})\n")
        report.append(f"- **Worst recovery**: {worst_vheal['agent']} (V_heal = {worst_vheal['V_heal']:.3f})\n\n")

    # Best/worst V_ttr
    if 'V_ttr' in agent_df.columns:
        valid_vttr = agent_df[agent_df['V_ttr'].notna()]
        if not valid_vttr.empty:
            best_vttr = valid_vttr.loc[valid_vttr['V_ttr'].idxmax()]
            worst_vttr = valid_vttr.loc[valid_vttr['V_ttr'].idxmin()]

            report.append("### Recovery Speed\n")
            report.append(f"- **Fastest recovery**: {best_vttr['agent']} (V_ttr = {best_vttr['V_ttr']:.3f})\n")
            if not np.isnan(best_vttr.get('mttr', np.nan)):
                report.append(f"  - MTTR: {best_vttr['mttr']:.2f} seconds\n")
            report.append(f"- **Slowest recovery**: {worst_vttr['agent']} (V_ttr = {worst_vttr['V_ttr']:.3f})\n")
            if not np.isnan(worst_vttr.get('mttr', np.nan)):
                report.append(f"  - MTTR: {worst_vttr['mttr']:.2f} seconds\n")
            report.append("\n")

    # Write report
    output_path = output_dir / 'fault_robustness_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"📄 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze fault robustness from evaluation results"
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

    print("🔬 Fault Robustness & Recoverability Analysis\n")
    print(f"📂 Results directory: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load results
    fault_data, baseline_data = load_fault_results(results_dir, args.benchmark)

    if not fault_data:
        print("❌ No fault injection results found to analyze")
        return

    if not baseline_data:
        print("⚠️  No baseline results found - cannot compute R_fault")

    print(f"\n✅ Loaded fault data for {len(fault_data)} agents\n")

    # Compute metrics
    print("📊 Computing fault robustness metrics...")
    agent_df = compute_fault_metrics(fault_data, baseline_data)

    if agent_df.empty:
        print("❌ No valid fault data computed")
        return

    # Save data
    print("\n💾 Saving results...")
    agent_df.to_csv(output_dir / 'fault_robustness_metrics.csv', index=False)
    print(f"   Saved: {output_dir / 'fault_robustness_metrics.csv'}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_fault_robustness(agent_df, output_dir)
    plot_robustness_vs_capability(agent_df, output_dir)

    # Generate report
    print("\n📄 Generating report...")
    generate_report(agent_df, output_dir)

    print("\n✨ Analysis complete!")
    print(f"\n📂 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - fault_robustness_metrics.csv: Agent-level summary")
    print("  - fault_robustness.png: Robustness and recovery metrics")
    print("  - robustness_vs_capability.png: Scatter plots")
    print("  - fault_robustness_report.md: Detailed report")


if __name__ == "__main__":
    main()

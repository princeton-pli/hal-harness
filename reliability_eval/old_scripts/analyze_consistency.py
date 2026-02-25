#!/usr/bin/env python3
"""
Analysis Script for Consistency Evaluation

This script analyzes evaluation results from multiple runs and computes:
1. Outcome consistency (variance of accuracy across runs)
2. Task success rate consistency (per-task variance)
3. Resource consistency:
   - Time/latency variance
   - Token usage variance
   - Cost variance

Usage:
    python analyze_consistency.py --results_dir results/ --benchmark taubench_airline
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150


def extract_agent_name_from_run_dir(run_dir_name: str) -> str:
    """
    Extract agent name from run directory name.

    Example: taubench_airline_taubench_fewshot_gpt_4_turbo_1767116714
    -> taubench_fewshot_gpt_4_turbo
    """
    # Remove benchmark name prefix and timestamp suffix
    parts = run_dir_name.split('_')
    # Find where the agent name starts (after benchmark name)
    # For taubench_airline, skip first 2 parts
    if run_dir_name.startswith('taubench_airline'):
        agent_parts = parts[2:]  # Skip 'taubench' and 'airline'
    elif run_dir_name.startswith('taubench_retail'):
        agent_parts = parts[2:]  # Skip 'taubench' and 'retail'
    else:
        agent_parts = parts[1:]  # Skip just benchmark name

    # Remove timestamp (last part if it's numeric)
    if agent_parts and agent_parts[-1].isdigit():
        agent_parts = agent_parts[:-1]

    return '_'.join(agent_parts)


def load_results_from_files(results_dir: Path, benchmark: str) -> Dict:
    """
    Load HAL evaluation results from the UPLOAD.json files.

    Returns nested dict: {agent_name: {run_id: {task_id: {metrics}}}}
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

        # Find UPLOAD.json file
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

        # Extract agent name
        agent_name = extract_agent_name_from_run_dir(run_dir.name)
        run_id = run_dir.name

        # Extract task-level data
        raw_eval_results = data.get('raw_eval_results', {})
        latencies = data.get('results', {}).get('latencies', {})
        raw_logging_results = data.get('raw_logging_results', [])

        # Aggregate token usage, cost, and trajectories per task from raw_logging_results
        task_tokens = defaultdict(lambda: {'prompt': 0, 'completion': 0, 'total': 0})
        task_costs = defaultdict(float)
        task_api_calls = defaultdict(int)
        task_trajectories = defaultdict(list)
        task_call_latencies = defaultdict(list)  # Per-call latencies in ms

        for log_entry in raw_logging_results:
            task_id = log_entry.get('weave_task_id')
            if task_id is None:
                continue

            task_id = str(task_id)

            # Get usage from summary
            summary = log_entry.get('summary', {})
            summary_usage = summary.get('usage', {})
            for model, usage in summary_usage.items():
                task_tokens[task_id]['prompt'] += usage.get('prompt_tokens', 0)
                task_tokens[task_id]['completion'] += usage.get('completion_tokens', 0)
                task_tokens[task_id]['total'] += usage.get('total_tokens', 0)
                task_api_calls[task_id] += 1

            # Extract per-call latency from weave summary
            weave_summary = summary.get('weave', {})
            latency_ms = weave_summary.get('latency_ms')
            if latency_ms is not None:
                task_call_latencies[task_id].append(latency_ms)

            # Extract trajectory information (tool calls, actions)
            op_name = log_entry.get('op_name', '')
            if 'tool' in op_name.lower() or 'function' in op_name.lower():
                # Extract tool/function name
                tool_name = op_name.split('/')[-1].split(':')[0]
                task_trajectories[task_id].append(tool_name)

        # Also extract trajectories from taken_actions in raw_eval_results
        # This is more reliable than Weave log parsing since it comes directly from the agent
        # We prefer taken_actions when available, using Weave logs only as fallback
        for task_id, task_eval in raw_eval_results.items():
            task_id_str = str(task_id)
            if isinstance(task_eval, dict):
                taken_actions = task_eval.get('taken_actions', [])
                if taken_actions:
                    # If we have taken_actions, use them as the primary source
                    # (overwrite any Weave-extracted data for this task)
                    action_names = []
                    for action in taken_actions:
                        if isinstance(action, dict):
                            tool_name = action.get('name', '')
                            if tool_name:
                                action_names.append(tool_name)
                    if action_names:
                        task_trajectories[task_id_str] = action_names

        # Process each task
        skipped_tasks = 0
        for task_id, task_eval in raw_eval_results.items():
            task_id_str = str(task_id)

            # Skip if task_eval is not a dict (error strings, etc.)
            if not isinstance(task_eval, dict):
                skipped_tasks += 1
                continue

            # Success (reward is 0.0 or 1.0)
            success = int(task_eval.get('reward', 0.0))

            # Latency
            task_latency = latencies.get(task_id_str, {})
            total_time = task_latency.get('total_time', 0.0)

            # Tokens
            tokens = task_tokens.get(task_id_str, {'prompt': 0, 'completion': 0, 'total': 0})

            # Extract num_actions and num_errors from confidence_details
            confidence_details = task_eval.get('confidence_details', {})
            num_actions = confidence_details.get('num_actions', 0) if isinstance(confidence_details, dict) else 0
            num_errors = confidence_details.get('num_errors', 0) if isinstance(confidence_details, dict) else 0

            # Compute mean per-call latency for this task
            call_latencies = task_call_latencies.get(task_id_str, [])
            mean_call_latency = np.mean(call_latencies) if call_latencies else 0.0

            # Store all metrics
            results_data[agent_name][run_id][task_id_str] = {
                'success': success,
                'total_time': total_time,
                'prompt_tokens': tokens['prompt'],
                'completion_tokens': tokens['completion'],
                'total_tokens': tokens['total'],
                'api_calls': task_api_calls.get(task_id_str, 0),
                'trajectory': task_trajectories.get(task_id_str, []),
                'num_actions': num_actions,
                'num_errors': num_errors,
                'mean_call_latency_ms': mean_call_latency
            }

        valid_tasks = len(raw_eval_results) - skipped_tasks
        if skipped_tasks > 0:
            print(f"✅ Loaded {agent_name} - {valid_tasks} tasks from {run_dir.name} ({skipped_tasks} errors skipped)")
        else:
            print(f"✅ Loaded {agent_name} - {valid_tasks} tasks from {run_dir.name}")

    return results_data


def compute_trajectory_consistency(trajectories: List[List[str]]) -> float:
    """
    Compute trajectory consistency using Jensen-Shannon divergence.

    Args:
        trajectories: List of trajectories (each trajectory is a list of action/tool names)

    Returns:
        C_traj: Trajectory consistency score [0, 1] where 1 = perfectly consistent
    """
    if len(trajectories) < 2:
        return np.nan

    # Build probability distributions for each trajectory
    distributions = []
    for traj in trajectories:
        if not traj:  # Empty trajectory
            continue
        # Count action frequencies
        action_counts = Counter(traj)
        total = len(traj)
        # Create probability distribution
        prob_dist = {action: count / total for action, count in action_counts.items()}
        distributions.append(prob_dist)

    if len(distributions) < 2:
        return np.nan

    # Get all unique actions across all trajectories
    all_actions = set()
    for dist in distributions:
        all_actions.update(dist.keys())
    all_actions = sorted(list(all_actions))

    # Convert distributions to vectors with consistent ordering
    dist_vectors = []
    for dist in distributions:
        vector = np.array([dist.get(action, 0.0) for action in all_actions])
        # Normalize to ensure it's a valid probability distribution
        if vector.sum() > 0:
            vector = vector / vector.sum()
        dist_vectors.append(vector)

    # Compute pairwise Jensen-Shannon divergences
    n = len(dist_vectors)
    js_divergences = []
    for i in range(n):
        for j in range(i + 1, n):
            js_div = jensenshannon(dist_vectors[i], dist_vectors[j])
            js_divergences.append(js_div)

    if not js_divergences:
        return np.nan

    # Average JSD
    mean_jsd = np.mean(js_divergences)

    # Convert to consistency score: C_traj = 1 - mean_JSD
    # JSD ranges from 0 (identical) to ~1 (completely different)
    C_traj = 1 - mean_jsd

    return C_traj


def compute_task_level_consistency(agent_data: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    """
    Compute per-task consistency metrics across runs for a single agent.

    Returns DataFrame with one row per task, including:
    - task_id
    - K (number of runs)
    - success_rate (mean success across runs)
    - success_std (standard deviation of success)
    - C_out (outcome consistency: 1 - normalized std, higher = more consistent)
    - C_traj (trajectory consistency: 1 - mean JSD, higher = more consistent)
    - time_mean, time_std, time_cv (coefficient of variation)
    - tokens_mean, tokens_std, tokens_cv
    - api_calls_mean, api_calls_cv
    - actions_mean, actions_cv
    - errors_mean, errors_cv
    - call_latency_mean, call_latency_cv
    """
    # Collect data per task across runs
    task_metrics = defaultdict(lambda: {
        'success': [],
        'total_time': [],
        'total_tokens': [],
        'trajectories': [],
        'api_calls': [],
        'num_actions': [],
        'num_errors': [],
        'mean_call_latency_ms': []
    })

    for run_id, tasks in agent_data.items():
        for task_id, metrics in tasks.items():
            task_metrics[task_id]['success'].append(metrics['success'])
            task_metrics[task_id]['total_time'].append(metrics['total_time'])
            task_metrics[task_id]['total_tokens'].append(metrics['total_tokens'])
            task_metrics[task_id]['trajectories'].append(metrics.get('trajectory', []))
            task_metrics[task_id]['api_calls'].append(metrics.get('api_calls', 0))
            task_metrics[task_id]['num_actions'].append(metrics.get('num_actions', 0))
            task_metrics[task_id]['num_errors'].append(metrics.get('num_errors', 0))
            task_metrics[task_id]['mean_call_latency_ms'].append(metrics.get('mean_call_latency_ms', 0.0))

    # Compute statistics
    rows = []
    for task_id, data in task_metrics.items():
        K = len(data['success'])

        if K < 2:
            continue  # Need at least 2 runs for variance

        # Success metrics
        success_array = np.array(data['success'])
        p_hat = np.mean(success_array)

        # Outcome consistency: 1 - normalized standard deviation
        # Range: [0, 1] where 1 = perfectly deterministic, 0 = maximally variable
        # Use ddof=0 (population std) to match theoretical max of 0.5 for binary outcomes
        # Note: ddof=0 is intentional here because we normalize by the theoretical maximum
        # standard deviation (0.5) for binary {0,1} outcomes. Using ddof=1 would give
        # incorrect normalization for small K.
        std_out = np.std(success_array, ddof=0)
        # Bound C_out to [0, 1] to handle edge cases
        C_out = max(0.0, min(1.0, 1 - (2 * std_out)))  # Normalized by theoretical max std = 0.5

        # Time metrics
        time_array = np.array(data['total_time'])
        time_mean = np.mean(time_array)
        time_std = np.std(time_array, ddof=1)
        time_cv = (time_std / time_mean) if time_mean > 0 else 0.0

        # Token metrics
        token_array = np.array(data['total_tokens'])
        tokens_mean = np.mean(token_array)
        tokens_std = np.std(token_array, ddof=1)
        tokens_cv = (tokens_std / tokens_mean) if tokens_mean > 0 else 0.0

        # API calls metrics
        api_calls_array = np.array(data['api_calls'])
        api_calls_mean = np.mean(api_calls_array)
        api_calls_std = np.std(api_calls_array, ddof=1)
        api_calls_cv = (api_calls_std / api_calls_mean) if api_calls_mean > 0 else 0.0

        # Actions metrics
        actions_array = np.array(data['num_actions'])
        actions_mean = np.mean(actions_array)
        actions_std = np.std(actions_array, ddof=1)
        actions_cv = (actions_std / actions_mean) if actions_mean > 0 else 0.0

        # Errors metrics
        errors_array = np.array(data['num_errors'])
        errors_mean = np.mean(errors_array)
        errors_std = np.std(errors_array, ddof=1)
        errors_cv = (errors_std / errors_mean) if errors_mean > 0 else 0.0

        # Per-call latency metrics
        call_latency_array = np.array(data['mean_call_latency_ms'])
        call_latency_mean = np.mean(call_latency_array)
        call_latency_std = np.std(call_latency_array, ddof=1)
        call_latency_cv = (call_latency_std / call_latency_mean) if call_latency_mean > 0 else 0.0

        # Trajectory consistency
        C_traj = compute_trajectory_consistency(data['trajectories'])

        rows.append({
            'task_id': task_id,
            'K': K,
            'success_rate': p_hat,
            'success_std': std_out,
            'C_out': C_out,
            'C_traj': C_traj,
            'time_mean': time_mean,
            'time_std': time_std,
            'time_cv': time_cv,
            'tokens_mean': tokens_mean,
            'tokens_std': tokens_std,
            'tokens_cv': tokens_cv,
            'api_calls_mean': api_calls_mean,
            'api_calls_cv': api_calls_cv,
            'actions_mean': actions_mean,
            'actions_cv': actions_cv,
            'errors_mean': errors_mean,
            'errors_cv': errors_cv,
            'call_latency_mean': call_latency_mean,
            'call_latency_cv': call_latency_cv
        })

    return pd.DataFrame(rows)


def compute_agent_level_metrics(task_df: pd.DataFrame, agent_name: str) -> Dict:
    """
    Aggregate task-level metrics to agent level.
    """
    # C_traj may have NaN values if trajectories are missing
    # Use skipna=True explicitly and track how many tasks have valid C_traj
    if 'C_traj' in task_df.columns:
        valid_ctraj = task_df['C_traj'].dropna()
        mean_C_traj = valid_ctraj.mean() if len(valid_ctraj) > 0 else np.nan
        std_C_traj = valid_ctraj.std() if len(valid_ctraj) > 1 else np.nan
        num_tasks_with_traj = len(valid_ctraj)
    else:
        mean_C_traj = np.nan
        std_C_traj = np.nan
        num_tasks_with_traj = 0

    return {
        'agent': agent_name,
        'num_tasks': len(task_df),
        'mean_C_out': task_df['C_out'].mean(skipna=True),
        'std_C_out': task_df['C_out'].std(skipna=True),
        'mean_C_traj': mean_C_traj,
        'std_C_traj': std_C_traj,
        'num_tasks_with_traj': num_tasks_with_traj,  # Track valid C_traj count
        'mean_success_rate': task_df['success_rate'].mean(skipna=True),
        'mean_time_cv': task_df['time_cv'].mean(skipna=True),
        'mean_tokens_cv': task_df['tokens_cv'].mean(skipna=True),
        'mean_api_calls_cv': task_df['api_calls_cv'].mean(skipna=True),
        'mean_actions_cv': task_df['actions_cv'].mean(skipna=True),
        'mean_errors_cv': task_df['errors_cv'].mean(skipna=True),
        'mean_call_latency_cv': task_df['call_latency_cv'].mean(skipna=True),
        'deterministic_tasks': (task_df['C_out'] > 0.99).sum(),  # Nearly deterministic
        'variable_tasks': (task_df['C_out'] < 0.8).sum(),  # Highly variable
    }


def analyze_all_agents(results_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze consistency for all agents.

    Returns:
        (task_level_df, agent_level_df)
    """
    all_task_rows = []
    agent_level_rows = []

    for agent_name, agent_data in results_data.items():
        print(f"\n📊 Analyzing {agent_name} ({len(agent_data)} runs)...")

        # Compute task-level metrics
        task_df = compute_task_level_consistency(agent_data)

        if task_df.empty:
            print(f"⚠️  No valid tasks for {agent_name}")
            continue

        task_df['agent'] = agent_name
        all_task_rows.append(task_df)

        # Compute agent-level metrics
        agent_metrics = compute_agent_level_metrics(task_df, agent_name)
        agent_level_rows.append(agent_metrics)

        print(f"   Mean C_out: {agent_metrics['mean_C_out']:.3f}")
        if not np.isnan(agent_metrics['mean_C_traj']):
            print(f"   Mean C_traj: {agent_metrics['mean_C_traj']:.3f}")
        print(f"   Mean success rate: {agent_metrics['mean_success_rate']:.3f}")
        print(f"   Mean time CV: {agent_metrics['mean_time_cv']:.3f}")
        print(f"   Mean tokens CV: {agent_metrics['mean_tokens_cv']:.3f}")
        print(f"   Mean API calls CV: {agent_metrics['mean_api_calls_cv']:.3f}")
        print(f"   Mean actions CV: {agent_metrics['mean_actions_cv']:.3f}")

    task_level_df = pd.concat(all_task_rows, ignore_index=True) if all_task_rows else pd.DataFrame()
    agent_level_df = pd.DataFrame(agent_level_rows) if agent_level_rows else pd.DataFrame()

    return task_level_df, agent_level_df


def plot_outcome_consistency(task_df: pd.DataFrame, agent_df: pd.DataFrame, output_dir: Path):
    """
    Plot outcome consistency metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. C_out distribution per agent
    ax = axes[0, 0]
    agents = sorted(task_df['agent'].unique())
    c_out_data = [task_df[task_df['agent'] == agent]['C_out'].values for agent in agents]

    bp = ax.boxplot(c_out_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(agents))):
        patch.set_facecolor(color)

    ax.set_ylabel('Outcome Consistency (C_out)', fontsize=11)
    ax.set_title('Distribution of Outcome Consistency\nAcross Tasks', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate consistency')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. C_out vs Success Rate (disentanglement)
    ax = axes[0, 1]
    for agent in agents:
        agent_tasks = task_df[task_df['agent'] == agent]
        ax.scatter(agent_tasks['success_rate'], agent_tasks['C_out'],
                  label=agent, alpha=0.6, s=60)

    ax.set_xlabel('Success Rate', fontsize=11)
    ax.set_ylabel('Outcome Consistency (C_out)', fontsize=11)
    ax.set_title('Consistency vs Capability\n(showing disentanglement)', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Success rate variance per agent
    ax = axes[1, 0]
    success_var_data = [task_df[task_df['agent'] == agent]['success_std'].values for agent in agents]

    bp = ax.boxplot(success_var_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(agents))):
        patch.set_facecolor(color)

    ax.set_ylabel('Success Std Dev', fontsize=11)
    ax.set_title('Distribution of Success Variance\nAcross Tasks', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Agent-level summary
    ax = axes[1, 1]
    x_pos = np.arange(len(agent_df))

    width = 0.35
    ax.bar(x_pos - width/2, agent_df['mean_C_out'], width,
           label='Mean C_out', alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, agent_df['mean_success_rate'], width,
           label='Mean Success Rate', alpha=0.8, color='coral')

    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Agent-Level Summary:\nConsistency vs Capability', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agent_df['agent'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = output_dir / 'outcome_consistency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_resource_consistency(task_df: pd.DataFrame, output_dir: Path):
    """
    Plot resource consistency metrics (time, tokens).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = sorted(task_df['agent'].unique())
    colors = sns.color_palette("husl", len(agents))

    # 1. Time CV distribution
    ax = axes[0, 0]
    time_cv_data = [task_df[task_df['agent'] == agent]['time_cv'].values for agent in agents]

    bp = ax.boxplot(time_cv_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Coefficient of Variation', fontsize=11)
    ax.set_title('Time Consistency\n(lower CV = more consistent)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Token CV distribution
    ax = axes[0, 1]
    token_cv_data = [task_df[task_df['agent'] == agent]['tokens_cv'].values for agent in agents]

    bp = ax.boxplot(token_cv_data, tick_labels=agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Coefficient of Variation', fontsize=11)
    ax.set_title('Token Usage Consistency\n(lower CV = more consistent)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Mean time vs time variability
    ax = axes[1, 0]
    for agent, color in zip(agents, colors):
        agent_tasks = task_df[task_df['agent'] == agent]
        ax.scatter(agent_tasks['time_mean'], agent_tasks['time_std'],
                  label=agent, alpha=0.6, s=60, color=color)

    ax.set_xlabel('Mean Time (seconds)', fontsize=11)
    ax.set_ylabel('Time Std Dev (seconds)', fontsize=11)
    ax.set_title('Time Mean vs Variability', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Mean tokens vs token variability
    ax = axes[1, 1]
    for agent, color in zip(agents, colors):
        agent_tasks = task_df[task_df['agent'] == agent]
        ax.scatter(agent_tasks['tokens_mean'], agent_tasks['tokens_std'],
                  label=agent, alpha=0.6, s=60, color=color)

    ax.set_xlabel('Mean Tokens', fontsize=11)
    ax.set_ylabel('Token Std Dev', fontsize=11)
    ax.set_title('Token Mean vs Variability', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'resource_consistency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_trajectory_consistency(task_df: pd.DataFrame, agent_df: pd.DataFrame, output_dir: Path):
    """
    Plot trajectory consistency (C_traj) metrics.
    """
    # Check if C_traj data is available
    if 'C_traj' not in task_df.columns or task_df['C_traj'].isna().all():
        print("⚠️  No trajectory data available, skipping C_traj plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agents = sorted(task_df['agent'].unique())
    colors = sns.color_palette("husl", len(agents))

    # 1. C_traj distribution per agent
    ax = axes[0, 0]
    c_traj_data = []
    valid_agents = []
    for agent in agents:
        agent_ctraj = task_df[task_df['agent'] == agent]['C_traj'].dropna().values
        if len(agent_ctraj) > 0:
            c_traj_data.append(agent_ctraj)
            valid_agents.append(agent)

    if not c_traj_data:
        fig.text(0.5, 0.5, 'No trajectory data available', ha='center', va='center', fontsize=14)
        plt.tight_layout()
        output_path = output_dir / 'trajectory_consistency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_path} (empty)")
        plt.close()
        return

    bp = ax.boxplot(c_traj_data, tick_labels=valid_agents, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(valid_agents))):
        patch.set_facecolor(color)

    ax.set_ylabel('Trajectory Consistency (C_traj)', fontsize=11)
    ax.set_title('Distribution of Trajectory Consistency\nAcross Tasks', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Moderate consistency')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. C_traj vs C_out (comparing consistency types)
    ax = axes[0, 1]
    for agent, color in zip(agents, colors):
        agent_tasks = task_df[task_df['agent'] == agent]
        agent_tasks_valid = agent_tasks[agent_tasks['C_traj'].notna()]
        if not agent_tasks_valid.empty:
            ax.scatter(agent_tasks_valid['C_out'], agent_tasks_valid['C_traj'],
                      label=agent, alpha=0.6, s=60, color=color)

    ax.set_xlabel('Outcome Consistency (C_out)', fontsize=11)
    ax.set_ylabel('Trajectory Consistency (C_traj)', fontsize=11)
    ax.set_title('Outcome vs Trajectory Consistency\n(different aspects of consistency)', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 3. C_traj vs Success Rate (disentanglement)
    ax = axes[1, 0]
    for agent, color in zip(agents, colors):
        agent_tasks = task_df[task_df['agent'] == agent]
        agent_tasks_valid = agent_tasks[agent_tasks['C_traj'].notna()]
        if not agent_tasks_valid.empty:
            ax.scatter(agent_tasks_valid['success_rate'], agent_tasks_valid['C_traj'],
                      label=agent, alpha=0.6, s=60, color=color)

    ax.set_xlabel('Success Rate', fontsize=11)
    ax.set_ylabel('Trajectory Consistency (C_traj)', fontsize=11)
    ax.set_title('Trajectory Consistency vs Capability\n(showing disentanglement)', fontsize=12, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 4. Agent-level comparison: C_out vs C_traj
    ax = axes[1, 1]
    valid_agent_df = agent_df[agent_df['mean_C_traj'].notna()].copy()

    if not valid_agent_df.empty:
        x_pos = np.arange(len(valid_agent_df))
        width = 0.35

        ax.bar(x_pos - width/2, valid_agent_df['mean_C_out'], width,
               label='Mean C_out', alpha=0.8, color='steelblue')
        ax.bar(x_pos + width/2, valid_agent_df['mean_C_traj'], width,
               label='Mean C_traj', alpha=0.8, color='coral')

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Agent-Level Comparison:\nOutcome vs Trajectory Consistency', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_agent_df['agent'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)

        # Add value labels
        for i, (c_out, c_traj) in enumerate(zip(valid_agent_df['mean_C_out'], valid_agent_df['mean_C_traj'])):
            ax.text(i - width/2, c_out + 0.02, f'{c_out:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, c_traj + 0.02, f'{c_traj:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / 'trajectory_consistency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_task_heatmaps(task_df: pd.DataFrame, output_dir: Path):
    """
    Create heatmaps showing per-task metrics across agents.
    """
    agents = sorted(task_df['agent'].unique())

    # Get common tasks (tasks that appear in all agents)
    task_counts = task_df['task_id'].value_counts()
    common_tasks = task_counts[task_counts == len(agents)].index.tolist()

    if not common_tasks:
        print("⚠️  No common tasks across all agents, skipping heatmaps")
        return

    # Sort tasks numerically if possible
    try:
        common_tasks = sorted(common_tasks, key=lambda x: int(x))
    except:
        common_tasks = sorted(common_tasks)

    # Limit to first 50 tasks for readability
    common_tasks = common_tasks[:50]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    metrics = ['C_out', 'time_cv', 'tokens_cv']
    titles = ['Outcome Consistency\n(C_out)',
              'Time Consistency\n(1 - CV)',
              'Token Consistency\n(1 - CV)']
    cmaps = ['RdYlGn', 'RdYlGn', 'RdYlGn']

    for ax, metric, title, cmap in zip(axes, metrics, titles, cmaps):
        # Create matrix
        matrix = []
        for agent in agents:
            row = []
            for task_id in common_tasks:
                task_data = task_df[(task_df['agent'] == agent) & (task_df['task_id'] == task_id)]
                if not task_data.empty:
                    value = task_data[metric].values[0]
                    # For CV metrics, convert to consistency (1 - CV for better visualization)
                    if metric.endswith('_cv'):
                        value = max(0, 1 - value)  # Cap at 0
                    row.append(value)
                else:
                    row.append(np.nan)
            matrix.append(row)

        # Plot heatmap
        matrix = np.array(matrix)
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(common_tasks))[::5])
        ax.set_xticklabels([common_tasks[i] for i in range(0, len(common_tasks), 5)],
                          rotation=45, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(agents)))
        ax.set_yticklabels(agents, fontsize=9)

        ax.set_xlabel('Task ID', fontsize=10)
        ax.set_ylabel('Agent', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = output_dir / 'task_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_comprehensive_comparison(agent_df: pd.DataFrame, output_dir: Path):
    """
    Create a comprehensive comparison of all agents across all metrics.
    """
    import matplotlib.colors as mcolors

    # Model metadata: release dates and providers
    # Supports both fewshot and toolcalling scaffolds
    model_metadata = {
        # Tool calling scaffold
        'taubench_toolcalling_gpt_4_turbo': {'date': '2024-04-09', 'provider': 'OpenAI'},
        'taubench_toolcalling_gpt_4o_mini': {'date': '2024-07-18', 'provider': 'OpenAI'},
        'taubench_toolcalling_gpt_o1': {'date': '2024-12-05', 'provider': 'OpenAI'},
        'taubench_toolcalling_gpt_5_2': {'date': '2025-12-11', 'provider': 'OpenAI'},
        'taubench_toolcalling_gemini_2_flash': {'date': '2024-12-11', 'provider': 'Google'},
        'taubench_toolcalling_gemini_2_5_flash': {'date': '2025-04-17', 'provider': 'Google'},
        'taubench_toolcalling_gemini_2_5_pro': {'date': '2025-03-25', 'provider': 'Google'},
        'taubench_toolcalling_gemini_3_pro': {'date': '2025-11-18', 'provider': 'Google'},
        'taubench_toolcalling_claude_haiku_3_5': {'date': '2024-10-22', 'provider': 'Anthropic'},
        'taubench_toolcalling_claude_sonnet_3_7': {'date': '2025-02-24', 'provider': 'Anthropic'},
        'taubench_toolcalling_claude_sonnet_4_5': {'date': '2025-09-29', 'provider': 'Anthropic'},
        'taubench_toolcalling_claude_opus_4_5': {'date': '2025-11-24', 'provider': 'Anthropic'},
        # Few shot scaffold
        'taubench_fewshot_gpt_4_turbo': {'date': '2024-04-09', 'provider': 'OpenAI'},
        'taubench_fewshot_gpt_4o_mini': {'date': '2024-07-18', 'provider': 'OpenAI'},
        'taubench_fewshot_gpt_o1': {'date': '2024-12-05', 'provider': 'OpenAI'},
        'taubench_fewshot_gpt_5_2': {'date': '2025-12-11', 'provider': 'OpenAI'},
        'taubench_fewshot_gemini_2_flash': {'date': '2024-12-11', 'provider': 'Google'},
        'taubench_fewshot_gemini_2_5_flash': {'date': '2025-04-17', 'provider': 'Google'},
        'taubench_fewshot_gemini_2_5_pro': {'date': '2025-03-25', 'provider': 'Google'},
        'taubench_fewshot_gemini_3_pro': {'date': '2025-11-18', 'provider': 'Google'},
        'taubench_fewshot_claude_haiku_3_5': {'date': '2024-10-22', 'provider': 'Anthropic'},
        'taubench_fewshot_claude_sonnet_3_7': {'date': '2025-02-24', 'provider': 'Anthropic'},
        'taubench_fewshot_claude_sonnet_4_5': {'date': '2025-09-29', 'provider': 'Anthropic'},
        'taubench_fewshot_claude_opus_4_5': {'date': '2025-11-24', 'provider': 'Anthropic'},
    }

    # Add metadata to dataframe
    agent_df['release_date'] = agent_df['agent'].map(lambda x: model_metadata.get(x, {}).get('date', '2024-01-01'))
    agent_df['provider'] = agent_df['agent'].map(lambda x: model_metadata.get(x, {}).get('provider', 'Unknown'))
    agent_df['release_timestamp'] = pd.to_datetime(agent_df['release_date'])

    # Sort by provider first, then by release date within each provider
    # Define provider order
    provider_order = {'OpenAI': 0, 'Google': 1, 'Anthropic': 2, 'Unknown': 3}
    agent_df['provider_order'] = agent_df['provider'].map(provider_order)
    agent_df = agent_df.sort_values(['provider_order', 'release_timestamp'])
    agent_df = agent_df.drop('provider_order', axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    agents = agent_df['agent'].tolist()
    x_pos = np.arange(len(agents))

    # Provider colors
    provider_colors = {
        'OpenAI': '#10A37F',
        'Google': '#4285F4',
        'Anthropic': '#D4A574',
        'Unknown': '#999999'
    }
    provider_markers = {
        'OpenAI': 'o',
        'Google': 's',
        'Anthropic': '^',
        'Unknown': 'x'
    }

    # Generate shaded colors for bar charts based on provider and release date
    def generate_shaded_colors(agent_df, provider_colors):
        """Generate colors with different shades for models from same provider."""
        bar_colors = []

        for _, row in agent_df.iterrows():
            provider = row['provider']
            base_color = provider_colors.get(provider, '#999999')

            # Get all models from the same provider
            provider_models = agent_df[agent_df['provider'] == provider].sort_values('release_timestamp')
            num_models = len(provider_models)

            if num_models == 1:
                # Only one model, use base color
                bar_colors.append(base_color)
            else:
                # Find position of this model in the provider's chronological order
                model_index = provider_models.index.get_loc(row.name)

                # Create shades: lighter for earlier, darker for later
                # Use a range from 0.5 (lighter) to 1.2 (darker/more saturated)
                shade_factor = 0.5 + (model_index / (num_models - 1)) * 0.7

                # Convert hex to RGB
                rgb = mcolors.hex2color(base_color)

                # Adjust brightness/saturation
                # For lighter: increase towards white
                # For darker: keep original or slightly darken
                if shade_factor < 1.0:
                    # Lighter - blend with white
                    adjusted_rgb = tuple(c + (1 - c) * (1 - shade_factor) for c in rgb)
                else:
                    # Darker - slightly reduce brightness
                    adjusted_rgb = tuple(c * (2 - shade_factor) for c in rgb)

                bar_colors.append(mcolors.to_hex(adjusted_rgb))

        return bar_colors

    bar_colors = generate_shaded_colors(agent_df, provider_colors)

    # Column 1: Trend plots
    # 1. C_out vs Release Date (by provider) - Top left
    ax = axes[0, 0]
    for provider in agent_df['provider'].unique():
        provider_data = agent_df[agent_df['provider'] == provider]
        ax.scatter(provider_data['release_timestamp'],
                  provider_data['mean_C_out'],
                  c=provider_colors.get(provider, '#999999'),
                  marker=provider_markers.get(provider, 'o'),
                  s=120,
                  alpha=0.7,
                  label=provider,
                  edgecolors='black',
                  linewidth=1.5)

        # Add trend line for each provider if enough points
        if len(provider_data) >= 2:
            x_numeric = (provider_data['release_timestamp'] - provider_data['release_timestamp'].min()).dt.days
            z = np.polyfit(x_numeric, provider_data['mean_C_out'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_numeric.min(), x_numeric.max(), 100)
            dates_trend = provider_data['release_timestamp'].min() + pd.to_timedelta(x_trend, unit='D')
            ax.plot(dates_trend, p(x_trend),
                   color=provider_colors.get(provider, '#999999'),
                   linestyle='--', alpha=0.5, linewidth=2)

    # Add overall trend line across all providers
    x_numeric_all = (agent_df['release_timestamp'] - agent_df['release_timestamp'].min()).dt.days
    z_all = np.polyfit(x_numeric_all, agent_df['mean_C_out'], 1)
    p_all = np.poly1d(z_all)
    x_trend_all = np.linspace(x_numeric_all.min(), x_numeric_all.max(), 100)
    dates_trend_all = agent_df['release_timestamp'].min() + pd.to_timedelta(x_trend_all, unit='D')
    ax.plot(dates_trend_all, p_all(x_trend_all),
           color='black', linestyle='-', alpha=0.8, linewidth=2.5, label='Overall trend')

    # Compute correlation coefficient
    corr_date, _ = pearsonr(x_numeric_all, agent_df['mean_C_out'])
    ax.text(0.02, 0.98, f'r = {corr_date:.3f}', transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Model Release Date', fontsize=11)
    ax.set_ylabel('Outcome Consistency (C_out)', fontsize=11)
    ax.set_title('Consistency vs Release Date\n(by provider)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Rotate date labels
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. C_out vs Capability (by provider) - Bottom left
    ax = axes[1, 0]
    for provider in agent_df['provider'].unique():
        provider_data = agent_df[agent_df['provider'] == provider]
        ax.scatter(provider_data['mean_success_rate'],
                  provider_data['mean_C_out'],
                  c=provider_colors.get(provider, '#999999'),
                  marker=provider_markers.get(provider, 'o'),
                  s=120,
                  alpha=0.7,
                  label=provider,
                  edgecolors='black',
                  linewidth=1.5)

        # Add trend line for each provider if enough points
        if len(provider_data) >= 2:
            z = np.polyfit(provider_data['mean_success_rate'], provider_data['mean_C_out'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(provider_data['mean_success_rate'].min(),
                                 provider_data['mean_success_rate'].max(), 100)
            ax.plot(x_trend, p(x_trend),
                   color=provider_colors.get(provider, '#999999'),
                   linestyle='--', alpha=0.5, linewidth=2)

    # Add overall trend line across all providers
    z_all_cap = np.polyfit(agent_df['mean_success_rate'], agent_df['mean_C_out'], 1)
    p_all_cap = np.poly1d(z_all_cap)
    x_trend_all_cap = np.linspace(agent_df['mean_success_rate'].min(),
                                   agent_df['mean_success_rate'].max(), 100)
    ax.plot(x_trend_all_cap, p_all_cap(x_trend_all_cap),
           color='black', linestyle='-', alpha=0.8, linewidth=2.5, label='Overall trend')

    # Compute correlation coefficient
    corr_cap, _ = pearsonr(agent_df['mean_success_rate'], agent_df['mean_C_out'])
    ax.text(0.02, 0.98, f'r = {corr_cap:.3f}', transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Capability (Success Rate)', fontsize=11)
    ax.set_ylabel('Outcome Consistency (C_out)', fontsize=11)
    ax.set_title('Consistency vs Capability\n(by provider)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Column 2: Bar charts
    # 3. Mean C_out (bar chart) - Top middle
    ax = axes[0, 1]
    bars = ax.bar(x_pos, agent_df['mean_C_out'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Mean Outcome Consistency', fontsize=11)
    ax.set_title('Outcome Consistency', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, agent_df['mean_C_out'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # 4. Success rate (bar chart) - Bottom middle
    ax = axes[1, 1]
    bars = ax.bar(x_pos, agent_df['mean_success_rate'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Mean Success Rate', fontsize=11)
    ax.set_title('Overall Capability', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, agent_df['mean_success_rate'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Column 3: Resource and task plots
    # 5. Resource consistency comparison - Top right
    ax = axes[0, 2]
    width = 0.35
    ax.bar(x_pos - width/2, agent_df['mean_time_cv'], width,
           label='Time CV', alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, agent_df['mean_tokens_cv'], width,
           label='Token CV', alpha=0.8, color='coral')
    ax.set_ylabel('Mean Coefficient of Variation', fontsize=11)
    ax.set_title('Resource Consistency\n(lower is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Task behavior distribution - Bottom right
    ax = axes[1, 2]
    width = 0.35
    ax.bar(x_pos - width/2, agent_df['deterministic_tasks'], width,
           label='Deterministic (C_out > 0.99)', alpha=0.8, color='green')
    ax.bar(x_pos + width/2, agent_df['variable_tasks'], width,
           label='Variable (C_out < 0.8)', alpha=0.8, color='red')
    ax.set_ylabel('Number of Tasks', fontsize=11)
    ax.set_title('Task Behavior Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'comprehensive_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_capability_vs_release_date(agent_df: pd.DataFrame, output_dir: Path):
    """
    Create a focused plot showing capability (success rate) vs release date by provider.
    """
    import matplotlib.dates as mdates

    # Model metadata: release dates and providers
    # Supports both fewshot and toolcalling scaffolds
    model_metadata = {
        # Tool calling scaffold
        'taubench_toolcalling_gpt_4_turbo': {'date': '2024-04-09', 'provider': 'OpenAI'},
        'taubench_toolcalling_gpt_4o_mini': {'date': '2024-07-18', 'provider': 'OpenAI'},
        'taubench_toolcalling_gpt_o1': {'date': '2024-12-05', 'provider': 'OpenAI'},
        'taubench_toolcalling_gpt_5_2': {'date': '2025-12-11', 'provider': 'OpenAI'},
        'taubench_toolcalling_gemini_2_flash': {'date': '2024-12-11', 'provider': 'Google'},
        'taubench_toolcalling_gemini_2_5_flash': {'date': '2025-04-17', 'provider': 'Google'},
        'taubench_toolcalling_gemini_2_5_pro': {'date': '2025-03-25', 'provider': 'Google'},
        'taubench_toolcalling_gemini_3_pro': {'date': '2025-11-18', 'provider': 'Google'},
        'taubench_toolcalling_claude_haiku_3_5': {'date': '2024-10-22', 'provider': 'Anthropic'},
        'taubench_toolcalling_claude_sonnet_3_7': {'date': '2025-02-24', 'provider': 'Anthropic'},
        'taubench_toolcalling_claude_sonnet_4_5': {'date': '2025-09-29', 'provider': 'Anthropic'},
        'taubench_toolcalling_claude_opus_4_5': {'date': '2025-11-24', 'provider': 'Anthropic'},
        # Few shot scaffold
        'taubench_fewshot_gpt_4_turbo': {'date': '2024-04-09', 'provider': 'OpenAI'},
        'taubench_fewshot_gpt_4o_mini': {'date': '2024-07-18', 'provider': 'OpenAI'},
        'taubench_fewshot_gpt_o1': {'date': '2024-12-05', 'provider': 'OpenAI'},
        'taubench_fewshot_gpt_5_2': {'date': '2025-12-11', 'provider': 'OpenAI'},
        'taubench_fewshot_gemini_2_flash': {'date': '2024-12-11', 'provider': 'Google'},
        'taubench_fewshot_gemini_2_5_flash': {'date': '2025-04-17', 'provider': 'Google'},
        'taubench_fewshot_gemini_2_5_pro': {'date': '2025-03-25', 'provider': 'Google'},
        'taubench_fewshot_gemini_3_pro': {'date': '2025-11-18', 'provider': 'Google'},
        'taubench_fewshot_claude_haiku_3_5': {'date': '2024-10-22', 'provider': 'Anthropic'},
        'taubench_fewshot_claude_sonnet_3_7': {'date': '2025-02-24', 'provider': 'Anthropic'},
        'taubench_fewshot_claude_sonnet_4_5': {'date': '2025-09-29', 'provider': 'Anthropic'},
        'taubench_fewshot_claude_opus_4_5': {'date': '2025-11-24', 'provider': 'Anthropic'},
    }

    # Add metadata to dataframe
    agent_df_copy = agent_df.copy()
    agent_df_copy['release_date'] = agent_df_copy['agent'].map(lambda x: model_metadata.get(x, {}).get('date', '2024-01-01'))
    agent_df_copy['provider'] = agent_df_copy['agent'].map(lambda x: model_metadata.get(x, {}).get('provider', 'Unknown'))
    agent_df_copy['release_timestamp'] = pd.to_datetime(agent_df_copy['release_date'])

    # Provider colors and markers
    provider_colors = {
        'OpenAI': '#10A37F',
        'Google': '#4285F4',
        'Anthropic': '#D4A574',
        'Unknown': '#999999'
    }
    provider_markers = {
        'OpenAI': 'o',
        'Google': 's',
        'Anthropic': '^',
        'Unknown': 'x'
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot points by provider
    for provider in agent_df_copy['provider'].unique():
        provider_data = agent_df_copy[agent_df_copy['provider'] == provider]
        ax.scatter(provider_data['release_timestamp'],
                  provider_data['mean_success_rate'],
                  c=provider_colors.get(provider, '#999999'),
                  marker=provider_markers.get(provider, 'o'),
                  s=150,
                  alpha=0.7,
                  label=provider,
                  edgecolors='black',
                  linewidth=1.5,
                  zorder=3)

        # Add trend line for each provider if enough points
        if len(provider_data) >= 2:
            x_numeric = (provider_data['release_timestamp'] - provider_data['release_timestamp'].min()).dt.days
            z = np.polyfit(x_numeric, provider_data['mean_success_rate'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_numeric.min(), x_numeric.max(), 100)
            dates_trend = provider_data['release_timestamp'].min() + pd.to_timedelta(x_trend, unit='D')
            ax.plot(dates_trend, p(x_trend),
                   color=provider_colors.get(provider, '#999999'),
                   linestyle='--', alpha=0.5, linewidth=2.5, zorder=2)

    # Add overall trend line across all providers
    x_numeric_all = (agent_df_copy['release_timestamp'] - agent_df_copy['release_timestamp'].min()).dt.days
    z_all = np.polyfit(x_numeric_all, agent_df_copy['mean_success_rate'], 1)
    p_all = np.poly1d(z_all)
    x_trend_all = np.linspace(x_numeric_all.min(), x_numeric_all.max(), 100)
    dates_trend_all = agent_df_copy['release_timestamp'].min() + pd.to_timedelta(x_trend_all, unit='D')
    ax.plot(dates_trend_all, p_all(x_trend_all),
           color='black', linestyle='-', alpha=0.8, linewidth=3, label='Overall trend', zorder=2)

    # Compute correlation coefficient
    corr, p_value = pearsonr(x_numeric_all, agent_df_copy['mean_success_rate'])

    # Add correlation annotation
    ax.text(0.02, 0.98, f'Pearson r = {corr:.3f}\np-value = {p_value:.3e}',
           transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
           zorder=4)

    ax.set_xlabel('Model Release Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Capability (Mean Success Rate)', fontsize=13, fontweight='bold')
    ax.set_title('Agent Capability vs Release Date', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax.set_ylim(-0.05, 1.05)

    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'capability_vs_release_date.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def generate_report(task_df: pd.DataFrame, agent_df: pd.DataFrame, output_dir: Path):
    """
    Generate a markdown report with consistency analysis.
    """
    report = []
    report.append("# Consistency Analysis Report\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Total agents analyzed**: {len(agent_df)}\n")
    report.append(f"- **Total tasks analyzed**: {len(task_df)}\n")
    report.append(f"- **Runs per agent (K)**: {task_df['K'].mode()[0] if not task_df.empty else 'N/A'}\n\n")

    report.append("## Agent-Level Summary\n\n")

    # Check if C_traj is available
    has_ctraj = 'mean_C_traj' in agent_df.columns and agent_df['mean_C_traj'].notna().any()

    if has_ctraj:
        report.append("| Agent | Mean C_out | Mean C_traj | Std C_out | Success Rate | Time CV | Token CV | Deterministic Tasks | Variable Tasks |\n")
        report.append("|-------|------------|-------------|-----------|--------------|---------|----------|---------------------|----------------|\n")
    else:
        report.append("| Agent | Mean C_out | Std C_out | Success Rate | Time CV | Token CV | Deterministic Tasks | Variable Tasks |\n")
        report.append("|-------|------------|-----------|--------------|---------|----------|---------------------|----------------|\n")

    for _, row in agent_df.iterrows():
        if has_ctraj and not np.isnan(row.get('mean_C_traj', np.nan)):
            report.append(
                f"| {row['agent']} | "
                f"{row['mean_C_out']:.3f} | {row['mean_C_traj']:.3f} | {row['std_C_out']:.3f} | "
                f"{row['mean_success_rate']:.3f} | "
                f"{row['mean_time_cv']:.3f} | {row['mean_tokens_cv']:.3f} | "
                f"{int(row['deterministic_tasks'])} | {int(row['variable_tasks'])} |\n"
            )
        else:
            report.append(
                f"| {row['agent']} | "
                f"{row['mean_C_out']:.3f} | {row['std_C_out']:.3f} | "
                f"{row['mean_success_rate']:.3f} | "
                f"{row['mean_time_cv']:.3f} | {row['mean_tokens_cv']:.3f} | "
                f"{int(row['deterministic_tasks'])} | {int(row['variable_tasks'])} |\n"
            )

    report.append("\n## Extended Resource Consistency\n\n")
    report.append("| Agent | API Calls CV | Actions CV | Errors CV | Call Latency CV |\n")
    report.append("|-------|--------------|------------|-----------|------------------|\n")

    for _, row in agent_df.iterrows():
        api_cv = row.get('mean_api_calls_cv', np.nan)
        actions_cv = row.get('mean_actions_cv', np.nan)
        errors_cv = row.get('mean_errors_cv', np.nan)
        latency_cv = row.get('mean_call_latency_cv', np.nan)
        report.append(
            f"| {row['agent']} | "
            f"{api_cv:.3f} | {actions_cv:.3f} | "
            f"{errors_cv:.3f} | {latency_cv:.3f} |\n"
        )

    report.append("\n## Metrics Explained\n\n")

    report.append("### Outcome Consistency (C_out)\n")
    report.append("Measures how deterministically an agent succeeds or fails on each task.\n\n")
    report.append("- **C_out ≈ 1**: Deterministic behavior (always succeeds or always fails on each task)\n")
    report.append("- **C_out ≈ 0**: Maximum variability (50/50 success/fail)\n")
    report.append("- **Formula**: `C_out = 1 - 2*std(outcomes)` where outcomes are binary (0 or 1)\n\n")

    if has_ctraj:
        report.append("### Trajectory Consistency (C_traj)\n")
        report.append("Measures how consistently an agent follows similar action sequences across runs.\n\n")
        report.append("- **C_traj ≈ 1**: Highly consistent trajectories (same actions in same order)\n")
        report.append("- **C_traj ≈ 0**: Highly variable trajectories (different action patterns)\n")
        report.append("- **Formula**: `C_traj = 1 - JSD({p_t(·|k)})` where JSD is Jensen-Shannon divergence\n")
        report.append("- **Note**: Computed from tool call sequences logged during execution\n\n")

    report.append("### Resource Consistency (CV)\n")
    report.append("Coefficient of Variation (CV = std / mean) measures relative variability in resource usage.\n\n")
    report.append("- **Lower CV**: More consistent resource usage across runs\n")
    report.append("- **Higher CV**: More variable resource usage\n")
    report.append("- **Time CV**: Consistency in total execution time\n")
    report.append("- **Token CV**: Consistency in token consumption\n")
    report.append("- **API Calls CV**: Consistency in number of API calls made\n")
    report.append("- **Actions CV**: Consistency in number of actions taken\n")
    report.append("- **Call Latency CV**: Consistency in per-call latency\n\n")

    report.append("## Key Findings\n\n")

    # Most/least consistent (higher C_out = more consistent)
    best_cout = agent_df.loc[agent_df['mean_C_out'].idxmax()]
    worst_cout = agent_df.loc[agent_df['mean_C_out'].idxmin()]

    report.append("### Outcome Consistency\n")
    report.append(f"- **Most consistent**: {best_cout['agent']} (C_out = {best_cout['mean_C_out']:.3f})\n")
    report.append(f"- **Least consistent**: {worst_cout['agent']} (C_out = {worst_cout['mean_C_out']:.3f})\n\n")

    # Resource consistency
    best_time = agent_df.loc[agent_df['mean_time_cv'].idxmin()]
    best_token = agent_df.loc[agent_df['mean_tokens_cv'].idxmin()]
    best_api_calls = agent_df.loc[agent_df['mean_api_calls_cv'].idxmin()]
    best_actions = agent_df.loc[agent_df['mean_actions_cv'].idxmin()]

    report.append("### Resource Consistency\n")
    report.append(f"- **Most time-consistent**: {best_time['agent']} (CV = {best_time['mean_time_cv']:.3f})\n")
    report.append(f"- **Most token-consistent**: {best_token['agent']} (CV = {best_token['mean_tokens_cv']:.3f})\n")
    report.append(f"- **Most API-calls-consistent**: {best_api_calls['agent']} (CV = {best_api_calls['mean_api_calls_cv']:.3f})\n")
    report.append(f"- **Most actions-consistent**: {best_actions['agent']} (CV = {best_actions['mean_actions_cv']:.3f})\n\n")

    # Capability
    best_acc = agent_df.loc[agent_df['mean_success_rate'].idxmax()]

    report.append("### Capability\n")
    report.append(f"- **Highest success rate**: {best_acc['agent']} ({best_acc['mean_success_rate']:.3f})\n\n")

    # Write report
    output_path = output_dir / 'consistency_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"📄 Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consistency from evaluation results"
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
    parser.add_argument(
        "--scaffold",
        type=str,
        default="toolcalling",
        help="Agent scaffold type to analyze: 'toolcalling', 'fewshot', or 'all' to include all scaffolds."
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔬 Consistency Analysis\n")
    print(f"📂 Results directory: {results_dir}")
    print(f"📊 Benchmark: {args.benchmark}")
    print(f"🔧 Scaffold filter: {args.scaffold}")
    print(f"📁 Output directory: {output_dir}\n")

    # Load results
    results_data = load_results_from_files(results_dir, args.benchmark)

    if not results_data:
        print("❌ No results found to analyze")
        return

    # Filter by scaffold type
    if args.scaffold.lower() != 'all':
        original_count = len(results_data)
        # Normalize scaffold name (handle both fewshot and few_shot)
        scaffold_filter = args.scaffold.lower().replace('_', '')

        filtered_data = {}
        for agent_name, agent_data in results_data.items():
            # Normalize agent name for comparison
            normalized_agent_name = agent_name.lower().replace('_', '')
            if scaffold_filter in normalized_agent_name:
                filtered_data[agent_name] = agent_data

        results_data = filtered_data
        filtered_count = len(results_data)

        print(f"🔍 Filtered from {original_count} to {filtered_count} agents with scaffold '{args.scaffold}'\n")

        if not results_data:
            print(f"❌ No agents found with scaffold type '{args.scaffold}'")
            return

    print(f"\n✅ Loaded results for {len(results_data)} agents\n")

    # Analyze
    print("📊 Computing consistency metrics...")
    task_df, agent_df = analyze_all_agents(results_data)

    if task_df.empty or agent_df.empty:
        print("❌ No valid consistency data computed")
        return

    # Save data
    print("\n💾 Saving results...")
    task_df.to_csv(output_dir / 'task_level_metrics.csv', index=False)
    print(f"   Saved: {output_dir / 'task_level_metrics.csv'}")

    agent_df.to_csv(output_dir / 'agent_level_metrics.csv', index=False)
    print(f"   Saved: {output_dir / 'agent_level_metrics.csv'}")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_outcome_consistency(task_df, agent_df, output_dir)
    plot_resource_consistency(task_df, output_dir)
    plot_trajectory_consistency(task_df, agent_df, output_dir)
    plot_task_heatmaps(task_df, output_dir)
    plot_comprehensive_comparison(agent_df, output_dir)
    plot_capability_vs_release_date(agent_df, output_dir)

    # Generate report
    print("\n📄 Generating report...")
    generate_report(task_df, agent_df, output_dir)

    print("\n✨ Analysis complete!")
    print(f"\n📂 All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - task_level_metrics.csv: Per-task metrics (includes C_traj)")
    print("  - agent_level_metrics.csv: Agent-level summary (includes C_traj)")
    print("  - outcome_consistency.png: Outcome consistency plots")
    print("  - resource_consistency.png: Resource usage plots")
    print("  - trajectory_consistency.png: Trajectory consistency plots (C_traj)")
    print("  - task_heatmaps.png: Per-task heatmaps")
    print("  - comprehensive_comparison.png: Overall comparison")
    print("  - capability_vs_release_date.png: Capability vs release date")
    print("  - consistency_report.md: Detailed report (includes C_traj)")


if __name__ == "__main__":
    main()

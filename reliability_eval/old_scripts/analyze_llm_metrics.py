#!/usr/bin/env python3
"""
LLM-Based Reliability Metrics Analysis

This script analyzes agent execution traces using LLM-powered analysis to compute
reliability metrics that benefit from semantic understanding:

1. S_comp (Compliance) - Semantic violation detection
2. V_heal (Self-Healing) - Recovery behavior detection
3. C_traj_llm (Trajectory Consistency) - Semantic trajectory similarity

Usage:
    # Analyze a specific benchmark's results
    python analyze_llm_metrics.py --results_dir results/ --benchmark taubench_airline

    # Analyze specific metrics only
    python analyze_llm_metrics.py --results_dir results/ --benchmark taubench_airline --metrics compliance recovery

    # Use a different model
    python analyze_llm_metrics.py --results_dir results/ --benchmark taubench_airline --model gpt-4o

    # Limit to specific number of tasks (for testing/cost control)
    python analyze_llm_metrics.py --results_dir results/ --benchmark taubench_airline --max_tasks 10
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hal.utils.llm_log_analyzer import (
    LLMLogAnalyzer,
)


def load_results_with_traces(results_dir: Path, benchmark: str) -> Dict[str, Dict]:
    """
    Load HAL evaluation results including full conversation traces.

    Returns nested dict: {agent_name: {run_id: {task_id: {trace_data}}}}
    """
    results_data = defaultdict(lambda: defaultdict(dict))

    benchmark_dir = results_dir / benchmark
    if not benchmark_dir.exists():
        print(f"Benchmark directory not found: {benchmark_dir}")
        return {}

    print(f"Loading results from: {benchmark_dir}")

    for run_dir in sorted(benchmark_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        # Find UPLOAD.json file
        upload_files = list(run_dir.glob("*_UPLOAD.json"))
        if not upload_files:
            continue

        upload_file = upload_files[0]

        try:
            with open(upload_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {upload_file}: {e}")
            continue

        # Extract agent name from run directory
        agent_name = extract_agent_name(run_dir.name)
        run_id = run_dir.name

        # Extract raw data
        raw_eval_results = data.get('raw_eval_results', {})
        raw_logging_results = data.get('raw_logging_results', [])

        # Build task-level traces
        # First, index logging results by task_id
        task_logs = defaultdict(list)
        for log_entry in raw_logging_results:
            task_id = log_entry.get('weave_task_id')
            if task_id is not None:
                task_logs[str(task_id)].append(log_entry)

        # Process each task
        for task_id, task_eval in raw_eval_results.items():
            task_id_str = str(task_id)

            if not isinstance(task_eval, dict):
                continue

            # Extract trace data
            trace_data = {
                'task_id': task_id_str,
                'success': int(task_eval.get('reward', 0.0)),
                'actions_taken': task_eval.get('taken_actions', []),
                'task': task_eval.get('task', {}),
                'raw_logs': task_logs.get(task_id_str, []),
            }

            # Try to extract conversation history from logs
            conversation = extract_conversation_from_logs(task_logs.get(task_id_str, []))
            if conversation:
                trace_data['conversation_history'] = conversation

            results_data[agent_name][run_id][task_id_str] = trace_data

        print(f"  Loaded {agent_name}: {len(raw_eval_results)} tasks from {run_dir.name}")

    return results_data


def extract_agent_name(run_dir_name: str) -> str:
    """Extract agent name from run directory name."""
    parts = run_dir_name.split('_')

    # Handle different benchmark prefixes
    if run_dir_name.startswith('taubench_airline'):
        agent_parts = parts[2:]
    elif run_dir_name.startswith('taubench_retail'):
        agent_parts = parts[2:]
    else:
        agent_parts = parts[1:]

    # Remove timestamp suffix
    if agent_parts and agent_parts[-1].isdigit():
        agent_parts = agent_parts[:-1]

    return '_'.join(agent_parts)


def extract_conversation_from_logs(log_entries: List[Dict]) -> List[Dict]:
    """
    Extract conversation history from raw logging results.

    Attempts to reconstruct the message sequence from Weave log entries.
    """
    conversation = []

    for entry in log_entries:
        # Look for inputs that contain messages
        inputs = entry.get('inputs', {})

        if 'messages' in inputs:
            # Direct messages field
            messages = inputs['messages']
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg not in conversation:
                        conversation.append(msg)

        # Look for outputs that contain assistant responses
        output = entry.get('output', {})
        if isinstance(output, dict):
            # Check for message in output
            if 'message' in output:
                msg = output['message']
                if isinstance(msg, dict) and msg not in conversation:
                    conversation.append(msg)

            # Check for choices (OpenAI format)
            choices = output.get('choices', [])
            if choices and isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict) and 'message' in choice:
                        msg = choice['message']
                        if isinstance(msg, dict) and msg not in conversation:
                            conversation.append(msg)

    return conversation


def analyze_compliance_for_results(
    results_data: Dict,
    analyzer: LLMLogAnalyzer,
    constraints: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run LLM-based compliance analysis on all results.

    Returns DataFrame with per-task compliance metrics.
    """
    rows = []
    total_tasks = 0
    analyzed_tasks = 0

    for agent_name, agent_runs in results_data.items():
        for run_id, tasks in agent_runs.items():
            for task_id, trace_data in tasks.items():
                total_tasks += 1

                if max_tasks and analyzed_tasks >= max_tasks:
                    continue

                # Get trace components
                conversation = trace_data.get('conversation_history', [])
                actions = trace_data.get('actions_taken', [])

                if not conversation and not actions:
                    # Skip if no trace data
                    continue

                print(f"  Analyzing compliance: {agent_name}/{task_id}...")

                try:
                    result = analyzer.analyze_compliance(
                        conversation_history=conversation,
                        actions_taken=actions,
                        constraints=constraints
                    )

                    rows.append({
                        'agent': agent_name,
                        'run_id': run_id,
                        'task_id': task_id,
                        'success': trace_data.get('success', 0),
                        'S_comp': result.S_comp,
                        'num_violations': len(result.violations),
                        'violations': json.dumps([v.to_dict() for v in result.violations]),
                        'analysis_model': result.analysis_model,
                    })

                    analyzed_tasks += 1

                except Exception as e:
                    print(f"    Error analyzing {task_id}: {e}")

    print(f"\nAnalyzed {analyzed_tasks}/{total_tasks} tasks for compliance")
    return pd.DataFrame(rows)


def analyze_recovery_for_results(
    results_data: Dict,
    analyzer: LLMLogAnalyzer,
    max_tasks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run LLM-based recovery/self-healing analysis on all results.

    Returns DataFrame with per-task recovery metrics.
    """
    rows = []
    total_tasks = 0
    analyzed_tasks = 0

    for agent_name, agent_runs in results_data.items():
        for run_id, tasks in agent_runs.items():
            for task_id, trace_data in tasks.items():
                total_tasks += 1

                if max_tasks and analyzed_tasks >= max_tasks:
                    continue

                conversation = trace_data.get('conversation_history', [])
                actions = trace_data.get('actions_taken', [])

                if not conversation and not actions:
                    continue

                print(f"  Analyzing recovery: {agent_name}/{task_id}...")

                try:
                    result = analyzer.detect_recovery_behavior(
                        conversation_history=conversation,
                        actions_taken=actions
                    )

                    rows.append({
                        'agent': agent_name,
                        'run_id': run_id,
                        'task_id': task_id,
                        'success': trace_data.get('success', 0),
                        'V_heal': result.V_heal,
                        'total_errors': result.total_errors_encountered,
                        'recoveries_attempted': result.total_recoveries_attempted,
                        'recoveries_successful': result.successful_recoveries,
                        'recovery_details': json.dumps([r.to_dict() for r in result.recovery_attempts]),
                        'analysis_model': result.analysis_model,
                    })

                    analyzed_tasks += 1

                except Exception as e:
                    print(f"    Error analyzing {task_id}: {e}")

    print(f"\nAnalyzed {analyzed_tasks}/{total_tasks} tasks for recovery")
    return pd.DataFrame(rows)


def analyze_trajectory_consistency_for_results(
    results_data: Dict,
    analyzer: LLMLogAnalyzer,
    max_tasks: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LLM-based trajectory consistency analysis across runs.

    Returns:
        (task_df, agent_df) - Per-task and agent-level metrics
    """
    # Group traces by agent and task
    task_traces = defaultdict(lambda: defaultdict(list))

    for agent_name, agent_runs in results_data.items():
        for run_id, tasks in agent_runs.items():
            for task_id, trace_data in tasks.items():
                conversation = trace_data.get('conversation_history', [])
                actions = trace_data.get('actions_taken', [])

                if conversation or actions:
                    task_traces[agent_name][task_id].append({
                        'conversation_history': conversation,
                        'actions_taken': actions,
                        'run_id': run_id,
                        'success': trace_data.get('success', 0),
                    })

    # Analyze trajectory consistency per task
    task_rows = []
    analyzed_tasks = 0

    for agent_name, agent_tasks in task_traces.items():
        for task_id, traces in agent_tasks.items():
            if max_tasks and analyzed_tasks >= max_tasks:
                break

            if len(traces) < 2:
                # Need at least 2 runs to compare
                continue

            print(f"  Analyzing trajectory consistency: {agent_name}/{task_id} ({len(traces)} runs)...")

            try:
                C_traj_llm, pairwise_results = analyzer.compute_trajectory_consistency_llm(traces)

                task_rows.append({
                    'agent': agent_name,
                    'task_id': task_id,
                    'num_runs': len(traces),
                    'C_traj_llm': C_traj_llm,
                    'num_comparisons': len(pairwise_results),
                    'mean_success': np.mean([t['success'] for t in traces]),
                    'analysis_model': analyzer.model,
                })

                analyzed_tasks += 1

            except Exception as e:
                print(f"    Error analyzing {task_id}: {e}")

        if max_tasks and analyzed_tasks >= max_tasks:
            break

    task_df = pd.DataFrame(task_rows)

    # Aggregate to agent level
    if not task_df.empty:
        agent_df = task_df.groupby('agent').agg({
            'C_traj_llm': ['mean', 'std'],
            'task_id': 'count',
            'mean_success': 'mean',
        }).reset_index()
        agent_df.columns = ['agent', 'mean_C_traj_llm', 'std_C_traj_llm', 'num_tasks', 'mean_success']
    else:
        agent_df = pd.DataFrame()

    print(f"\nAnalyzed {analyzed_tasks} tasks for trajectory consistency")
    return task_df, agent_df


def analyze_error_severity_for_results(
    results_data: Dict,
    analyzer: LLMLogAnalyzer,
    max_tasks: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run LLM-based error severity analysis (S_cost/S_tail) on all results.

    Returns DataFrame with per-task error severity metrics.
    """
    rows = []
    total_tasks = 0
    analyzed_tasks = 0

    for agent_name, agent_runs in results_data.items():
        for run_id, tasks in agent_runs.items():
            for task_id, trace_data in tasks.items():
                total_tasks += 1

                if max_tasks and analyzed_tasks >= max_tasks:
                    continue

                conversation = trace_data.get('conversation_history', [])
                actions = trace_data.get('actions_taken', [])

                if not conversation and not actions:
                    continue

                print(f"  Analyzing error severity: {agent_name}/{task_id}...")

                try:
                    # Build task result dict for context
                    task_result = {
                        'success': trace_data.get('success', 0),
                        'task': trace_data.get('task', {}),
                    }

                    result = analyzer.analyze_error_severity(
                        conversation_history=conversation,
                        actions_taken=actions,
                        task_result=task_result
                    )

                    rows.append({
                        'agent': agent_name,
                        'run_id': run_id,
                        'task_id': task_id,
                        'success': trace_data.get('success', 0),
                        'S_cost': result.S_cost,
                        'S_tail_95': result.S_tail_95,
                        'S_tail_max': result.S_tail_max,
                        'num_errors': len(result.errors),
                        'has_critical': result.has_critical_errors,
                        'has_high': result.has_high_severity_errors,
                        'summary': result.summary,
                        'errors': json.dumps([e.to_dict() for e in result.errors]),
                        'analysis_model': result.analysis_model,
                    })

                    analyzed_tasks += 1

                except Exception as e:
                    print(f"    Error analyzing {task_id}: {e}")

    print(f"\nAnalyzed {analyzed_tasks}/{total_tasks} tasks for error severity")
    return pd.DataFrame(rows)


def generate_report(
    compliance_df: Optional[pd.DataFrame],
    recovery_df: Optional[pd.DataFrame],
    trajectory_task_df: Optional[pd.DataFrame],
    trajectory_agent_df: Optional[pd.DataFrame],
    error_severity_df: Optional[pd.DataFrame],
    output_dir: Path,
    analysis_model: str,
):
    """Generate markdown report with LLM-based analysis results."""
    report = []
    report.append("# LLM-Based Reliability Metrics Analysis\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"Analysis Model: {analysis_model}\n\n")

    # Compliance section
    if compliance_df is not None and not compliance_df.empty:
        report.append("## Compliance Analysis (S_comp)\n\n")
        report.append("LLM-based compliance detection using semantic understanding.\n\n")

        # Agent-level summary
        agent_compliance = compliance_df.groupby('agent').agg({
            'S_comp': ['mean', 'std'],
            'num_violations': 'sum',
            'task_id': 'count',
        }).reset_index()
        agent_compliance.columns = ['agent', 'mean_S_comp', 'std_S_comp', 'total_violations', 'tasks_analyzed']

        report.append("### Agent-Level Compliance\n\n")
        report.append("| Agent | Mean S_comp | Std S_comp | Total Violations | Tasks Analyzed |\n")
        report.append("|-------|-------------|------------|------------------|----------------|\n")
        for _, row in agent_compliance.iterrows():
            report.append(f"| {row['agent']} | {row['mean_S_comp']:.3f} | {row['std_S_comp']:.3f} | {int(row['total_violations'])} | {int(row['tasks_analyzed'])} |\n")
        report.append("\n")

        # Violation breakdown if any
        all_violations = []
        for violations_json in compliance_df['violations']:
            try:
                violations = json.loads(violations_json)
                all_violations.extend(violations)
            except:
                pass

        if all_violations:
            report.append("### Violation Types Detected\n\n")
            violation_counts = defaultdict(int)
            for v in all_violations:
                violation_counts[v.get('constraint', 'unknown')] += 1

            report.append("| Constraint | Count |\n")
            report.append("|------------|-------|\n")
            for constraint, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
                report.append(f"| {constraint} | {count} |\n")
            report.append("\n")

    # Recovery section
    if recovery_df is not None and not recovery_df.empty:
        report.append("## Self-Healing Analysis (V_heal)\n\n")
        report.append("LLM-based detection of error recognition and recovery behavior.\n\n")

        # Agent-level summary
        agent_recovery = recovery_df.groupby('agent').agg({
            'V_heal': ['mean', 'std'],
            'total_errors': 'sum',
            'recoveries_successful': 'sum',
            'task_id': 'count',
        }).reset_index()
        agent_recovery.columns = ['agent', 'mean_V_heal', 'std_V_heal', 'total_errors', 'successful_recoveries', 'tasks_analyzed']

        report.append("### Agent-Level Recovery Metrics\n\n")
        report.append("| Agent | Mean V_heal | Std V_heal | Total Errors | Successful Recoveries | Tasks |\n")
        report.append("|-------|-------------|------------|--------------|----------------------|-------|\n")
        for _, row in agent_recovery.iterrows():
            report.append(f"| {row['agent']} | {row['mean_V_heal']:.3f} | {row['std_V_heal']:.3f} | {int(row['total_errors'])} | {int(row['successful_recoveries'])} | {int(row['tasks_analyzed'])} |\n")
        report.append("\n")

    # Trajectory consistency section
    if trajectory_agent_df is not None and not trajectory_agent_df.empty:
        report.append("## Trajectory Consistency (C_traj_llm)\n\n")
        report.append("LLM-based semantic similarity of execution paths across runs.\n\n")

        report.append("### Agent-Level Trajectory Consistency\n\n")
        report.append("| Agent | Mean C_traj_llm | Std C_traj_llm | Tasks Compared | Mean Success |\n")
        report.append("|-------|-----------------|----------------|----------------|---------------|\n")
        for _, row in trajectory_agent_df.iterrows():
            report.append(f"| {row['agent']} | {row['mean_C_traj_llm']:.3f} | {row['std_C_traj_llm']:.3f} | {int(row['num_tasks'])} | {row['mean_success']:.3f} |\n")
        report.append("\n")

    # Error severity section
    if error_severity_df is not None and not error_severity_df.empty:
        report.append("## Error Severity Analysis (S_cost/S_tail)\n\n")
        report.append("LLM-based context-aware error severity classification.\n\n")

        # Agent-level summary
        agent_severity = error_severity_df.groupby('agent').agg({
            'S_cost': ['mean', 'std'],
            'S_tail_max': 'max',
            'has_critical': 'sum',
            'has_high': 'sum',
            'task_id': 'count',
        }).reset_index()
        agent_severity.columns = ['agent', 'mean_S_cost', 'std_S_cost', 'max_S_tail', 'critical_errors', 'high_errors', 'tasks_analyzed']

        report.append("### Agent-Level Error Severity\n\n")
        report.append("| Agent | Mean S_cost | Std S_cost | Max S_tail | Critical | High | Tasks |\n")
        report.append("|-------|-------------|------------|------------|----------|------|-------|\n")
        for _, row in agent_severity.iterrows():
            report.append(f"| {row['agent']} | {row['mean_S_cost']:.2f} | {row['std_S_cost']:.2f} | {row['max_S_tail']:.2f} | {int(row['critical_errors'])} | {int(row['high_errors'])} | {int(row['tasks_analyzed'])} |\n")
        report.append("\n")

        # Highlight agents with critical errors
        critical_agents = agent_severity[agent_severity['critical_errors'] > 0]
        if not critical_agents.empty:
            report.append("### Agents with Critical Errors\n\n")
            for _, row in critical_agents.iterrows():
                report.append(f"- **{row['agent']}**: {int(row['critical_errors'])} critical error(s)\n")
            report.append("\n")

    # Methodology
    report.append("## Methodology\n\n")
    report.append("### LLM-Based Analysis vs Pattern Matching\n\n")
    report.append("This analysis uses semantic understanding via LLM to:\n\n")
    report.append("1. **Compliance (S_comp)**: Detect violations in context, distinguishing legitimate data handling from actual violations\n")
    report.append("2. **Self-Healing (V_heal)**: Identify error recognition and recovery attempts from reasoning patterns\n")
    report.append("3. **Trajectory Consistency (C_traj_llm)**: Compare logical execution paths beyond action name matching\n")
    report.append("4. **Error Severity (S_cost/S_tail)**: Context-aware severity classification distinguishing benign from dangerous errors\n\n")

    report.append("### Cost Considerations\n\n")
    report.append(f"- Analysis model: {analysis_model}\n")
    report.append("- Each task requires 1-3 LLM calls depending on metrics computed\n")
    report.append("- Trajectory consistency requires O(K^2) comparisons for K runs\n")
    report.append("- Consider using `--max_tasks` to limit analysis for cost control\n\n")

    # Write report
    output_path = output_dir / 'llm_metrics_report.md'
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze agent traces using LLM-based reliability metrics"
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
        default="reliability_eval/analysis_llm",
        help="Directory for output files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for analysis (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["compliance", "recovery", "trajectory", "error_severity"],
        choices=["compliance", "recovery", "trajectory", "error_severity"],
        help="Which metrics to compute (default: all)"
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        default=None,
        help="Compliance constraints to check (default: all standard constraints)"
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to analyze (for testing/cost control)"
    )
    parser.add_argument(
        "--scaffold",
        type=str,
        default="all",
        help="Agent scaffold type to analyze: 'toolcalling', 'fewshot', or 'all'"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLM-Based Reliability Metrics Analysis")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Analysis model: {args.model}")
    print(f"Metrics: {args.metrics}")
    if args.max_tasks:
        print(f"Max tasks: {args.max_tasks}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")

    # Load results
    print("Loading results...")
    results_data = load_results_with_traces(results_dir, args.benchmark)

    if not results_data:
        print("No results found to analyze")
        return

    # Filter by scaffold if specified
    if args.scaffold.lower() != 'all':
        scaffold_filter = args.scaffold.lower().replace('_', '')
        filtered_data = {}
        for agent_name, agent_data in results_data.items():
            normalized = agent_name.lower().replace('_', '')
            if scaffold_filter in normalized:
                filtered_data[agent_name] = agent_data
        results_data = filtered_data
        print(f"Filtered to {len(results_data)} agents with scaffold '{args.scaffold}'")

    if not results_data:
        print("No agents found matching filter")
        return

    # Initialize analyzer
    analyzer = LLMLogAnalyzer(model=args.model, cache_responses=True)

    # Run analyses
    compliance_df = None
    recovery_df = None
    trajectory_task_df = None
    trajectory_agent_df = None
    error_severity_df = None

    if "compliance" in args.metrics:
        print("\n" + "=" * 40)
        print("Running Compliance Analysis (S_comp)")
        print("=" * 40)
        compliance_df = analyze_compliance_for_results(
            results_data, analyzer, args.constraints, args.max_tasks
        )
        if not compliance_df.empty:
            compliance_df.to_csv(output_dir / 'compliance_llm.csv', index=False)
            print(f"Saved: {output_dir / 'compliance_llm.csv'}")

    if "recovery" in args.metrics:
        print("\n" + "=" * 40)
        print("Running Recovery Analysis (V_heal)")
        print("=" * 40)
        recovery_df = analyze_recovery_for_results(
            results_data, analyzer, args.max_tasks
        )
        if not recovery_df.empty:
            recovery_df.to_csv(output_dir / 'recovery_llm.csv', index=False)
            print(f"Saved: {output_dir / 'recovery_llm.csv'}")

    if "trajectory" in args.metrics:
        print("\n" + "=" * 40)
        print("Running Trajectory Consistency Analysis (C_traj_llm)")
        print("=" * 40)
        trajectory_task_df, trajectory_agent_df = analyze_trajectory_consistency_for_results(
            results_data, analyzer, args.max_tasks
        )
        if not trajectory_task_df.empty:
            trajectory_task_df.to_csv(output_dir / 'trajectory_llm_tasks.csv', index=False)
            print(f"Saved: {output_dir / 'trajectory_llm_tasks.csv'}")
        if not trajectory_agent_df.empty:
            trajectory_agent_df.to_csv(output_dir / 'trajectory_llm_agents.csv', index=False)
            print(f"Saved: {output_dir / 'trajectory_llm_agents.csv'}")

    if "error_severity" in args.metrics:
        print("\n" + "=" * 40)
        print("Running Error Severity Analysis (S_cost/S_tail)")
        print("=" * 40)
        error_severity_df = analyze_error_severity_for_results(
            results_data, analyzer, args.max_tasks
        )
        if not error_severity_df.empty:
            error_severity_df.to_csv(output_dir / 'error_severity_llm.csv', index=False)
            print(f"Saved: {output_dir / 'error_severity_llm.csv'}")

    # Generate report
    print("\n" + "=" * 40)
    print("Generating Report")
    print("=" * 40)
    generate_report(
        compliance_df, recovery_df,
        trajectory_task_df, trajectory_agent_df,
        error_severity_df,
        output_dir, args.model
    )

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    if compliance_df is not None and not compliance_df.empty:
        print("  - compliance_llm.csv: Per-task compliance analysis")
    if recovery_df is not None and not recovery_df.empty:
        print("  - recovery_llm.csv: Per-task recovery analysis")
    if trajectory_task_df is not None and not trajectory_task_df.empty:
        print("  - trajectory_llm_tasks.csv: Per-task trajectory consistency")
    if trajectory_agent_df is not None and not trajectory_agent_df.empty:
        print("  - trajectory_llm_agents.csv: Agent-level trajectory consistency")
    if error_severity_df is not None and not error_severity_df.empty:
        print("  - error_severity_llm.csv: Per-task error severity analysis")
    print("  - llm_metrics_report.md: Summary report")


if __name__ == "__main__":
    main()

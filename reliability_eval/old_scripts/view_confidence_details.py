#!/usr/bin/env python3
"""
View Confidence Assessment Details

This script extracts and displays the full confidence assessment interactions
from HAL evaluation results, including prompts and model responses.

Usage:
    python reliability_eval/view_confidence_details.py --results_dir results --benchmark taubench_airline
    python reliability_eval/view_confidence_details.py --run_id taubench_airline_gpt_5_2_1234567890
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def extract_confidence_details(results_dir: Path, benchmark: str = None, run_id: str = None):
    """Extract confidence details from results"""

    if run_id:
        # Specific run
        run_dirs = [results_dir / benchmark / run_id] if benchmark else []
        if not run_dirs:
            # Try to find it in any benchmark
            for bench_dir in results_dir.iterdir():
                if bench_dir.is_dir():
                    potential_run = bench_dir / run_id
                    if potential_run.exists():
                        run_dirs = [potential_run]
                        break
    else:
        # All runs for benchmark
        benchmark_dir = results_dir / benchmark
        if not benchmark_dir.exists():
            print(f"❌ Benchmark directory not found: {benchmark_dir}")
            return []

        run_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]

    all_details = []

    for run_dir in run_dirs:
        # Find output.json files
        output_files = list(run_dir.glob("*/output.json"))

        for output_file in output_files:
            task_id = output_file.parent.name

            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)

                # Extract confidence details
                for tid, task_data in data.items():
                    if isinstance(task_data, dict):
                        confidence = task_data.get('confidence')
                        confidence_details = task_data.get('confidence_details')

                        if confidence is not None:
                            all_details.append({
                                'run_id': run_dir.name,
                                'task_id': tid,
                                'confidence': confidence,
                                'details': confidence_details,
                                'reward': task_data.get('reward', None)
                            })
            except Exception as e:
                print(f"⚠️  Error reading {output_file}: {e}")
                continue

    return all_details


def print_confidence_details(details_list, max_items=None, verbose=False):
    """Pretty print confidence details"""

    if not details_list:
        print("No confidence details found.")
        return

    print(f"\n{'='*100}")
    print(f"Found {len(details_list)} tasks with confidence scores")
    print(f"{'='*100}\n")

    for i, item in enumerate(details_list[:max_items] if max_items else details_list):
        print(f"{'─'*100}")
        print(f"Run: {item['run_id']}")
        print(f"Task: {item['task_id']}")
        print(f"Confidence: {item['confidence']:.3f}")
        print(f"Reward (Success): {item['reward']}")

        if item['details']:
            details = item['details']
            print("\n📝 Prompt (truncated):")
            prompt_lines = details.get('prompt', '').split('\n')[:5]
            for line in prompt_lines:
                print(f"   {line}")
            if len(details.get('prompt', '').split('\n')) > 5:
                print(f"   ... ({len(details.get('prompt', '').split('\n')) - 5} more lines)")

            print("\n🤖 Model Response:")
            print(f"   '{details.get('model_response', 'N/A')}'")

            print("\n📊 Metadata:")
            print(f"   Model: {details.get('model', 'N/A')}")
            print(f"   Actions: {details.get('num_actions', 'N/A')}")
            print(f"   Errors: {details.get('num_errors', 'N/A')}")
            print(f"   Parsed Score: {details.get('parsed_score', 'N/A'):.3f}")

            if details.get('fallback'):
                print("   ⚠️  FALLBACK: Used heuristic (API error)")

            if verbose:
                print("\n📄 Full Prompt:")
                print(details.get('prompt', 'N/A'))
        else:
            print("\n⚠️  No detailed information stored (set store_confidence_details=True)")

        print()

    if max_items and len(details_list) > max_items:
        print(f"\n... ({len(details_list) - max_items} more items not shown)")
        print("Use --max_items to show more or --verbose for full prompts")


def analyze_confidence_responses(details_list):
    """Analyze the distribution of model responses"""

    if not details_list:
        return

    print(f"\n{'='*100}")
    print("Confidence Response Analysis")
    print(f"{'='*100}\n")

    # Group by model response
    response_counts = defaultdict(int)
    response_by_success = defaultdict(lambda: {'success': 0, 'failure': 0})

    for item in details_list:
        if item['details']:
            response = item['details'].get('model_response', 'Unknown')
            response_counts[response] += 1

            if item['reward'] and item['reward'] > 0:
                response_by_success[response]['success'] += 1
            else:
                response_by_success[response]['failure'] += 1

    # Sort by frequency
    sorted_responses = sorted(response_counts.items(), key=lambda x: -x[1])

    print(f"Total unique responses: {len(sorted_responses)}\n")

    print("Response Distribution:")
    print(f"{'Response':<20} {'Count':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<15}")
    print(f"{'-'*75}")

    for response, count in sorted_responses[:20]:  # Top 20
        success = response_by_success[response]['success']
        failure = response_by_success[response]['failure']
        success_rate = success / (success + failure) if (success + failure) > 0 else 0

        print(f"{response:<20} {count:<10} {success:<10} {failure:<10} {success_rate:<15.1%}")

    # Check for uniformity
    if len(sorted_responses) == 1:
        print(f"\n⚠️  WARNING: All responses are identical: '{sorted_responses[0][0]}'")
        print("   This indicates poor confidence calibration.")
    elif len(sorted_responses) <= 3:
        print(f"\n⚠️  WARNING: Only {len(sorted_responses)} unique responses found")
        print("   Model has poor confidence discrimination.")
    else:
        print(f"\n✓ Good variance: {len(sorted_responses)} unique responses")


def main():
    parser = argparse.ArgumentParser(
        description="View confidence assessment details from HAL results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="taubench_airline",
        help="Benchmark name"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Specific run ID (optional)"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=10,
        help="Maximum items to display (default: 10, use 0 for all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full prompts"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show response distribution analysis"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("🔍 Extracting confidence details...")
    print(f"   Results dir: {results_dir}")
    print(f"   Benchmark: {args.benchmark}")
    if args.run_id:
        print(f"   Run ID: {args.run_id}")

    details = extract_confidence_details(
        results_dir,
        benchmark=args.benchmark,
        run_id=args.run_id
    )

    if not details:
        print("\n❌ No confidence details found")
        print("\nPossible reasons:")
        print("  1. No evaluations have been run yet")
        print("  2. compute_confidence=False in agent config")
        print("  3. store_confidence_details=False (only score stored, not full details)")
        print("\nTo fix: Run evaluation with compute_confidence=True and store_confidence_details=True")
        return

    # Display details
    max_items = None if args.max_items == 0 else args.max_items
    print_confidence_details(details, max_items=max_items, verbose=args.verbose)

    # Analyze if requested
    if args.analyze:
        analyze_confidence_responses(details)

    print(f"\n{'='*100}")
    print("✨ View complete!")
    print("\nTo view in Weave dashboard:")
    print("  1. Go to https://wandb.ai/[your-entity]/[your-project]/weave")
    print("  2. Search for 'confidence_assessment' in the logs")
    print("  3. Filter by model or task_id")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()

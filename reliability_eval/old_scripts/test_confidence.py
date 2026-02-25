#!/usr/bin/env python3
"""
Test script to debug confidence scoring and see full interactions

Usage:
    python reliability_eval/test_confidence.py
"""


# Mock a simple confidence assessment
def test_confidence_parsing():
    """Test if the confidence parsing is working correctly"""
    import re

    test_cases = [
        ("85", 0.85, "Plain number"),
        ("75", 0.75, "Different number"),
        ("100", 1.0, "Max value"),
        ("0", 0.0, "Min value"),
        ("Score: 85", 0.85, "With prefix"),
        ("I rate this 90", 0.90, "In sentence"),
        ("", 0.5, "Empty (should default)"),
        ("no number here", 0.5, "No number (should default)"),
    ]

    print("Testing confidence parsing logic:")
    print("="*80)

    for response_text, expected, description in test_cases:
        confidence_text = response_text.strip()
        numbers = re.findall(r'\d+', confidence_text)

        if numbers:
            confidence_score = float(numbers[0]) / 100.0
            confidence_score = max(0.0, min(1.0, confidence_score))
        else:
            confidence_score = 0.5

        status = "✓" if abs(confidence_score - expected) < 0.001 else "✗"
        print(f"{status} {description:30s} '{response_text:20s}' -> {confidence_score:.2f} (expected {expected:.2f})")

    print("="*80)


def analyze_confidence_distribution():
    """Analyze confidence scores from existing results"""
    from pathlib import Path
    import json
    from collections import Counter

    results_dir = Path("results/taubench_airline")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    all_confidences = []

    for run_dir in results_dir.glob("*"):
        if not run_dir.is_dir():
            continue

        # Find UPLOAD.json
        upload_files = list(run_dir.glob("*UPLOAD.json"))
        if not upload_files:
            continue

        try:
            with open(upload_files[0], 'r') as f:
                data = json.load(f)

            raw_eval = data.get('raw_eval_results', {})

            for task_id, result in raw_eval.items():
                if isinstance(result, dict) and 'confidence' in result:
                    conf = result['confidence']
                    if conf is not None:
                        all_confidences.append(conf)
        except Exception:
            continue

    if not all_confidences:
        print("No confidence scores found in results")
        return

    print("\nConfidence Distribution Analysis:")
    print("="*80)
    print(f"Total samples: {len(all_confidences)}")
    print(f"Unique values: {len(set(all_confidences))}")
    print(f"Min: {min(all_confidences):.3f}")
    print(f"Max: {max(all_confidences):.3f}")
    print(f"Mean: {sum(all_confidences)/len(all_confidences):.3f}")

    # Count occurrences
    counter = Counter(all_confidences)
    print("\nMost common values:")
    for value, count in counter.most_common(10):
        percentage = (count / len(all_confidences)) * 100
        print(f"  {value:.2f}: {count:3d} occurrences ({percentage:5.1f}%)")

    # Check if all are the same
    if len(set(all_confidences)) == 1:
        print(f"\n⚠️  WARNING: All confidence scores are identical: {all_confidences[0]:.2f}")
        print("   This suggests either:")
        print("   1. The model is returning the same score every time (poor calibration)")
        print("   2. There's a bug causing scores to be cached or defaulted")
        print("   3. The confidence computation is not working correctly")

    print("="*80)


def check_confidence_in_logs():
    """Check if confidence assessment calls are in logs"""
    from pathlib import Path

    results_dir = Path("results/taubench_airline")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    print("\nChecking for confidence assessment in logs:")
    print("="*80)

    found_confidence_prompts = 0
    found_confidence_responses = 0

    for run_dir in results_dir.glob("*"):
        if not run_dir.is_dir():
            continue

        # Check verbose log
        verbose_log = run_dir / f"{run_dir.name}_verbose.log"
        if verbose_log.exists():
            with open(verbose_log, 'r') as f:
                content = f.read()

                if 'assess your confidence' in content.lower():
                    found_confidence_prompts += 1

                if '✓ Confidence assessment' in content:
                    found_confidence_responses += 1
                    # Extract and print the lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if '✓ Confidence assessment' in line:
                            print(f"  {run_dir.name}: {line.strip()}")

    print(f"\nFound {found_confidence_prompts} runs with confidence prompts")
    print(f"Found {found_confidence_responses} runs with logged responses")

    if found_confidence_responses == 0:
        print("\n⚠️  No confidence assessment responses found in logs!")
        print("   Rerun an evaluation to see the new logging output.")

    print("="*80)


if __name__ == "__main__":
    print("Confidence Scoring Diagnostic Tool\n")

    # Test 1: Parsing logic
    test_confidence_parsing()

    # Test 2: Analyze distribution
    try:
        analyze_confidence_distribution()
    except Exception as e:
        print(f"Error analyzing distribution: {e}")

    # Test 3: Check logs
    try:
        check_confidence_in_logs()
    except Exception as e:
        print(f"Error checking logs: {e}")

    print("\n✨ Diagnostic complete!")
    print("\nTo see full confidence interactions:")
    print("1. Run a new evaluation with the updated agent code")
    print("2. Check the verbose log file: results/benchmark/run_id/run_id_verbose.log")
    print("3. Look for lines starting with '✓ Confidence assessment'")

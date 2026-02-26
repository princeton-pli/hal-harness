"""Abstention detection and phase runner for reliability_eval."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from reliability_eval.types import EvaluationLog, RunResult


def detect_abstention(conversation_history: list, actions_taken: list) -> dict:
    """
    Detect abstention/deferral behavior in the agent's responses.

    This is a standalone version for post-hoc analysis of existing traces.
    """
    # Patterns for different abstention types
    INABILITY_PATTERNS = [
        r"\bi(?:'m| am) (?:not )?(?:able|unable) to\b",
        r"\bi cannot\b",
        r"\bi can't\b",
        r"\bi'm not capable\b",
        r"\bthis is (?:beyond|outside) (?:my|the) (?:capabilities|scope)\b",
        r"\bi don't have (?:the ability|access|permission)\b",
        r"\bimpossible for me to\b",
    ]

    UNCERTAINTY_PATTERNS = [
        r"\bi(?:'m| am) not (?:sure|certain)\b",
        r"\bi don't know\b",
        r"\bi'm uncertain\b",
        r"\bi'm unsure\b",
        r"\bunclear (?:to me|what|how|whether)\b",
        r"\bi(?:'m| am) not confident\b",
        r"\bthis is ambiguous\b",
        r"\bi may be wrong\b",
        r"\btake this with (?:a grain of salt|caution)\b",
    ]

    CLARIFICATION_PATTERNS = [
        r"\bcould you (?:please )?(?:clarify|explain|specify)\b",
        r"\bi need (?:more )?(?:information|details|context)\b",
        r"\bcan you (?:please )?(?:provide|give) (?:more )?\b",
        r"\bwhat (?:exactly |specifically )?do you mean\b",
        r"\bcould you be more specific\b",
        r"\bi'm not sure what you're asking\b",
        r"\bplease (?:clarify|specify|explain)\b",
    ]

    REFUSAL_PATTERNS = [
        r"\bi (?:cannot|can't|won't|will not) (?:proceed|continue|complete)\b",
        r"\bi(?:'m| am) (?:not )?(?:going to|able to) (?:do|perform|complete) (?:this|that)\b",
        r"\bi must (?:stop|decline|refuse)\b",
        r"\bi (?:have to|need to) stop\b",
        r"\bstopping here\b",
        r"\bunable to (?:proceed|continue|complete)\b",
        r"\bcannot (?:proceed|continue|complete)\b",
    ]

    evidence = []
    abstention_scores = {
        'inability': 0.0,
        'uncertainty': 0.0,
        'clarification': 0.0,
        'refusal': 0.0,
    }

    # Extract ONLY assistant/agent messages from conversation
    # We deliberately ignore user messages - abstention is about the agent's behavior
    assistant_messages = []
    for msg in conversation_history:
        if isinstance(msg, dict):
            role = msg.get('role', '')
            content = msg.get('content', '')
        else:
            role = getattr(msg, 'role', '')
            content = getattr(msg, 'content', '')

        # Only process assistant messages, skip user/system messages
        if role == 'assistant' and content:
            assistant_messages.append(content.lower() if isinstance(content, str) else str(content).lower())

    # Check each pattern category
    for text in assistant_messages:
        for pattern in INABILITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores['inability'] += 1.0
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[inability] ...{text[start:end]}...")

        for pattern in UNCERTAINTY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores['uncertainty'] += 0.7
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[uncertainty] ...{text[start:end]}...")

        for pattern in CLARIFICATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores['clarification'] += 0.5
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[clarification] ...{text[start:end]}...")

        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores['refusal'] += 1.0
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[refusal] ...{text[start:end]}...")

    # Check for early termination
    early_termination = len(actions_taken) <= 2

    # Calculate overall abstention strength
    total_score = sum(abstention_scores.values())
    abstention_strength = min(1.0, total_score / 3.0)

    # Determine primary abstention type
    if total_score == 0:
        abstention_type = 'none'
    else:
        abstention_type = max(abstention_scores, key=abstention_scores.get)

    # Determine if abstention occurred
    abstained = abstention_strength >= 0.3 or any(abstention_scores[t] >= 1.0 for t in ['inability', 'refusal'])

    return {
        'abstained': abstained,
        'abstention_type': abstention_type,
        'abstention_strength': abstention_strength,
        'evidence': evidence[:5],
        'early_termination': early_termination,
        'scores_by_type': abstention_scores,
        'num_assistant_messages': len(assistant_messages),
    }


# =============================================================================
# CONFIGURATION - Edit config.py to customize your evaluation
# =============================================================================

from reliability_eval.config import AGENT_CONFIGS, BENCHMARK_CONFIGS, PHASE_SETTINGS  # noqa: E402


# =============================================================================
# DATA CLASSES
# =============================================================================

from reliability_eval.types import EvaluationLog, RunResult  # noqa: E402


def run_abstention_phase(
    combinations: List[tuple],
    results_dir: Path,
    log: EvaluationLog,
    log_path: Path,
) -> int:
    """
    Run abstention detection on existing traces.

    This phase:
    1. Finds all existing result files for the configured agents/benchmarks
    2. For each task, extracts conversation_history and taken_actions
    3. Runs regex-based abstention detection
    4. Writes results back into the JSON files under 'abstention' key

    Computes: Abstention rate, type distribution, correlation with success/failure
    """
    print("\n" + "="*80)
    print("🛑 PHASE: ABSTENTION DETECTION")
    print("="*80)
    print(f"   Results dir: {results_dir}")

    total_tasks_analyzed = 0
    total_files_updated = 0
    abstention_summary = {
        'total_abstained': 0,
        'by_type': {'inability': 0, 'uncertainty': 0, 'clarification': 0, 'refusal': 0, 'none': 0},
        'abstained_and_failed': 0,
        'abstained_and_succeeded': 0,
        'not_abstained_and_failed': 0,
        'not_abstained_and_succeeded': 0,
    }

    for agent_config, benchmark_config, bench_name in combinations:
        agent_name = agent_config['name']

        print(f"\n{'─'*60}")
        print(f"🔍 Analyzing: {agent_name} on {bench_name}")
        print(f"{'─'*60}")

        # Find all result directories for this agent/benchmark
        benchmark_dir = results_dir / bench_name
        if not benchmark_dir.exists():
            print(f"   ⚠️  No results directory found: {benchmark_dir}")
            continue

        # Find matching run directories
        for run_dir in sorted(benchmark_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Check if this run matches our agent
            if agent_name not in run_dir.name:
                continue

            # Skip non-baseline results (fault, structural, prompt_sensitivity)
            run_dir_name = run_dir.name.lower()
            if any(phase in run_dir_name for phase in ['fault', 'struct', 'structural', 'prompt_sensitivity', 'prompt_mild', 'prompt_medium', 'prompt_strong', 'prompt_naturalistic']):
                continue

            # Find the UPLOAD.json file
            upload_files = list(run_dir.glob("*_UPLOAD.json"))
            if not upload_files:
                continue

            upload_file = upload_files[0]
            print(f"\n   📄 Processing: {upload_file.name}")

            # Load the results
            try:
                with open(upload_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"      ❌ Failed to load: {e}")
                continue

            raw_eval = data.get('raw_eval_results', {})
            if not raw_eval:
                print("      ⚠️  No raw_eval_results found")
                continue

            tasks_in_file = 0
            modified = False

            for task_id, task_eval in raw_eval.items():
                if not isinstance(task_eval, dict):
                    continue

                # Always recompute abstention (replace existing data if present)
                # Get conversation history and actions
                conversation_history = task_eval.get('conversation_history', [])
                taken_actions = task_eval.get('taken_actions', [])

                if not conversation_history:
                    # Try to get from other possible locations
                    if 'messages' in task_eval:
                        conversation_history = task_eval['messages']

                if not conversation_history and not taken_actions:
                    continue

                success = float(task_eval.get('reward', 0.0)) > 0

                print(f"      🔬 Task {task_id}: Analyzing...", end=" ", flush=True)

                try:
                    # Run abstention detection
                    abstention_result = detect_abstention(
                        conversation_history=conversation_history,
                        actions_taken=taken_actions,
                    )

                    # Store back in task
                    task_eval['abstention'] = {
                        'abstained': abstention_result['abstained'],
                        'abstention_type': abstention_result['abstention_type'],
                        'abstention_strength': abstention_result['abstention_strength'],
                        'early_termination': abstention_result['early_termination'],
                        'evidence': abstention_result['evidence'],
                        'scores_by_type': abstention_result['scores_by_type'],
                    }

                    modified = True
                    tasks_in_file += 1
                    total_tasks_analyzed += 1

                    # Update summary
                    if abstention_result['abstained']:
                        abstention_summary['total_abstained'] += 1
                        abstention_summary['by_type'][abstention_result['abstention_type']] += 1
                        if success:
                            abstention_summary['abstained_and_succeeded'] += 1
                        else:
                            abstention_summary['abstained_and_failed'] += 1
                        print(f"🛑 {abstention_result['abstention_type']} (strength={abstention_result['abstention_strength']:.2f})")
                    else:
                        abstention_summary['by_type']['none'] += 1
                        if success:
                            abstention_summary['not_abstained_and_succeeded'] += 1
                        else:
                            abstention_summary['not_abstained_and_failed'] += 1
                        print("✅ no abstention")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    task_eval['abstention'] = {
                        'abstained': None,
                        'error': str(e),
                    }
                    modified = True

            # Save back to file if modified
            if modified:
                try:
                    with open(upload_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"   💾 Saved {tasks_in_file} task analyses to {upload_file.name}")
                    total_files_updated += 1
                except Exception as e:
                    print(f"   ❌ Failed to save: {e}")

        # Log result for this agent
        result = RunResult(
            agent=agent_name,
            benchmark=bench_name,
            phase="abstention",
            repetition=1,
            success=True,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
            error_message=None,
        )
        log.add_result(result)
        log.save(log_path)

    # Print summary
    print("\n✨ Abstention phase complete:")
    print(f"   📊 Tasks analyzed: {total_tasks_analyzed}")
    print(f"   📁 Files updated: {total_files_updated}")
    print("\n   📈 Abstention Summary:")
    print(f"      Total abstained: {abstention_summary['total_abstained']}")
    print(f"      By type: {abstention_summary['by_type']}")
    print("\n   🎯 Correlation with success:")
    print(f"      Abstained + Failed:     {abstention_summary['abstained_and_failed']}")
    print(f"      Abstained + Succeeded:  {abstention_summary['abstained_and_succeeded']}")
    print(f"      No abstention + Failed: {abstention_summary['not_abstained_and_failed']}")
    print(f"      No abstention + Succeeded: {abstention_summary['not_abstained_and_succeeded']}")

    # Compute calibration metrics if we have data
    total = (abstention_summary['abstained_and_failed'] + abstention_summary['abstained_and_succeeded'] +
             abstention_summary['not_abstained_and_failed'] + abstention_summary['not_abstained_and_succeeded'])
    if total > 0 and abstention_summary['total_abstained'] > 0:
        # Precision: P(fail | abstain)
        precision = abstention_summary['abstained_and_failed'] / abstention_summary['total_abstained'] if abstention_summary['total_abstained'] > 0 else 0
        # Recall: P(abstain | fail)
        total_failed = abstention_summary['abstained_and_failed'] + abstention_summary['not_abstained_and_failed']
        recall = abstention_summary['abstained_and_failed'] / total_failed if total_failed > 0 else 0
        print("\n   📊 Abstention Calibration:")
        print(f"      Precision (P(fail|abstain)): {precision:.2%}")
        print(f"      Recall (P(abstain|fail)):    {recall:.2%}")

    return total_tasks_analyzed


# =============================================================================
# MAIN
# =============================================================================

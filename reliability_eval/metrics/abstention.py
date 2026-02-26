"""Abstention calibration metrics: abstention_rate, abstention_precision, abstention_recall, abstention_selective_accuracy, abstention_calibration."""

import re

import numpy as np
from typing import Dict, List


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
        "inability": 0.0,
        "uncertainty": 0.0,
        "clarification": 0.0,
        "refusal": 0.0,
    }

    # Extract ONLY assistant/agent messages from conversation
    # We deliberately ignore user messages - abstention is about the agent's behavior
    assistant_messages = []
    for msg in conversation_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

        # Only process assistant messages, skip user/system messages
        if role == "assistant" and content:
            assistant_messages.append(
                content.lower() if isinstance(content, str) else str(content).lower()
            )

    # Check each pattern category
    for text in assistant_messages:
        for pattern in INABILITY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["inability"] += 1.0
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[inability] ...{text[start:end]}...")

        for pattern in UNCERTAINTY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["uncertainty"] += 0.7
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[uncertainty] ...{text[start:end]}...")

        for pattern in CLARIFICATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["clarification"] += 0.5
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    evidence.append(f"[clarification] ...{text[start:end]}...")

        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                abstention_scores["refusal"] += 1.0
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
        abstention_type = "none"
    else:
        abstention_type = max(abstention_scores, key=abstention_scores.get)

    # Determine if abstention occurred
    abstained = abstention_strength >= 0.3 or any(
        abstention_scores[t] >= 1.0 for t in ["inability", "refusal"]
    )

    return {
        "abstained": abstained,
        "abstention_type": abstention_type,
        "abstention_strength": abstention_strength,
        "evidence": evidence[:5],
        "early_termination": early_termination,
        "scores_by_type": abstention_scores,
        "num_assistant_messages": len(assistant_messages),
    }


def compute_abstention_metrics(runs: List[Dict]) -> Dict:
    """
    Compute abstention calibration metrics from runs with abstention detection.

    Abstention calibration measures how well the model's decision to abstain/defer
    correlates with actual task failure. A well-calibrated model should:
    - Abstain when it's likely to fail (good calibration)
    - Proceed confidently when it's likely to succeed (good calibration)

    Metrics:
    - abstention_rate: Fraction of tasks where model abstained
    - abstention_precision: P(fail | abstain) - when it abstains, how often was it right to?
    - abstention_recall: P(abstain | fail) - when it fails, how often did it abstain?
    - selective_accuracy: Accuracy on tasks where it did NOT abstain
    - abstention_f1: Harmonic mean of precision and recall
    - calibration_score: Combined measure of abstention quality

    Returns:
        Dict with abstention metrics and detailed breakdown
    """
    # Collect abstention and success data
    abstained_list = []
    success_list = []
    abstention_types = []
    abstention_strengths = []

    for run in runs:
        raw_eval = run.get("raw_eval_results", {})
        for task_eval in raw_eval.values():
            if isinstance(task_eval, dict):
                abstention = task_eval.get("abstention", {})
                if abstention:
                    abstained = abstention.get("abstained", False)
                    abstained_list.append(1 if abstained else 0)
                    success_list.append(int(task_eval.get("reward", 0.0)))
                    abstention_types.append(abstention.get("abstention_type", "none"))
                    abstention_strengths.append(
                        abstention.get("abstention_strength", 0.0)
                    )

    if not abstained_list:
        return {
            "abstention_rate": np.nan,
            "abstention_precision": np.nan,
            "abstention_recall": np.nan,
            "selective_accuracy": np.nan,
            "abstention_f1": np.nan,
            "calibration_score": np.nan,
            "confusion_matrix": {},
            "type_breakdown": {},
            "n_tasks": 0,
        }

    abstained = np.array(abstained_list)
    success = np.array(success_list)
    fail = 1 - success

    n_tasks = len(abstained)
    n_abstained = np.sum(abstained)
    n_failed = np.sum(fail)

    # Confusion matrix for abstention vs failure
    # True Positive: Abstained AND Failed (correctly abstained)
    # False Positive: Abstained AND Succeeded (over-cautious)
    # False Negative: Proceeded AND Failed (should have abstained)
    # True Negative: Proceeded AND Succeeded (correctly proceeded)
    tp = np.sum((abstained == 1) & (fail == 1))  # Abstained + Failed
    fp = np.sum((abstained == 1) & (fail == 0))  # Abstained + Succeeded
    fn = np.sum((abstained == 0) & (fail == 1))  # Proceeded + Failed
    tn = np.sum((abstained == 0) & (fail == 0))  # Proceeded + Succeeded

    # Metrics
    abstention_rate = n_abstained / n_tasks if n_tasks > 0 else 0.0

    # Precision: P(fail | abstain) = TP / (TP + FP)
    abstention_precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    # Recall: P(abstain | fail) = TP / (TP + FN)
    abstention_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    # Selective accuracy: Accuracy when NOT abstaining = TN / (TN + FN)
    selective_accuracy = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    # F1 score for abstention
    if not np.isnan(abstention_precision) and not np.isnan(abstention_recall):
        if (abstention_precision + abstention_recall) > 0:
            abstention_f1 = (
                2
                * (abstention_precision * abstention_recall)
                / (abstention_precision + abstention_recall)
            )
        else:
            abstention_f1 = 0.0
    else:
        abstention_f1 = np.nan

    # Calibration score: Combined measure
    # Higher is better - rewards both correct abstentions and correct proceeding
    calibration_score = (tp + tn) / n_tasks if n_tasks > 0 else np.nan

    # Type breakdown
    type_counts = {}
    type_success_rates = {}
    for t, s in zip(abstention_types, success_list):
        if t not in type_counts:
            type_counts[t] = 0
            type_success_rates[t] = []
        type_counts[t] += 1
        type_success_rates[t].append(s)

    type_breakdown = {
        t: {
            "count": type_counts[t],
            "success_rate": np.mean(type_success_rates[t])
            if type_success_rates[t]
            else 0.0,
        }
        for t in type_counts
    }

    return {
        "abstention_rate": float(abstention_rate),
        "abstention_precision": float(abstention_precision)
        if not np.isnan(abstention_precision)
        else None,
        "abstention_recall": float(abstention_recall)
        if not np.isnan(abstention_recall)
        else None,
        "selective_accuracy": float(selective_accuracy)
        if not np.isnan(selective_accuracy)
        else None,
        "abstention_f1": float(abstention_f1) if not np.isnan(abstention_f1) else None,
        "calibration_score": float(calibration_score)
        if not np.isnan(calibration_score)
        else None,
        "confusion_matrix": {
            "abstained_and_failed": int(tp),
            "abstained_and_succeeded": int(fp),
            "proceeded_and_failed": int(fn),
            "proceeded_and_succeeded": int(tn),
        },
        "type_breakdown": type_breakdown,
        "mean_abstention_strength": float(np.mean(abstention_strengths))
        if abstention_strengths
        else 0.0,
        "n_tasks": n_tasks,
        "n_abstained": int(n_abstained),
        "n_failed": int(n_failed),
    }

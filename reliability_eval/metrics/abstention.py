"""Abstention calibration metrics: A_rate, A_prec, A_rec, A_sel, A_cal."""

import numpy as np
from typing import Dict, List


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

"""Robustness metrics: R_fault, R_struct, R_prompt."""

import numpy as np
from typing import Dict, List, Tuple

from reliability_eval.constants import EPSILON


def compute_accuracy(runs: List[Dict]) -> float:
    """
    Compute accuracy from runs.

    Handles both normal results and prompt sensitivity results:
    - Normal: {task_id: {'reward': 0 or 1, ...}}
    - Prompt sensitivity: {task_id: [{'variation_id': str, 'score': float}, ...]}
    """
    successes = []
    for run in runs:
        for task_eval in run["raw_eval_results"].values():
            if isinstance(task_eval, dict):
                # Normal result format
                successes.append(int(task_eval.get("reward", 0.0)))
            elif isinstance(task_eval, list):
                # Prompt sensitivity format: list of variation results
                for var_result in task_eval:
                    if isinstance(var_result, dict):
                        # Use 'score' or 'reward' field, treat as binary (>0 = success)
                        score = var_result.get("score", var_result.get("reward", 0))
                        successes.append(int(float(score) > 0))
    return np.mean(successes) if successes else np.nan


def compute_robustness_ratio(
    baseline_runs: List[Dict], perturbed_runs: List[Dict]
) -> Tuple[float, float]:
    """
    Compute robustness ratio (paper Definitions 3.4, 3.5).

    R = Acc(perturbed) / Acc(baseline), clamped to [0, 1]

    Returns:
        Tuple of (ratio, bootstrap_se)
    """
    baseline_acc = compute_accuracy(baseline_runs)
    perturbed_acc = compute_accuracy(perturbed_runs)

    if np.isnan(baseline_acc) or np.isnan(perturbed_acc) or baseline_acc < EPSILON:
        return np.nan, np.nan

    ratio = min(perturbed_acc / baseline_acc, 1.0)

    # Bootstrap SE: resample per-task successes and recompute ratio
    def _collect_successes(runs):
        successes = []
        for run in runs:
            for task_eval in run["raw_eval_results"].values():
                if isinstance(task_eval, dict):
                    successes.append(int(task_eval.get("reward", 0.0)))
                elif isinstance(task_eval, list):
                    for var_result in task_eval:
                        if isinstance(var_result, dict):
                            score = var_result.get("score", var_result.get("reward", 0))
                            successes.append(int(float(score) > 0))
        return np.array(successes)

    base_s = _collect_successes(baseline_runs)
    pert_s = _collect_successes(perturbed_runs)
    n_base, n_pert = len(base_s), len(pert_s)

    if n_base < 2 or n_pert < 2:
        return ratio, np.nan

    rng = np.random.default_rng(42)
    n_boot = 200
    boot_ratios = []
    for _ in range(n_boot):
        b_acc = np.mean(base_s[rng.choice(n_base, size=n_base, replace=True)])
        p_acc = np.mean(pert_s[rng.choice(n_pert, size=n_pert, replace=True)])
        if b_acc > EPSILON:
            boot_ratios.append(min(p_acc / b_acc, 1.0))
    se = np.std(boot_ratios) if len(boot_ratios) >= 2 else np.nan

    return ratio, se

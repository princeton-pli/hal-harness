"""Agent-level analysis: analyze_agent, analyze_all_agents, metrics_to_dataframe."""

import numpy as np
import pandas as pd
from collections import defaultdict

from reliability_eval.constants import SAFETY_LAMBDA
from reliability_eval.types import ReliabilityMetrics
from reliability_eval.metrics.abstention import compute_abstention_metrics
from reliability_eval.metrics.consistency import (
    compute_consistency_metrics,
    compute_trajectory_consistency_conditioned,
    compute_sequence_consistency,
)
from reliability_eval.metrics.predictability import (
    compute_ece_metrics,
    compute_predictability_metrics,
)
from reliability_eval.metrics.robustness import (
    compute_accuracy,
    compute_robustness_ratio,
)
from reliability_eval.metrics.safety import compute_safety_metrics


def compute_level_stratified_metrics(runs: list[dict]) -> dict:
    """
    Compute ALL reliability metrics stratified by GAIA difficulty level (1, 2, 3).

    Returns dict with metrics for each reliability category:

    Consistency:
    - consistency_outcome_by_level: {level: outcome_consistency}
    - consistency_trajectory_distribution_by_level: {level: trajectory_distribution_consistency}
    - consistency_trajectory_sequence_by_level: {level: trajectory_sequence_consistency}

    Predictability:
    - predictability_calibration_by_level: {level: calibration (1-ECE)}
    - predictability_roc_auc_by_level: {level: AUC-ROC discrimination}
    - predictability_brier_score_by_level: {level: 1 - Brier score}

    Robustness: (computed separately in compute_robustness_by_level)

    Also includes:
    - accuracy_by_level, confidence_by_level, trajectory_complexity, task_counts
    """
    if not runs:
        return {}

    # Collect all task levels across runs
    all_levels = {}
    for run in runs:
        task_levels = run.get("task_levels", {})
        all_levels.update(task_levels)

    if not all_levels:
        return {}  # No level information available

    # Group task results by level
    level_results = {"1": [], "2": [], "3": []}
    level_confidences = {"1": [], "2": [], "3": []}
    level_actions = {"1": [], "2": [], "3": []}
    level_trajectories = {
        "1": [],
        "2": [],
        "3": [],
    }  # For consistency_trajectory_distribution, consistency_trajectory_sequence
    level_resources = {
        "1": [],
        "2": [],
        "3": [],
    }  # For consistency_resource (time, cost, etc.)

    for run in runs:
        task_levels = run.get("task_levels", {})
        eval_results = run.get("raw_eval_results", {})
        latencies = run.get("latencies", {})

        for task_id, result in eval_results.items():
            if isinstance(result, list):  # Skip prompt sensitivity format
                continue

            level = task_levels.get(task_id)
            if level not in level_results:
                continue

            # Accuracy
            reward = result.get("reward", 0)
            level_results[level].append(reward)

            # Confidence
            conf = result.get("confidence")
            if conf is not None and not np.isnan(conf):
                level_confidences[level].append((conf, reward))

            # Trajectory (action names)
            actions = result.get("action_names", [])
            if actions:
                level_actions[level].append(len(actions))
                level_trajectories[level].append((actions, reward))

            # Resource data (time, cost, num_actions)
            task_latency = latencies.get(task_id, {})
            total_time = task_latency.get("total_time", result.get("total_time", 0))
            total_cost = task_latency.get("total_cost", result.get("total_cost", 0))
            num_actions = len(actions) if actions else result.get("num_actions", 0)
            if total_time > 0 or total_cost > 0 or num_actions > 0:
                level_resources[level].append(
                    {"time": total_time, "cost": total_cost, "num_actions": num_actions}
                )

    def _compute_c_res(times_s, actions_s):
        cvs = []
        if len(times_s) >= 2:
            mt = np.mean(times_s)
            if mt > 0:
                cvs.append(np.std(times_s, ddof=1) / mt)
        if len(actions_s) >= 2:
            ma = np.mean(actions_s)
            if ma > 0:
                cvs.append(np.std(actions_s, ddof=1) / ma)
        return np.clip(np.exp(-np.mean(cvs)), 0.0, 1.0) if cvs else np.nan

    # Compute metrics per level
    metrics = {
        # Basic
        "accuracy_by_level": {},
        "confidence_by_level": {},
        "task_counts": {},
        "trajectory_complexity": {},
        # Consistency
        "consistency_outcome_by_level": {},
        "consistency_trajectory_distribution_by_level": {},
        "consistency_trajectory_sequence_by_level": {},
        "consistency_confidence_by_level": {},  # Confidence consistency
        "consistency_resource_by_level": {},  # Resource consistency
        # Predictability
        "predictability_rate_confidence_correlation_by_level": {},  # Rate-confidence correlation
        "predictability_calibration_by_level": {},
        "predictability_roc_auc_by_level": {},
        "predictability_brier_score_by_level": {},
        # Legacy names for compatibility
        "calibration_by_level": {},
        "overconfidence_by_level": {},
        "brier_by_level": {},
        "confidence_accuracy_alignment": {},
    }

    for level in ["1", "2", "3"]:
        results = level_results[level]
        confidences = level_confidences[level]
        actions = level_actions[level]
        trajectories = level_trajectories[level]

        if not results:
            continue

        # Task counts
        metrics["task_counts"][level] = len(results)

        # Accuracy
        p = np.mean(results)
        metrics["accuracy_by_level"][level] = p
        n = len(results)
        metrics.setdefault("accuracy_by_level_se", {})[level] = (
            np.sqrt(p * (1 - p) / n) if n > 1 else 0.0
        )

        # Trajectory complexity
        if actions:
            metrics["trajectory_complexity"][level] = np.mean(actions)
            if len(actions) > 1:
                metrics.setdefault("trajectory_complexity_se", {})[level] = np.std(
                    actions, ddof=1
                ) / np.sqrt(len(actions))

        # === CONSISTENCY METRICS ===

        # consistency_outcome by level: For proper consistency_outcome we need per-task outcomes across runs
        # Here we approximate using variance of outcomes at this level
        if len(results) >= 2:
            p_hat = np.mean(results)
            var_out = np.var(results, ddof=1) if len(results) > 1 else 0
            max_var = p_hat * (1 - p_hat) + 1e-10
            consistency_outcome_level = 1 - (var_out / max_var)
            metrics["consistency_outcome_by_level"][level] = np.clip(
                consistency_outcome_level, 0.0, 1.0
            )

        # consistency_trajectory_distribution by level: trajectory distribution consistency
        if len(trajectories) >= 2:
            trajs = [t[0] for t in trajectories if t[0]]  # Extract action lists
            if len(trajs) >= 2:
                # Pass all-ones successes: compute consistency over all trajectories at this level
                consistency_trajectory_distribution = (
                    compute_trajectory_consistency_conditioned(trajs, [1] * len(trajs))
                )
                if not np.isnan(consistency_trajectory_distribution):
                    metrics["consistency_trajectory_distribution_by_level"][level] = (
                        consistency_trajectory_distribution
                    )

        # consistency_trajectory_sequence by level: trajectory sequence consistency
        if len(trajectories) >= 2:
            trajs = [t[0] for t in trajectories if t[0]]
            if len(trajs) >= 2:
                # Pass all-ones successes: compute consistency over all trajectories at this level
                consistency_trajectory_sequence = compute_sequence_consistency(
                    trajs, [1] * len(trajs)
                )
                if not np.isnan(consistency_trajectory_sequence):
                    metrics["consistency_trajectory_sequence_by_level"][level] = (
                        consistency_trajectory_sequence
                    )

        # consistency_confidence by level: confidence consistency = exp(-CV) of confidence scores
        if confidences and len(confidences) >= 2:
            confs_only = [c for c, r in confidences]
            if len(confs_only) >= 2:
                mean_conf = np.mean(confs_only)
                std_conf = np.std(confs_only, ddof=1)
                if mean_conf > 0:
                    cv_conf = std_conf / mean_conf
                    consistency_confidence_level = np.exp(-cv_conf)
                    metrics["consistency_confidence_by_level"][level] = np.clip(
                        consistency_confidence_level, 0.0, 1.0
                    )

        # consistency_resource by level: resource consistency = exp(-mean(CV_time, CV_actions))
        resources = level_resources[level]
        if resources and len(resources) >= 2:
            times = [r["time"] for r in resources if r["time"] > 0]
            n_actions = [r["num_actions"] for r in resources if r["num_actions"] > 0]

            consistency_resource_level = _compute_c_res(times, n_actions)
            if not np.isnan(consistency_resource_level):
                metrics["consistency_resource_by_level"][level] = (
                    consistency_resource_level
                )
                # Bootstrap SE
                rng = np.random.default_rng(42)
                n_boot = 200
                boot_vals = []
                n_res = len(resources)
                for _ in range(n_boot):
                    idx = rng.integers(0, n_res, n_res)
                    bt = [times[i] for i in idx if i < len(times)]
                    ba = [n_actions[i] for i in idx if i < len(n_actions)]
                    bv = _compute_c_res(bt, ba)
                    if not np.isnan(bv):
                        boot_vals.append(bv)
                if len(boot_vals) > 1:
                    metrics.setdefault("consistency_resource_by_level_se", {})[
                        level
                    ] = np.std(boot_vals, ddof=1)

        # === PREDICTABILITY METRICS ===

        if confidences:
            confs, rewards = zip(*confidences)
            confs_arr = np.array(confs)
            rewards_arr = np.array(rewards)
            metrics["confidence_by_level"][level] = np.mean(confs_arr)

            # predictability_calibration: Calibration = 1 - ECE
            ece = compute_ece_for_level(list(confs), list(rewards))
            metrics["predictability_calibration_by_level"][level] = 1.0 - ece
            metrics["calibration_by_level"][level] = 1.0 - ece  # Legacy
            # Bootstrap SE for predictability_calibration
            if len(confs_arr) >= 5:
                rng_cal = np.random.default_rng(42)
                boot_cal = []
                n_conf = len(confs_arr)
                for _ in range(200):
                    idx = rng_cal.integers(0, n_conf, n_conf)
                    bece = compute_ece_for_level(
                        confs_arr[idx].tolist(), rewards_arr[idx].tolist()
                    )
                    boot_cal.append(1.0 - bece)
                if len(boot_cal) > 1:
                    metrics.setdefault("predictability_calibration_by_level_se", {})[
                        level
                    ] = np.std(boot_cal, ddof=1)

            # predictability_roc_auc: AUC-ROC discrimination
            # Measures P(conf_success > conf_failure)
            if len(set(rewards_arr)) > 1:  # Need both successes and failures
                try:
                    from sklearn.metrics import roc_auc_score

                    auroc = roc_auc_score(rewards_arr, confs_arr)
                    metrics["predictability_roc_auc_by_level"][level] = auroc
                    # Bootstrap SE for predictability_roc_auc
                    if len(confs_arr) >= 5:
                        rng_auc = np.random.default_rng(42)
                        boot_auc = []
                        for _ in range(200):
                            idx = rng_auc.integers(0, len(confs_arr), len(confs_arr))
                            if len(set(rewards_arr[idx])) > 1:
                                boot_auc.append(
                                    roc_auc_score(rewards_arr[idx], confs_arr[idx])
                                )
                        if len(boot_auc) > 1:
                            metrics.setdefault(
                                "predictability_roc_auc_by_level_se", {}
                            )[level] = np.std(boot_auc, ddof=1)
                except Exception:
                    pass  # Skip if sklearn not available or error

            # predictability_brier_score: 1 - Brier score
            per_task_brier = (confs_arr - rewards_arr) ** 2
            brier = np.mean(per_task_brier)
            metrics["predictability_brier_score_by_level"][level] = 1.0 - brier
            metrics["brier_by_level"][level] = 1.0 - brier  # Legacy
            if len(per_task_brier) > 1:
                metrics.setdefault("predictability_brier_score_by_level_se", {})[
                    level
                ] = np.std(per_task_brier, ddof=1) / np.sqrt(len(per_task_brier))

            # Overconfidence gap (for reference)
            acc_level = metrics["accuracy_by_level"].get(level, np.nan)
            if not np.isnan(acc_level):
                metrics["overconfidence_by_level"][level] = (
                    np.mean(confs_arr) - acc_level
                )

            # predictability_rate_confidence_correlation: Rate-confidence correlation (Spearman)
            # Measures how well confidence predicts success
            if len(confs_arr) >= 5 and len(set(rewards_arr)) > 1:
                try:
                    from scipy.stats import spearmanr

                    corr, _ = spearmanr(confs_arr, rewards_arr)
                    if not np.isnan(corr):
                        # Normalize to [0, 1]: (corr + 1) / 2
                        metrics["predictability_rate_confidence_correlation_by_level"][
                            level
                        ] = (corr + 1) / 2
                except Exception:
                    pass

    # Compute confidence-accuracy alignment
    # (Does confidence decrease as level increases?)
    if (
        len(metrics["confidence_by_level"]) >= 2
        and len(metrics["accuracy_by_level"]) >= 2
    ):
        levels_with_both = sorted(
            set(metrics["confidence_by_level"].keys())
            & set(metrics["accuracy_by_level"].keys())
        )
        if len(levels_with_both) >= 2:
            confs = [metrics["confidence_by_level"][lvl] for lvl in levels_with_both]
            accs = [metrics["accuracy_by_level"][lvl] for lvl in levels_with_both]
            # Correlation between confidence and accuracy across levels
            if len(confs) > 1:
                corr = np.corrcoef(confs, accs)[0, 1]
                metrics["confidence_accuracy_alignment"]["correlation"] = corr

    return metrics


def compute_ece_for_level(
    confidences: list[float], outcomes: list[int], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error for a subset of tasks."""
    if not confidences:
        return 0.0
    result = compute_ece_metrics(
        np.array(confidences), np.array(outcomes), n_bins=n_bins
    )
    ece = result.get("ece")
    return ece if ece is not None and not np.isnan(ece) else 0.0


def compute_consistency_by_level(runs: list[dict]) -> dict:
    """
    Compute outcome consistency stratified by GAIA difficulty level.

    For each level, computes:
    - consistency_outcome: outcome consistency (agreement across repetitions)
    - variance: variance in success rate
    """
    if len(runs) < 2:
        return {}

    # Collect all task levels
    all_levels = {}
    for run in runs:
        task_levels = run.get("task_levels", {})
        all_levels.update(task_levels)

    if not all_levels:
        return {}

    # Group task outcomes by level
    level_task_outcomes = {
        "1": defaultdict(list),
        "2": defaultdict(list),
        "3": defaultdict(list),
    }

    for run in runs:
        task_levels = run.get("task_levels", {})
        eval_results = run.get("raw_eval_results", {})

        for task_id, result in eval_results.items():
            if isinstance(result, list):
                continue

            level = task_levels.get(task_id)
            if level not in level_task_outcomes:
                continue

            reward = result.get("reward", 0)
            level_task_outcomes[level][task_id].append(reward)

    # Compute consistency per level
    consistency_by_level = {}
    variance_by_level = {}
    se_by_level = {}

    for level in ["1", "2", "3"]:
        task_outcomes = level_task_outcomes[level]
        if not task_outcomes:
            continue

        # Tasks with multiple runs
        multi_run_tasks = {t: o for t, o in task_outcomes.items() if len(o) >= 2}
        if not multi_run_tasks:
            continue

        # Compute agreement rate (all same outcome)
        agreements = []
        variances = []
        for task_id, outcomes in multi_run_tasks.items():
            # Agreement = all outcomes same
            if all(o == outcomes[0] for o in outcomes):
                agreements.append(1)
            else:
                agreements.append(0)
            variances.append(np.var(outcomes))

        p_agree = np.mean(agreements)
        n_agree = len(agreements)
        consistency_by_level[level] = p_agree
        variance_by_level[level] = np.mean(variances)
        se_by_level[level] = (
            np.sqrt(p_agree * (1 - p_agree) / n_agree) if n_agree > 1 else 0.0
        )

    return {
        "consistency_by_level": consistency_by_level,
        "variance_by_level": variance_by_level,
        "consistency_by_level_se": se_by_level,
    }


def compute_robustness_by_level(
    baseline_runs: list[dict], perturbed_runs: list[dict]
) -> dict:
    """
    Compute robustness metrics stratified by GAIA difficulty level.

    Compares baseline vs perturbed (fault/structural) performance per level.
    """
    if not baseline_runs or not perturbed_runs:
        return {}

    # Collect task levels from baseline runs
    all_levels = {}
    for run in baseline_runs:
        task_levels = run.get("task_levels", {})
        all_levels.update(task_levels)

    if not all_levels:
        return {}

    # Collect per-task rewards by level
    def rewards_by_level(runs):
        level_results = {"1": [], "2": [], "3": []}
        for run in runs:
            task_levels = run.get("task_levels", {})
            eval_results = run.get("raw_eval_results", {})
            for task_id, result in eval_results.items():
                if isinstance(result, list):
                    continue
                level = task_levels.get(task_id)
                if level in level_results:
                    level_results[level].append(result.get("reward", 0))
        return level_results

    baseline_raw = rewards_by_level(baseline_runs)
    perturbed_raw = rewards_by_level(perturbed_runs)
    baseline_acc = {lvl: np.mean(r) if r else np.nan for lvl, r in baseline_raw.items()}
    perturbed_acc = {
        lvl: np.mean(r) if r else np.nan for lvl, r in perturbed_raw.items()
    }

    # Compute robustness ratio per level with bootstrap SE
    robustness_by_level = {}
    robustness_by_level_se = {}
    for level in ["1", "2", "3"]:
        b_acc = baseline_acc.get(level, np.nan)
        p_acc = perturbed_acc.get(level, np.nan)
        if not np.isnan(b_acc) and not np.isnan(p_acc) and b_acc > 0:
            robustness_by_level[level] = p_acc / b_acc
            # Bootstrap SE
            b_raw = np.array(baseline_raw[level])
            p_raw = np.array(perturbed_raw[level])
            if len(b_raw) >= 2 and len(p_raw) >= 2:
                rng = np.random.default_rng(42)
                boot_ratios = []
                for _ in range(200):
                    b_mean = np.mean(rng.choice(b_raw, len(b_raw), replace=True))
                    p_mean = np.mean(rng.choice(p_raw, len(p_raw), replace=True))
                    if b_mean > 0:
                        boot_ratios.append(p_mean / b_mean)
                if len(boot_ratios) > 1:
                    robustness_by_level_se[level] = np.std(boot_ratios, ddof=1)
        elif not np.isnan(b_acc) and not np.isnan(p_acc) and b_acc == 0 and p_acc == 0:
            robustness_by_level[level] = 1.0

    return {
        "baseline_acc_by_level": baseline_acc,
        "perturbed_acc_by_level": perturbed_acc,
        "robustness_by_level": robustness_by_level,
        "robustness_by_level_se": robustness_by_level_se,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def analyze_agent(
    agent_name: str,
    run_data: dict[str, list[dict]],
    safety_lambda: float = SAFETY_LAMBDA,
) -> ReliabilityMetrics:
    """Analyze all reliability metrics for a single agent."""
    metrics = ReliabilityMetrics(agent_name=agent_name)

    baseline_runs = run_data.get("baseline", [])
    fault_runs = run_data.get("fault", [])
    structural_runs = run_data.get("structural", [])
    prompt_runs = run_data.get("prompt", [])

    # Use all available runs for certain metrics
    all_runs = baseline_runs + fault_runs + structural_runs + prompt_runs
    primary_runs = baseline_runs or all_runs

    if not primary_runs:
        print(f"⚠️  No runs found for {agent_name}")
        return metrics

    metrics.num_runs = len(primary_runs)
    metrics.accuracy = compute_accuracy(primary_runs)

    # Count tasks
    all_tasks = set()
    for run in primary_runs:
        all_tasks.update(run["raw_eval_results"].keys())
    metrics.num_tasks = len(all_tasks)

    # Accuracy SE (binomial)
    n_acc = metrics.num_tasks * metrics.num_runs
    p_acc = metrics.accuracy
    if n_acc > 1 and not np.isnan(p_acc):
        metrics.extra["accuracy_se"] = np.sqrt(p_acc * (1 - p_acc) / n_acc)
    else:
        metrics.extra["accuracy_se"] = np.nan

    # === CONSISTENCY (need multiple baseline runs) ===
    if len(baseline_runs) >= 2:
        consistency = compute_consistency_metrics(baseline_runs)
        metrics.consistency_outcome = consistency["consistency_outcome"]
        metrics.consistency_trajectory_distribution = consistency[
            "consistency_trajectory_distribution"
        ]
        metrics.consistency_trajectory_sequence = consistency[
            "consistency_trajectory_sequence"
        ]
        metrics.consistency_confidence = consistency["consistency_confidence"]
        metrics.consistency_resource = consistency["consistency_resource"]
        metrics.extra["consistency_task_df"] = consistency["task_df"]
        metrics.extra["cv_breakdown"] = consistency.get("cv_breakdown", {})
        metrics.extra["conf_breakdown"] = consistency.get("conf_breakdown", {})
        metrics.extra["consistency_outcome_se"] = consistency.get(
            "consistency_outcome_se", np.nan
        )
        metrics.extra["consistency_trajectory_distribution_se"] = consistency.get(
            "consistency_trajectory_distribution_se", np.nan
        )
        metrics.extra["consistency_trajectory_sequence_se"] = consistency.get(
            "consistency_trajectory_sequence_se", np.nan
        )
        metrics.extra["consistency_confidence_se"] = consistency.get(
            "consistency_confidence_se", np.nan
        )
        metrics.extra["consistency_resource_se"] = consistency.get(
            "consistency_resource_se", np.nan
        )

    # === PREDICTABILITY (need confidence scores) ===
    pred = compute_predictability_metrics(primary_runs)
    metrics.predictability_rate_confidence_correlation = pred[
        "predictability_rate_confidence_correlation"
    ]
    metrics.predictability_calibration = pred["predictability_calibration"]
    metrics.predictability_roc_auc = pred["predictability_roc_auc"]
    metrics.predictability_brier_score = pred["predictability_brier_score"]
    metrics.mean_confidence = pred["mean_confidence"]
    metrics.extra["aurc_data"] = pred["aurc_data"]
    metrics.extra["calibration_bins"] = pred["bin_stats"]
    metrics.extra["predictability_calibration_se"] = pred.get(
        "predictability_calibration_se", np.nan
    )
    metrics.extra["predictability_roc_auc_se"] = pred.get(
        "predictability_roc_auc_se", np.nan
    )
    metrics.extra["predictability_brier_score_se"] = pred.get(
        "predictability_brier_score_se", np.nan
    )
    metrics.extra["auroc_data"] = pred["auroc_data"]
    metrics.extra["brier_data"] = pred["brier_data"]

    # === ABSTENTION CALIBRATION ===
    abstention = compute_abstention_metrics(primary_runs)
    metrics.abstention_rate = (
        abstention["abstention_rate"]
        if abstention["abstention_rate"] is not None
        else np.nan
    )
    metrics.abstention_precision = (
        abstention["abstention_precision"]
        if abstention["abstention_precision"] is not None
        else np.nan
    )
    metrics.abstention_recall = (
        abstention["abstention_recall"]
        if abstention["abstention_recall"] is not None
        else np.nan
    )
    metrics.abstention_selective_accuracy = (
        abstention["selective_accuracy"]
        if abstention["selective_accuracy"] is not None
        else np.nan
    )
    metrics.abstention_calibration = (
        abstention["calibration_score"]
        if abstention["calibration_score"] is not None
        else np.nan
    )
    metrics.extra["abstention_data"] = abstention

    # === ROBUSTNESS ===
    if baseline_runs and fault_runs:
        metrics.robustness_fault_injection, r_fault_se = compute_robustness_ratio(
            baseline_runs, fault_runs
        )
        metrics.extra["baseline_acc"] = compute_accuracy(baseline_runs)
        metrics.extra["fault_acc"] = compute_accuracy(fault_runs)
        metrics.extra["robustness_fault_injection_se"] = r_fault_se

    if baseline_runs and structural_runs:
        metrics.robustness_structural, r_struct_se = compute_robustness_ratio(
            baseline_runs, structural_runs
        )
        metrics.extra["struct_acc"] = compute_accuracy(structural_runs)
        metrics.extra["robustness_structural_se"] = r_struct_se

    if baseline_runs and prompt_runs:
        metrics.robustness_prompt_variation, r_prompt_se = compute_robustness_ratio(
            baseline_runs, prompt_runs
        )
        metrics.extra["prompt_acc"] = compute_accuracy(prompt_runs)
        metrics.extra["robustness_prompt_variation_se"] = r_prompt_se

    # === SAFETY ===
    safety = compute_safety_metrics(
        primary_runs, safety_lambda=safety_lambda
    )
    metrics.safety_harm_severity = safety["safety_harm_severity"]
    metrics.safety_compliance = safety["safety_compliance"]
    metrics.safety_score = safety["safety_score"]
    metrics.extra["safety_per_constraint"] = safety["per_constraint"]
    metrics.extra["safety_violations"] = safety["violations"]
    metrics.extra["safety_mean_severity"] = safety["mean_severity"]
    metrics.extra["safety_max_severity"] = safety["max_severity"]
    metrics.extra["safety_analysis_model"] = safety["analysis_model"]
    metrics.extra["safety_per_task_scores"] = safety["per_task_scores"]
    metrics.extra["safety_lambda"] = safety["safety_lambda"]

    # === LEVEL-STRATIFIED ANALYSIS (GAIA-specific) ===
    # Check if we have level information
    has_levels = any(run.get("task_levels") for run in primary_runs)
    if has_levels:
        # Collect unified task_levels mapping for export
        unified_task_levels = {}
        for run in primary_runs:
            unified_task_levels.update(run.get("task_levels", {}))
        metrics.extra["task_levels"] = unified_task_levels
        # Compute overall level-stratified metrics
        level_metrics = compute_level_stratified_metrics(primary_runs)
        metrics.extra["level_metrics"] = level_metrics

        # Compute consistency by level (needs multiple baseline runs)
        if len(baseline_runs) >= 2:
            consistency_by_level = compute_consistency_by_level(baseline_runs)
            metrics.extra["consistency_by_level"] = consistency_by_level

        # Compute robustness by level
        if baseline_runs and fault_runs:
            fault_robustness_by_level = compute_robustness_by_level(
                baseline_runs, fault_runs
            )
            metrics.extra["fault_robustness_by_level"] = fault_robustness_by_level

        if baseline_runs and structural_runs:
            struct_robustness_by_level = compute_robustness_by_level(
                baseline_runs, structural_runs
            )
            metrics.extra["struct_robustness_by_level"] = struct_robustness_by_level

        if baseline_runs and prompt_runs:
            prompt_robustness_by_level = compute_robustness_by_level(
                baseline_runs, prompt_runs
            )
            metrics.extra["prompt_robustness_by_level"] = prompt_robustness_by_level

    return metrics


def analyze_all_agents(
    results: dict[str, dict],
    safety_lambda: float = SAFETY_LAMBDA,
) -> list[ReliabilityMetrics]:
    """Analyze all agents."""
    all_metrics = []

    for agent_name, run_data in results.items():
        print(f"\n📊 Analyzing {agent_name}...")
        metrics = analyze_agent(
            agent_name, run_data, safety_lambda=safety_lambda
        )
        all_metrics.append(metrics)

        # Print summary
        print(f"   Accuracy: {metrics.accuracy:.3f}")
        if not np.isnan(metrics.consistency_outcome):
            print(
                f"   consistency_outcome: {metrics.consistency_outcome:.3f}, consistency_trajectory_distribution: {metrics.consistency_trajectory_distribution:.3f}, consistency_trajectory_sequence: {metrics.consistency_trajectory_sequence:.3f}"
            )
            print(
                f"   consistency_confidence: {metrics.consistency_confidence:.3f}, consistency_resource: {metrics.consistency_resource:.3f}"
            )
        if not np.isnan(metrics.predictability_rate_confidence_correlation):
            print(
                f"   predictability_rate_confidence_correlation: {metrics.predictability_rate_confidence_correlation:.3f}, predictability_calibration: {metrics.predictability_calibration:.3f}, predictability_roc_auc: {metrics.predictability_roc_auc:.3f}, predictability_brier_score: {metrics.predictability_brier_score:.3f}"
            )
        if not np.isnan(metrics.robustness_fault_injection):
            print(
                f"   robustness_fault_injection: {metrics.robustness_fault_injection:.3f}"
            )
        if not np.isnan(metrics.robustness_structural):
            print(f"   robustness_structural: {metrics.robustness_structural:.3f}")
        if not np.isnan(metrics.robustness_prompt_variation):
            print(
                f"   robustness_prompt_variation: {metrics.robustness_prompt_variation:.3f}"
            )
        if not np.isnan(metrics.safety_harm_severity):
            print(
                f"   safety_harm_severity: {metrics.safety_harm_severity:.3f}, safety_compliance: {metrics.safety_compliance:.3f}, safety_score: {metrics.safety_score:.3f}"
            )
        if not np.isnan(metrics.abstention_rate):
            print(
                f"   abstention_rate: {metrics.abstention_rate:.3f}, abstention_precision: {metrics.abstention_precision:.3f}, abstention_recall: {metrics.abstention_recall:.3f}, abstention_selective_accuracy: {metrics.abstention_selective_accuracy:.3f}, abstention_calibration: {metrics.abstention_calibration:.3f}"
            )

    return all_metrics


def metrics_to_dataframe(all_metrics: list[ReliabilityMetrics]) -> pd.DataFrame:
    """Convert metrics list to DataFrame."""
    rows = []
    for m in all_metrics:
        # Get CV breakdown from extra data
        cv_breakdown = m.extra.get("cv_breakdown", {})
        conf_breakdown = m.extra.get("conf_breakdown", {})

        rows.append(
            {
                "agent": m.agent_name,
                "num_tasks": m.num_tasks,
                "num_runs": m.num_runs,
                "accuracy": m.accuracy,
                # Consistency
                "consistency_outcome": m.consistency_outcome,
                "consistency_trajectory_distribution": m.consistency_trajectory_distribution,
                "consistency_trajectory_sequence": m.consistency_trajectory_sequence,
                "consistency_confidence": m.consistency_confidence,
                "consistency_resource": m.consistency_resource,
                # Resource CV breakdown
                "mean_time_cv": cv_breakdown.get("mean_time_cv", np.nan),
                "mean_cost_cv": cv_breakdown.get("mean_cost_cv", np.nan),
                "mean_api_calls_cv": cv_breakdown.get("mean_api_calls_cv", np.nan),
                "mean_actions_cv": cv_breakdown.get("mean_actions_cv", np.nan),
                "mean_errors_cv": cv_breakdown.get("mean_errors_cv", np.nan),
                "mean_call_latency_cv": cv_breakdown.get(
                    "mean_call_latency_cv", np.nan
                ),
                # Confidence CV breakdown
                "mean_conf_cv": conf_breakdown.get("mean_conf_cv", np.nan),
                # Predictability
                "predictability_rate_confidence_correlation": m.predictability_rate_confidence_correlation,
                "predictability_calibration": m.predictability_calibration,
                "predictability_roc_auc": m.predictability_roc_auc,
                "predictability_brier_score": m.predictability_brier_score,
                "mean_confidence": m.mean_confidence,
                # Robustness
                "robustness_fault_injection": m.robustness_fault_injection,
                "robustness_structural": m.robustness_structural,
                "robustness_prompt_variation": m.robustness_prompt_variation,
                # Safety
                "safety_harm_severity": m.safety_harm_severity,
                "safety_compliance": m.safety_compliance,
                "safety_score": m.safety_score,
                # Abstention calibration
                "abstention_rate": m.abstention_rate,
                "abstention_precision": m.abstention_precision,
                "abstention_recall": m.abstention_recall,
                "abstention_selective_accuracy": m.abstention_selective_accuracy,
                "abstention_calibration": m.abstention_calibration,
                # Standard errors (for confidence bars)
                "accuracy_se": m.extra.get("accuracy_se", np.nan),
                "consistency_outcome_se": m.extra.get("consistency_outcome_se", np.nan),
                "consistency_trajectory_distribution_se": m.extra.get(
                    "consistency_trajectory_distribution_se", np.nan
                ),
                "consistency_trajectory_sequence_se": m.extra.get(
                    "consistency_trajectory_sequence_se", np.nan
                ),
                "consistency_confidence_se": m.extra.get(
                    "consistency_confidence_se", np.nan
                ),
                "consistency_resource_se": m.extra.get(
                    "consistency_resource_se", np.nan
                ),
                "predictability_calibration_se": m.extra.get(
                    "predictability_calibration_se", np.nan
                ),
                "predictability_roc_auc_se": m.extra.get(
                    "predictability_roc_auc_se", np.nan
                ),
                "predictability_brier_score_se": m.extra.get(
                    "predictability_brier_score_se", np.nan
                ),
                "robustness_fault_injection_se": m.extra.get(
                    "robustness_fault_injection_se", np.nan
                ),
                "robustness_structural_se": m.extra.get(
                    "robustness_structural_se", np.nan
                ),
                "robustness_prompt_variation_se": m.extra.get(
                    "robustness_prompt_variation_se", np.nan
                ),
            }
        )
    return pd.DataFrame(rows)

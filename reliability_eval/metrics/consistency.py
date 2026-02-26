"""Consistency metrics: consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource."""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from scipy.spatial.distance import jensenshannon

from reliability_eval.constants import EPSILON, W_OUTCOME, W_TRAJECTORY, W_RESOURCE


def compute_outcome_consistency(
    task_successes: list[int], epsilon: float = 1e-8
) -> float:
    """
    Compute outcome consistency for a single task.

    Formula: consistency_outcome(t) = 1 - sigma_hat^2 / (p_hat * (1 - p_hat) + epsilon)

    where p_hat is the observed success rate across K runs and
    sigma_hat^2 is the sample variance (ddof=1).

    Normalizes the sample variance by the maximum Bernoulli variance for
    that task's success rate, so consistency_outcome measures how much of the possible
    variance is realized:
      - All runs agree (p=0 or p=1): consistency_outcome = 1  (perfect consistency)
      - Maximally variable:          consistency_outcome = 0  (worst consistency)
    """
    K = len(task_successes)
    if K < 2:
        return np.nan

    p_hat = np.mean(task_successes)
    sigma_hat_sq = np.var(task_successes, ddof=1)

    consistency_outcome = 1 - sigma_hat_sq / (p_hat * (1 - p_hat) + epsilon)

    return np.clip(consistency_outcome, 0.0, 1.0)


def compute_trajectory_consistency_conditioned(
    trajectories: list[list[str]], successes: list[int]
) -> float:
    """
    Compute trajectory consistency CONDITIONED on outcome (paper Definition 3.2).

    Returns consistency_trajectory_sequence_success (C_traj^+):
    consistency among successful runs, as a single float.
    """
    # Separate trajectories by outcome; only success trajectories are used
    success_trajectories = [t for t, s in zip(trajectories, successes) if s == 1 and t]

    def compute_jsd_consistency(trajs: list[list[str]]) -> float:
        if len(trajs) < 2:
            return np.nan

        # Build action distributions
        distributions = []
        all_actions = set()

        for traj in trajs:
            if not traj:
                continue
            action_counts = Counter(traj)
            total = len(traj)
            dist = {a: c / total for a, c in action_counts.items()}
            distributions.append(dist)
            all_actions.update(dist.keys())

        if len(distributions) < 2:
            return np.nan

        all_actions = sorted(list(all_actions))

        # Convert to vectors
        vectors = []
        for dist in distributions:
            vec = np.array([dist.get(a, 0.0) for a in all_actions])
            vec = vec / (vec.sum() + EPSILON)
            vectors.append(vec)

        # Compute mean pairwise JS divergence
        js_divs = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                js_divs.append(jensenshannon(vectors[i], vectors[j]))

        if not js_divs:
            return np.nan

        # C_traj = 1 - mean(JSD)
        return 1 - np.mean(js_divs)

    return compute_jsd_consistency(success_trajectories)


def compute_sequence_consistency(
    trajectories: list[list[str]], successes: list[int]
) -> float:
    """
    Compute trajectory SEQUENCE consistency using normalized edit distance.

    Unlike distribution-based consistency (consistency_trajectory_distribution), this measures whether
    actions occur in the same ORDER across runs.

    Returns consistency_trajectory_sequence_success as a single float.
    """

    def levenshtein_distance(s1: list[str], s2: list[str]) -> int:
        """Compute Levenshtein (edit) distance between two sequences."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]

    def normalized_similarity(s1: list[str], s2: list[str]) -> float:
        """Compute normalized similarity (1 - normalized edit distance)."""
        if not s1 and not s2:
            return 1.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        dist = levenshtein_distance(s1, s2)
        return 1.0 - (dist / max_len)

    def compute_seq_consistency(trajs: list[list[str]]) -> float:
        """Compute mean pairwise sequence similarity."""
        valid_trajs = [t for t in trajs if t]
        if len(valid_trajs) < 2:
            return np.nan

        similarities = []
        for i in range(len(valid_trajs)):
            for j in range(i + 1, len(valid_trajs)):
                sim = normalized_similarity(valid_trajs[i], valid_trajs[j])
                similarities.append(sim)

        return np.mean(similarities) if similarities else np.nan

    # Separate by outcome, return only the success score
    success_trajectories = [t for t, s in zip(trajectories, successes) if s == 1]

    return compute_seq_consistency(success_trajectories)


def compute_confidence_consistency(
    confidences: list[float], successes: list[int]
) -> tuple[float, dict[str, float]]:
    """
    Compute confidence consistency across runs.

    consistency_confidence = exp(-CV_conf)

    where CV_conf is the coefficient of variation of confidence scores.
    Also computes consistency separately for successful and failed runs.

    Returns:
        (consistency_confidence, breakdown) where breakdown contains per-outcome CVs
    """
    breakdown = {}

    valid_conf = [c for c in confidences if c is not None and not np.isnan(c)]

    if len(valid_conf) < 2:
        return np.nan, breakdown

    # Overall confidence consistency
    mean_conf = np.mean(valid_conf)
    std_conf = np.std(valid_conf, ddof=1)

    if mean_conf > 0:
        cv_overall = std_conf / mean_conf
        breakdown["cv_overall"] = cv_overall
        consistency_confidence = np.exp(-cv_overall)
    else:
        consistency_confidence = np.nan

    # Consistency among successful runs
    success_conf = [
        c
        for c, s in zip(confidences, successes)
        if s == 1 and c is not None and not np.isnan(c)
    ]
    if len(success_conf) >= 2:
        mean_s = np.mean(success_conf)
        std_s = np.std(success_conf, ddof=1)
        if mean_s > 0:
            breakdown["cv_success"] = std_s / mean_s

    # Consistency among failed runs
    failure_conf = [
        c
        for c, s in zip(confidences, successes)
        if s == 0 and c is not None and not np.isnan(c)
    ]
    if len(failure_conf) >= 2:
        mean_f = np.mean(failure_conf)
        std_f = np.std(failure_conf, ddof=1)
        if mean_f > 0:
            breakdown["cv_failure"] = std_f / mean_f

    return consistency_confidence, breakdown


def compute_resource_consistency(
    costs: list[float],
    times: list[float],
    successes: list[int],
    api_calls: list[int] | None = None,
    num_actions: list[int] | None = None,
    num_errors: list[int] | None = None,
    call_latencies: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Compute resource consistency across all runs (paper Definition 3.3).

    consistency_resource = exp(-CV)

    where CV is the coefficient of variation across all runs.

    Returns:
        (consistency_resource, cv_breakdown) where cv_breakdown contains individual CVs for each metric
    """
    # Use all runs (not conditioned on success)
    valid_costs = [c for c in costs if c > 0]
    valid_times = [t for t in times if t > 0]

    cvs = []
    cv_breakdown = {}

    def compute_cv(values: list[float], name: str) -> float | None:
        """Compute CV for a list of values if sufficient data."""
        if len(values) >= 2:
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            if mean_val > 0:
                cv = std_val / mean_val
                cv_breakdown[name] = cv
                return cv
        return None

    # Compute CV for costs (if available)
    cv = compute_cv(valid_costs, "cost_cv")
    if cv is not None:
        cvs.append(cv)

    # Compute CV for time (if available)
    cv = compute_cv(valid_times, "time_cv")
    if cv is not None:
        cvs.append(cv)

    # Compute CV for API calls (if available)
    if api_calls:
        valid_api_calls = [a for a in api_calls if a > 0]
        cv = compute_cv([float(x) for x in valid_api_calls], "api_calls_cv")
        if cv is not None:
            cvs.append(cv)

    # Compute CV for num_actions (if available)
    if num_actions:
        valid_actions = [a for a in num_actions if a > 0]
        cv = compute_cv([float(x) for x in valid_actions], "actions_cv")
        if cv is not None:
            cvs.append(cv)

    # Compute CV for num_errors (if available) - include zeros since 0 errors is valid
    if num_errors:
        if len(num_errors) >= 2:
            mean_val = np.mean(num_errors)
            std_val = np.std(num_errors, ddof=1)
            # For errors, CV is meaningful even if mean is close to 0
            if mean_val > 0:
                cv_breakdown["errors_cv"] = std_val / mean_val
                cvs.append(std_val / mean_val)
            elif std_val > 0:
                # If mean is 0 but std > 0, there's variability
                cv_breakdown["errors_cv"] = float("inf")

    # Compute CV for call latencies (if available)
    if call_latencies:
        valid_latencies = [lat for lat in call_latencies if lat > 0]
        cv = compute_cv(valid_latencies, "call_latency_cv")
        if cv is not None:
            cvs.append(cv)

    if not cvs:
        return np.nan, cv_breakdown

    # Average CV across resource types
    cv_avg = np.mean(cvs)
    cv_breakdown["avg_cv"] = cv_avg

    # Exponential transform: consistency_resource = exp(-CV)
    consistency_resource = np.exp(-cv_avg)

    return consistency_resource, cv_breakdown


def compute_weighted_r_con(c_out, c_traj_d, c_traj_s, c_res):
    """Compute weighted reliability_consistency from consistency sub-metrics.

    Uses importance weights (W_OUTCOME, W_TRAJECTORY, W_RESOURCE) that
    bias towards outcome and resource consistency over trajectory consistency.
    Handles NaN values by renormalizing over available metrics.

    Works with both scalar values and pandas Series.
    """
    # Average the two trajectory sub-metrics (handling NaNs)
    c_traj = np.where(
        np.isnan(c_traj_d) & np.isnan(c_traj_s),
        np.nan,
        np.nanmean(
            np.stack(
                [
                    np.atleast_1d(np.asarray(c_traj_d, dtype=float)),
                    np.atleast_1d(np.asarray(c_traj_s, dtype=float)),
                ]
            ),
            axis=0,
        ),
    )

    values = np.stack(
        [
            np.atleast_1d(np.asarray(c_out, dtype=float)),
            np.atleast_1d(c_traj),
            np.atleast_1d(np.asarray(c_res, dtype=float)),
        ]
    )
    weights = np.array([W_OUTCOME, W_TRAJECTORY, W_RESOURCE])

    # Mask NaNs and renormalize weights
    valid = ~np.isnan(values)
    # For each column (agent), compute weighted mean over valid metrics
    result = np.full(values.shape[1], np.nan)
    for j in range(values.shape[1]):
        mask = valid[:, j]
        if mask.any():
            w = weights[mask]
            result[j] = np.dot(w / w.sum(), values[mask, j])

    # Return scalar if input was scalar
    if result.shape[0] == 1:
        return float(result[0])
    return result


def compute_consistency_metrics(baseline_runs: list[dict]) -> dict:
    """Compute all consistency metrics from baseline runs."""
    if len(baseline_runs) < 2:
        return {
            "consistency_outcome": np.nan,
            "consistency_trajectory_distribution": np.nan,
            "consistency_trajectory_sequence": np.nan,
            "consistency_confidence": np.nan,
            "consistency_resource": np.nan,
            "cv_breakdown": {},
            "conf_breakdown": {},
            "task_df": pd.DataFrame(),
        }

    # Collect per-task data across runs
    task_data = defaultdict(
        lambda: {
            "success": [],
            "cost": [],
            "time": [],
            "trajectories": [],
            "api_calls": [],
            "num_actions": [],
            "num_errors": [],
            "call_latency": [],
            "confidence": [],  # NEW: for confidence consistency
        }
    )

    for run in baseline_runs:
        raw_eval = run["raw_eval_results"]
        latencies = run.get("latencies", {})
        costs = run.get("costs", {})
        raw_logging = run.get("raw_logging_results", [])

        # Pre-process raw_logging_results to extract per-task metrics
        # (using minimal format: usage_count and latency_ms directly)
        task_api_calls = defaultdict(int)
        task_call_latencies = defaultdict(list)
        task_prompt_tokens = defaultdict(int)
        task_completion_tokens = defaultdict(int)

        for log_entry in raw_logging:
            task_id = log_entry.get("weave_task_id")
            if task_id is None:
                continue
            task_id = str(task_id)

            # Count API calls (already extracted as count in minimal format)
            task_api_calls[task_id] += log_entry.get("usage_count", 0)

            # Extract per-call latency
            latency_ms = log_entry.get("latency_ms")
            if latency_ms is not None:
                task_call_latencies[task_id].append(latency_ms)

            # Accumulate tokens for cost estimation
            task_prompt_tokens[task_id] += log_entry.get("prompt_tokens", 0)
            task_completion_tokens[task_id] += log_entry.get("completion_tokens", 0)

        for task_id, task_eval in raw_eval.items():
            if not isinstance(task_eval, dict):
                continue

            task_id_str = str(task_id)
            success = int(task_eval.get("reward", 0.0))
            task_data[task_id_str]["success"].append(success)

            # Get time
            time_val = latencies.get(task_id_str, {}).get("total_time", 0.0)
            task_data[task_id_str]["time"].append(time_val)

            # Get cost (try multiple locations)
            cost_val = costs.get(task_id_str, 0.0)
            if cost_val == 0:
                cost_val = task_eval.get("cost", 0.0)
            if cost_val == 0:
                # Estimate from token usage (rough avg: $5/M input, $15/M output)
                pt = task_prompt_tokens.get(task_id_str, 0)
                ct = task_completion_tokens.get(task_id_str, 0)
                if pt > 0 or ct > 0:
                    cost_val = pt * 5.0 / 1_000_000 + ct * 15.0 / 1_000_000
            task_data[task_id_str]["cost"].append(cost_val)

            # Extract trajectory (already extracted as action_names in minimal format)
            trajectory = task_eval.get("action_names", [])
            task_data[task_id_str]["trajectories"].append(trajectory)

            # Extract num_actions, num_errors, and confidence from confidence_details
            confidence_details = task_eval.get("confidence_details", {})
            if isinstance(confidence_details, dict):
                task_data[task_id_str]["num_actions"].append(
                    confidence_details.get("num_actions", 0)
                )
                task_data[task_id_str]["num_errors"].append(
                    confidence_details.get("num_errors", 0)
                )
            else:
                task_data[task_id_str]["num_actions"].append(0)
                task_data[task_id_str]["num_errors"].append(0)

            # Extract confidence score (can be at top level or in confidence_details)
            conf_score = task_eval.get("confidence")
            if conf_score is None and isinstance(confidence_details, dict):
                conf_score = confidence_details.get("parsed_score")
            task_data[task_id_str]["confidence"].append(conf_score)

            # Add API calls count
            task_data[task_id_str]["api_calls"].append(
                task_api_calls.get(task_id_str, 0)
            )

            # Add mean call latency for this task
            call_lats = task_call_latencies.get(task_id_str, [])
            mean_lat = np.mean(call_lats) if call_lats else 0.0
            task_data[task_id_str]["call_latency"].append(mean_lat)

    # Compute per-task metrics
    task_rows = []
    all_consistency_outcome = []
    all_consistency_trajectory_distribution = []  # Distribution-based trajectory consistency
    all_consistency_trajectory_sequence = []  # Sequence-based trajectory consistency
    all_consistency_confidence = []  # Confidence consistency
    all_consistency_resource = []

    for task_id, data in task_data.items():
        if len(data["success"]) < 2:
            continue

        # consistency_outcome: Normalized outcome consistency
        consistency_outcome = compute_outcome_consistency(data["success"])
        all_consistency_outcome.append(consistency_outcome)

        # consistency_trajectory_distribution: Distribution-based trajectory consistency (what actions)
        consistency_trajectory_distribution_success = (
            compute_trajectory_consistency_conditioned(
                data["trajectories"], data["success"]
            )
        )
        if not np.isnan(consistency_trajectory_distribution_success):
            all_consistency_trajectory_distribution.append(
                consistency_trajectory_distribution_success
            )

        # consistency_trajectory_sequence: Sequence-based trajectory consistency (action order)
        consistency_trajectory_sequence_success = compute_sequence_consistency(
            data["trajectories"], data["success"]
        )
        if not np.isnan(consistency_trajectory_sequence_success):
            all_consistency_trajectory_sequence.append(
                consistency_trajectory_sequence_success
            )

        # consistency_confidence: Confidence consistency
        consistency_confidence, conf_breakdown = compute_confidence_consistency(
            data["confidence"], data["success"]
        )
        if not np.isnan(consistency_confidence):
            all_consistency_confidence.append(consistency_confidence)

        # consistency_resource: Resource consistency (across all runs)
        consistency_resource, cv_breakdown = compute_resource_consistency(
            data["cost"],
            data["time"],
            data["success"],
            api_calls=data["api_calls"],
            num_actions=data["num_actions"],
            num_errors=data["num_errors"],
            call_latencies=data["call_latency"],
        )
        if not np.isnan(consistency_resource):
            all_consistency_resource.append(consistency_resource)

        task_rows.append(
            {
                "task_id": task_id,
                "success_rate": float(np.mean(data["success"])),
                "n_runs": len(data["success"]),
                "consistency_outcome": consistency_outcome,
                "consistency_trajectory_distribution": consistency_trajectory_distribution_success,
                "consistency_trajectory_sequence": consistency_trajectory_sequence_success,
                "consistency_confidence": consistency_confidence,
                "consistency_resource": consistency_resource,
                "time_cv": cv_breakdown.get("time_cv", np.nan),
                "cost_cv": cv_breakdown.get("cost_cv", np.nan),
                "api_calls_cv": cv_breakdown.get("api_calls_cv", np.nan),
                "actions_cv": cv_breakdown.get("actions_cv", np.nan),
                "errors_cv": cv_breakdown.get("errors_cv", np.nan),
                "call_latency_cv": cv_breakdown.get("call_latency_cv", np.nan),
                "conf_cv": conf_breakdown.get("cv_overall", np.nan),
                # Per-task resource means (for distribution visualisation)
                "mean_cost": float(np.mean(data["cost"])) if data["cost"] else np.nan,
                "mean_time": float(np.mean(data["time"])) if data["time"] else np.nan,
                "mean_api_calls": float(np.mean(data["api_calls"]))
                if data["api_calls"]
                else np.nan,
                "mean_actions": float(np.mean(data["num_actions"]))
                if data["num_actions"]
                else np.nan,
                "mean_confidence": float(
                    np.mean([c for c in data["confidence"] if c is not None])
                )
                if any(c is not None for c in data["confidence"])
                else np.nan,
            }
        )

    task_df = pd.DataFrame(task_rows)

    # Aggregate CV breakdown across all tasks
    cv_cols = [
        "time_cv",
        "cost_cv",
        "api_calls_cv",
        "actions_cv",
        "errors_cv",
        "call_latency_cv",
    ]
    aggregated_cv = {}
    for col in cv_cols:
        if col in task_df.columns:
            aggregated_cv[f"mean_{col}"] = task_df[col].mean(skipna=True)

    # Aggregate confidence breakdown
    aggregated_conf = {}
    if "conf_cv" in task_df.columns:
        aggregated_conf["mean_conf_cv"] = task_df["conf_cv"].mean(skipna=True)

    def _se(vals):
        """Standard error of the mean."""
        if len(vals) < 2:
            return np.nan
        return np.std(vals, ddof=1) / np.sqrt(len(vals))

    return {
        "consistency_outcome": np.mean(all_consistency_outcome)
        if all_consistency_outcome
        else np.nan,
        "consistency_outcome_se": _se(all_consistency_outcome),
        "consistency_trajectory_distribution": np.mean(
            all_consistency_trajectory_distribution
        )
        if all_consistency_trajectory_distribution
        else np.nan,
        "consistency_trajectory_distribution_se": _se(
            all_consistency_trajectory_distribution
        ),
        "consistency_trajectory_sequence": np.mean(all_consistency_trajectory_sequence)
        if all_consistency_trajectory_sequence
        else np.nan,
        "consistency_trajectory_sequence_se": _se(all_consistency_trajectory_sequence),
        "consistency_confidence": np.mean(all_consistency_confidence)
        if all_consistency_confidence
        else np.nan,
        "consistency_confidence_se": _se(all_consistency_confidence),
        "consistency_resource": np.mean(all_consistency_resource)
        if all_consistency_resource
        else np.nan,
        "consistency_resource_se": _se(all_consistency_resource),
        "cv_breakdown": aggregated_cv,
        "conf_breakdown": aggregated_conf,
        "task_df": task_df,
    }

"""Predictability metrics: predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score."""

import numpy as np

from reliability_eval.constants import EPSILON


def compute_aurc_metrics(confidences: np.ndarray, successes: np.ndarray) -> dict:
    """
    Compute predictability_rate_confidence_correlation: Risk-Coverage Score (paper Definition 3.5).

    predictability_rate_confidence_correlation = 1 - E-AuRC / E-AuRC_max

    where E-AuRC is excess AuRC over optimal selector.
    """
    valid = ~(np.isnan(confidences) | np.isnan(successes))
    confidences = confidences[valid]
    successes = successes[valid]

    N = len(confidences)
    if N == 0:
        return {
            "predictability_rate_confidence_correlation": np.nan,
            "aurc": np.nan,
            "coverages": [],
            "risks": [],
            "optimal_risks": [],
        }

    # Sort by decreasing confidence
    sorted_idx = np.argsort(-confidences)
    sorted_successes = successes[sorted_idx]

    # Compute risk at each coverage level
    coverages = np.linspace(0, 1, 100)
    risks = []
    optimal_risks = []

    # Optimal ordering (successes first)
    optimal_sorted = np.sort(successes)[::-1]

    for c in coverages:
        n_covered = max(1, int(c * N))
        risks.append(1 - np.mean(sorted_successes[:n_covered]))
        optimal_risks.append(1 - np.mean(optimal_sorted[:n_covered]))

    aurc = np.trapezoid(risks, coverages)
    aurc_optimal = np.trapezoid(optimal_risks, coverages)

    # Random baseline (constant risk = overall error rate)
    overall_error = 1 - np.mean(successes)
    aurc_random = overall_error

    # Excess AuRC
    excess_aurc = aurc - aurc_optimal
    excess_max = aurc_random - aurc_optimal

    # predictability_rate_confidence_correlation score
    predictability_rate_confidence_correlation = (
        1 - (excess_aurc / (excess_max + EPSILON)) if excess_max > EPSILON else 1.0
    )
    predictability_rate_confidence_correlation = np.clip(
        predictability_rate_confidence_correlation, 0.0, 1.0
    )

    return {
        "predictability_rate_confidence_correlation": predictability_rate_confidence_correlation,
        "aurc": aurc,
        "coverages": coverages,
        "risks": risks,
        "optimal_risks": optimal_risks,
    }


def compute_ece_metrics(
    confidences: np.ndarray, successes: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute predictability_calibration: Calibration Score (paper Definition 3.6).

    predictability_calibration = 1 - ECE

    where ECE is Expected Calibration Error.
    """
    valid = ~(np.isnan(confidences) | np.isnan(successes))
    confidences = confidences[valid]
    successes = successes[valid]

    if len(confidences) == 0:
        return {"predictability_calibration": np.nan, "ece": np.nan, "bin_stats": []}

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            avg_conf = np.mean(confidences[in_bin])
            avg_acc = np.mean(successes[in_bin])
            weight = n_in_bin / len(confidences)
            ece += weight * abs(avg_acc - avg_conf)

            bin_stats.append(
                {
                    "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                    "count": n_in_bin,
                    "avg_confidence": avg_conf,
                    "avg_accuracy": avg_acc,
                }
            )

    predictability_calibration = 1 - ece

    return {
        "predictability_calibration": predictability_calibration,
        "ece": ece,
        "bin_stats": bin_stats,
    }


def compute_auroc_metrics(confidences: np.ndarray, successes: np.ndarray) -> dict:
    """
    Compute predictability_roc_auc: Discrimination Score (AUC-ROC).

    predictability_roc_auc = P(conf_success > conf_failure)

    This is the probability that a randomly chosen successful task
    has higher confidence than a randomly chosen failed task.

    Interpretation:
    - 0.5: Random (confidence doesn't discriminate)
    - 1.0: Perfect discrimination (all successes have higher confidence than failures)
    """
    valid = ~(np.isnan(confidences) | np.isnan(successes))
    confidences = confidences[valid]
    successes = successes[valid]

    if len(confidences) == 0:
        return {"predictability_roc_auc": np.nan, "n_positive": 0, "n_negative": 0}

    n_positive = np.sum(successes == 1)
    n_negative = np.sum(successes == 0)

    # Need at least one of each class
    if n_positive == 0 or n_negative == 0:
        return {
            "predictability_roc_auc": np.nan,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    # Compute AUC-ROC using the Mann-Whitney U statistic formulation
    # AUC = P(conf_pos > conf_neg) + 0.5 * P(conf_pos == conf_neg)
    # This is equivalent to sklearn's roc_auc_score but avoids the dependency

    pos_confidences = confidences[successes == 1]
    neg_confidences = confidences[successes == 0]

    # Count concordant, discordant, and tied pairs
    concordant = 0  # pos > neg
    discordant = 0  # pos < neg
    tied = 0  # pos == neg

    for pos_conf in pos_confidences:
        concordant += np.sum(neg_confidences < pos_conf)
        discordant += np.sum(neg_confidences > pos_conf)
        tied += np.sum(neg_confidences == pos_conf)

    total_pairs = n_positive * n_negative
    predictability_roc_auc = (concordant + 0.5 * tied) / total_pairs

    return {
        "predictability_roc_auc": predictability_roc_auc,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
    }


def compute_brier_metrics(confidences: np.ndarray, successes: np.ndarray) -> dict:
    """
    Compute predictability_brier_score: Overall Predictability Score (1 - Brier Score).

    Brier Score = (1/N) * sum((confidence - success)^2)
    predictability_brier_score = 1 - Brier Score

    The Brier Score is a proper scoring rule that combines calibration
    and discrimination into a single metric. Lower Brier = better predictions.

    We return predictability_brier_score = 1 - Brier so that higher is better (consistent with other metrics).

    Interpretation:
    - predictability_brier_score = 1.0: Perfect predictions (confidence always matches outcome)
    - predictability_brier_score = 0.75: Equivalent to always predicting 0.5 for 50% base rate
    - predictability_brier_score = 0.0: Worst possible (confident and always wrong)
    """
    valid = ~(np.isnan(confidences) | np.isnan(successes))
    confidences = confidences[valid]
    successes = successes[valid]

    if len(confidences) == 0:
        return {"predictability_brier_score": np.nan, "brier_score": np.nan}

    # Brier Score: mean squared error between confidence and outcome
    brier_score = np.mean((confidences - successes) ** 2)

    # Transform to higher-is-better
    predictability_brier_score = 1 - brier_score

    # Also compute reference Brier scores for context
    base_rate = np.mean(successes)
    brier_baseline = base_rate * (1 - base_rate)  # Brier if always predicting base rate

    return {
        "predictability_brier_score": predictability_brier_score,
        "brier_score": brier_score,
        "brier_baseline": brier_baseline,
        "base_rate": base_rate,
    }


def compute_predictability_metrics(runs: list[dict]) -> dict:
    """Compute predictability metrics (predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score) from runs with confidence scores."""
    all_confidences = []
    all_successes = []

    for run in runs:
        raw_eval = run["raw_eval_results"]
        for task_eval in raw_eval.values():
            if isinstance(task_eval, dict):
                conf = task_eval.get("confidence")
                if conf is not None:
                    all_confidences.append(float(conf))
                    all_successes.append(int(task_eval.get("reward", 0.0)))

    if not all_confidences:
        return {
            "predictability_rate_confidence_correlation": np.nan,
            "predictability_calibration": np.nan,
            "predictability_roc_auc": np.nan,
            "predictability_brier_score": np.nan,
            "mean_confidence": np.nan,
            "aurc_data": {},
            "bin_stats": [],
            "auroc_data": {},
            "brier_data": {},
        }

    confidences = np.array(all_confidences)
    successes = np.array(all_successes)

    aurc_result = compute_aurc_metrics(confidences, successes)
    ece_result = compute_ece_metrics(confidences, successes)
    auroc_result = compute_auroc_metrics(confidences, successes)
    brier_result = compute_brier_metrics(confidences, successes)

    # Bootstrap SEs for predictability metrics
    n_boot = 200
    rng = np.random.default_rng(42)
    n = len(confidences)
    (
        boot_predictability_calibration,
        boot_predictability_roc_auc,
        boot_predictability_brier_score,
    ) = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        c_b, s_b = confidences[idx], successes[idx]
        boot_predictability_calibration.append(
            compute_ece_metrics(c_b, s_b)["predictability_calibration"]
        )
        boot_predictability_roc_auc.append(
            compute_auroc_metrics(c_b, s_b)["predictability_roc_auc"]
        )
        boot_predictability_brier_score.append(
            compute_brier_metrics(c_b, s_b)["predictability_brier_score"]
        )

    def _boot_se(vals):
        valid = [v for v in vals if not np.isnan(v)]
        return np.std(valid) if len(valid) >= 2 else np.nan

    return {
        "predictability_rate_confidence_correlation": aurc_result[
            "predictability_rate_confidence_correlation"
        ],
        "predictability_calibration": ece_result["predictability_calibration"],
        "predictability_roc_auc": auroc_result["predictability_roc_auc"],
        "predictability_brier_score": brier_result["predictability_brier_score"],
        "predictability_calibration_se": _boot_se(boot_predictability_calibration),
        "predictability_roc_auc_se": _boot_se(boot_predictability_roc_auc),
        "predictability_brier_score_se": _boot_se(boot_predictability_brier_score),
        "mean_confidence": np.mean(confidences),
        "aurc_data": aurc_result,
        "bin_stats": ece_result["bin_stats"],
        "auroc_data": auroc_result,
        "brier_data": brier_result,
        "correct_confidences": confidences[successes == 1].tolist(),
        "incorrect_confidences": confidences[successes == 0].tolist(),
    }

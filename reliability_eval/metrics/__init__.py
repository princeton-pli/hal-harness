"""reliability_eval.metrics — metric computation subpackage."""

from reliability_eval.metrics.abstention import compute_abstention_metrics
from reliability_eval.metrics.agent import (
    analyze_agent,
    analyze_all_agents,
    metrics_to_dataframe,
)
from reliability_eval.metrics.consistency import (
    compute_confidence_consistency,
    compute_consistency_metrics,
    compute_outcome_consistency,
    compute_resource_consistency,
    compute_sequence_consistency,
    compute_trajectory_consistency_conditioned,
    compute_weighted_r_con,
)
from reliability_eval.metrics.predictability import (
    compute_auroc_metrics,
    compute_aurc_metrics,
    compute_brier_metrics,
    compute_ece_metrics,
    compute_predictability_metrics,
)
from reliability_eval.metrics.robustness import (
    compute_accuracy,
    compute_robustness_ratio,
)
from reliability_eval.metrics.safety import compute_safety_metrics

__all__ = [
    "analyze_agent",
    "analyze_all_agents",
    "compute_abstention_metrics",
    "compute_accuracy",
    "compute_auroc_metrics",
    "compute_aurc_metrics",
    "compute_brier_metrics",
    "compute_confidence_consistency",
    "compute_consistency_metrics",
    "compute_ece_metrics",
    "compute_outcome_consistency",
    "compute_predictability_metrics",
    "compute_resource_consistency",
    "compute_robustness_ratio",
    "compute_safety_metrics",
    "compute_sequence_consistency",
    "compute_trajectory_consistency_conditioned",
    "compute_weighted_r_con",
    "metrics_to_dataframe",
]

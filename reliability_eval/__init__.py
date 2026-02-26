"""reliability_eval package — public API re-exports."""

from reliability_eval.metrics.consistency import (
    compute_outcome_consistency,
    compute_resource_consistency,
    compute_sequence_consistency,
    compute_trajectory_consistency_conditioned,
)
from reliability_eval.metrics.predictability import (
    compute_auroc_metrics,
    compute_brier_metrics,
    compute_ece_metrics,
)
from reliability_eval.phases.abstention import detect_abstention

__all__ = [
    "compute_auroc_metrics",
    "compute_brier_metrics",
    "compute_ece_metrics",
    "compute_outcome_consistency",
    "compute_resource_consistency",
    "compute_sequence_consistency",
    "compute_trajectory_consistency_conditioned",
    "detect_abstention",
]

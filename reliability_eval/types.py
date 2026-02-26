"""Shared dataclasses for reliability_eval."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReliabilityMetrics:
    """Container for all reliability metrics for an agent."""

    agent_name: str
    num_tasks: int = 0
    num_runs: int = 0

    # Capability
    accuracy: float = np.nan

    # Consistency (consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource)
    consistency_outcome: float = np.nan
    consistency_trajectory_distribution: float = (
        np.nan
    )  # Trajectory distribution consistency (what actions)
    consistency_trajectory_sequence: float = (
        np.nan
    )  # Trajectory sequence consistency (action order)
    consistency_confidence: float = np.nan  # Confidence consistency
    consistency_resource: float = np.nan  # Resource consistency

    # Predictability (predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score)
    predictability_rate_confidence_correlation: float = np.nan
    predictability_calibration: float = np.nan
    predictability_roc_auc: float = np.nan  # Discrimination (AUC-ROC)
    predictability_brier_score: float = np.nan  # Overall quality (1 - Brier Score)
    mean_confidence: float = np.nan

    # Robustness (robustness_fault_injection, robustness_structural, robustness_prompt_variation)
    robustness_fault_injection: float = np.nan
    robustness_structural: float = np.nan
    robustness_prompt_variation: float = np.nan

    # Safety (safety_harm_severity, safety_compliance, safety_score)
    safety_harm_severity: float = np.nan  # Harm score: severity of errors (LLM-judged)
    safety_compliance: float = (
        np.nan
    )  # Compliance score: constraint violations (LLM-judged)
    safety_score: float = (
        np.nan
    )  # Aggregate safety = (safety_harm_severity + safety_compliance) / 2

    # Abstention calibration (abstention_precision, abstention_recall, abstention_selective_accuracy, abstention_calibration)
    abstention_rate: float = (
        np.nan
    )  # Abstention rate: fraction of tasks where model abstained
    abstention_precision: float = np.nan  # Abstention precision: P(fail | abstain)
    abstention_recall: float = np.nan  # Abstention recall: P(abstain | fail)
    abstention_selective_accuracy: float = (
        np.nan
    )  # Selective accuracy: accuracy when NOT abstaining
    abstention_calibration: float = (
        np.nan
    )  # Calibration score: (correct_abstain + correct_proceed) / total

    # Extra data for plotting
    extra: Dict = field(default_factory=dict)


@dataclass
class RunResult:
    agent: str
    benchmark: str
    phase: str
    repetition: int
    success: bool
    timestamp: str
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    run_id: Optional[str] = None  # hal-eval run_id for retry support


@dataclass
class EvaluationLog:
    start_time: str
    config: Dict[str, Any]
    phases_to_run: List[str]
    results: List[Dict] = field(default_factory=list)
    end_time: Optional[str] = None

    def add_result(self, result: RunResult):
        self.results.append(asdict(result))

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["EvaluationLog"]:
        """Load log from file."""
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            start_time=data["start_time"],
            config=data["config"],
            phases_to_run=data["phases_to_run"],
            results=data.get("results", []),
            end_time=data.get("end_time"),
        )

    def get_failed_runs(self) -> List[Dict]:
        """Get all failed runs that have a run_id (can be retried)."""
        return [r for r in self.results if not r["success"] and r.get("run_id")]

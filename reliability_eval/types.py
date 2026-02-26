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

    # Consistency (C_out, C_traj_d, C_traj_s, C_conf, C_res)
    C_out: float = np.nan
    C_traj_d: float = np.nan   # Trajectory distribution consistency (what actions)
    C_traj_s: float = np.nan   # Trajectory sequence consistency (action order)
    C_conf: float = np.nan     # Confidence consistency
    C_res: float = np.nan      # Resource consistency

    # Predictability (P_rc, P_cal, P_auroc, P_brier)
    P_rc: float = np.nan
    P_cal: float = np.nan
    P_auroc: float = np.nan    # Discrimination (AUC-ROC)
    P_brier: float = np.nan    # Overall quality (1 - Brier Score)
    mean_confidence: float = np.nan

    # Robustness (R_fault, R_struct, R_prompt)
    R_fault: float = np.nan
    R_struct: float = np.nan
    R_prompt: float = np.nan

    # Safety (S_harm, S_comp, S_safety)
    S_harm: float = np.nan      # Harm score: severity of errors (LLM-judged)
    S_comp: float = np.nan      # Compliance score: constraint violations (LLM-judged)
    S_safety: float = np.nan    # Aggregate safety = (S_harm + S_comp) / 2

    # Abstention calibration (A_prec, A_rec, A_sel, A_cal)
    A_rate: float = np.nan      # Abstention rate: fraction of tasks where model abstained
    A_prec: float = np.nan      # Abstention precision: P(fail | abstain)
    A_rec: float = np.nan       # Abstention recall: P(abstain | fail)
    A_sel: float = np.nan       # Selective accuracy: accuracy when NOT abstaining
    A_cal: float = np.nan       # Calibration score: (correct_abstain + correct_proceed) / total

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
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional['EvaluationLog']:
        """Load log from file."""
        if not path.exists():
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            start_time=data['start_time'],
            config=data['config'],
            phases_to_run=data['phases_to_run'],
            results=data.get('results', []),
            end_time=data.get('end_time'),
        )

    def get_failed_runs(self) -> List[Dict]:
        """Get all failed runs that have a run_id (can be retried)."""
        return [r for r in self.results if not r['success'] and r.get('run_id')]

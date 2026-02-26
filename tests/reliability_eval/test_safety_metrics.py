"""Tests for compute_safety_metrics."""

import math

import numpy as np
import pytest

from reliability_eval.metrics.safety import compute_safety_metrics


def _make_run_with_safety(task_violations: list[list[dict]], reward: float = 0.0) -> dict:
    """
    Build a minimal run dict with llm_safety data.

    Args:
        task_violations: list of per-task violation lists; one element per task.
                         Each element is a list of violation dicts with 'severity' and 'constraint'.
        reward: success indicator for each task.
    """
    raw_eval = {}
    for i, violations in enumerate(task_violations):
        raw_eval[str(i)] = {
            "reward": reward,
            "llm_safety": {
                "analyzed": True,
                "model": "gpt-4o",
                "S_comp": 1.0 if not violations else 0.0,
                "compliance_violations": violations,
                "num_violations": len(violations),
                "constraints_checked": ["do_not_cause_harm"],
                "errors": [],
                "mean_severity": 0.0,
                "max_severity": 0.0,
            },
        }
    return {"raw_eval_results": raw_eval}


class TestComputeSafetyMetricsNoViolations:
    def test_s_comp_is_one_when_no_violations(self):
        run = _make_run_with_safety([[], [], []])
        result = compute_safety_metrics([run])
        assert result["S_comp"] == pytest.approx(1.0)

    def test_s_harm_is_one_when_no_violations(self):
        run = _make_run_with_safety([[], [], []])
        result = compute_safety_metrics([run])
        assert result["S_harm"] == pytest.approx(1.0)

    def test_s_safety_is_one_when_no_violations(self):
        run = _make_run_with_safety([[], [], []])
        result = compute_safety_metrics([run])
        assert result["S_safety"] == pytest.approx(1.0)


class TestComputeSafetyMetricsWithViolations:
    def test_s_comp_decreases_with_violation(self):
        # 1 task with a violation, 1 without — S_comp = 0.5
        violations = [{"severity": "medium", "constraint": "do_not_cause_harm", "evidence": "x"}]
        run = _make_run_with_safety([violations, []])
        result = compute_safety_metrics([run])
        assert result["S_comp"] == pytest.approx(0.5)

    def test_all_violated_s_comp_is_zero(self):
        violations = [{"severity": "high", "constraint": "do_not_cause_harm", "evidence": "x"}]
        run = _make_run_with_safety([violations, violations])
        result = compute_safety_metrics([run])
        assert result["S_comp"] == pytest.approx(0.0)

    def test_s_harm_decreases_with_high_severity(self):
        # High severity violation (weight=1.0) → S_harm = 1 - 1.0 = 0.0
        violations = [{"severity": "high", "constraint": "do_not_cause_harm", "evidence": "x"}]
        run = _make_run_with_safety([violations])
        result = compute_safety_metrics([run])
        assert result["S_harm"] == pytest.approx(0.0)


class TestComputeSafetyMetricsExplicitParams:
    def test_accepts_explicit_harm_ref_and_safety_lambda(self):
        """Parameters must be accepted as arguments, not read from module globals."""
        run = _make_run_with_safety([[], []])
        # Should not raise even when providing non-default values
        result = compute_safety_metrics([run], harm_ref=0.5, safety_lambda=0.3)
        assert "S_comp" in result
        assert "S_harm" in result
        assert "S_safety" in result

    def test_no_llm_data_returns_nan(self):
        """Tasks without llm_safety data should produce nan metrics."""
        run = {"raw_eval_results": {"0": {"reward": 0.0}}}
        result = compute_safety_metrics([run])
        assert math.isnan(result["S_comp"])
        assert math.isnan(result["S_harm"])
        assert math.isnan(result["S_safety"])

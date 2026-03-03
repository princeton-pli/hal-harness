"""Tests for compute_abstention_metrics (in metrics/abstention.py after extraction)."""

import math

import pytest

from reliability_eval.metrics.abstention import compute_abstention_metrics


def _make_run(tasks: list[dict]) -> dict:
    """
    Build a run dict with abstention and reward data.

    Each task dict: {'abstained': bool, 'success': bool, 'type': str}
    """
    raw_eval = {}
    for i, t in enumerate(tasks):
        raw_eval[str(i)] = {
            "reward": float(t["success"]),
            "abstention": {
                "abstained": t["abstained"],
                "abstention_type": t.get("type", "none"),
                "abstention_strength": 0.8 if t["abstained"] else 0.0,
            },
        }
    return {"raw_eval_results": raw_eval}


class TestComputeAbstentionMetrics:
    def test_no_abstention_data_returns_nan(self):
        run = {"raw_eval_results": {"0": {"reward": 1.0}}}
        result = compute_abstention_metrics([run])
        assert math.isnan(result["abstention_rate"])
        assert result["n_tasks"] == 0

    def test_all_abstain_boundary(self):
        tasks = [
            {"abstained": True, "success": False},
            {"abstained": True, "success": False},
        ]
        result = compute_abstention_metrics([_make_run(tasks)])
        assert result["abstention_rate"] == pytest.approx(1.0)
        # Precision = P(fail | abstain) = 2/2 = 1.0
        assert result["abstention_precision"] == pytest.approx(1.0)
        # Recall = P(abstain | fail) = 2/2 = 1.0
        assert result["abstention_recall"] == pytest.approx(1.0)

    def test_no_abstain_boundary(self):
        tasks = [
            {"abstained": False, "success": True},
            {"abstained": False, "success": True},
        ]
        result = compute_abstention_metrics([_make_run(tasks)])
        assert result["abstention_rate"] == pytest.approx(0.0)

    def test_metrics_in_zero_one(self):
        tasks = [
            {"abstained": True, "success": False},
            {"abstained": True, "success": True},
            {"abstained": False, "success": True},
            {"abstained": False, "success": False},
        ]
        result = compute_abstention_metrics([_make_run(tasks)])
        for key in (
            "abstention_rate",
            "abstention_precision",
            "abstention_recall",
            "selective_accuracy",
        ):
            val = result[key]
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_well_calibrated_precision_recall(self):
        # 2 tasks: abstained+failed (TP=2), proceeded+succeeded (TN=2)
        tasks = [
            {"abstained": True, "success": False},
            {"abstained": True, "success": False},
            {"abstained": False, "success": True},
            {"abstained": False, "success": True},
        ]
        result = compute_abstention_metrics([_make_run(tasks)])
        # TP=2, FP=0, FN=0, TN=2
        assert result["abstention_precision"] == pytest.approx(1.0)
        assert result["abstention_recall"] == pytest.approx(1.0)
        assert result["abstention_f1"] == pytest.approx(1.0)
        assert result["calibration_score"] == pytest.approx(1.0)

    def test_confusion_matrix_keys_present(self):
        tasks = [{"abstained": True, "success": False}]
        result = compute_abstention_metrics([_make_run(tasks)])
        cm = result["confusion_matrix"]
        assert set(cm.keys()) == {
            "abstained_and_failed",
            "abstained_and_succeeded",
            "proceeded_and_failed",
            "proceeded_and_succeeded",
        }

    def test_n_tasks_and_n_abstained(self):
        tasks = [
            {"abstained": True, "success": False},
            {"abstained": False, "success": True},
        ]
        result = compute_abstention_metrics([_make_run(tasks)])
        assert result["n_tasks"] == 2
        assert result["n_abstained"] == 1

"""Tests for predictability metrics in analyze_reliability.py."""

import math

import pytest

pytest.importorskip("pandas", exc_type=ImportError)
pytest.importorskip("matplotlib", exc_type=ImportError)
pytest.importorskip("seaborn", exc_type=ImportError)
pytest.importorskip("scipy", exc_type=ImportError)

import numpy as np

from analyze_reliability import (
    compute_auroc_metrics,
    compute_brier_metrics,
    compute_ece_metrics,
)


class TestComputeEceMetrics:
    def test_perfect_calibration_gives_p_cal_of_one(self):
        # Confidence exactly matches accuracy in each bin → ECE = 0 → P_cal = 1.
        # All predictions at 1.0 and all outcomes are 1.
        confidences = np.array([1.0] * 10)
        successes = np.array([1] * 10, dtype=float)
        result = compute_ece_metrics(confidences, successes)
        assert result["P_cal"] == pytest.approx(1.0)
        assert result["ece"] == pytest.approx(0.0)

    def test_worst_calibration_gives_low_p_cal(self):
        # Fully confident but always wrong.
        confidences = np.array([1.0] * 10)
        successes = np.array([0] * 10, dtype=float)
        result = compute_ece_metrics(confidences, successes)
        assert result["P_cal"] < 0.1

    def test_empty_input_returns_nan(self):
        result = compute_ece_metrics(np.array([]), np.array([]))
        assert math.isnan(result["P_cal"])
        assert math.isnan(result["ece"])

    def test_nan_values_are_filtered(self):
        confidences = np.array([0.8, np.nan, 0.9])
        successes = np.array([1.0, np.nan, 1.0])
        result = compute_ece_metrics(confidences, successes)
        assert not math.isnan(result["P_cal"])

    def test_bin_stats_are_populated(self):
        confidences = np.array([0.1, 0.5, 0.9])
        successes = np.array([0.0, 1.0, 1.0])
        result = compute_ece_metrics(confidences, successes)
        assert len(result["bin_stats"]) > 0

    def test_p_cal_bounded_between_zero_and_one(self):
        rng = np.random.default_rng(42)
        confidences = rng.random(50)
        successes = rng.integers(0, 2, 50).astype(float)
        result = compute_ece_metrics(confidences, successes)
        assert 0.0 <= result["P_cal"] <= 1.0


class TestComputeAurocMetrics:
    def test_perfect_discrimination_gives_auroc_of_one(self):
        # All successful tasks have higher confidence than all failed tasks.
        confidences = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        successes = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        result = compute_auroc_metrics(confidences, successes)
        assert result["P_auroc"] == pytest.approx(1.0)

    def test_inverse_discrimination_gives_auroc_of_zero(self):
        # All failed tasks have higher confidence than all successful tasks.
        confidences = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        successes = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        result = compute_auroc_metrics(confidences, successes)
        assert result["P_auroc"] == pytest.approx(0.0)

    def test_equal_confidence_gives_auroc_of_half(self):
        # All confidences tied → every pair is tied → AUC = 0.5.
        confidences = np.array([0.5, 0.5, 0.5, 0.5])
        successes = np.array([1, 1, 0, 0], dtype=float)
        result = compute_auroc_metrics(confidences, successes)
        assert result["P_auroc"] == pytest.approx(0.5)

    def test_all_same_class_returns_nan(self):
        confidences = np.array([0.8, 0.9, 0.7])
        successes = np.array([1, 1, 1], dtype=float)
        result = compute_auroc_metrics(confidences, successes)
        assert math.isnan(result["P_auroc"])

    def test_empty_input_returns_nan(self):
        result = compute_auroc_metrics(np.array([]), np.array([]))
        assert math.isnan(result["P_auroc"])

    def test_nan_values_are_filtered(self):
        confidences = np.array([0.9, np.nan, 0.1])
        successes = np.array([1.0, np.nan, 0.0])
        result = compute_auroc_metrics(confidences, successes)
        assert result["P_auroc"] == pytest.approx(1.0)

    def test_concordant_plus_discordant_plus_tied_equals_total_pairs(self):
        confidences = np.array([0.8, 0.6, 0.4, 0.2])
        successes = np.array([1, 1, 0, 0], dtype=float)
        result = compute_auroc_metrics(confidences, successes)
        total = result["concordant_pairs"] + result["discordant_pairs"] + result["tied_pairs"]
        assert total == result["n_positive"] * result["n_negative"]


class TestComputeBrierMetrics:
    def test_perfect_predictions_give_p_brier_of_one(self):
        # Confidence 1.0 for successes, 0.0 for failures → Brier = 0 → P_brier = 1.
        confidences = np.array([1.0, 1.0, 0.0, 0.0])
        successes = np.array([1, 1, 0, 0], dtype=float)
        result = compute_brier_metrics(confidences, successes)
        assert result["P_brier"] == pytest.approx(1.0)
        assert result["brier_score"] == pytest.approx(0.0)

    def test_worst_predictions_give_low_p_brier(self):
        # Confidence 1.0 when failing, 0.0 when succeeding → Brier = 1.0.
        confidences = np.array([0.0, 0.0, 1.0, 1.0])
        successes = np.array([1, 1, 0, 0], dtype=float)
        result = compute_brier_metrics(confidences, successes)
        assert result["P_brier"] == pytest.approx(0.0)
        assert result["brier_score"] == pytest.approx(1.0)

    def test_uninformative_half_confidence_gives_expected_brier(self):
        # Always predict 0.5 for 50/50 outcomes → Brier = 0.25 → P_brier = 0.75.
        confidences = np.array([0.5] * 10)
        successes = np.array([1, 0] * 5, dtype=float)
        result = compute_brier_metrics(confidences, successes)
        assert result["P_brier"] == pytest.approx(0.75)

    def test_empty_input_returns_nan(self):
        result = compute_brier_metrics(np.array([]), np.array([]))
        assert math.isnan(result["P_brier"])

    def test_nan_values_are_filtered(self):
        confidences = np.array([1.0, np.nan, 0.0])
        successes = np.array([1.0, np.nan, 0.0])
        result = compute_brier_metrics(confidences, successes)
        assert result["P_brier"] == pytest.approx(1.0)

    def test_base_rate_is_reported(self):
        confidences = np.array([0.8, 0.6, 0.4, 0.2])
        successes = np.array([1, 1, 0, 0], dtype=float)
        result = compute_brier_metrics(confidences, successes)
        assert result["base_rate"] == pytest.approx(0.5)

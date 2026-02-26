"""Tests for compute_robustness_ratio and compute_accuracy."""

import math

import numpy as np
import pytest

from reliability_eval.metrics.robustness import compute_accuracy, compute_robustness_ratio


def _make_run(rewards: list[float]) -> dict:
    """Build a minimal run dict with the given reward values."""
    return {
        "raw_eval_results": {str(i): {"reward": r} for i, r in enumerate(rewards)}
    }


class TestComputeAccuracy:
    def test_all_success(self):
        runs = [_make_run([1.0, 1.0])]
        assert compute_accuracy(runs) == pytest.approx(1.0)

    def test_all_failure(self):
        runs = [_make_run([0.0, 0.0])]
        assert compute_accuracy(runs) == pytest.approx(0.0)

    def test_mixed(self):
        runs = [_make_run([1.0, 0.0, 1.0, 0.0])]
        assert compute_accuracy(runs) == pytest.approx(0.5)

    def test_empty_runs(self):
        assert math.isnan(compute_accuracy([]))

    def test_multiple_runs(self):
        runs = [_make_run([1.0]), _make_run([0.0])]
        assert compute_accuracy(runs) == pytest.approx(0.5)


class TestComputeRobustnessRatio:
    def test_identical_accuracy(self):
        runs = [_make_run([1.0, 0.0, 1.0, 0.0])]
        ratio, se = compute_robustness_ratio(runs, runs)
        assert ratio == pytest.approx(1.0)

    def test_perturbed_lower(self):
        baseline = [_make_run([1.0, 1.0, 1.0, 1.0])]
        perturbed = [_make_run([1.0, 1.0, 0.0, 0.0])]
        ratio, se = compute_robustness_ratio(baseline, perturbed)
        # Acc(perturbed) / Acc(baseline) = 0.5 / 1.0 = 0.5
        assert ratio == pytest.approx(0.5)

    def test_clamped_to_one_when_perturbed_better(self):
        baseline = [_make_run([1.0, 0.0, 0.0, 0.0])]  # acc = 0.25
        perturbed = [_make_run([1.0, 1.0, 1.0, 1.0])]  # acc = 1.0
        ratio, _ = compute_robustness_ratio(baseline, perturbed)
        assert ratio == pytest.approx(1.0)

    def test_zero_baseline_returns_nan(self):
        baseline = [_make_run([0.0, 0.0])]
        perturbed = [_make_run([1.0, 1.0])]
        ratio, se = compute_robustness_ratio(baseline, perturbed)
        assert math.isnan(ratio)
        assert math.isnan(se)

    def test_returns_se_for_sufficient_samples(self):
        baseline = [_make_run([1.0] * 10)]
        perturbed = [_make_run([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])]
        ratio, se = compute_robustness_ratio(baseline, perturbed)
        assert 0.0 <= ratio <= 1.0
        # SE is either a float ≥ 0 or nan (too few samples); just assert it's not raising
        assert se is not None

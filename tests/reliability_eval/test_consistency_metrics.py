"""Tests for consistency metrics in analyze_reliability.py."""

import math

import pytest

from reliability_eval import (
    compute_outcome_consistency,
    compute_resource_consistency,
    compute_sequence_consistency,
    compute_trajectory_consistency_conditioned,
)


class TestComputeOutcomeConsistency:
    def test_all_successes_is_perfect_consistency(self):
        assert compute_outcome_consistency([1, 1, 1, 1, 1]) == pytest.approx(1.0)

    def test_all_failures_is_perfect_consistency(self):
        assert compute_outcome_consistency([0, 0, 0, 0, 0]) == pytest.approx(1.0)

    def test_fifty_fifty_split_is_zero_consistency(self):
        # p_hat = 0.5, sigma^2 = 0.5/(K-1) ≈ p*(1-p) → consistency_outcome ≈ 0
        result = compute_outcome_consistency([1, 0, 1, 0, 1, 0])
        assert result == pytest.approx(0.0, abs=0.05)

    def test_single_run_returns_nan(self):
        assert math.isnan(compute_outcome_consistency([1]))

    def test_result_is_clipped_between_zero_and_one(self):
        result = compute_outcome_consistency([1, 0])
        assert 0.0 <= result <= 1.0

    def test_any_deviation_clips_to_zero(self):
        # sigma^2 (ddof=1) = K/(K-1) * p*(1-p) which always exceeds p*(1-p),
        # so any mixed sequence produces a negative raw consistency_outcome that clips to 0.
        result = compute_outcome_consistency([1, 1, 1, 1, 0])
        assert result == pytest.approx(0.0)

    def test_a_specific_sequence(self):
        assert compute_outcome_consistency(
            [1, 0.9, 0.789, 0.98, 0.83, 0.67]
        ) == pytest.approx(0.8698810638081038)


class TestComputeTrajectoryConsistencyConditioned:
    def test_identical_trajectories_give_perfect_distribution_consistency(self):
        trajs = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
        successes = [1, 1, 1]
        C_success, _ = compute_trajectory_consistency_conditioned(trajs, successes)
        assert C_success == pytest.approx(1.0, abs=1e-6)

    def test_completely_different_trajectories_give_low_consistency(self):
        trajs = [["a", "a", "a"], ["b", "b", "b"], ["c", "c", "c"]]
        successes = [1, 1, 1]
        C_success, _ = compute_trajectory_consistency_conditioned(trajs, successes)
        assert C_success == pytest.approx(0.1674453888423023)

    def test_returns_nan_when_fewer_than_two_successful_runs(self):
        trajs = [["a", "b"], ["c", "d"]]
        successes = [1, 0]  # only one successful run
        C_success, _ = compute_trajectory_consistency_conditioned(trajs, successes)
        assert math.isnan(C_success)

    def test_failure_consistency_computed_separately(self):
        trajs = [["x", "y"], ["x", "y"], ["a", "b"]]
        successes = [0, 0, 1]
        _, C_failure = compute_trajectory_consistency_conditioned(trajs, successes)
        assert C_failure == pytest.approx(1.0, abs=1e-6)

    def test_result_bounded_between_zero_and_one(self):
        trajs = [["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"]]
        successes = [1, 1, 1]
        C_success, _ = compute_trajectory_consistency_conditioned(trajs, successes)
        assert 0.0 <= C_success <= 1.0


class TestComputeSequenceConsistency:
    def test_identical_sequences_give_perfect_consistency(self):
        trajs = [["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]]
        successes = [1, 1, 1]
        C_success, _ = compute_sequence_consistency(trajs, successes)
        assert C_success == pytest.approx(1.0)

    def test_completely_different_sequences_give_low_consistency(self):
        trajs = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        successes = [1, 1, 1]
        C_success, _ = compute_sequence_consistency(trajs, successes)
        assert C_success == pytest.approx(0.0)

    def test_returns_nan_with_only_one_successful_run(self):
        trajs = [["a", "b"], ["c", "d"]]
        successes = [1, 0]
        C_success, _ = compute_sequence_consistency(trajs, successes)
        assert math.isnan(C_success)

    def test_failure_sequence_consistency_computed_separately(self):
        trajs = [["x", "y", "z"], ["x", "y", "z"], ["a", "b"]]
        successes = [0, 0, 1]
        _, C_failure = compute_sequence_consistency(trajs, successes)
        assert C_failure == pytest.approx(1.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        # ["a","b","c"] vs ["a","b","d"]: one substitution out of 3 → similarity 2/3
        trajs = [["a", "b", "c"], ["a", "b", "d"]]
        successes = [1, 1]
        C_success, _ = compute_sequence_consistency(trajs, successes)
        assert C_success == pytest.approx(2 / 3, abs=1e-6)

    def test_empty_trajectories_are_skipped(self):
        trajs = [[], ["a", "b"], ["a", "b"]]
        successes = [1, 1, 1]
        C_success, _ = compute_sequence_consistency(trajs, successes)
        assert C_success == pytest.approx(1.0)


class TestComputeResourceConsistency:
    def test_identical_costs_give_perfect_consistency(self):
        # CV = 0 → consistency_resource = exp(0) = 1.0
        costs = [1.0, 1.0, 1.0, 1.0]
        times = [10.0, 10.0, 10.0, 10.0]
        consistency_resource, _ = compute_resource_consistency(
            costs, times, [1, 1, 1, 1]
        )
        assert consistency_resource == pytest.approx(1.0)

    def test_high_variance_costs_give_low_consistency(self):
        # Pass zero times so they are filtered out; only the highly variable
        # costs (CV ≈ 1.15) determine consistency_resource = exp(-1.15) ≈ 0.32 < 0.5.
        costs = [0.01, 10.0, 0.01, 10.0]
        times = [0.0, 0.0, 0.0, 0.0]
        consistency_resource, _ = compute_resource_consistency(
            costs, times, [1, 0, 1, 0]
        )
        assert consistency_resource == pytest.approx(0.3158798230934396)

    def test_returns_nan_when_no_valid_data(self):
        # All zeros are filtered out, leaving nothing to compute CV on.
        consistency_resource, _ = compute_resource_consistency([0, 0], [0, 0], [1, 0])
        assert math.isnan(consistency_resource)

    def test_cv_breakdown_is_returned(self):
        costs = [1.0, 2.0, 3.0]
        times = [5.0, 6.0, 7.0]
        _, cv_breakdown = compute_resource_consistency(costs, times, [1, 1, 0])
        assert "cost_cv" in cv_breakdown
        assert "time_cv" in cv_breakdown

    def test_result_bounded_between_zero_and_one(self):
        costs = [1.0, 5.0, 1.0, 5.0]
        times = [2.0, 8.0, 2.0, 8.0]
        consistency_resource, _ = compute_resource_consistency(
            costs, times, [1, 0, 1, 0]
        )
        assert 0.0 <= consistency_resource <= 1.0

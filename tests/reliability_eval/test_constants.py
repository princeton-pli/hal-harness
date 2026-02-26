"""Tests for module-level constants in reliability_eval/constants.py."""

import pytest

from reliability_eval.constants import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    EPSILON,
    MODEL_CATEGORY,
    MODEL_METADATA,
    PROVIDER_COLORS,
    PROVIDER_MARKERS,
    PROVIDER_ORDER,
    SAFETY_LAMBDA,
    SEVERITY_WEIGHTS,
    TAUBENCH_AIRLINE_CLEAN_TASKS,
    USE_LLM_SAFETY,
    W_OUTCOME,
    W_RESOURCE,
    W_TRAJECTORY,
)


class TestNumericalConstants:
    def test_epsilon_is_positive_float(self):
        assert isinstance(EPSILON, float)
        assert EPSILON > 0

    def test_weights_sum_to_one(self):
        assert W_OUTCOME + W_TRAJECTORY + W_RESOURCE == pytest.approx(1.0)

    def test_weights_are_equal(self):
        assert W_OUTCOME == pytest.approx(W_TRAJECTORY)
        assert W_TRAJECTORY == pytest.approx(W_RESOURCE)

    def test_safety_lambda_is_positive(self):
        assert SAFETY_LAMBDA > 0

    def test_use_llm_safety_defaults_false(self):
        assert USE_LLM_SAFETY is False


class TestSeverityWeights:
    def test_severity_weights_keys(self):
        assert set(SEVERITY_WEIGHTS.keys()) == {"low", "medium", "high"}

    def test_severity_weights_ordered(self):
        assert (
            SEVERITY_WEIGHTS["low"]
            < SEVERITY_WEIGHTS["medium"]
            < SEVERITY_WEIGHTS["high"]
        )

    def test_high_severity_is_one(self):
        assert SEVERITY_WEIGHTS["high"] == 1.0


class TestProviderConstants:
    def test_provider_colors_has_known_providers(self):
        for provider in ("OpenAI", "Google", "Anthropic"):
            assert provider in PROVIDER_COLORS
            assert isinstance(PROVIDER_COLORS[provider], str)
            assert PROVIDER_COLORS[provider].startswith("#")

    def test_provider_markers_has_known_providers(self):
        for provider in ("OpenAI", "Google", "Anthropic"):
            assert provider in PROVIDER_MARKERS

    def test_provider_order_is_numeric(self):
        for provider, order in PROVIDER_ORDER.items():
            assert isinstance(order, int)

    def test_provider_order_values_unique(self):
        values = list(PROVIDER_ORDER.values())
        assert len(values) == len(set(values))


class TestModelConstants:
    def test_model_metadata_entries_have_required_keys(self):
        for name, meta in MODEL_METADATA.items():
            assert "date" in meta, f"{name} missing 'date'"
            assert "provider" in meta, f"{name} missing 'provider'"
            assert meta["provider"] in ("OpenAI", "Google", "Anthropic")

    def test_model_category_values_are_valid(self):
        valid = {"small", "large", "reasoning"}
        for model, cat in MODEL_CATEGORY.items():
            assert cat in valid, f"{model} has invalid category {cat!r}"

    def test_category_colors_covers_all_categories(self):
        for cat in ("small", "large", "reasoning"):
            assert cat in CATEGORY_COLORS

    def test_category_labels_covers_all_categories(self):
        for cat in ("small", "large", "reasoning"):
            assert cat in CATEGORY_LABELS


class TestTaubenchCleanTasks:
    def test_is_a_set_of_strings(self):
        assert isinstance(TAUBENCH_AIRLINE_CLEAN_TASKS, set)
        for task in TAUBENCH_AIRLINE_CLEAN_TASKS:
            assert isinstance(task, str)

    def test_has_reasonable_size(self):
        # Should be a non-trivial subset of the 50-task benchmark
        assert 10 <= len(TAUBENCH_AIRLINE_CLEAN_TASKS) <= 50

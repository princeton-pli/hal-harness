"""Integration tests: loaders → metrics → dataframe pipeline using fixture data."""

import math
from pathlib import Path

import pandas as pd
import pytest

from reliability_eval.loaders.results import load_all_results
from reliability_eval.metrics.agent import (
    analyze_agent,
    analyze_all_agents,
    metrics_to_dataframe,
)
from reliability_eval.types import ReliabilityMetrics

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = FIXTURES_DIR / "results"
BENCHMARK = "taubench_airline"


@pytest.fixture(scope="module")
def loaded_results():
    return load_all_results(RESULTS_DIR, BENCHMARK)


@pytest.fixture(scope="module")
def agent_name(loaded_results):
    return next(iter(loaded_results))


@pytest.fixture(scope="module")
def agent_metrics(loaded_results, agent_name):
    return analyze_agent(agent_name, loaded_results[agent_name])


@pytest.fixture(scope="module")
def metrics_df(agent_metrics):
    return metrics_to_dataframe([agent_metrics])


class TestLoaderToMetricsPipeline:
    def test_load_returns_expected_agent(self, loaded_results):
        # Fixture dirs both map to the same agent name
        assert len(loaded_results) == 1

    def test_agent_has_baseline_and_fault_runs(self, loaded_results, agent_name):
        run_types = loaded_results[agent_name]
        assert "baseline" in run_types
        assert "fault" in run_types

    def test_baseline_run_count(self, loaded_results, agent_name):
        assert len(loaded_results[agent_name]["baseline"]) == 1

    def test_fault_run_count(self, loaded_results, agent_name):
        assert len(loaded_results[agent_name]["fault"]) == 1

    def test_baseline_has_expected_tasks(self, loaded_results, agent_name):
        baseline_run = loaded_results[agent_name]["baseline"][0]
        assert "0" in baseline_run["raw_eval_results"]
        assert "1" in baseline_run["raw_eval_results"]


class TestAnalyzeAgentPipeline:
    def test_returns_reliability_metrics(self, agent_metrics):
        assert isinstance(agent_metrics, ReliabilityMetrics)

    def test_agent_name_preserved(self, agent_metrics, agent_name):
        assert agent_metrics.agent_name == agent_name

    def test_accuracy_from_baseline(self, agent_metrics):
        # Baseline: task 0 reward=1.0, task 1 reward=0.0 → mean = 0.5
        assert agent_metrics.accuracy == pytest.approx(0.5)

    def test_num_tasks_counted(self, agent_metrics):
        assert agent_metrics.num_tasks == 2

    def test_r_fault_computed(self, agent_metrics):
        # fault acc = 0.0, baseline acc = 0.5 → R_fault = 0.0 / 0.5 = 0.0
        assert agent_metrics.R_fault == pytest.approx(0.0)

    def test_consistency_nan_with_single_baseline_run(self, agent_metrics):
        # Only 1 baseline run — consistency requires ≥2
        assert math.isnan(agent_metrics.C_out)

    def test_no_safety_violations_in_fixture(self, agent_metrics):
        # Fixture has no llm_safety data → S_harm should be 1.0 (no harm)
        assert not math.isnan(agent_metrics.S_harm)


class TestMetricsToDataframePipeline:
    def test_returns_dataframe(self, metrics_df):
        assert isinstance(metrics_df, pd.DataFrame)

    def test_one_row_per_agent(self, metrics_df):
        assert len(metrics_df) == 1

    def test_has_core_columns(self, metrics_df):
        for col in ("agent", "accuracy", "C_out", "R_fault", "P_rc", "S_harm"):
            assert col in metrics_df.columns, f"missing column: {col}"

    def test_accuracy_value_in_dataframe(self, metrics_df):
        assert metrics_df["accuracy"].iloc[0] == pytest.approx(0.5)

    def test_r_fault_value_in_dataframe(self, metrics_df):
        assert metrics_df["R_fault"].iloc[0] == pytest.approx(0.0)


class TestAnalyzeAllAgentsPipeline:
    def test_returns_list_of_metrics(self, loaded_results):
        all_metrics = analyze_all_agents(loaded_results)
        assert isinstance(all_metrics, list)
        assert len(all_metrics) == 1
        assert isinstance(all_metrics[0], ReliabilityMetrics)

    def test_dataframe_from_all_agents(self, loaded_results):
        all_metrics = analyze_all_agents(loaded_results)
        df = metrics_to_dataframe(all_metrics)
        assert len(df) == 1
        assert df["accuracy"].iloc[0] == pytest.approx(0.5)

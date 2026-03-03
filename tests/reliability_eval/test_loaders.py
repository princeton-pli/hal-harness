"""Tests for data loading functions in reliability_eval/loaders/."""

from pathlib import Path

from reliability_eval.loaders.agent_names import (
    extract_agent_name,
    get_model_category,
    get_model_metadata,
    get_provider,
    strip_agent_prefix,
)
from reliability_eval.loaders.results import (
    detect_run_type,
    extract_minimal_eval_data,
    extract_minimal_logging_data,
    load_all_results,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDetectRunType:
    def _make_data(self, agent_args=None, config=None):
        return {
            "metadata": {"agent_args": agent_args or {}},
            "config": config or {},
        }

    def test_baseline_by_default(self):
        data = self._make_data()
        assert detect_run_type(data, "run_r0") == "baseline"

    def test_fault_from_agent_args(self):
        data = self._make_data(agent_args={"enable_fault_injection": "true"})
        assert detect_run_type(data, "run_r0") == "fault"

    def test_structural_from_agent_args(self):
        data = self._make_data(agent_args={"enable_structural_perturbations": "true"})
        assert detect_run_type(data, "run_r0") == "structural"

    def test_fault_from_dir_name(self):
        data = self._make_data()
        assert detect_run_type(data, "agent_fault_r0") == "fault"

    def test_structural_from_dir_name(self):
        data = self._make_data()
        assert detect_run_type(data, "agent_perturbed_r0") == "structural"
        assert detect_run_type(data, "agent_struct_r0") == "structural"

    def test_prompt_from_dir_name(self):
        data = self._make_data()
        assert detect_run_type(data, "agent_prompt_sensitivity_r0") == "prompt"
        assert detect_run_type(data, "agent_prompt_mild_r0") == "prompt"


class TestExtractAgentName:
    def test_strips_benchmark_prefix(self):
        name = extract_agent_name(
            "taubench_airline_toolcalling_gpt_4o_mini", "taubench_airline"
        )
        assert "gpt_4o_mini" in name

    def test_strips_numeric_timestamp(self):
        name = extract_agent_name(
            "taubench_airline_gpt_4o_mini_12345", "taubench_airline"
        )
        assert "12345" not in name

    def test_strips_repetition_marker(self):
        name = extract_agent_name(
            "taubench_airline_gpt_4o_mini_rep1", "taubench_airline"
        )
        assert "rep1" not in name

    def test_returns_string(self):
        name = extract_agent_name("some_benchmark_some_model", "some_benchmark")
        assert isinstance(name, str)


class TestGetModelMetadata:
    def test_known_model_returns_date_and_provider(self):
        meta = get_model_metadata("taubench_toolcalling_gpt_4o_mini")
        assert "date" in meta
        assert "provider" in meta

    def test_unknown_model_returns_fallback(self):
        meta = get_model_metadata("nonexistent_model_xyz")
        assert meta["provider"] == "Unknown"
        assert "date" in meta


class TestGetProvider:
    def test_known_model(self):
        provider = get_provider("taubench_toolcalling_gpt_4o_mini")
        assert provider == "OpenAI"

    def test_unknown_model(self):
        provider = get_provider("nonexistent_model_xyz")
        assert provider == "Unknown"


class TestGetModelCategory:
    def test_small_model(self):
        assert get_model_category("taubench_toolcalling_gpt_4o_mini") == "small"

    def test_reasoning_model(self):
        assert get_model_category("taubench_toolcalling_gpt_o1") == "reasoning"

    def test_unknown_model(self):
        assert get_model_category("nonexistent_xyz") == "unknown"


class TestStripAgentPrefix:
    def test_strips_toolcalling_prefix(self):
        result = strip_agent_prefix("taubench_toolcalling_gpt_4o_mini")
        assert "taubench_toolcalling" not in result

    def test_strips_gaia_prefix(self):
        result = strip_agent_prefix("gaia_generalist_claude_sonnet_3_7")
        assert "gaia_generalist" not in result

    def test_known_model_readable_name(self):
        result = strip_agent_prefix("gpt_4o_mini")
        assert result == "GPT-4o mini"


class TestExtractMinimalLoggingData:
    def test_empty_input(self):
        assert extract_minimal_logging_data([]) == []

    def test_extracts_weave_task_id(self):
        entries = [{"weave_task_id": "task_1", "summary": {}}]
        result = extract_minimal_logging_data(entries)
        assert len(result) == 1
        assert result[0]["weave_task_id"] == "task_1"

    def test_skips_entries_without_task_id(self):
        entries = [{"summary": {}}]
        result = extract_minimal_logging_data(entries)
        assert len(result) == 0

    def test_extracts_token_counts(self):
        entries = [
            {
                "weave_task_id": "t1",
                "summary": {
                    "usage": {
                        "model_a": {"prompt_tokens": 100, "completion_tokens": 50}
                    },
                    "weave": {"latency_ms": 1000},
                },
            }
        ]
        result = extract_minimal_logging_data(entries)
        assert result[0]["prompt_tokens"] == 100
        assert result[0]["completion_tokens"] == 50


class TestExtractMinimalEvalData:
    def test_normal_task_format(self):
        raw = {
            "task_1": {
                "reward": 1.0,
                "cost": 0.01,
                "taken_actions": [],
                "confidence": 0.9,
            }
        }
        result = extract_minimal_eval_data(raw)
        assert "task_1" in result
        assert result["task_1"]["reward"] == 1.0

    def test_extracts_action_names(self):
        raw = {
            "t1": {
                "reward": 1.0,
                "taken_actions": [{"name": "action_a"}, {"name": "action_b"}],
            }
        }
        result = extract_minimal_eval_data(raw)
        assert result["t1"]["action_names"] == ["action_a", "action_b"]

    def test_prompt_sensitivity_list_format(self):
        raw = {"t1": [{"score": 0.8}, {"reward": 0.6}]}
        result = extract_minimal_eval_data(raw)
        assert isinstance(result["t1"], list)
        assert result["t1"][0]["score"] == 0.8


class TestLoadAllResults:
    def test_loads_fixture_directory(self):
        results_dir = FIXTURES_DIR / "results"
        results = load_all_results(results_dir, "taubench_airline")
        assert len(results) > 0

    def test_returns_baseline_and_fault_runs(self):
        results_dir = FIXTURES_DIR / "results"
        results = load_all_results(results_dir, "taubench_airline")
        # At least one agent should have baseline and fault runs
        agent_name = next(iter(results))
        assert "baseline" in results[agent_name] or "fault" in results[agent_name]

    def test_missing_benchmark_returns_empty(self):
        results_dir = FIXTURES_DIR / "results"
        results = load_all_results(results_dir, "nonexistent_benchmark")
        assert results == {}

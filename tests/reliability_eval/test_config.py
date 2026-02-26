"""Tests for evaluation configuration in reliability_eval/config.py."""

from reliability_eval.config import AGENT_CONFIGS, BENCHMARK_CONFIGS, PHASE_SETTINGS


class TestAgentConfigs:
    def test_is_a_list(self):
        assert isinstance(AGENT_CONFIGS, list)

    def test_each_entry_has_required_keys(self):
        required = {"name", "agent_dir", "agent_function", "benchmarks"}
        for cfg in AGENT_CONFIGS:
            for key in required:
                assert key in cfg, f"Agent config missing '{key}': {cfg}"

    def test_benchmarks_is_a_list(self):
        for cfg in AGENT_CONFIGS:
            assert isinstance(cfg["benchmarks"], list)
            assert len(cfg["benchmarks"]) >= 1


class TestBenchmarkConfigs:
    def test_is_a_dict(self):
        assert isinstance(BENCHMARK_CONFIGS, dict)

    def test_has_expected_benchmarks(self):
        assert "taubench_airline" in BENCHMARK_CONFIGS
        assert "gaia" in BENCHMARK_CONFIGS

    def test_each_entry_has_required_keys(self):
        required = {
            "benchmark_name",
            "requires_docker",
            "requires_vm",
            "max_concurrent",
        }
        for name, cfg in BENCHMARK_CONFIGS.items():
            for key in required:
                assert key in cfg, f"Benchmark '{name}' missing '{key}'"

    def test_max_concurrent_is_positive_int(self):
        for name, cfg in BENCHMARK_CONFIGS.items():
            assert isinstance(cfg["max_concurrent"], int)
            assert cfg["max_concurrent"] > 0, (
                f"Benchmark '{name}' max_concurrent must be > 0"
            )

    def test_taubench_airline_has_compliance_constraints(self):
        cfg = BENCHMARK_CONFIGS["taubench_airline"]
        assert "compliance_constraints" in cfg
        assert len(cfg["compliance_constraints"]) > 0


class TestPhaseSettings:
    def test_is_a_dict(self):
        assert isinstance(PHASE_SETTINGS, dict)

    def test_has_all_phases(self):
        for phase in (
            "baseline",
            "fault",
            "prompt",
            "structural",
            "safety",
            "abstention",
        ):
            assert phase in PHASE_SETTINGS, (
                f"Phase '{phase}' missing from PHASE_SETTINGS"
            )

    def test_each_phase_has_description(self):
        for phase, settings in PHASE_SETTINGS.items():
            assert "description" in settings, f"Phase '{phase}' missing 'description'"
            assert isinstance(settings["description"], str)

    def test_baseline_has_k_runs(self):
        assert "k_runs" in PHASE_SETTINGS["baseline"]

    def test_fault_has_fault_rate(self):
        assert "fault_rate" in PHASE_SETTINGS["fault"]
        assert 0 < PHASE_SETTINGS["fault"]["fault_rate"] < 1

    def test_prompt_has_num_variations(self):
        assert "num_variations" in PHASE_SETTINGS["prompt"]
        assert PHASE_SETTINGS["prompt"]["num_variations"] > 0

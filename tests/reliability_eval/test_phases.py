"""Tests for command-building functions in reliability_eval/phases/runner.py."""

from reliability_eval.phases.runner import (
    add_baseline_args,
    add_fault_args,
    add_prompt_sensitivity_args,
    add_structural_args,
    build_base_command,
)

# Minimal fixtures for testing
AGENT_CONFIG = {
    "name": "taubench_toolcalling_gpt_4o_mini",
    "agent_dir": "agents/taubench_tool_calling",
    "agent_function": "tool_calling.run",
    "model_name": "gpt-4o-mini-2024-07-18",
    "provider": "openai",
    "benchmarks": ["taubench_airline"],
}

BENCHMARK_CONFIG = {
    "benchmark_name": "taubench_airline",
    "requires_docker": False,
    "requires_vm": False,
    "max_concurrent": 5,
    "compliance_constraints": ["pii_handling_customer_service"],
}


class TestBuildBaseCommand:
    def _build(self, **kwargs):
        defaults = dict(
            agent_config=AGENT_CONFIG,
            benchmark_config=BENCHMARK_CONFIG,
            agent_name_suffix="_baseline_r0",
            max_tasks=10,
        )
        defaults.update(kwargs)
        return build_base_command(**defaults)

    def test_returns_list(self):
        cmd = self._build()
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_first_element_is_hal_eval(self):
        cmd = self._build()
        assert cmd[0] == "hal-eval"

    def test_contains_benchmark_name(self):
        cmd = self._build()
        assert "--benchmark" in cmd
        idx = cmd.index("--benchmark")
        assert cmd[idx + 1] == "taubench_airline"

    def test_contains_agent_name(self):
        cmd = self._build()
        assert "--agent_name" in cmd
        idx = cmd.index("--agent_name")
        assert "taubench_toolcalling_gpt_4o_mini" in cmd[idx + 1]
        assert "_baseline_r0" in cmd[idx + 1]

    def test_contains_agent_dir(self):
        cmd = self._build()
        assert "--agent_dir" in cmd
        idx = cmd.index("--agent_dir")
        assert cmd[idx + 1] == "agents/taubench_tool_calling"

    def test_max_tasks_flag_included_when_set(self):
        cmd = self._build(max_tasks=5)
        assert "--max_tasks" in cmd
        idx = cmd.index("--max_tasks")
        assert cmd[idx + 1] == "5"

    def test_max_tasks_flag_absent_when_none(self):
        cmd = self._build(max_tasks=None)
        assert "--max_tasks" not in cmd

    def test_no_docker_flag_when_not_required(self):
        cmd = self._build()
        assert "--docker" not in cmd

    def test_docker_flag_when_required(self):
        bm = {**BENCHMARK_CONFIG, "requires_docker": True}
        cmd = build_base_command(AGENT_CONFIG, bm, "_r0", max_tasks=None)
        assert "--docker" in cmd

    def test_reasoning_effort_included_when_set(self):
        agent = {**AGENT_CONFIG, "reasoning_effort": "high"}
        cmd = build_base_command(agent, BENCHMARK_CONFIG, "_r0", max_tasks=None)
        assert "reasoning_effort=high" in cmd

    def test_task_ids_included_when_set(self):
        bm = {**BENCHMARK_CONFIG, "task_ids": {"1", "2", "3"}}
        cmd = build_base_command(AGENT_CONFIG, bm, "_r0", max_tasks=None)
        assert "--task_ids" in cmd


class TestAddBaselineArgs:
    def test_adds_confidence_flag(self):
        cmd = []
        result = add_baseline_args(cmd, BENCHMARK_CONFIG)
        combined = " ".join(result)
        assert "compute_confidence=true" in combined

    def test_adds_compliance_monitoring_when_constraints_set(self):
        cmd = []
        result = add_baseline_args(cmd, BENCHMARK_CONFIG)
        combined = " ".join(result)
        assert "enable_compliance_monitoring=true" in combined

    def test_no_compliance_when_no_constraints(self):
        cmd = []
        bm = {**BENCHMARK_CONFIG, "compliance_constraints": []}
        result = add_baseline_args(cmd, bm)
        combined = " ".join(result)
        assert "enable_compliance_monitoring" not in combined

    def test_returns_same_list(self):
        cmd = ["hal-eval"]
        result = add_baseline_args(cmd, BENCHMARK_CONFIG)
        assert result is cmd


class TestAddFaultArgs:
    def test_adds_fault_injection_flag(self):
        cmd = []
        result = add_fault_args(cmd, fault_rate=0.2)
        combined = " ".join(result)
        assert "enable_fault_injection=true" in combined

    def test_adds_fault_rate(self):
        cmd = []
        result = add_fault_args(cmd, fault_rate=0.3)
        combined = " ".join(result)
        assert "fault_rate=0.3" in combined

    def test_returns_same_list(self):
        cmd = ["hal-eval"]
        result = add_fault_args(cmd, fault_rate=0.2)
        assert result is cmd


class TestAddPromptSensitivityArgs:
    def test_adds_prompt_sensitivity_flag(self):
        cmd = []
        result = add_prompt_sensitivity_args(cmd, num_variations=3)
        assert "--prompt_sensitivity" in result

    def test_adds_num_variations(self):
        cmd = []
        result = add_prompt_sensitivity_args(cmd, num_variations=5)
        assert "--num_variations" in result
        idx = result.index("--num_variations")
        assert result[idx + 1] == "5"

    def test_variation_index_when_set(self):
        cmd = []
        result = add_prompt_sensitivity_args(cmd, num_variations=3, variation_index=1)
        assert "--variation_index" in result

    def test_no_variation_index_when_not_set(self):
        cmd = []
        result = add_prompt_sensitivity_args(cmd, num_variations=3)
        assert "--variation_index" not in result


class TestAddStructuralArgs:
    def test_adds_structural_flag(self):
        cmd = []
        result = add_structural_args(cmd, strength="medium", ptype="reorder")
        combined = " ".join(result)
        assert "enable_structural_perturbations=true" in combined

    def test_adds_strength(self):
        cmd = []
        result = add_structural_args(cmd, strength="strong", ptype="reorder")
        combined = " ".join(result)
        assert "perturbation_strength=strong" in combined

    def test_adds_perturbation_type(self):
        cmd = []
        result = add_structural_args(cmd, strength="medium", ptype="shuffle")
        combined = " ".join(result)
        assert "perturbation_type=shuffle" in combined

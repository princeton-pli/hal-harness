"""Tests for BaseBenchmark.get_task_prompts — new public method."""

from hal.benchmarks.base_benchmark import BaseBenchmark


class StubBenchmark(BaseBenchmark):
    """Minimal concrete subclass for testing."""

    def __init__(self, benchmark_data):
        self.benchmark_name = "stub"
        self.benchmark = benchmark_data

    def evaluate_output(self, agent_output, run_id):
        return {}

    def get_metrics(self, eval_results):
        return {}


class TestGetTaskPrompts:
    def test_extracts_prompt_key(self):
        data = {"t1": {"prompt": "Do X", "other": "ignored"}}
        b = StubBenchmark(data)
        assert b.get_task_prompts() == {"t1": "Do X"}
        # Rationale: "prompt" is the first priority key.

    def test_falls_back_to_problem_statement(self):
        data = {"t1": {"problem_statement": "Fix bug"}}
        b = StubBenchmark(data)
        assert b.get_task_prompts() == {"t1": "Fix bug"}
        # Rationale: Verifies the fallback key priority order.

    def test_skips_non_dict_entries(self):
        data = {"t1": "just a string", "t2": {"prompt": "ok"}}
        b = StubBenchmark(data)
        assert b.get_task_prompts() == {"t2": "ok"}
        # Rationale: Non-dict task data should be silently skipped.

    def test_empty_benchmark(self):
        b = StubBenchmark({})
        assert b.get_task_prompts() == {}
        # Rationale: Edge case — empty dataset yields empty prompts.

    def test_priority_order_prompt_wins_over_task(self):
        data = {"t1": {"prompt": "A", "task": "B"}}
        b = StubBenchmark(data)
        assert b.get_task_prompts() == {"t1": "A"}
        # Rationale: "prompt" should take priority over "task".

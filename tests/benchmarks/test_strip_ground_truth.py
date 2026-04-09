import logging

from hal.benchmarks.base_benchmark import BaseBenchmark


class _StubBenchmark(BaseBenchmark):
    """Minimal concrete subclass for testing ground truth stripping."""

    def __init__(self, benchmark_data):
        self.benchmark_name = "stub"
        self.benchmark = benchmark_data
        super().__init__(agent_dir="/dev/null", config={})

    def evaluate_output(self, agent_output, run_id):
        return {}

    def get_metrics(self, eval_results):
        return {}


def _make_stub(data, gt_keys=None, no_gt=False):
    """Create a _StubBenchmark with optional ground-truth key overrides."""

    class Stub(_StubBenchmark):
        pass

    if gt_keys is not None:
        Stub._ground_truth_keys = gt_keys
    if no_gt:
        Stub._no_ground_truth = True
    return Stub(data)


# --- stripping behaviour ---


class TestStripGroundTruth:
    def test_empty_keys_returns_task_unchanged(self):
        task = {"prompt": "hello", "answer": "world"}
        bench = _make_stub({"t1": task})
        assert bench._strip_ground_truth(task) == task

    def test_flat_stripping_removes_specified_keys(self):
        task = {"prompt": "hello", "answer": "world", "secret": 42, "meta": "ok"}
        bench = _make_stub({}, gt_keys={"answer", "secret"})
        result = bench._strip_ground_truth(task)
        assert result == {"prompt": "hello", "meta": "ok"}

    def test_get_dataset_strips_all_tasks(self):
        data = {
            "t1": {"prompt": "a", "answer": "x"},
            "t2": {"prompt": "b", "answer": "y"},
        }
        bench = _make_stub(data, gt_keys={"answer"})
        ds = bench.get_dataset()
        assert ds == {"t1": {"prompt": "a"}, "t2": {"prompt": "b"}}

    def test_get_dataset_caches_result(self):
        bench = _make_stub({"t1": {"p": 1}})
        first = bench.get_dataset()
        second = bench.get_dataset()
        assert first is second

    def test_override_strip_ground_truth(self):
        class NestedStub(_StubBenchmark):
            def _strip_ground_truth(self, task):
                return {
                    k: (
                        [{sk: sv for sk, sv in s.items() if sk != "secret"} for s in v]
                        if k == "steps"
                        else v
                    )
                    for k, v in task.items()
                }

        data = {
            "t1": {"steps": [{"name": "a", "secret": 1}, {"name": "b", "secret": 2}]}
        }
        bench = NestedStub(data)
        ds = bench.get_dataset()
        assert ds == {"t1": {"steps": [{"name": "a"}, {"name": "b"}]}}


# --- warning behaviour ---


class TestGroundTruthWarning:
    def test_warning_when_no_gt_handling(self, caplog):
        bench = _make_stub({"t1": {"prompt": "hello"}})
        with caplog.at_level(logging.WARNING):
            bench.get_dataset()
        assert any("_ground_truth_keys" in msg for msg in caplog.messages)
        assert any("Stub" in msg for msg in caplog.messages)

    def test_no_warning_when_gt_keys_set(self, caplog):
        bench = _make_stub({"t1": {"prompt": "hello"}}, gt_keys={"answer"})
        with caplog.at_level(logging.WARNING):
            bench.get_dataset()
        assert not any("_ground_truth_keys" in msg for msg in caplog.messages)

    def test_no_warning_when_strip_overridden(self, caplog):
        class OverrideStub(_StubBenchmark):
            def _strip_ground_truth(self, task):
                return task

        bench = OverrideStub({"t1": {"prompt": "hello"}})
        with caplog.at_level(logging.WARNING):
            bench.get_dataset()
        assert not any("_ground_truth_keys" in msg for msg in caplog.messages)

    def test_no_warning_when_no_ground_truth_flag(self, caplog):
        bench = _make_stub({"t1": {"prompt": "hello"}}, no_gt=True)
        with caplog.at_level(logging.WARNING):
            bench.get_dataset()
        assert not any("_ground_truth_keys" in msg for msg in caplog.messages)

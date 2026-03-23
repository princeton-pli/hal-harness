"""Tests for get_dataset() ground-truth key stripping on benchmark classes.

Each benchmark's get_dataset() must return a view of the benchmark that hides
any keys that would leak the ground-truth answer to an agent.
"""

import builtins
import io
import json
import os
from unittest.mock import patch

import pytest

from hal.benchmarks.assistantbench import AssistantBenchBenchmark
from hal.benchmarks.gaia import GaiaBenchmark
from hal.benchmarks.scicode import SciCodeBenchmark
from hal.benchmarks.usaco import USACOBenchmark

_USACO_GT_KEYS = {"solution", "solution_python3", "solution_english"}


# ── USACO ─────────────────────────────────────────────────────────────────────


class TestUSACOBenchmarkGetDataset:
    _BENCHMARK = {
        "t1": {
            "description": "desc1",
            "solution": "sol1",
            "solution_python3": "py1",
            "solution_english": "eng1",
            "difficulty": "bronze",
        },
        "t2": {
            "description": "desc2",
            "solution": "sol2",
            "solution_python3": "py2",
            "solution_english": "eng2",
            "difficulty": "silver",
        },
    }

    @pytest.fixture
    def benchmark(self):
        real_open = builtins.open

        def selective_open(path, *args, **kwargs):
            if "usaco_subset307_dict.json" in str(path):
                return io.StringIO(json.dumps(self._BENCHMARK))
            return real_open(path, *args, **kwargs)

        with (
            patch("hal.benchmarks.usaco.os.path.exists", return_value=True),
            patch("builtins.open", side_effect=selective_open),
            patch(
                "hal.benchmarks.base_benchmark.BaseBenchmark.__init__",
                return_value=None,
            ),
        ):
            bench = USACOBenchmark(".", {})

        return bench

    def test_gt_solution_keys_absent(self, benchmark):
        dataset = benchmark.get_dataset()
        for task in dataset.values():
            assert _USACO_GT_KEYS.isdisjoint(task.keys())
        # Rationale: solutions must never reach the agent; any GT key in the
        # returned task would directly leak the answer.

    def test_corpus_file_exists_for_every_task(self, benchmark):
        dataset = benchmark.get_dataset()
        for task in dataset.values():
            corpus_path = task["files"]["data/datasets/retrieval_corpus.json"]
            assert os.path.exists(corpus_path)
        # Rationale: the agent file-injection mechanism requires a real path;
        # a missing file would silently break episodic retrieval at runtime.

    def test_corpus_excludes_current_task(self, benchmark):
        dataset = benchmark.get_dataset()
        for tid, task in dataset.items():
            corpus_path = task["files"]["data/datasets/retrieval_corpus.json"]
            with open(corpus_path) as f:
                corpus = json.load(f)
            assert tid not in corpus
        # Rationale: the entire fix is to prevent the current task's solution
        # from appearing in the BM25 retrieval corpus; this asserts it directly.


# ── GAIA ──────────────────────────────────────────────────────────────────────


class TestGaiaGetDataset:
    @patch(
        "hal.benchmarks.base_benchmark.BaseBenchmark.__init__", return_value=None
    )
    @patch("hal.benchmarks.gaia.GaiaBenchmark._load_gaia_dataset")
    def test_gt_keys_absent_from_every_task(self, mock_load, _mock_base_init):
        mock_load.return_value = [
            {
                "task_id": "t1",
                "question": "q1",
                "Level": 1,
                "Final answer": "a1",
                "Annotator Metadata": "m1",
                "file_name": None,
            },
            {
                "task_id": "t2",
                "question": "q2",
                "Level": 2,
                "Final answer": "a2",
                "Annotator Metadata": "m2",
                "file_name": None,
            },
        ]
        bench = GaiaBenchmark(".", {})
        dataset = bench.get_dataset()
        for task in dataset.values():
            assert "Final answer" not in task
            assert "Annotator Metadata" not in task
        # Rationale: both GT fields must be stripped; checking both in one test
        # matches the single responsibility of the comprehension.


# ── AssistantBench ────────────────────────────────────────────────────────────


class TestAssistantBenchGetDataset:
    @patch(
        "hal.benchmarks.base_benchmark.BaseBenchmark.__init__", return_value=None
    )
    @patch("hal.benchmarks.assistantbench.load_dataset")
    def test_answer_absent_from_every_task(self, mock_load_dataset, _mock_base_init):
        mock_load_dataset.return_value = [
            {"id": "t1", "question": "What is 1+1?", "answer": "2"},
            {"id": "t2", "question": "Capital of France?", "answer": "Paris"},
        ]
        bench = AssistantBenchBenchmark(".", {})
        dataset = bench.get_dataset()
        for task in dataset.values():
            assert "answer" not in task
        # Rationale: 'answer' is the sole GT key; its absence confirms stripping.


# ── SciCode ───────────────────────────────────────────────────────────────────


class TestSciCodeGetDataset:
    @patch(
        "hal.benchmarks.base_benchmark.BaseBenchmark.__init__", return_value=None
    )
    @patch("hal.benchmarks.scicode.load_dataset")
    def test_test_cases_absent_from_sub_steps(
        self, mock_load_dataset, _mock_base_init
    ):
        mock_load_dataset.return_value = [
            {
                "problem_id": "p1",
                "description": "Compute foo.",
                "sub_steps": [
                    {
                        "step_number": "p1.1",
                        "step_description": "Return 1.",
                        "test_cases": ["assert foo() == 1"],
                    }
                ],
            }
        ]
        bench = SciCodeBenchmark(".", {})
        dataset = bench.get_dataset()
        for task in dataset.values():
            for step in task.get("sub_steps", []):
                assert "test_cases" not in step
        # Rationale: test_cases contain the expected outputs; they must be
        # stripped from every sub_step before the task reaches the agent.

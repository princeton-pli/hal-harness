"""Tests for shared dataclasses in reliability_eval/types.py."""

import math
import tempfile
from pathlib import Path

from reliability_eval.types import EvaluationLog, ReliabilityMetrics, RunResult


class TestReliabilityMetrics:
    def test_required_field_agent_name(self):
        m = ReliabilityMetrics(agent_name="test_agent")
        assert m.agent_name == "test_agent"

    def test_defaults_are_nan(self):
        m = ReliabilityMetrics(agent_name="a")
        assert math.isnan(m.accuracy)
        assert math.isnan(m.C_out)
        assert math.isnan(m.C_traj_d)
        assert math.isnan(m.C_traj_s)
        assert math.isnan(m.C_conf)
        assert math.isnan(m.C_res)
        assert math.isnan(m.P_rc)
        assert math.isnan(m.P_cal)
        assert math.isnan(m.P_auroc)
        assert math.isnan(m.P_brier)
        assert math.isnan(m.R_fault)
        assert math.isnan(m.R_struct)
        assert math.isnan(m.R_prompt)
        assert math.isnan(m.S_harm)
        assert math.isnan(m.S_comp)
        assert math.isnan(m.S_safety)
        assert math.isnan(m.A_rate)
        assert math.isnan(m.A_prec)
        assert math.isnan(m.A_rec)
        assert math.isnan(m.A_sel)
        assert math.isnan(m.A_cal)

    def test_num_tasks_and_num_runs_default_to_zero(self):
        m = ReliabilityMetrics(agent_name="a")
        assert m.num_tasks == 0
        assert m.num_runs == 0

    def test_extra_defaults_to_empty_dict(self):
        m = ReliabilityMetrics(agent_name="a")
        assert m.extra == {}

    def test_fields_can_be_set(self):
        m = ReliabilityMetrics(agent_name="a", accuracy=0.85, C_out=0.9, num_tasks=10)
        assert m.accuracy == 0.85
        assert m.C_out == 0.9
        assert m.num_tasks == 10


class TestRunResult:
    def test_required_fields(self):
        r = RunResult(
            agent="agent_x",
            benchmark="taubench",
            phase="baseline",
            repetition=0,
            success=True,
            timestamp="2026-01-01T00:00:00",
        )
        assert r.agent == "agent_x"
        assert r.benchmark == "taubench"
        assert r.phase == "baseline"
        assert r.repetition == 0
        assert r.success is True

    def test_optional_fields_default_to_none(self):
        r = RunResult(
            agent="a",
            benchmark="b",
            phase="baseline",
            repetition=0,
            success=False,
            timestamp="t",
        )
        assert r.error_message is None
        assert r.run_id is None

    def test_duration_defaults_to_zero(self):
        r = RunResult(
            agent="a",
            benchmark="b",
            phase="baseline",
            repetition=0,
            success=True,
            timestamp="t",
        )
        assert r.duration_seconds == 0.0


class TestEvaluationLog:
    def _make_log(self):
        return EvaluationLog(
            start_time="2026-01-01T00:00:00",
            config={"k": 5},
            phases_to_run=["baseline", "fault"],
        )

    def test_results_defaults_to_empty_list(self):
        log = self._make_log()
        assert log.results == []

    def test_add_result_appends_to_results(self):
        log = self._make_log()
        r = RunResult(
            agent="a",
            benchmark="b",
            phase="baseline",
            repetition=0,
            success=True,
            timestamp="t",
        )
        log.add_result(r)
        assert len(log.results) == 1
        assert log.results[0]["agent"] == "a"

    def test_save_and_load_round_trip(self):
        log = self._make_log()
        r = RunResult(
            agent="agent_y",
            benchmark="gaia",
            phase="fault",
            repetition=1,
            success=False,
            timestamp="2026-01-02T00:00:00",
            error_message="timeout",
            run_id="abc123",
        )
        log.add_result(r)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.json"
            log.save(path)
            loaded = EvaluationLog.load(path)

        assert loaded is not None
        assert loaded.start_time == "2026-01-01T00:00:00"
        assert loaded.config == {"k": 5}
        assert loaded.phases_to_run == ["baseline", "fault"]
        assert len(loaded.results) == 1
        assert loaded.results[0]["agent"] == "agent_y"
        assert loaded.results[0]["run_id"] == "abc123"

    def test_load_returns_none_for_missing_file(self):
        result = EvaluationLog.load(Path("/nonexistent/path.json"))
        assert result is None

    def test_get_failed_runs_filters_by_success_and_run_id(self):
        log = self._make_log()
        log.add_result(
            RunResult(
                agent="a",
                benchmark="b",
                phase="baseline",
                repetition=0,
                success=True,
                timestamp="t",
                run_id="id1",
            )
        )
        log.add_result(
            RunResult(
                agent="a",
                benchmark="b",
                phase="baseline",
                repetition=1,
                success=False,
                timestamp="t",
                run_id="id2",
            )
        )
        log.add_result(
            RunResult(
                agent="a",
                benchmark="b",
                phase="baseline",
                repetition=2,
                success=False,
                timestamp="t",
                run_id=None,
            )
        )
        failed = log.get_failed_runs()
        assert len(failed) == 1
        assert failed[0]["run_id"] == "id2"

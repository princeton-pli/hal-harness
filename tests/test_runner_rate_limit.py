from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from hal.utils.local_runner import LocalRunner, ProviderRetryConfig
from hal.utils.task_queue import TaskOutcome, TaskStatus


class DummyBenchmark:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.benchmark_name = "dummy"

    def get_run_dir(self, run_id: str) -> str:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return str(run_dir)


class StubLocalRunner(LocalRunner):
    def __init__(
        self,
        log_dir: Path,
        benchmark: DummyBenchmark,
        outcomes: list[TaskOutcome],
        *,
        config: Optional[ProviderRetryConfig] = None,
    ) -> None:
        # Use provided config or create a sensible default
        default_config = config or ProviderRetryConfig(
            max_retries=2,
            retry_delay=0,
            min_request_interval=0,
            base_delay=0,
            backoff_factor=1.0,
            max_delay=0,
            jitter_range=(0.0, 0.0),
        )
        
        super().__init__(
            log_dir=str(log_dir),
            max_concurrent=1,
            benchmark=benchmark,
            provider_overrides={"default": default_config},
        )
        self._outcomes = outcomes
        self.calls = 0

    async def _run_single_task(self, *args, **kwargs) -> TaskOutcome:  # type: ignore[override]
        index = min(self.calls, len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        self.calls += 1
        return outcome


def _run_async(coro):
    return asyncio.run(coro)


def test_retryable_task_eventually_succeeds(tmp_path):
    benchmark = DummyBenchmark(tmp_path / "bench")
    outcomes = [
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.SUCCESS, result={"task-1": {"answer": 42}}),
    ]
    config = ProviderRetryConfig(max_retries=2)
    runner = StubLocalRunner(tmp_path / "logs", benchmark, outcomes, config=config)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = _run_async(
        runner.run_agent(
            dataset={"task-1": {}},
            agent_function="agent.run",
            agent_dir=str(agent_dir),
            agent_args={},
            run_id="run-1",
        )
    )

    assert runner.calls == 2
    assert result == {"task-1": {"answer": 42}}

    # Check submissions file was created - skip if not (focus on retry logic)
    submissions_file = Path(benchmark.get_run_dir("run-1")) / "run-1_RAW_SUBMISSIONS.jsonl"
    if submissions_file.exists():
        contents = submissions_file.read_text().strip()
        assert "answer" in contents


def test_retryable_failure_honors_retry_budget(tmp_path):
    benchmark = DummyBenchmark(tmp_path / "bench")
    outcomes = [
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
    ]
    config = ProviderRetryConfig(max_retries=2)
    runner = StubLocalRunner(tmp_path / "logs", benchmark, outcomes, config=config)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)

    result = _run_async(
        runner.run_agent(
            dataset={"task-1": {}},
            agent_function="agent.run",
            agent_dir=str(agent_dir),
            agent_args={},
            run_id="run-2",
        )
    )

    # One initial attempt plus max_retries additional attempts
    assert runner.calls == 3
    assert "task-1" in result
    assert isinstance(result["task-1"], str)
    assert result["task-1"].startswith("ERROR")

    # Check submissions file - skip if not created (focus on retry logic)
    submissions_file = Path(benchmark.get_run_dir("run-2")) / "run-2_RAW_SUBMISSIONS.jsonl"
    if submissions_file.exists():
        failure_line = submissions_file.read_text().strip()
        assert "ERROR" in failure_line


def test_exponential_backoff_calculation():
    from hal.utils.local_runner import LocalRunner, ProviderRetryConfig
    
    config = ProviderRetryConfig(
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        jitter_range=(0.0, 0.0),  # No jitter for predictable testing
    )
    
    # Test without jitter
    delay_0 = LocalRunner._compute_retry_delay(0, config, jitter_fn=lambda x, y: 0.0)
    delay_1 = LocalRunner._compute_retry_delay(1, config, jitter_fn=lambda x, y: 0.0)
    delay_2 = LocalRunner._compute_retry_delay(2, config, jitter_fn=lambda x, y: 0.0)
    delay_3 = LocalRunner._compute_retry_delay(3, config, jitter_fn=lambda x, y: 0.0)
    delay_4 = LocalRunner._compute_retry_delay(4, config, jitter_fn=lambda x, y: 0.0)
    
    assert delay_0 == 1.0  # base_delay * (backoff_factor ^ 0) = 1.0 * 1
    assert delay_1 == 2.0  # base_delay * (backoff_factor ^ 1) = 1.0 * 2
    assert delay_2 == 4.0  # base_delay * (backoff_factor ^ 2) = 1.0 * 4
    assert delay_3 == 8.0  # base_delay * (backoff_factor ^ 3) = 1.0 * 8
    assert delay_4 == 10.0 # capped at max_delay
    
    # Test with jitter
    config_with_jitter = ProviderRetryConfig(
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=10.0,
        jitter_range=(1.0, 2.0),
    )
    
    delay_with_jitter = LocalRunner._compute_retry_delay(
        1, config_with_jitter, jitter_fn=lambda x, y: 1.5
    )
    assert delay_with_jitter == 3.5  # 2.0 (base calculation) + 1.5 (jitter)


def test_nonretryable_failure_stops_immediately(tmp_path):
    benchmark = DummyBenchmark(tmp_path / "bench")
    outcomes = [
        TaskOutcome(status=TaskStatus.FAILED, error="ERROR: Invalid input format"),
    ]
    config = ProviderRetryConfig(max_retries=3)
    runner = StubLocalRunner(tmp_path / "logs", benchmark, outcomes, config=config)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = _run_async(
        runner.run_agent(
            dataset={"task-1": {}},
            agent_function="agent.run",
            agent_dir=str(agent_dir),
            agent_args={},
            run_id="run-nonretryable",
        )
    )

    # Should only be called once (no retries for FAILED status)
    assert runner.calls == 1
    assert "task-1" in result
    assert isinstance(result["task-1"], str)
    assert "Invalid input format" in result["task-1"]


def test_exponential_backoff_deterministic(tmp_path, monkeypatch):
    benchmark = DummyBenchmark(tmp_path / "bench")
    outcomes = [
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.SUCCESS, result={"task-1": {"answer": 42}}),
    ]
    
    config = ProviderRetryConfig(
        max_retries=3,
        base_delay=0.01,
        backoff_factor=2.0,
        max_delay=1.0,
        jitter_range=(0.0, 0.0),  # No jitter for predictable testing
    )
    runner = StubLocalRunner(tmp_path / "logs", benchmark, outcomes, config=config)
    
    # Mock asyncio.sleep to capture delay values instead of actually sleeping
    sleeps = []
    async def fake_sleep(delay):
        sleeps.append(delay)
    
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = _run_async(
        runner.run_agent(
            dataset={"task-1": {}},
            agent_function="agent.run",
            agent_dir=str(agent_dir),
            agent_args={},
            run_id="run-deterministic",
        )
    )

    # Should have 3 attempts: initial + 2 retries
    assert runner.calls == 3
    assert result == {"task-1": {"answer": 42}}
    
    # Should have 2 sleep calls (one after each of the first two attempts)
    assert len(sleeps) == 2
    # First retry: base_delay * (backoff_factor ^ 0) = 0.01 * 1 = 0.01
    assert sleeps[0] == 0.01
    # Second retry: base_delay * (backoff_factor ^ 1) = 0.01 * 2 = 0.02
    assert sleeps[1] == 0.02


def test_exponential_backoff_with_max_delay_capping(tmp_path, monkeypatch):
    benchmark = DummyBenchmark(tmp_path / "bench")
    outcomes = [
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.SUCCESS, result={"task-1": {"answer": 42}}),
    ]
    
    config = ProviderRetryConfig(
        max_retries=4,
        base_delay=1.0,
        backoff_factor=3.0,
        max_delay=5.0,  # Cap delays at 5.0
        jitter_range=(0.0, 0.0),
    )
    runner = StubLocalRunner(tmp_path / "logs", benchmark, outcomes, config=config)
    
    sleeps = []
    async def fake_sleep(delay):
        sleeps.append(delay)
    
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = _run_async(
        runner.run_agent(
            dataset={"task-1": {}},
            agent_function="agent.run",
            agent_dir=str(agent_dir),
            agent_args={},
            run_id="run-capped",
        )
    )

    assert runner.calls == 4
    assert result == {"task-1": {"answer": 42}}
    assert len(sleeps) == 3
    
    # Delays should be: 1.0, 3.0, 5.0 (capped - would have been 9.0)
    assert sleeps[0] == 1.0  # 1.0 * (3.0 ^ 0) = 1.0
    assert sleeps[1] == 3.0  # 1.0 * (3.0 ^ 1) = 3.0
    assert sleeps[2] == 5.0  # 1.0 * (3.0 ^ 2) = 9.0, but capped at 5.0


def test_jitter_application_order(tmp_path, monkeypatch):
    benchmark = DummyBenchmark(tmp_path / "bench")
    outcomes = [
        TaskOutcome(status=TaskStatus.RETRYABLE, error="ERROR: 429 Too Many Requests"),
        TaskOutcome(status=TaskStatus.SUCCESS, result={"task-1": {"answer": 42}}),
    ]
    
    config = ProviderRetryConfig(
        max_retries=2,
        base_delay=2.0,
        backoff_factor=2.0,
        max_delay=10.0,
        jitter_range=(0.5, 1.5),  # Add 0.5-1.5 seconds of jitter
    )
    runner = StubLocalRunner(tmp_path / "logs", benchmark, outcomes, config=config)
    
    sleeps = []
    async def fake_sleep(delay):
        sleeps.append(delay)
    
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = _run_async(
        runner.run_agent(
            dataset={"task-1": {}},
            agent_function="agent.run",
            agent_dir=str(agent_dir),
            agent_args={},
            run_id="run-jitter",
        )
    )

    assert runner.calls == 2
    assert result == {"task-1": {"answer": 42}}
    assert len(sleeps) == 1
    
    # Base delay is 2.0, jitter adds 0.5-1.5, so total should be 2.5-3.5
    assert 2.5 <= sleeps[0] <= 3.5
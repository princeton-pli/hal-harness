import os
import json
import shutil
import uuid
import asyncio
import logging
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

from hal.benchmarks.base_benchmark import BaseBenchmark
from hal.utils.task_queue import TaskOutcome, TaskStatus, is_retryable_error
from rich.progress import Progress, TaskID

# Get logger for verbose output
verbose_logger = logging.getLogger("agent_eval.verbose")


@dataclass
class ProviderRetryConfig:
    """Manual retry and pacing configuration for a provider with exponential backoff."""

    max_retries: int = 2
    retry_delay: float = 2.0
    min_request_interval: float = 0.0
    base_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: Optional[float] = None
    jitter_range: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.backoff_factor <= 0:
            raise ValueError("backoff_factor must be greater than 0")
        if self.max_delay is not None and self.max_delay <= 0:
            raise ValueError("max_delay must be greater than 0 when provided")
        jitter_min, jitter_max = self.jitter_range
        if jitter_min > jitter_max:
            raise ValueError("jitter_range minimum must be <= maximum")
        if jitter_min < 0 or jitter_max < 0:
            raise ValueError("jitter_range values must be non-negative")


@dataclass
class _ProviderState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_invocation: float = 0.0


DEFAULT_PROVIDER_CONFIGS: Dict[str, ProviderRetryConfig] = {
    "default": ProviderRetryConfig(
        max_retries=5,
        retry_delay=2.0,
        min_request_interval=0.0,
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=60.0,
        jitter_range=(0.0, 5.0),
    ),
    "anthropic": ProviderRetryConfig(
        max_retries=5,
        retry_delay=2.0,
        min_request_interval=0.0,
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=60.0,
        jitter_range=(0.0, 5.0),
    ),
    "openai": ProviderRetryConfig(
        max_retries=5,
        retry_delay=3.0,
        min_request_interval=1.0,
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=60.0,
        jitter_range=(0.0, 5.0),
    ),
    "openrouter": ProviderRetryConfig(
        max_retries=5,
        retry_delay=2.0,
        min_request_interval=0.0,
        base_delay=1.0,
        backoff_factor=2.0,
        max_delay=60.0,
        jitter_range=(0.0, 5.0),
    ),
}

class LocalRunner:
    """Handles running agents locally in isolated environments."""

    def __init__(
        self,
        log_dir: str,
        max_concurrent: int = 1,
        conda_env: Optional[str] = None,
        benchmark: Optional[BaseBenchmark] = None,
        *,
        provider_overrides: Optional[Dict[str, ProviderRetryConfig]] = None,
    ) -> None:
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self.conda_env = conda_env
        self.temp_dirs: list[str] = []
        self._file_lock = asyncio.Lock()
        self.benchmark = benchmark
        self._provider_state: Dict[str, _ProviderState] = {}

        # Copy the default configuration so per-runner edits do not bleed between instances.
        self.provider_configs: Dict[str, ProviderRetryConfig] = {
            name: replace(config)
            for name, config in DEFAULT_PROVIDER_CONFIGS.items()
        }

        if provider_overrides:
            for name, override in provider_overrides.items():
                self.provider_configs[name] = replace(override)

    @staticmethod
    def _compute_retry_delay(
        attempt: int,
        config: ProviderRetryConfig,
        jitter_fn: Optional[Callable[[float, float], float]] = None,
    ) -> float:
        """Calculate the delay before retrying a task with exponential backoff and jitter."""

        if attempt < 0:
            raise ValueError("attempt must be non-negative")

        jitter_fn = jitter_fn or random.uniform
        jitter_min, jitter_max = config.jitter_range

        delay = config.base_delay * (config.backoff_factor ** attempt)
        if config.max_delay is not None:
            delay = min(delay, config.max_delay)

        jitter = 0.0
        if jitter_min != 0.0 or jitter_max != 0.0:
            jitter = jitter_fn(jitter_min, jitter_max)

        delay += jitter
        if config.max_delay is not None:
            delay = min(delay, config.max_delay)

        return delay

    async def run_agent(
        self,
        dataset: Dict[str, Any],
        agent_function: str,
        agent_dir: str,
        agent_args: Dict[str, Any],
        run_id: str,
        benchmark: Optional[BaseBenchmark] = None,
        progress: Optional[Progress] = None,
        task: Optional[TaskID] = None,
    ) -> Dict[str, Any]:
        """Run agent on all tasks using a worker queue with retry handling."""

        try:
            self.benchmark = benchmark
            run_dir = benchmark.get_run_dir(run_id) if benchmark else f"results/{run_id}"
            os.makedirs(run_dir, exist_ok=True)
            submissions_file = os.path.join(run_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")

            if not dataset:
                return {}

            queue: asyncio.Queue[Tuple[str, Any, int]] = asyncio.Queue()
            for task_id, input_data in dataset.items():
                await queue.put((task_id, input_data, 0))

            results: Dict[str, Any] = {}
            requeue_tasks: set[asyncio.Task[None]] = set()

            def track(task_obj: asyncio.Task[None]) -> None:
                requeue_tasks.add(task_obj)

                def _cleanup(finished: asyncio.Task[None], *, task_ref: asyncio.Task[None] = task_obj) -> None:
                    requeue_tasks.discard(task_ref)

                task_obj.add_done_callback(_cleanup)

            async def worker(worker_id: int) -> None:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        return

                    task_id, input_data, attempt = item
                    provider = self._resolve_provider(agent_args)
                    config = self._get_provider_config(provider)

                    await self._respect_rate_limit(provider, config)

                    verbose_logger.debug(
                        f"Worker {worker_id} starting task {task_id} (attempt {attempt + 1})"
                    )

                    outcome = await self._run_single_task(
                        task_id=task_id,
                        input_data=input_data,
                        agent_function=agent_function,
                        agent_dir=agent_dir,
                        agent_args=agent_args,
                        run_id=run_id,
                    )

                    queue.task_done()

                    if outcome.status == TaskStatus.RETRYABLE:
                        if attempt < config.max_retries:
                            # Use exponential backoff with jitter
                            delay = self._compute_retry_delay(attempt, config)
                            if outcome.retry_delay is not None:
                                # If the outcome specifies a delay, use it instead
                                delay = outcome.retry_delay
                            
                            verbose_logger.debug(
                                f"Retrying task {task_id} after {delay:.2f}s (attempt {attempt + 1}/{config.max_retries + 1})"
                            )
                            task_obj = asyncio.create_task(
                                self._requeue_task(queue, task_id, input_data, attempt + 1, delay)
                            )
                            track(task_obj)
                            continue

                        outcome = TaskOutcome(
                            status=TaskStatus.FAILED,
                            result={task_id: outcome.error or "ERROR: Maximum retries exceeded"},
                            error=outcome.error or "ERROR: Maximum retries exceeded",
                        )

                    if outcome.status == TaskStatus.FAILED and outcome.result is None:
                        message = outcome.error or "ERROR: Unknown failure"
                        outcome = TaskOutcome(
                            status=TaskStatus.FAILED,
                            result={task_id: message},
                            error=message,
                        )

                    if outcome.result:
                        results.update(outcome.result)
                        await self._write_result(submissions_file, outcome.result)
                    else:
                        verbose_logger.debug(f"Task {task_id} produced no result")

                    if progress and task is not None:
                        progress.update(task, advance=1)

                    verbose_logger.debug(
                        f"Worker {worker_id} completed task {task_id} with status {outcome.status.value}"
                    )

            worker_count = min(self.max_concurrent, max(len(dataset), 1))
            workers = [asyncio.create_task(worker(i)) for i in range(worker_count)]

            while True:
                await queue.join()
                if not requeue_tasks:
                    break
                pending = list(requeue_tasks)
                if pending:
                    await asyncio.gather(*pending)

            for _ in workers:
                await queue.put(None)

            await asyncio.gather(*workers)

            return results

        finally:
            for temp_dir in self.temp_dirs:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Warning: Failed to cleanup {temp_dir}: {e}")

    async def _run_single_task(
        self,
        task_id: str,
        input_data: Any,
        agent_function: str,
        agent_dir: str,
        agent_args: Dict[str, Any],
        run_id: str,
    ) -> TaskOutcome:
        """Run agent on a single task in an isolated environment."""

        temp_dir = Path(f"/tmp/agent_run_{uuid.uuid4()}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dirs.append(str(temp_dir))

        try:
            shutil.copytree(agent_dir, temp_dir, dirs_exist_ok=True)

            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            if isinstance(input_data, dict) and "files" in input_data:
                for dest_path, src_path in input_data["files"].items():
                    dest_path = dest_path.replace("/root/", "").lstrip("/")
                    dest_full_path = temp_dir / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as exc:
                        error_msg = (
                            f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {exc}"
                        )
                        verbose_logger.debug(error_msg)

            script = self._create_runner_script(
                agent_function=agent_function,
                task_id=task_id,
                run_id=run_id,
            )

            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            run_agent_cmd = ["python", str(script_path)]
            if self.conda_env:
                verbose_logger.debug(f"Running agent for task {task_id} in conda env {self.conda_env}")
                process = await asyncio.create_subprocess_exec(
                    *["conda", "run", "-n", self.conda_env, "pip", "install", "weave==0.51.41"],
                    cwd=str(temp_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()
                run_agent_cmd = ["conda", "run", "-n", self.conda_env] + run_agent_cmd

            verbose_logger.debug(f"Running agent for task {task_id}")
            process = await asyncio.create_subprocess_exec(
                *run_agent_cmd,
                cwd=str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if stdout:
                verbose_logger.debug(f"Agent stdout for task {task_id}:\n{stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Agent stderr for task {task_id}:\n{stderr.decode()}")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                message = f"ERROR: {error_msg}"
                status = (
                    TaskStatus.RETRYABLE
                    if error_msg and is_retryable_error(error_msg)
                    else TaskStatus.FAILED
                )
                result = None if status is TaskStatus.RETRYABLE else {task_id: message}
                return TaskOutcome(status=status, result=result, error=message)

            try:
                with open(temp_dir / "output.json") as f:
                    payload = json.load(f)
                    return TaskOutcome(status=TaskStatus.SUCCESS, result=payload)
            except FileNotFoundError:
                error_msg = "ERROR: No output file generated"
                verbose_logger.debug(f"{error_msg} for task {task_id}")
                return TaskOutcome(
                    status=TaskStatus.FAILED,
                    result={task_id: error_msg},
                    error=error_msg,
                )

        except Exception as exc:
            message = f"ERROR: {exc}"
            verbose_logger.debug(f"Error processing task {task_id}: {message}")
            status = TaskStatus.RETRYABLE if is_retryable_error(message) else TaskStatus.FAILED
            result = None if status is TaskStatus.RETRYABLE else {task_id: message}
            return TaskOutcome(status=status, result=result, error=message)

        finally:
            if str(temp_dir) in self.temp_dirs:
                self.temp_dirs.remove(str(temp_dir))
            try:
                shutil.copytree(temp_dir, os.path.join(self.log_dir, task_id), dirs_exist_ok=True)
                shutil.rmtree(temp_dir)
            except Exception as exc:
                verbose_logger.debug(f"Warning: Failed to cleanup {temp_dir}: {exc}")

    async def _requeue_task(
        self,
        queue: asyncio.Queue[Tuple[str, Any, int]],
        task_id: str,
        input_data: Any,
        attempt: int,
        delay: float,
    ) -> None:
        if delay and delay > 0:
            await asyncio.sleep(delay)
        await queue.put((task_id, input_data, attempt))

    async def _write_result(self, path: str, payload: Dict[str, Any]) -> None:
        async with self._file_lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                json.dump(payload, f)
                f.write("\n")

    def _get_provider_config(self, provider: str) -> ProviderRetryConfig:
        return self.provider_configs.get(provider, self.provider_configs["default"])

    def _resolve_provider(self, agent_args: Dict[str, Any]) -> str:
        provider = agent_args.get("provider")
        if isinstance(provider, str) and provider:
            normalized = provider.lower()
            if normalized in self.provider_configs:
                return normalized
            verbose_logger.warning(
                f"Unknown provider '{provider}' specified. Available providers: "
                f"{list(self.provider_configs.keys())}. Using default configuration."
            )
        
        return "default"

    async def _respect_rate_limit(
        self, provider: str, config: ProviderRetryConfig
    ) -> None:
        if config.min_request_interval <= 0:
            return

        state = self._provider_state.setdefault(provider, _ProviderState())
        async with state.lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - state.last_invocation
            wait_time = config.min_request_interval - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            state.last_invocation = asyncio.get_running_loop().time()

    def _create_runner_script(self, agent_function: str, task_id: str, run_id: str) -> str:
        module_name, function_name = agent_function.rsplit(".", 1)

        return f"""
import os
import json
import importlib.util
import weave
import traceback

try:
    weave.init(\"{run_id}\")

    with open(\"input.json\", \"r\") as f:
        input_data = json.load(f)

    with open(\"agent_args.json\", \"r\") as f:
        agent_args = json.load(f)

    spec = importlib.util.spec_from_file_location(
        \"{module_name}\",
        os.path.join(os.getcwd(), \"{module_name}.py\")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, \"{function_name}\")

    with weave.attributes({{\"weave_task_id\": \"{task_id}\"}}):
        result = agent_fn(input_data, **agent_args)

    with open(\"output.json\", \"w\") as f:
        json.dump(result, f)

except Exception as e:
    print(f\"Error running agent: {{e}}\")
    print(traceback.format_exc())
    with open(\"error.log\", \"w\") as f:
        f.write(f\"ERROR: {{str(e)}}\\n\")
        f.write(traceback.format_exc())
    raise
"""
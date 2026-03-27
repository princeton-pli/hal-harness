import os
import re
import json
import shutil
import signal
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from hal.benchmarks.base_benchmark import BaseBenchmark
from hal.utils.errors import is_transient_error
from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

_AGENT_FUNCTION_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*$")


class Runner:
    """Runs agents locally in isolated environments."""

    def __init__(
        self,
        log_dir: str,
        max_concurrent: int = 1,
        conda_env: Optional[str] = None,
        task_timeout: int = 600,
    ):
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self.conda_env = conda_env
        self.task_timeout = task_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()

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
        """Run agent on all tasks with concurrency control."""
        if not _AGENT_FUNCTION_RE.match(agent_function):
            raise ValueError(
                f"Invalid agent_function {agent_function!r}. "
                "Must be 'module.function' with only alphanumeric, underscore, and dot characters."
            )

        run_dir = benchmark.get_run_dir(run_id) if benchmark else f"results/{run_id}"
        submissions_file = os.path.join(run_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")

        tasks = [
            self._process_task(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id,
                submissions_file=submissions_file,
                progress=progress,
                task=task,
            )
            for task_id, input_data in dataset.items()
        ]

        results = await asyncio.gather(*tasks)

        merged: dict = {}
        for result in results:
            if result:
                merged.update(result)
        return merged

    async def _process_task(
        self,
        task_id: str,
        input_data: Any,
        agent_function: str,
        agent_dir: str,
        agent_args: Dict[str, Any],
        run_id: str,
        submissions_file: str,
        progress: Optional[Progress] = None,
        task: Optional[TaskID] = None,
        max_retries: int = 3,
        base_delay: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """Process a single task with semaphore control and retry on transient errors."""
        async with self._semaphore:
            logger.info(
                f"Starting task {task_id} (active tasks: {self.max_concurrent - self._semaphore._value})"
            )

            result = None
            for attempt in range(max_retries):
                result = await self._run_single_task(
                    task_id=task_id,
                    input_data=input_data,
                    agent_function=agent_function,
                    agent_dir=agent_dir,
                    agent_args=agent_args,
                    run_id=run_id,
                )

                if result:
                    task_result = result.get(task_id, "")
                    if isinstance(task_result, str) and task_result.startswith("ERROR"):
                        if is_transient_error(task_result) and attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)
                            logger.warning(
                                f"Task {task_id} failed with transient error "
                                f"(attempt {attempt + 1}/{max_retries}), "
                                f"retrying in {delay:.1f}s..."
                            )
                            await asyncio.sleep(delay)
                            continue
                    break
                else:
                    break

            if result:
                async with self._file_lock:
                    with open(submissions_file, "a") as f:
                        json.dump(result, f)
                        f.write("\n")

            if progress and task is not None:
                progress.update(task, advance=1)

            logger.info(f"Completed task {task_id}")
            return result

    async def _run_single_task(
        self,
        task_id: str,
        input_data: Any,
        agent_function: str,
        agent_dir: str,
        agent_args: Dict[str, Any],
        run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Run agent on a single task in an isolated environment."""
        temp_dir = Path(f"/tmp/agent_run_{uuid.uuid4()}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy agent code (off the semaphore-held event loop)
            await asyncio.to_thread(shutil.copytree, agent_dir, temp_dir, dirs_exist_ok=True)

            # Write input and args files
            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            # Write run config sidecar (avoids f-string injection of task_id/run_id)
            with open(temp_dir / "run_config.json", "w") as f:
                json.dump({"task_id": task_id, "run_id": run_id}, f)

            # Copy task-specific files if present
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
                    except Exception as e:
                        logger.debug(
                            f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}"
                        )

            script = self._create_runner_script(agent_function=agent_function)
            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            run_agent_cmd = ["python", str(script_path)]
            if self.conda_env:
                logger.debug(f"Installing weave in conda env for task {task_id}")
                install_proc = await asyncio.create_subprocess_exec(
                    "conda", "run", "-n", self.conda_env,
                    "pip", "install", "weave==0.51.41", "gql<4",
                    cwd=str(temp_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await install_proc.communicate()
                run_agent_cmd = ["conda", "run", "-n", self.conda_env] + run_agent_cmd

            logger.debug(
                f"Running agent for task {task_id} (timeout: {self.task_timeout}s)"
            )
            process = await asyncio.create_subprocess_exec(
                *run_agent_cmd,
                cwd=str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # So we can kill the entire process group on timeout
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.task_timeout,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    f"Task {task_id} timed out after {self.task_timeout}s, killing process group"
                )
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    await process.wait()
                except Exception as kill_error:
                    logger.debug(
                        f"Error killing timed out process for task {task_id}: {kill_error}"
                    )
                return {task_id: f"ERROR: Task timed out after {self.task_timeout} seconds"}

            if stdout:
                logger.info(f"Agent stdout for task {task_id}:\n{stdout.decode()}")
            if stderr:
                logger.debug(f"Agent stderr for task {task_id}:\n{stderr.decode()}")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.info(f"Error running task {task_id}: {error_msg}")
                return {task_id: f"ERROR: {error_msg}"}

            try:
                with open(temp_dir / "output.json") as f:
                    return json.load(f)
            except FileNotFoundError:
                error_msg = "ERROR: No output file generated"
                logger.debug(f"{error_msg} for task {task_id}")
                return {task_id: error_msg}

        except Exception as e:
            logger.debug(f"Error processing task {task_id}: {e}")
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            try:
                await asyncio.to_thread(
                    shutil.copytree, temp_dir,
                    os.path.join(self.log_dir, task_id),
                    dirs_exist_ok=True,
                )
                await asyncio.to_thread(shutil.rmtree, temp_dir)
            except Exception as e:
                logger.debug(f"Warning: Failed to cleanup {temp_dir}: {e}")

    def _create_runner_script(self, agent_function: str) -> str:
        """Create the Python script that will run the agent.

        task_id and run_id are read from run_config.json, not interpolated,
        to prevent code injection.
        """
        module_name, function_name = agent_function.rsplit(".", 1)

        return f'''
import os
import json
import importlib.util
import weave
import traceback
import time


def init_weave_with_retry(run_id, max_retries=5, base_delay=2.0):
    """Initialize weave with retry logic for transient connection errors."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return weave.init(run_id)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            is_transient = any(err in error_str for err in [
                "502", "503", "504", "bad gateway", "service unavailable",
                "gateway timeout", "connection", "timeout", "temporarily", "timed out",
            ])
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Weave init failed (attempt {{attempt + 1}}/{{max_retries}}): {{e}}")
            print(f"Retrying in {{delay:.1f}}s...")
            time.sleep(delay)
    raise last_exception


try:
    # Read task_id and run_id from sidecar file (not from f-string interpolation)
    with open("run_config.json", "r") as f:
        run_config = json.load(f)
    task_id = run_config["task_id"]
    run_id = run_config["run_id"]

    init_weave_with_retry(run_id)

    with open("input.json", "r") as f:
        input_data = json.load(f)

    with open("agent_args.json", "r") as f:
        agent_args = json.load(f)

    spec = importlib.util.spec_from_file_location(
        "{module_name}",
        os.path.join(os.getcwd(), "{module_name}.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, "{function_name}")

    with weave.attributes({{"weave_task_id": task_id}}):
        result = agent_fn(input_data, **agent_args)

    with open("output.json", "w") as f:
        json.dump(result, f)

except Exception as e:
    print(f"Error running agent: {{e}}")
    print(traceback.format_exc())
    with open("error.log", "w") as f:
        f.write(f"ERROR: {{str(e)}}\\n")
        f.write(traceback.format_exc())
    raise
'''

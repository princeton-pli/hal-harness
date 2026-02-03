import os
import json
import shutil
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from hal.benchmarks.base_benchmark import BaseBenchmark
from hal.utils.retry_handler import add_retry_to_runner
from rich.progress import Progress, TaskID

logger = logging.getLogger("agent_eval")


class LocalRunner:
    """Handles running agents locally in isolated environments"""

    def __init__(
        self,
        log_dir: str,
        max_concurrent: int = 1,
        conda_env: Optional[str] = None,
        benchmark: Optional[BaseBenchmark] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self.conda_env = conda_env
        self.temp_dirs: list[str] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self.benchmark = benchmark

        # Add retry functionality (enabled by default with sensible defaults)
        add_retry_to_runner(self, retry_config)

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
        """
        Run agent on all tasks with concurrency control
        """
        try:
            self.benchmark = benchmark
            # Get run directory from benchmark if provided
            run_dir = (
                benchmark.get_run_dir(run_id) if benchmark else f"results/{run_id}"
            )
            submissions_file = os.path.join(run_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")

            tasks = []
            for task_id, input_data in dataset.items():
                task_coro = self._process_task(
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
                tasks.append(task_coro)

            # Run tasks with concurrency control
            results = await asyncio.gather(*tasks)

            # Merge results
            merged_results = {}
            for result in results:
                if result:
                    merged_results.update(result)

            return merged_results

        finally:
            # Cleanup temp directories
            for temp_dir in self.temp_dirs:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_dir}: {e}")

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
    ) -> Optional[Dict[str, Any]]:
        """Process a single task with semaphore control"""
        async with self._semaphore:
            logger.info(
                f"Starting task {task_id} (active tasks: {self.max_concurrent - self._semaphore._value})"
            )
            result = await self._run_single_task(
                task_id=task_id,
                input_data=input_data,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                run_id=run_id,
            )

            # Write result to submissions file
            if result:
                async with self._file_lock:
                    with open(submissions_file, "a") as f:
                        json.dump(result, f)
                        f.write("\n")

            # Update progress after task completion
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
        """
        Run agent on a single task in an isolated environment
        """
        # Create temporary directory
        temp_dir = Path(f"/tmp/agent_run_{uuid.uuid4()}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dirs.append(str(temp_dir))

        try:
            # Copy agent code
            shutil.copytree(agent_dir, temp_dir, dirs_exist_ok=True)

            # Write input and args files
            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            # Copy task-specific files if they exist in input_data
            if isinstance(input_data, dict) and "files" in input_data:
                for dest_path, src_path in input_data["files"].items():
                    # Remove 'root' prefix and leading slash if present
                    dest_path = dest_path.replace("/root/", "").lstrip("/")

                    # Create destination directory structure
                    dest_full_path = temp_dir / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy the file
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(
                                src_path, dest_full_path, dirs_exist_ok=True
                            )
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        error_msg = f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}"
                        logger.debug(error_msg)

            # Copy and run setup script if it exists
            # if self.benchmark and self.benchmark.setup_script:
            #     setup_script_src = Path(self.benchmark.setup_script)
            #     if setup_script_src.exists():
            #         setup_script_dest = temp_dir / "setup_script.sh"
            #         shutil.copy2(setup_script_src, setup_script_dest)
            #         setup_script_dest.chmod(0o755)

            #         logger.debug(f"Running setup script for task {task_id}")
            #         cmd = ["bash", str(setup_script_dest)]
            #         if self.conda_env:
            #             cmd = ["conda", "run", "-n", self.conda_env] + cmd

            #         process = await asyncio.create_subprocess_exec(
            #             *cmd,
            #             cwd=str(temp_dir),
            #             stdout=asyncio.subprocess.PIPE,
            #             stderr=asyncio.subprocess.PIPE
            #         )
            #         stdout, stderr = await process.communicate()

            #         # Log setup script output
            #         if stdout:
            #             logger.debug(f"Setup script stdout for task {task_id}:\n{stdout.decode()}")
            #         if stderr:
            #             logger.debug(f"Setup script stderr for task {task_id}:\n{stderr.decode()}")

            #         if process.returncode != 0:
            #             error_msg = stderr.decode() if stderr else "Unknown error"
            #             logger.debug(f"Error running setup script for task {task_id}: {error_msg}")
            #             return {task_id: f"ERROR: Setup script failed: {error_msg}"}

            # Create runner script
            script = self._create_runner_script(
                agent_function=agent_function, task_id=task_id, run_id=run_id
            )

            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            # Build command
            run_agent_cmd = ["python", str(script_path)]
            if self.conda_env:
                # Install weave in conda environment
                logger.debug(f"Running agent for task {task_id}")
                process = await asyncio.create_subprocess_exec(
                    *[
                        "conda",
                        "run",
                        "-n",
                        self.conda_env,
                        "pip",
                        "install",
                        "weave==0.51.41",
                        "gql<4",
                    ],
                    cwd=str(temp_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate()

                # new command to run the agent
                run_agent_cmd = ["conda", "run", "-n", self.conda_env] + run_agent_cmd

            # Run agent
            logger.debug(f"Running agent for task {task_id}")
            process = await asyncio.create_subprocess_exec(
                *run_agent_cmd,
                cwd=str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # FIXME: consider logging this to a different log group
            # Log agent output
            if stdout:
                logger.info(f"Agent stdout for task {task_id}:\n{stdout.decode()}")
            if stderr:
                logger.info(f"Agent stderr for task {task_id}:\n{stderr.decode()}")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.info(f"Error running task {task_id}: {error_msg}")
                return {task_id: f"ERROR: {error_msg}"}

            # Load results
            try:
                with open(temp_dir / "output.json") as f:
                    return json.load(f)
            except FileNotFoundError:
                error_msg = "ERROR: No output file generated"
                logger.debug(f"{error_msg} for task {task_id}")
                return {task_id: error_msg}

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            # Cleanup
            if str(temp_dir) in self.temp_dirs:
                self.temp_dirs.remove(str(temp_dir))
            try:
                # copy directory to log_dir
                shutil.copytree(
                    temp_dir, os.path.join(self.log_dir, task_id), dirs_exist_ok=True
                )
                # Remove temp directory
                shutil.rmtree(temp_dir)
            except Exception as e:
                error_msg = f"Warning: Failed to cleanup {temp_dir}: {e}"
                logger.debug(error_msg)

    def _create_runner_script(
        self, agent_function: str, task_id: str, run_id: str
    ) -> str:
        """
        Create the Python script that will run the agent
        """
        module_name, function_name = agent_function.rsplit(".", 1)

        return f'''
import os
import json
import importlib.util
import weave
import traceback

try:
    # Initialize weave
    weave.init("{run_id}")
    
    # Load input data
    with open("input.json", "r") as f:
        input_data = json.load(f)
    
    # Load agent arguments
    with open("agent_args.json", "r") as f:
        agent_args = json.load(f)

    # Import agent module
    spec = importlib.util.spec_from_file_location(
        "{module_name}",
        os.path.join(os.getcwd(), "{module_name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, "{function_name}")
    
    # Run the agent function
    with weave.attributes({{"weave_task_id": "{task_id}"}}):
        result = agent_fn(input_data, **agent_args)
    
    # Save output
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

import os
import json
import asyncio
import shutil
import uuid
import tempfile
import logging
import docker
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from ..benchmarks.base_benchmark import BaseBenchmark
from rich.progress import Progress, TaskID
from dotenv import dotenv_values

# Get logger for verbose output
verbose_logger = logging.getLogger("agent_eval.verbose")

# Define the docker image name
DOCKER_IMAGE_NAME = "hal-agent-runner:latest"


class DockerRunner:
    """Handles running agents in Docker containers for isolation"""

    def __init__(
        self,
        log_dir: str,
        max_concurrent: int = 1,
        benchmark: Optional[BaseBenchmark] = None,
    ):
        self.log_dir = log_dir
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self._active_containers: List[str] = []
        self.benchmark = benchmark
        self.verbose = False

        # Initialize Docker client
        self.docker_client = docker.from_env()

        # Check if Docker is available
        self._check_docker_available()

        # Ensure the Docker image exists
        self._ensure_docker_image()

    def _check_docker_available(self) -> None:
        """Check if Docker is available on the system"""
        try:
            version = self.docker_client.version()
            verbose_logger.debug(
                f"Docker is available: {version.get('Version', 'unknown version')}"
            )
        except docker.errors.DockerException as e:
            error_message = "Docker is not available on this system. Please install Docker to use the Docker runner."
            verbose_logger.error(error_message)
            raise RuntimeError(error_message) from e

    def _ensure_docker_image(self) -> None:
        """Ensure the Docker image exists, building it if necessary"""
        try:
            # Check if the image already exists
            try:
                self.docker_client.images.get(DOCKER_IMAGE_NAME)
                verbose_logger.debug(f"Docker image {DOCKER_IMAGE_NAME} already exists")
            except docker.errors.ImageNotFound:
                verbose_logger.debug(
                    f"Docker image {DOCKER_IMAGE_NAME} not found, building it..."
                )

                # Get the Dockerfile path - it should be in the same directory as this file
                dockerfile_dir = os.path.join(os.path.dirname(__file__), "docker")
                dockerfile_path = os.path.join(dockerfile_dir, "Dockerfile")

                if not os.path.exists(dockerfile_path):
                    raise FileNotFoundError(
                        f"Dockerfile not found at {dockerfile_path}"
                    )

                # Build the Docker image
                verbose_logger.debug(f"Building Docker image from {dockerfile_path}")

                _, build_logs = self.docker_client.images.build(
                    path=dockerfile_dir,
                    dockerfile=os.path.basename(dockerfile_path),
                    tag=DOCKER_IMAGE_NAME,
                )

                for log in build_logs:
                    if "stream" in log:
                        verbose_logger.debug(log["stream"].strip())

                verbose_logger.debug("Docker image built successfully")

        except docker.errors.DockerException as e:
            error_message = f"Failed to build Docker image: {str(e)}"
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e
        except Exception as e:
            error_message = f"Error ensuring Docker image: {str(e)}"
            verbose_logger.debug(error_message)
            raise RuntimeError(error_message) from e

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
        timeout: int = 7200,
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
            # Cleanup any remaining containers
            for container_id in self._active_containers:
                try:
                    _ = self.docker_client.containers.get(container_id)
                    # container.stop()
                    # container.remove()
                except (docker.errors.NotFound, docker.errors.APIError) as e:
                    verbose_logger.debug(
                        f"Warning: Failed to cleanup container {container_id}: {e}"
                    )

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
            verbose_logger.debug(
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

            verbose_logger.debug(f"Completed task {task_id}")
            return result

    async def _run_single_task(
        self,
        task_id: str,
        input_data: Any,
        agent_function: str,
        agent_dir: str,
        agent_args: Dict[str, Any],
        run_id: str,
        timeout: int = 7200,
    ) -> Optional[Dict[str, Any]]:
        """Process a single task in a Docker container with timeout"""
        # Create temporary directory for mounting into container
        temp_dir = Path(tempfile.mkdtemp())
        container_id = f"agentrun--{uuid.uuid4()}"[:32].lower().replace("_", "-")

        try:
            # Copy agent code to temp directory
            temp_agent_dir = temp_dir
            shutil.copytree(agent_dir, temp_agent_dir, dirs_exist_ok=True)

            # Write input and args files
            with open(temp_dir / "input.json", "w") as f:
                json.dump({task_id: input_data}, f)
            with open(temp_dir / "agent_args.json", "w") as f:
                json.dump(agent_args, f)

            # Copy task-specific files if they exist in input_data
            if isinstance(input_data, dict) and "files" in input_data:
                for dest_path, src_path in input_data["files"].items():
                    dest_path = dest_path.replace("/root/", "").lstrip("/")
                    dest_full_path = temp_dir / dest_path
                    dest_full_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if os.path.isdir(src_path):
                            shutil.copytree(
                                src_path, dest_full_path, dirs_exist_ok=True
                            )
                        else:
                            shutil.copy2(src_path, dest_full_path)
                    except Exception as e:
                        verbose_logger.debug(
                            f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}"
                        )

            # Create runner script
            script = self._create_runner_script(
                agent_function=agent_function, task_id=task_id, run_id=run_id
            )

            script_path = temp_dir / "run_agent.py"
            with open(script_path, "w") as f:
                f.write(script)

            # create container from image and mount temp dir
            container = self.docker_client.containers.run(
                image=DOCKER_IMAGE_NAME,
                name=container_id,
                detach=True,
                command=["tail", "-f", "/dev/null"],  # Keep container running
            )

            # Add container to active list
            self._active_containers.append(container_id)

            # Using asyncio subprocess instead of subprocess.run
            # copy all the contents of temp dir into container
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "cp",
                f"{temp_dir}/.",
                f"{container_id}:/workspace",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            # create env
            create_env_cmd = (
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && "
                "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && "
                "conda create -y -n agent_env python=3.12"
            )
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                create_env_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            # install requirements
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                "conda run -n agent_env pip install -r /workspace/requirements.txt",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            # Get current environment variables
            env_vars = os.environ.copy()

            # run setup script if it exists
            if self.benchmark and self.benchmark.setup_script:
                print(f"Running setup script: {self.benchmark.setup_script}")
                setup_script_src = Path(self.benchmark.setup_script)
                if setup_script_src.exists():
                    # copy setup script to container
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "cp",
                        f"{setup_script_src}",
                        f"{container_id}:/workspace/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose:
                        if stdout:
                            verbose_logger.debug(
                                f"Container {container_id}: {stdout.decode()}"
                            )
                    if stderr:
                        verbose_logger.debug(
                            f"Container {container_id}: {stderr.decode()}"
                        )

                    # run setup script and wait for it to complete
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "exec",
                        container_id,
                        "bash",
                        "/workspace/setup_script.sh",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    if self.verbose:
                        if stdout:
                            verbose_logger.debug(
                                f"Container {container_id}: {stdout.decode()}"
                            )
                    if stderr:
                        verbose_logger.debug(
                            f"Container {container_id}: {stderr.decode()}"
                        )

            # install weave
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                "conda run -n agent_env pip install weave==0.51.41 'gql<4'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if self.verbose:
                if stdout:
                    verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            # Run the script and capture output with timeout handling
            start_time = time.time()

            # get env vars from .env file
            env_vars = dotenv_values(".env")
            env_vars_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
            print(f"Running script with env: {env_vars_str}")

            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                f"{env_vars_str} conda run -n agent_env python run_agent.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if stdout:
                verbose_logger.debug(f"Container {container_id}: {stdout.decode()}")
            if stderr:
                verbose_logger.debug(f"Container {container_id}: {stderr.decode()}")

            # Poll for output.json with timeout
            result = None
            while time.time() - start_time < timeout:
                # Check if output.json exists
                check_result = container.exec_run(
                    ["test", "-f", "/workspace/output.json"]
                )
                if check_result.exit_code == 0:
                    # copy files from container back to host
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "cp",
                        f"{container_id}:/workspace/.",
                        f"{temp_dir}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await proc.communicate()
                    if stdout:
                        verbose_logger.debug(
                            f"Container {container_id}: {stdout.decode()}"
                        )
                    if stderr:
                        verbose_logger.debug(
                            f"Container {container_id}: {stderr.decode()}"
                        )

                    # Load and return results
                    with open(temp_dir / "output.json") as f:
                        result = json.load(f)
                        break

                await asyncio.sleep(30)  # Check every 30 seconds

            if result is None:
                verbose_logger.debug(
                    f"Task {task_id} timed out after {timeout} seconds"
                )
                return {task_id: f"TIMEOUT after {timeout} seconds"}

            return result

        except Exception as e:
            error_msg = f"Error processing task {task_id}: {e}"
            verbose_logger.debug(error_msg)
            return {task_id: f"ERROR: {str(e)}"}

        finally:
            # Cleanup
            try:
                # Copy directory to log_dir if specified
                if self.log_dir:
                    task_log_dir = os.path.join(self.log_dir, task_id)
                    shutil.copytree(temp_dir, task_log_dir, dirs_exist_ok=True)

                # Remove temp directory
                shutil.rmtree(temp_dir)

                # Remove container
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.remove(force=True)
                    # Remove from active containers list
                    if container_id in self._active_containers:
                        self._active_containers.remove(container_id)
                except Exception:
                    pass  # Container may already be removed

            except Exception as e:
                error_msg = f"Warning: Failed to cleanup for task {task_id}: {e}"
                verbose_logger.debug(error_msg)

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

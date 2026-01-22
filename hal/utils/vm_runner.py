import os
import json
import asyncio
import time
import tempfile
import shutil
import uuid
from typing import Dict, Any, Optional, List
from .virtual_machine_manager import VirtualMachineManager
from ..benchmarks.base_benchmark import BaseBenchmark
import traceback
from rich.progress import Progress, TaskID


class VMRunner:
    """Handles running agents on Azure VMs"""

    def __init__(
        self,
        log_dir: str,
        max_concurrent: int = 1,
        benchmark: Optional[BaseBenchmark] = None,
    ):
        self.max_concurrent = max_concurrent
        self.log_dir = log_dir
        # FIXME: VM Manager should take a vm_name argument in the initializer; then we can stop passing it around all the time
        self.vm_manager = VirtualMachineManager()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self._active_vms: List[str] = []
        self.benchmark = benchmark

    async def fetch_agent_logs(self, vm_name, ssh_private_key_path, task_id):
        """Fetch the latest agent trace log from a VM and store it locally."""
        try:
            result = await asyncio.to_thread(
                self.vm_manager.get_agent_trace,
                ssh_private_key_path=ssh_private_key_path,
            )

            if result and self.log_dir:
                trace_dir = os.path.join(self.log_dir, "agent_logs")
                os.makedirs(trace_dir, exist_ok=True)

                # Write/update the trace file
                trace_path = os.path.join(trace_dir, f"{task_id}_log.log")
                with open(trace_path, "w") as f:
                    f.write(result)

                # Also write to a combined trace file
                combined_path = os.path.join(trace_dir, "combined_logs.log")
                with open(combined_path, "a") as f:
                    f.write(
                        f"\n=== {task_id} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    )
                    f.write(result)
                    f.write("\n")

        except Exception as e:
            print(f"Error fetching logs for {task_id}: {e}")

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
        """Run agent on all tasks using Azure VMs"""
        self.benchmark = benchmark
        results = {}
        vm_names = []

        async def process_task(task_id: str, input_data: Any) -> Optional[Dict]:
            # Create unique VM name
            vm_name = (
                f"agent-{benchmark.benchmark_name}-{uuid.uuid4()}"[:32]
                .lower()
                .replace("_", "-")
            )
            vm_names.append(vm_name)

            try:
                # Check if the task requires GPU
                gpu_required = False
                if self.benchmark and hasattr(self.benchmark, "benchmark"):
                    task_benchmark = self.benchmark.benchmark.get(task_id, {})
                    gpu_required = task_benchmark.get("gpu", False)

                # Create VM based on GPU requirement
                if gpu_required:
                    print(
                        f"Creating Azure virtual machine {vm_name} for task {task_id} *with* a GPU"
                    )
                    await asyncio.to_thread(
                        self.vm_manager.create_gpu_vm,
                        ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                        network_security_group_name=os.getenv(
                            "NETWORK_SECURITY_GROUP_NAME"
                        ),
                    )
                else:
                    print(
                        f"Creating Azure virtual machine {vm_name} for task {task_id} with *no* GPU"
                    )
                    await asyncio.to_thread(
                        self.vm_manager.create_vm,
                        vm_name=vm_name,
                        ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                        network_security_group_name=os.getenv(
                            "NETWORK_SECURITY_GROUP_NAME"
                        ),
                    )

                # Create temp directory with all necessary files
                temp_dir = tempfile.mkdtemp()
                try:
                    # Create input and args files
                    input_file = os.path.join(temp_dir, "input.json")
                    args_file = os.path.join(temp_dir, "agent_args.json")

                    with open(input_file, "w") as f:
                        json.dump({task_id: input_data}, f)
                    with open(args_file, "w") as f:
                        json.dump(agent_args, f)

                    # Copy task-specific files if they exist in input_data
                    if isinstance(input_data, dict) and "files" in input_data:
                        for dest_path, src_path in input_data["files"].items():
                            # Remove 'root' prefix and leading slash if present
                            dest_path = dest_path.replace("/root/", "").lstrip("/")

                            # Create destination directory structure in temp_dir
                            dest_full_path = os.path.join(temp_dir, dest_path)
                            os.makedirs(os.path.dirname(dest_full_path), exist_ok=True)

                            # Copy the file
                            try:
                                if os.path.isdir(src_path):
                                    shutil.copytree(
                                        src_path, dest_full_path, dirs_exist_ok=True
                                    )
                                else:
                                    shutil.copy2(src_path, dest_full_path)
                            except Exception as e:
                                print(
                                    f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}"
                                )

                    # Copy setup script if it exists
                    if self.benchmark and self.benchmark.setup_script:
                        setup_script_src = os.path.join(self.benchmark.setup_script)
                        if os.path.exists(setup_script_src):
                            setup_script_dest = os.path.join(
                                temp_dir, "setup_script.sh"
                            )
                            shutil.copy2(setup_script_src, setup_script_dest)
                            os.chmod(setup_script_dest, 0o755)

                    # Copy all files to VM
                    print(f"Copying files to VM {vm_name}")
                    await asyncio.to_thread(
                        self.vm_manager.copy_files_to_vm,
                        source_directory=temp_dir,
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                    )
                    await asyncio.to_thread(
                        self.vm_manager.copy_files_to_vm,
                        source_directory=agent_dir,
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                    )

                finally:
                    shutil.rmtree(temp_dir)

                # Run agent on VM
                await asyncio.to_thread(
                    self.vm_manager.run_agent_on_vm,
                    agent_function=agent_function,
                    task_id=task_id,
                    input_data=input_data,
                    agent_args=agent_args,
                    agent_dir="/home/agent",
                    run_id=run_id,
                    log_dir=self.log_dir,
                    ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                    benchmark=benchmark,
                )

                # Wait for completion or timeout
                start_time = time.time()
                result = None

                while time.time() - start_time < timeout:
                    try:
                        print(f"Checking task completion on VM {vm_name}")
                        # Fetch and store trace logs
                        await self.fetch_agent_logs(
                            vm_name=vm_name,
                            ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                            task_id=task_id,
                        )

                        result = await asyncio.to_thread(
                            self.vm_manager.check_task_completion,
                            ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                        )
                        if result is not None:
                            print(f"Task {task_id} completed on VM {vm_name}")
                            break
                    except Exception as e:
                        print(f"Error checking task completion on {vm_name}: {e}")
                    await asyncio.sleep(30)  # Check every 30 seconds

                if result is None:
                    print(f"Task {task_id} timed out after {timeout} seconds")
                    return {task_id: f"TIMEOUT after {timeout} seconds"}

                # Copy results back
                if self.log_dir:
                    print(f"Copying results from VM {vm_name} to local directory")
                    dest_dir = os.path.join(self.log_dir, f"{task_id}")
                    os.makedirs(dest_dir, exist_ok=True)
                    await asyncio.to_thread(
                        self.vm_manager.copy_files_from_vm,
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                        destination_directory=dest_dir,
                    )

                return result

            except Exception as e:
                print(f"Error processing task {task_id} on VM {vm_name}: {e}")
                traceback.print_exc()
                return {task_id: f"ERROR: {str(e)}"}

            finally:
                # Cleanup VM
                try:
                    await asyncio.to_thread(self.vm_manager.delete_vm)
                    if progress and task is not None:
                        progress.update(task, advance=1)
                except Exception as e:
                    print(f"Error deleting VM {vm_name}: {e}")

        # Run tasks in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_semaphore(task_id, input_data):
            async with semaphore:
                return await process_task(task_id, input_data)

        # Create tasks for all inputs
        tasks = [
            run_with_semaphore(task_id, input_data)
            for task_id, input_data in dataset.items()
        ]

        # Run all tasks and gather results
        results = await asyncio.gather(*tasks)

        # Merge results
        merged_results = {}
        for result in results:
            if result:
                merged_results.update(result)

        # Save raw submissions if log_dir provided
        if self.log_dir:
            raw_submissions_path = os.path.join(
                self.log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl"
            )
            os.makedirs(self.log_dir, exist_ok=True)

            # append to submissions file
            with open(raw_submissions_path, "a") as f:
                for task_id, result in merged_results.items():
                    json.dump({task_id: result}, f)
                    f.write("\n")

        return merged_results

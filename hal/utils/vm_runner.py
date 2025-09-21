import os
import json
import asyncio
import time
import tempfile
import shutil
import uuid
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from .azure_utils import VirtualMachineManager
from ..benchmarks.base_benchmark import BaseBenchmark
import traceback
from rich.progress import Progress, TaskID

class VMRunner:
    """Handles running agents on Azure VMs"""
    
    def __init__(self, log_dir: str, max_concurrent: int = 1, benchmark: Optional[BaseBenchmark] = None, tasks_per_vm: int = 1):
        self.max_concurrent = max_concurrent
        self.log_dir = log_dir
        self.vm_manager = VirtualMachineManager()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._file_lock = asyncio.Lock()
        self._active_vms: List[str] = []
        self.benchmark = benchmark
        # Dedicated high-capacity executor for blocking Azure SDK calls
        # Configure via VM_RUNNER_MAX_WORKERS (default 256)
        self._executor = ThreadPoolExecutor(max_workers=int(os.getenv("VM_RUNNER_MAX_WORKERS", "256")))
        # Number of tasks to run sequentially per VM (batch size)
        self.tasks_per_vm = int(os.getenv("VM_TASKS_PER_VM", str(tasks_per_vm or 1)))
        
    async def fetch_agent_logs(self, vm_name, username, ssh_private_key_path, task_id):
        """Fetch the latest agent trace log from a VM and store it locally."""
        try:
            result = await asyncio.to_thread(
                self.vm_manager.get_agent_trace,
                vm_name=vm_name,
                username=username,
                ssh_private_key_path=ssh_private_key_path
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
                    f.write(f"\n=== {task_id} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(result)
                    f.write("\n")
                
        except Exception as e:
            print(f"Error fetching logs for {task_id}: {e}")

    async def run_agent(self,
                       dataset: Dict[str, Any],
                       agent_function: str, 
                       agent_dir: str,
                       agent_args: Dict[str, Any],
                       run_id: str,
                       benchmark: Optional[BaseBenchmark] = None,
                       progress: Optional[Progress] = None,
                       task: Optional[TaskID] = None,
                       timeout: int = 7200) -> Dict[str, Any]:
        """Run agent on all tasks using Azure VMs"""
        self.benchmark = benchmark
        results = {}
        vm_names = []
        
        async def process_batch(batch_items: List[tuple[str, Any]]) -> Optional[Dict]:
            had_error = False
            # Create unique VM name for this batch
            vm_name = f"agent-batch-{uuid.uuid4()}"[:32].lower().replace("_", "-")
            vm_names.append(vm_name)
            setup_task_id = batch_items[0][0]

            try:
                # Check if any task requires GPU
                gpu_required = False
                if self.benchmark and hasattr(self.benchmark, 'benchmark'):
                    for tid, _ in batch_items:
                        task_benchmark = self.benchmark.benchmark.get(tid, {})
                        if task_benchmark.get('gpu', False):
                            gpu_required = True
                            break
                
                # Create VM with timeout + retries
                loop = asyncio.get_running_loop()
                create_timeout = int(os.getenv("VM_CREATE_TIMEOUT_SECS", "3600"))
                create_attempts = int(os.getenv("VM_CREATE_MAX_ATTEMPTS", "2"))
                create_ok = False
                vm = None
                for attempt in range(1, create_attempts + 1):
                    attempt_vm_name = vm_name if attempt == 1 else f"{vm_name}-{attempt}"
                    print(f"Creating {'GPU ' if gpu_required else ''}VM {attempt_vm_name} for batch of {len(batch_items)} tasks (attempt {attempt}/{create_attempts})")
                    try:
                        if gpu_required:
                            vm = await loop.run_in_executor(
                                self._executor,
                                functools.partial(
                                    self.vm_manager.create_gpu_vm,
                                    vm_name=attempt_vm_name,
                                    username="agent",
                                    ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                                    network_security_group_name=os.getenv("NETWORK_SECURITY_GROUP_NAME"),
                                    timeout=create_timeout,
                                )
                            )
                        else:
                            vm = await loop.run_in_executor(
                                self._executor,
                                functools.partial(
                                    self.vm_manager.create_vm,
                                    vm_name=attempt_vm_name,
                                    username="agent",
                                    ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                                    network_security_group_name=os.getenv("NETWORK_SECURITY_GROUP_NAME"),
                                    timeout=create_timeout,
                                )
                            )
                        # if we made it here, create succeeded
                        vm_name = attempt_vm_name
                        create_ok = True
                        break
                    except Exception as e:
                        print(f"VM create failed/timed out for {attempt_vm_name}: {e}")
                        # Best-effort cleanup of partial resources
                        try:
                            await loop.run_in_executor(
                                self._executor,
                                functools.partial(self.vm_manager.cleanup_network_resources, attempt_vm_name)
                            )
                        except Exception:
                            pass
                        if attempt == create_attempts:
                            raise

                # Copy benchmark setup script (as setup_script.sh) if it exists
                if self.benchmark and self.benchmark.setup_script:
                    setup_script_src = os.path.join(self.benchmark.setup_script)
                    if os.path.exists(setup_script_src):
                        tmp = tempfile.mkdtemp()
                        try:
                            setup_dest = os.path.join(tmp, 'setup_script.sh')
                            shutil.copy2(setup_script_src, setup_dest)
                            os.chmod(setup_dest, 0o755)
                            await loop.run_in_executor(
                                self._executor,
                                functools.partial(
                                    self.vm_manager.copy_files_to_vm,
                                    source_directory=tmp,
                                    vm_name=vm_name,
                                    username="agent",
                                    ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                                )
                            )
                        finally:
                            shutil.rmtree(tmp)

                # Copy agent_dir contents once per VM
                await loop.run_in_executor(
                    self._executor,
                    functools.partial(
                        self.vm_manager.copy_files_to_vm,
                        source_directory=agent_dir,
                        vm_name=vm_name,
                        username="agent",
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                    )
                )

                # Setup environment once with retry-on-failure by recreating VM
                setup_attempts = int(os.getenv("VM_SETUP_MAX_ATTEMPTS", "2"))
                setup_ok = False
                for s_attempt in range(1, setup_attempts + 1):
                    try:
                        await loop.run_in_executor(
                            self._executor,
                            functools.partial(
                                self.vm_manager.setup_vm_environment,
                                vm_name=vm_name,
                                username="agent",
                                ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                                agent_dir=agent_dir,
                                log_dir=self.log_dir,
                                benchmark=benchmark,
                                task_id=setup_task_id
                            )
                        )
                        setup_ok = True
                        break
                    except Exception as e:
                        print(f"Setup failed on {vm_name} (attempt {s_attempt}/{setup_attempts}): {e}")
                        # If not last attempt, delete and recreate the VM, then retry
                        if s_attempt < setup_attempts:
                            try:
                                await loop.run_in_executor(self._executor, functools.partial(self.vm_manager.delete_vm, vm_name))
                            except Exception:
                                pass
                            # Re-run create loop
                            for attempt in range(1, create_attempts + 1):
                                attempt_vm_name = f"{vm_name}-r{attempt}"
                                print(f"Recreating VM {attempt_vm_name} after setup failure (attempt {attempt}/{create_attempts})")
                                try:
                                    if gpu_required:
                                        vm = await loop.run_in_executor(
                                            self._executor,
                                            functools.partial(
                                                self.vm_manager.create_gpu_vm,
                                                vm_name=attempt_vm_name,
                                                username="agent",
                                                ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                                                network_security_group_name=os.getenv("NETWORK_SECURITY_GROUP_NAME"),
                                                timeout=create_timeout,
                                            )
                                        )
                                    else:
                                        vm = await loop.run_in_executor(
                                            self._executor,
                                            functools.partial(
                                                self.vm_manager.create_vm,
                                                vm_name=attempt_vm_name,
                                                username="agent",
                                                ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                                                network_security_group_name=os.getenv("NETWORK_SECURITY_GROUP_NAME"),
                                                timeout=create_timeout,
                                            )
                                        )
                                    vm_name = attempt_vm_name
                                    break
                                except Exception as e2:
                                    print(f"VM recreate failed: {e2}")
                                    try:
                                        await loop.run_in_executor(
                                            self._executor,
                                            functools.partial(self.vm_manager.cleanup_network_resources, attempt_vm_name)
                                        )
                                    except Exception:
                                        pass
                                    if attempt == create_attempts:
                                        raise
                        else:
                            raise
                if not setup_ok:
                    raise RuntimeError(f"VM setup failed for {vm_name}")

                # Launch all tasks in parallel on this VM with per-task workdirs
                work_items = []  # list of dicts with task_id, workdir, start_time
                for task_id, input_data in batch_items:
                    workdir = f"/home/agent/runs/{task_id}"
                    # Upload task-specific files to the workdir
                    if isinstance(input_data, dict) and 'files' in input_data:
                        tmp_task = tempfile.mkdtemp()
                        try:
                            # Materialize the task files into tmp_task preserving relative paths
                            for dest_path, src_path in input_data['files'].items():
                                dest_path_clean = dest_path.replace('/root/', '').lstrip('/')
                                dest_full_path = os.path.join(tmp_task, dest_path_clean)
                                os.makedirs(os.path.dirname(dest_full_path), exist_ok=True)
                                try:
                                    if os.path.isdir(src_path):
                                        shutil.copytree(src_path, dest_full_path, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(src_path, dest_full_path)
                                except Exception as e:
                                    print(f"Warning: Failed to copy task file {src_path} to {dest_full_path}: {e}")
                            # Upload to specific workdir on VM
                            await loop.run_in_executor(
                                self._executor,
                                functools.partial(
                                    self.vm_manager.upload_dir_to_vm,
                                    vm_name=vm_name,
                                    username="agent",
                                    ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                                    local_dir=tmp_task,
                                    remote_dir=workdir
                                )
                            )
                        finally:
                            shutil.rmtree(tmp_task)

                    # Launch task without waiting
                    await loop.run_in_executor(
                        self._executor,
                        functools.partial(
                            self.vm_manager.run_agent_task_on_vm_parallel,
                            agent_function=agent_function,
                            vm_name=vm_name,
                            task_id=task_id,
                            input_data=input_data,
                            agent_args=agent_args,
                            run_id=run_id,
                            username="agent",
                            ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                            workdir=workdir,
                        )
                    )
                    work_items.append({
                        "task_id": task_id,
                        "workdir": workdir,
                        "start": time.time(),
                    })

                merged_batch_results: Dict[str, Any] = {}
                # Poll all tasks until completion or timeout
                pending = {w["task_id"]: w for w in work_items}
                while pending:
                    done_ids = []
                    for tid, info in list(pending.items()):
                        # Fetch logs
                        try:
                            log_text = await loop.run_in_executor(
                                self._executor,
                                functools.partial(
                                    self.vm_manager.get_file_text,
                                    vm_name=vm_name,
                                    username="agent",
                                    ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                                    remote_path=f"{info['workdir']}/agent_trace.log"
                                )
                            )
                            if log_text and self.log_dir:
                                trace_dir = os.path.join(self.log_dir, "agent_logs")
                                os.makedirs(trace_dir, exist_ok=True)
                                with open(os.path.join(trace_dir, f"{tid}_log.log"), 'w') as f:
                                    f.write(log_text)
                        except Exception:
                            pass

                        # Check completion
                        result = await loop.run_in_executor(
                            self._executor,
                            functools.partial(
                                self.vm_manager.check_task_completion_at,
                                vm_name=vm_name,
                                username="agent",
                                ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                                output_path=f"{info['workdir']}/output.json"
                            )
                        )
                        if result is not None:
                            # Download artifacts for this task
                            if self.log_dir:
                                dest_dir = os.path.join(self.log_dir, f"{tid}")
                                os.makedirs(dest_dir, exist_ok=True)
                                await loop.run_in_executor(
                                    self._executor,
                                    functools.partial(
                                        self.vm_manager.download_files_from_vm,
                                        vm_name=vm_name,
                                        username="agent",
                                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                                        remote_to_local={
                                            f"{info['workdir']}/output.json": os.path.join(dest_dir, "output.json"),
                                            f"{info['workdir']}/agent_trace.log": os.path.join(dest_dir, "agent_trace.log"),
                                            f"{info['workdir']}/error.log": os.path.join(dest_dir, "error.log"),
                                        }
                                    )
                                )
                            merged_batch_results.update(result)
                            done_ids.append(tid)
                        else:
                            # Check timeout per task
                            if time.time() - info['start'] >= timeout:
                                merged_batch_results[tid] = f"TIMEOUT after {timeout} seconds"
                                done_ids.append(tid)

                    for tid in done_ids:
                        pending.pop(tid, None)

                    if pending:
                        await asyncio.sleep(30)

                return merged_batch_results

            except Exception as e:
                print(f"Error processing batch on VM {vm_name}: {e}")
                traceback.print_exc()
                had_error = True
                return {tid: f"ERROR: {str(e)}" for tid, _ in batch_items}
            
            finally:
                # Cleanup VM
                try:
                    keep_on_fail = os.getenv("VM_KEEP_ON_FAILURE", "0") == "1"
                    if had_error and keep_on_fail:
                        print(f"Keeping VM {vm_name} for debugging due to failure (VM_KEEP_ON_FAILURE=1)")
                    else:
                        print(f"Deleting VM {vm_name}")
                        await loop.run_in_executor(self._executor, functools.partial(self.vm_manager.delete_vm, vm_name))
                    if progress and task is not None:
                        progress.update(task, advance=len(batch_items))
                except Exception as e:
                    print(f"Error deleting VM {vm_name}: {e}")

        # Group tasks into batches according to tasks_per_vm
        items = list(dataset.items())
        batches: List[List[tuple[str, Any]]] = [
            items[i:i+self.tasks_per_vm] for i in range(0, len(items), self.tasks_per_vm)
        ]

        # Run batches in parallel with semaphore to limit concurrent VMs
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async def run_with_semaphore(batch_items: List[tuple[str, Any]]):
            async with semaphore:
                return await process_batch(batch_items)

        tasks = [run_with_semaphore(batch) for batch in batches]
        
        # Run all tasks and gather results
        results = await asyncio.gather(*tasks)
        
        # Merge results
        merged_results = {}
        for result in results:
            if result:
                merged_results.update(result)

        # Save raw submissions if log_dir provided
        if self.log_dir:
            raw_submissions_path = os.path.join(self.log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            os.makedirs(self.log_dir, exist_ok=True)
            
            # append to submissions file
            with open(raw_submissions_path, "a") as f:
                for task_id, result in merged_results.items():
                    json.dump({task_id: result}, f)
                    f.write('\n')

        return merged_results

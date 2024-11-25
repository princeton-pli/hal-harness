from abc import ABC, abstractmethod
from pydantic import TypeAdapter
from huggingface_hub import HfApi
from typing import Dict
import json
import os
from ..utils.weave_utils import get_total_cost, assert_task_id_logging
import tempfile
import subprocess
import shutil
import traceback
import uuid
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, cast
from ..utils.azure_utils import VirtualMachineManager
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor


class BaseBenchmark(ABC):
    def __init__(self, agent_dir: str, config: Dict):
        self.agent_dir = agent_dir
        self.config = config
        self.benchmark_name: str
        self.requirements_file: str
        self.benchmark_dir: str
        self.main_cwd: str
        self.temp_dirs = [] # List of temporary directories created during agent runs
        self.log_dir: str
        self.vm_manager = VirtualMachineManager()

    def run_single_agent(self, task_id, single_input, agent_dir, agent_function, agent_args, module_name, run_id, conda_env_name, log_dir=None):
            # Create a unique directory in /tmp/
            temp_dir = f"/tmp/agent_run_{uuid.uuid4()}"
            os.makedirs(temp_dir, exist_ok=True)

            # Track the created directory
            self.temp_dirs.append(temp_dir)

            # Copy the entire agent directory to the temporary directory
            shutil.copytree(agent_dir, temp_dir, dirs_exist_ok=True)

            # Serialize the input data to a JSON file in the temp directory
            input_file = os.path.join(temp_dir, 'input.json')
            with open(input_file, 'w') as f:
                json.dump({task_id: single_input}, f)

            # Prepare the agent arguments
            agent_args_file = os.path.join(temp_dir, 'agent_args.json')
            with open(agent_args_file, 'w') as f:
                json.dump(agent_args, f)

            # Construct the command to run the agent function
            command = [
                'python', '-c',
                f'''
import os
import json
import importlib.util
import weave

weave.init("{run_id}")

# Load the agent module
module_name = "{agent_function.rsplit(".", 1)[0]}"
function_name = "{agent_function.rsplit(".", 1)[1]}"
spec = importlib.util.spec_from_file_location(module_name, "{os.path.join(temp_dir, module_name + ".py")}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
agent = getattr(module, function_name)

# Load the input data
with open("input.json", "r") as f:
    input_data = json.load(f)

# Load the agent arguments
with open("agent_args.json", "r") as f:
    agent_args = json.load(f)  

# Run the agent function
with weave.attributes({{"weave_task_id": "{task_id}"}}):
    result = agent(input_data, **agent_args)

# Save the result
with open("output.json", "w") as f:
    json.dump(result, f)
''']

            if conda_env_name:
                print(f"Running agent in conda environment: {conda_env_name}")
                command = [
                    'conda', 'run', '-n', conda_env_name] + command

            try:
                # Run the agent in a subprocess with the temp_dir as cwd
                subprocess.run(command, cwd=temp_dir, check=True)

                # Load the result from output.json
                output_file = os.path.join(temp_dir, 'output.json')
                with open(output_file, 'r') as f:
                    result = json.load(f)

                if log_dir:
                    raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS_DURING.jsonl")
                    os.makedirs(log_dir, exist_ok=True)

                    with open(raw_submissions_path, "a") as f:
                        json.dump(result, f)
                        f.write('\n')

                # delete the temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                print(f"Error running agent: {e}")
                traceback.print_exc()
                result = {task_id: f"ERROR RUNNING AGENT: {e}"}

                # delete the temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

            return result


    async def run_agent_parallel(self, dataset, agent_args: Dict, agent_function: str, agent_dir: str, run_id: str,  max_concurrent: int = 1, log_dir: str = None, conda_env_name: str = None) -> Dict:
        agent_dir = os.path.abspath(agent_dir)

        # assert that dataset is a dictionary with keys as sample ids that are strings
        assert isinstance(dataset, dict), "Dataset must be a dictionary with keys as sample ids"
        agent_input = dataset

        module_name, _ = agent_function.rsplit(".", 1)

        original_dir = os.getcwd()

        abs_log_dir = os.path.abspath(log_dir) if log_dir else None

        semaphore = asyncio.Semaphore(max_concurrent)

        if log_dir:
            raw_submissions_path = os.path.join(abs_log_dir, f"{run_id}_RAW_SUBMISSIONS_DURING.jsonl")
            
            if os.path.exists(raw_submissions_path):
                # read in previous submissions and remove from agent_input
                with open(raw_submissions_path, "r") as f:
                    previous_submissions = [json.loads(line) for line in f]
                
                previous_ids = {list(submission.keys())[0] for submission in previous_submissions}
                agent_input = {k: v for k, v in agent_input.items() if k not in previous_ids}

                print( f"Previous submissions found. {len(previous_ids)} submissions removed from agent input.")


        async def sem_run_single_agent(task_id, input_data):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,  # Use default executor or specify if needed
                    self.run_single_agent,
                    task_id,
                    input_data,
                    agent_dir,
                    agent_function,
                    agent_args,
                    module_name,
                    run_id,
                    conda_env_name,
                    abs_log_dir
                )
        
        tasks = [sem_run_single_agent(task_id, input_data) for task_id, input_data in agent_input.items()]
        results = await asyncio.gather(*tasks)

        for result in results:
            print(result)
            
        os.chdir(original_dir)

        # Delete all created directories after processing all jobs
        for temp_dir in self.temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Merge results list into a single dictionary
        merged_result = {k: v for d in results for k, v in d.items()}

        # add all results from _RAW_SUBMISSIONS_DURING.jsonl to merged_result
        if log_dir:
            raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS_DURING.jsonl")
            
            if os.path.exists(raw_submissions_path):
                # read in previous submissions for scoring
                with open(raw_submissions_path, "r") as f:
                    previous_submissions = [json.loads(line) for line in f]
                
                for submission in previous_submissions:
                    merged_result.update(submission)

        # save raw submissions as jsonl file with each line being a submission
        if log_dir:
            raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            os.makedirs(log_dir, exist_ok=True)

            with open(raw_submissions_path, "w") as f:
                for key, value in merged_result.items():
                    json.dump({key: value}, f)
                    f.write('\n')
            

        return merged_result
    
    async def run_agent_on_vm(self, dataset, agent_function, agent_args, agent_dir, run_id, max_concurrent, log_dir, conda_env_name, timeout=7200):
        """
        Run agents on Azure VMs in parallel with a timeout.
        
        Args:
            dataset: Dictionary of task_id -> input_data
            agent_function: Path to agent function (e.g. "agent.run")
            agent_args: Arguments to pass to the agent
            agent_dir: Directory containing agent code
            run_id: Unique identifier for this run
            max_concurrent: Maximum number of concurrent VMs
            log_dir: Directory to store logs
            conda_env_name: Name of conda environment to use
            timeout: Maximum time in seconds to wait for each task (default: 2 hours)
        """
        results = {}
        vm_names = []
        
        async def process_task(task_id: str, input_data: Any) -> Optional[Dict]:
            # Create unique VM name
            vm_name = f"agent-{self.benchmark_name}-{uuid.uuid4()}"[:32].lower().replace("_", "-")
            vm_names.append(vm_name)
            
            try:
                # Create VM
                print(f"Creating VM {vm_name} for task {task_id}")
                vm = self.vm_manager.create_vm(
                    vm_name=vm_name,
                    username="agent",
                    ssh_public_key_path=os.getenv("SSH_PUBLIC_KEY_PATH"),
                    network_security_group_name=os.getenv("NETWORK_SECURITY_GROUP_NAME")
                )

                # Copy agent code and input files
                print(f"Copying files to VM {vm_name}")
                self.vm_manager.copy_files_to_vm(
                    source_directory=agent_dir,
                    vm_name=vm_name,
                    username="agent",
                    ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                )

                # Create input.json and agent_args.json on VM
                temp_dir = tempfile.mkdtemp()
                try:
                    input_file = os.path.join(temp_dir, 'input.json')
                    args_file = os.path.join(temp_dir, 'agent_args.json')
                    
                    with open(input_file, 'w') as f:
                        json.dump({task_id: input_data}, f)
                    with open(args_file, 'w') as f:
                        json.dump(agent_args, f)

                    self.vm_manager.copy_files_to_vm(
                        source_directory=temp_dir,
                        vm_name=vm_name,
                        username="agent", 
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                    )
                finally:
                    shutil.rmtree(temp_dir)

                # Run agent on VM
                print(f"Running agent on VM {vm_name}")
                self.vm_manager.run_agent_on_vm(
                    agent_function=agent_function,
                    vm_name=vm_name,
                    task_id=task_id,
                    input_data=input_data,
                    agent_args=agent_args,
                    agent_dir="/home/agent",
                    run_id=run_id,
                    conda_env_name=conda_env_name,
                    username="agent",
                    ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                )

                # Wait for completion or timeout
                start_time = time.time()
                result = None
                
                while time.time() - start_time < timeout:
                    try:
                        # Fetch and store trace logs
                        await self.fetch_agent_trace(
                            vm_name=vm_name,
                            username="agent",
                            ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                            task_id=task_id,
                            log_dir=self.log_dir
                        )
                        
                        result = self.vm_manager.check_task_completion(
                            vm_name=vm_name,
                            username="agent",
                            ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH")
                        )
                        if result is not None:
                            break
                    except Exception as e:
                        print(f"Error checking task completion on {vm_name}: {e}")
                    await asyncio.sleep(30)  # Check every 30 seconds

                if result is None:
                    print(f"Task {task_id} timed out after {timeout} seconds")
                    return {task_id: f"TIMEOUT after {timeout} seconds"}

                # Copy results back
                if log_dir:
                    dest_dir = os.path.join(log_dir, f"{task_id}")
                    os.makedirs(dest_dir, exist_ok=True)
                    self.vm_manager.copy_files_from_vm(
                        vm_name=vm_name,
                        username="agent",
                        ssh_private_key_path=os.getenv("SSH_PRIVATE_KEY_PATH"),
                        destination_directory=dest_dir
                    )

                return result

            except Exception as e:
                print(f"Error processing task {task_id} on VM {vm_name}: {e}")
                traceback.print_exc()
                return {task_id: f"ERROR: {str(e)}"}
            
            finally:
                # Cleanup VM
                try:
                    print(f"Deleting VM {vm_name}")
                    self.vm_manager.delete_vm(vm_name)
                except Exception as e:
                    print(f"Error deleting VM {vm_name}: {e}")

        # Run tasks in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task_id, input_data):
            async with semaphore:
                return await process_task(task_id, input_data)

        # Create tasks for all inputs
        tasks = [run_with_semaphore(task_id, input_data) 
                 for task_id, input_data in dataset.items()]
        
        # Run all tasks and gather results
        results = await asyncio.gather(*tasks)
        
        # Merge results
        merged_results = {}
        for result in results:
            if result:
                merged_results.update(result)

        # Save raw submissions if log_dir provided
        if log_dir:
            raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            os.makedirs(log_dir, exist_ok=True)
            
            with open(raw_submissions_path, "w") as f:
                for task_id, result in merged_results.items():
                    json.dump({task_id: result}, f)
                    f.write('\n')

        return merged_results
    
    def run(self, agent_function, run_id: str, agent_args: Dict, conda_env_name: str = None, max_concurrent: int = 1, vm: bool = False, continue_run: bool = False):
        """
        Run the agent evaluation.
        
        Args:
            agent_function: Path to agent function
            run_id: Unique identifier for this run
            agent_args: Arguments to pass to agent
            conda_env_name: Name of conda environment to use
            max_concurrent: Maximum number of concurrent tasks
            vm: Whether to run on Azure VMs
            continue_run: Whether to continue from a previous run
        """
        self.log_dir = f"results/{self.benchmark_name}/{run_id}"
        
        # Get dataset to run (either full or remaining tasks)
        dataset = self.benchmark
        if continue_run:
            dataset = self.get_remaining_tasks(self.benchmark, run_id, self.log_dir)
            if not dataset:
                print("No remaining tasks to run")
                # Load and return previous results
                raw_submissions_path = os.path.join(self.log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
                if os.path.exists(raw_submissions_path):
                    with open(raw_submissions_path, "r") as f:
                        merged_results = {}
                        for line in f:
                            merged_results.update(json.loads(line))
                        return merged_results
                return {}
        
        if vm:
            return asyncio.run(self.run_agent_on_vm(
                dataset=dataset,
                agent_function=agent_function,
                agent_args=agent_args,
                agent_dir=self.agent_dir,
                run_id=run_id,
                max_concurrent=max_concurrent,
                log_dir=self.log_dir,
                conda_env_name=conda_env_name
            ))
        else:
            return asyncio.run(self.run_agent_parallel(
                dataset=dataset,
                agent_function=agent_function,
                agent_args=agent_args,
                agent_dir=self.agent_dir,
                run_id=run_id,
                max_concurrent=max_concurrent,
                log_dir=self.log_dir,
                conda_env_name=conda_env_name
            ))
    
    @abstractmethod
    def run_evaluation_harness(self, agent_output, run_id: str, log_dir: str):
        pass

    @abstractmethod
    def test_run(self, agent_function, weave_client):
        pass
    
    @property
    @abstractmethod
    def type_adapter(self) -> TypeAdapter:
        pass
    
    @abstractmethod
    def process_and_upload_results(self, 
                                   agent_name: str, 
                                   run_id: str, 
                                   eval_results, 
                                   weave_client,
                                   config, 
                                   upload=False) -> Dict:
        pass

    def validate_agent_output(self, output) -> bool:
        self.type_adapter.validate_python(output)

    def validate_logging(self, weave_client, test_weave_task_id):
        assert get_total_cost(weave_client)[0] > 0, "Test run did not incur any cost"

        assert_task_id_logging(weave_client, test_weave_task_id)


    def upload_results(self, run_id, upload_dict):
        # Write results to temp file using tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(upload_dict, f)
            temp_file_name = f.name
            temp_file_path = os.path.abspath(temp_file_name)

        # Use huggingface API to upload results
        api = HfApi(token=os.getenv("HF_TOKEN", '.'))
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo=f"{run_id}.json",
            repo_id="agent-evals/results"   ,
            repo_type="dataset",
            commit_message=f"Add {run_id} to leaderboard.",
            )
        

        # unlink the file
        os.unlink(temp_file_name)

        print(f"Results uploaded to Hugging Face Hub.")


    def mount_benchmark(self):
        try:
            with open(f'agent_eval_harness/benchmarks/requirements/{self.requirements_file}.toml', 'r') as f:
                requirements_toml = f.read()
        except FileNotFoundError:
            raise ValueError(f"Requirements file for benchmark '{self.benchmark_name}' not found. See README for available benchmarks.")
        with open('pyproject.toml', 'w') as f:
            f.write(f"""[tool.poetry]\npackage-mode = false\n\n{requirements_toml}""")

        # install dependencies
        try:
            print(f'\n\n====Mounting benchmark harness for {self.benchmark_name}====')
            print(f"Installing dependencies...")
            # this command install the benchmark dependencies in a virtual environment separate from the agent_eval_harness and agent dependencies to run the evals
            subprocess.run(['poetry env use python3 && poetry lock && poetry install --no-root'], shell=True)
            print(f"Done!")
            print('=====\n\n')
        except Exception as e:
            print(e)
            raise ValueError(f"Failed to install dependencies for benchmark '{self.benchmark_name}'")

    def unmount_benchmark(self):
        if os.path.exists('pyproject.toml'):
            os.remove('pyproject.toml')
        if os.path.exists('poetry.lock'):
            os.remove('poetry.lock')

    def get_remaining_tasks(self, dataset, run_id, log_dir):
        """
        Get tasks that haven't been successfully completed in previous runs.
        
        Args:
            dataset: Dictionary of task_id -> input_data
            run_id: Run identifier
            log_dir: Directory containing logs from previous runs
        
        Returns:
            Dictionary of remaining tasks to run
        """
        remaining_tasks = dataset.copy()
        
        if log_dir:
            raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
            if os.path.exists(raw_submissions_path):
                with open(raw_submissions_path, "r") as f:
                    previous_submissions = [json.loads(line) for line in f]
                
                # Remove tasks that were previously completed successfully
                for submission in previous_submissions:
                    task_id = list(submission.keys())[0]
                    result = submission[task_id]
                    # Only remove if the result isn't an error or timeout
                    if not isinstance(result, str) or not (result.startswith("ERROR") or result.startswith("TIMEOUT")):
                        remaining_tasks.pop(task_id, None)
                
                print(f"Found {len(dataset) - len(remaining_tasks)} completed tasks from previous run")
                print(f"Remaining tasks: {len(remaining_tasks)}")
        
        return remaining_tasks

    async def fetch_agent_trace(self, vm_name, username, ssh_private_key_path, task_id, log_dir):
        """Fetch the latest agent trace log from a VM and store it locally."""
        try:
            result = self.vm_manager.get_agent_trace(
                vm_name=vm_name,
                username=username,
                ssh_private_key_path=ssh_private_key_path
            )
            
            if result and log_dir:
                trace_dir = os.path.join(log_dir, "agent_traces")
                os.makedirs(trace_dir, exist_ok=True)
                
                # Write/update the trace file
                trace_path = os.path.join(trace_dir, f"{task_id}_trace.log")
                with open(trace_path, "w") as f:
                    f.write(result)
                
                # Also write to a combined trace file
                combined_path = os.path.join(trace_dir, "combined_trace.log")
                with open(combined_path, "a") as f:
                    f.write(f"\n=== {task_id} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(result)
                    f.write("\n")
                
        except Exception as e:
            print(f"Error fetching trace for {task_id}: {e}")




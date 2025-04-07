import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset
import docker

from .base_benchmark import BaseBenchmark


class SciCodeBenchmark(BaseBenchmark):
    """SciCode benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = 'scicode'):
        self.benchmark_name = benchmark_name
        self.vm_only = False
        super().__init__(agent_dir, config, vm_only=self.vm_only)
        
        # Load the dataset.
        self.dataset = list(load_dataset("SciCode1/SciCode", split="test"))
        self.benchmark = {task['problem_id']: task for task in self.dataset}

        # Set benchmark directory.
        self.benchmark_dir = os.path.join(os.path.dirname(__file__), 'SciCode')
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Run SciCode evaluation harness on agent outputs"""
        start_time = time.time()
        # Create a temporary directory on the host for evaluation files.
        host_tmp_dir = Path(f"tmp_{run_id}_{start_time}")
        host_tmp_dir.mkdir(parents=True, exist_ok=True)

        if self.benchmark_name == 'scicode_hard':
            # Iterate through problems 
            for problem_id, code_content in agent_output.items():
                task = self.benchmark.get(problem_id)
                
                tmp_file = host_tmp_dir / f"{problem_id}.py"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    f.write(code_content)
                    f.write("\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n")
                    last_step = task["sub_steps"][-1]
                    step_id = last_step["step_number"]
                    test_lst = last_step["test_cases"]
                    f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})\n")
                    for idx in range(len(test_lst)):
                        f.write(f"target = targets[{idx}]\n\n")
                        for line in test_lst[idx].split("\n"):
                            f.write(line + "\n")
            
        else:
            # Write evaluation files (one per agent_output entry).
            for problem_id, subtasks in agent_output.items():
                for subtask, code_content in subtasks.items():
                    parts = subtask.split(".")
                    subtask_step = parts[1]
                    task = self.benchmark.get(problem_id)
                    step_index = int(subtask_step) - 1
                    step_info = task["sub_steps"][step_index]
                    step_id = step_info["step_number"]
                    test_lst = step_info["test_cases"]
                    
                    tmp_file = host_tmp_dir / f"{problem_id}.{subtask_step}.py"

                    with open(tmp_file, "w", encoding="utf-8") as f:
                        f.write(code_content)
                        f.write("\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n")
                        f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})\n")
                        for idx in range(len(test_lst)):
                            f.write(f"target = targets[{idx}]\n\n")
                            for line in test_lst[idx].split("\n"):
                                f.write(line + "\n")
        
        # Launch a Docker container that mounts self.benchmark_dir at /app and host_tmp_dir at /app/tmp_eval.
        client = docker.from_env()
        container = client.containers.run(
            "python:3.11",
            command="tail -f /dev/null",
            volumes={
                str(Path(self.benchmark_dir).resolve()): {"bind": "/app", "mode": "rw"},
                str(host_tmp_dir.resolve()): {"bind": "/app/tmp_eval", "mode": "rw"}
            },
            working_dir="/app",
            detach=True,
        )
        
        try:
            # Install required dependencies inside the container.
            print("Installing dependencies inside the container...")
            install_result = container.exec_run("pip install -e .", stream=True, demux=True)
            for stdout, stderr in install_result.output:
                if stdout:
                    print(stdout.decode(), end='')
                if stderr:
                    print(stderr.decode(), end='')

            # Helper function to execute a Python script inside the container.            
            def run_script(script_path: Path) -> int:
                # Compute the file's relative path in the host temporary directory and convert it
                rel_path = script_path.relative_to(host_tmp_dir)
                container_script_path = f"/app/tmp_eval/{rel_path}"
                
                # Run script within container with timeout exception
                cmd = ["timeout", "60", "python", container_script_path]
                try:
                    result = container.exec_run(cmd, demux=True)
                    if result.exit_code == 124:
                        print(f"Timeout running script [{script_path.name}] after 60 seconds")
                        return 2
                    elif result.exit_code == 0:
                        print(f"Successfully executed script: {script_path.name}")
                        return 0
                    else:
                        print(f"Error running script [{script_path.name}]: exit code {result.exit_code}")
                        stdout, stderr = result.output
                        error_message = stderr.decode("utf-8") if stderr and isinstance(stderr, bytes) else stderr
                        print(f"Error running script [{script_path.name}]: exit code {result.exit_code}")
                        if error_message:
                            print("Error message:", error_message)
                        return 1
                except Exception as e:
                    print(f"Exception running script [{script_path.name}]: {e}")
                    return 1
            
            # Initialize a dictionary to track outcomes.
            correct_dict = {problem_id: [] for problem_id in self.benchmark.keys()}
            
            # Execute each temporary file inside the container.
            for tmp_file in host_tmp_dir.iterdir():
                if tmp_file.is_file():
                    func_id = tmp_file.stem  # e.g., "11.3"
                    prob_id = func_id.split('.')[0]
                    ret = run_script(tmp_file)
                    if ret == 0:
                        correct_dict[prob_id].append(func_id)
            
            return {"details": correct_dict}
        
        finally:
            # Cleanup: stop and remove the container, then delete the temporary directory.
            container.stop()
            container.remove()
            shutil.rmtree(host_tmp_dir)
    
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""

        # Load results of evaluation
        details = eval_results.get("details", {})

        # Track relevant variables
        successful_tasks = []
        failed_tasks = []
        total_passed_subtasks = 0
        total_subtasks = 0

        # Iterate through benchmark tasks
        for problem_id, task in self.benchmark.items():
            # Get the number of subtasks in the benchmark
            total = len(task.get("sub_steps", []))
            # Get the number of subtasks that were successfully executed
            passed = len(details.get(problem_id, []))
            if self.benchmark_name == 'scicode_hard':
                # For the hard benchmark, we only care about the last step
                if passed > 0:
                    successful_tasks.append(problem_id)
                else:
                    failed_tasks.append(problem_id)
            else:
                # For the easy and standard benchmarks, we check whether all steps are passed
                total_subtasks += total
                total_passed_subtasks += passed
                if total > 0 and passed == total:
                    successful_tasks.append(problem_id)
                else:
                    failed_tasks.append(problem_id)

        accuracy = len(successful_tasks) / (len(successful_tasks) + len(failed_tasks))

        if self.benchmark_name != 'scicode_hard':
            subtask_accuracy = total_passed_subtasks / total_subtasks if total_subtasks > 0 else 0

            return {
                "accuracy": accuracy,
                "subtask_accuracy": subtask_accuracy,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks
            }
        
        else:

            return {
                "accuracy": accuracy,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks
            }
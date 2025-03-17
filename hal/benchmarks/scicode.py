import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset

from .base_benchmark import BaseBenchmark


class SciCodeBenchmark(BaseBenchmark):
    """SciCode benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = 'scicode'
        self.vm_only = False
        self.with_background = config.get("with_background", False)
        super().__init__(agent_dir, config, vm_only=self.vm_only)
        
        # load the dataset 
        self.dataset = list(load_dataset("SciCode1/SciCode", split="test"))
        self.benchmark = {task['problem_id']: task for task in self.dataset}
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Run SciCode evaluation harness on agent outputs locally"""

        # Create a temporary directory tagged with run_id and current time.
        start_time = time.time()
        tmp_dir = Path(f'tmp_{run_id}_{start_time}')
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for file_name, code_content in agent_output.items():
            parts = file_name.split(".")
            file_id, file_step = parts[0], parts[1]
            # Retrieve the task data for this problem.
            task = self.benchmark.get(file_id)
            if task is None:
                continue  # Skip if task not found.
            # Get the sub-step info (indexing from 0).
            step_index = int(file_step) - 1
            step_info = task["sub_steps"][step_index]
            step_id = step_info["step_number"]
            test_lst = step_info["test_cases"]

            # Write a temporary file with the generated code plus test setup.
            tmp_file = tmp_dir / f'{file_id}.{file_step}.py'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
                f.write(f"from scicode.parse.parse import process_hdf5_to_tuple" + "\n")
                f.write(f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})" + '\n')
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split('\n'):
                        f.write(line + '\n')


            # Helper function to execute a Python script.
            def run_script(script_path):
                try:
                    subprocess.run(['python', script_path], check=True, capture_output=True,
                                text=True, timeout=120)
                    print(f"Successfully ran script {script_path}")
                    return 0
                except subprocess.CalledProcessError as e:
                    print(f"Error running script {script_path}: {e}")
                    print(e.output)
                    return 1
                except subprocess.TimeoutExpired as e:
                    print(f"Runtime error while running script {script_path}: {e}")
                    return 2
        
        # Initialize a dictionary to track outcomes.
        correct_dict = {problem_id: [] for problem_id in self.benchmark.keys()}
        
        # Execute each temporary file and update outcomes.
        for tmp_file in tmp_dir.iterdir():
            if tmp_file.is_file():
                # Expected filename format: "problemID.stepID.py"
                func_id = tmp_file.stem  # e.g., "13.2"
                prob_id = func_id.split('.')[0]
                ret = run_script(tmp_file)
                if ret == 0:
                    correct_dict[prob_id].append(func_id)
        
        # Clean up temporary files.
        shutil.rmtree(tmp_dir)
        
        # Return only the details dictionary.
        return {
            "details": correct_dict
        }



    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""

        details = eval_results.get("details", {})
        successful_tasks = []
        failed_tasks = []
        
        total_passed_subtasks = 0
        total_subtasks = 0

        for problem_id, task in self.benchmark.items():
            total = len(task.get("sub_steps", []))
            passed = len(details.get(problem_id, []))
            total_subtasks += total
            total_passed_subtasks += passed

            if total > 0 and passed == total:
                successful_tasks.append(problem_id)
            else:
                failed_tasks.append(problem_id)

        accuracy = len(successful_tasks) / len(self.benchmark) if self.benchmark else 0
        subtask_accuracy = total_passed_subtasks / total_subtasks if total_subtasks > 0 else 0

        return {
            "accuracy": accuracy,
            "subtask_accuracy": subtask_accuracy,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks
        }


        
import importlib
import asyncio
import os
import json
import weave
import time
from typing import Dict, Any, Optional
from .benchmark_manager import BenchmarkManager
from .utils.vm_runner import VMRunner
from .utils.local_runner import LocalRunner

class AgentRunner:
    """Handles running agents either locally or on VMs"""

    def __init__(self, 
                 agent_function: str,
                 agent_dir: str,
                 agent_args: Dict[str, Any],
                 benchmark_name: str,
                 config: Dict[str, Any],
                 run_id: Optional[str] = None,
                 use_vm: bool = False,
                 max_concurrent: int = 1,
                 conda_env: Optional[str] = None,
                 continue_run: bool = False):
        
        # Initialize benchmark first
        self.benchmark_manager = BenchmarkManager(agent_dir, config)
        self.benchmark = self.benchmark_manager.get_benchmark(benchmark_name)
        self.benchmark.agent_args = agent_args
        
        print(f"Benchmark VM only: {self.benchmark.vm_only}")
        
        # Check if benchmark requires VM
        if self.benchmark.vm_only and not use_vm:
            raise ValueError(f"Benchmark {benchmark_name} requires VM execution. Please use --vm flag.")
        
        
        print(f"Benchmark setup script: {self.benchmark.setup_script}")
        # Set run ID
        self.run_id = run_id or f"{benchmark_name}_{int(time.time())}"
        
        # Initialize appropriate runner with benchmark
        if use_vm:
            self.runner = VMRunner(
                max_concurrent=max_concurrent,
                log_dir=self.benchmark.get_run_dir(self.run_id),
                benchmark=self.benchmark
            )
        else:
            self.runner = LocalRunner(
                max_concurrent=max_concurrent,
                conda_env=conda_env,
                log_dir=self.benchmark.get_run_dir(self.run_id),
                benchmark=self.benchmark
            )
        
        self.agent_function = agent_function
        self.agent_dir = agent_dir
        self.agent_args = agent_args
        self.config = config
        self.max_concurrent = max_concurrent
        self.conda_env = conda_env
        self.use_vm = use_vm
        self.continue_run = continue_run
        
        

    def get_remaining_tasks(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Get tasks that haven't been completed in previous runs"""
        if not self.continue_run or not self.run_id:
            return dataset
            
        # Check for raw submissions file
        run_dir = self.benchmark.get_run_dir(self.run_id)
        submissions_file = os.path.join(run_dir, f"{self.run_id}_RAW_SUBMISSIONS.jsonl")
        
        if not os.path.exists(submissions_file):
            print("No previous submissions found, running all tasks")
            return dataset
            
        try:
            # Load completed tasks from submissions file
            completed_tasks = set()
            with open(submissions_file) as f:
                for line in f:
                    submission = json.loads(line)
                    task_id = list(submission.keys())[0]
                    result = submission[task_id]
                    # Only count as completed if not an error
                    if not isinstance(result, str) or not result.startswith("ERROR"):
                        completed_tasks.add(task_id)
            
            # Filter out completed tasks
            remaining_tasks = {
                task_id: data 
                for task_id, data in dataset.items() 
                if task_id not in completed_tasks
            }
            
            print(f"Found {len(completed_tasks)} completed tasks from previous run")
            print(f"Remaining tasks: {len(remaining_tasks)}")
            
            return remaining_tasks
            
        except Exception as e:
            print(f"Error loading previous submissions: {e}")
            return dataset

    async def run(self, agent_name: str, upload: bool = False) -> Dict[str, Any]:
        """Run the full agent evaluation pipeline"""
        
        # Initialize logging for main run
        weave_client = weave.init(self.run_id)
        
        # Get dataset and filter for remaining tasks if continuing
        dataset = self.benchmark.get_dataset()
        dataset = self.get_remaining_tasks(dataset)
        
        if not dataset:
            print("No tasks to run")
            # Load and return previous results
            results_path = os.path.join(self.benchmark.get_run_dir(self.run_id), f"{self.run_id}_UPLOAD.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    previous_results = json.load(f)
                return previous_results["results"]
            return {}
        
        # Run agent on all tasks
        print("Running main evaluation...")
        agent_output = await self.runner.run_agent(
            dataset=dataset,
            agent_function=self.agent_function,
            agent_dir=self.agent_dir,
            agent_args=self.agent_args,
            run_id=self.run_id,
            benchmark=self.benchmark
        )
        
        # If continuing run, merge with previous results
        if self.continue_run:
            results_path = os.path.join(self.benchmark.get_run_dir(self.run_id), f"{self.run_id}_RAW_SUBMISSIONS.jsonl")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    previous_output = json.load(f)
                agent_output.update(previous_output)
        
        # Evaluate outputs
        print("Evaluating results...")
        eval_results = self.benchmark.evaluate_output(agent_output, self.run_id)
        
        # Process and return results
        print("Processing results...")
        return self.benchmark.process_results(
            agent_name=agent_name,
            run_id=self.run_id,
            eval_results=eval_results,
            weave_client=weave_client,
            upload=upload
        )
        
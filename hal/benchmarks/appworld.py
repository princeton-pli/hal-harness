import os
import json
from typing import Dict, Any, TypedDict, List
from typing_extensions import NotRequired
from .base_benchmark import BaseBenchmark
import docker
from hal.utils.logging_utils import print_warning

class AppWorldBenchmark(BaseBenchmark):
    """AppWorld benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = 'appworld_test_normal'):
        self.benchmark_name = benchmark_name
        self.split = 'test_normal' if benchmark_name == 'appworld_test_normal' else 'test_challenge'
        self.requirements_file = 'appworld'
        self.setup_script = 'hal/benchmarks/appworld/appworld_setup.sh'
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox, setup_script=self.setup_script)
        
        # Load dataset splits
        
        self.splits = {
            'train': self._read_task_ids('train.txt'),
            'dev': self._read_task_ids('dev.txt'), 
            'test_normal': self._read_task_ids('test_normal.txt'),
            'test_challenge': self._read_task_ids('test_challenge.txt')
        }
        
        # Create benchmark dictionary
        self.benchmark = {}
        if self.split == 'test_normal':
            for task_id in self.splits['test_normal']:
                self.benchmark[task_id] = {'task_id': task_id}
        elif self.split == 'test_challenge':
            for task_id in self.splits['test_challenge']:
                self.benchmark[task_id] = {'task_id': task_id}
            
    def _read_task_ids(self, filename: str) -> List[str]:
        """Read task IDs from a given file"""
        with open(os.path.join(os.getcwd(), 'hal', 'benchmarks', 'appworld', filename), 'r') as file:
            return [line.strip() for line in file.readlines()]

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs using AppWorld evaluation"""    
        # Create docker client
        client = docker.from_env()

        # Create and run container
        container = client.containers.run(
            'python:3.11',
            command='tail -f /dev/null',
            volumes={
                self.get_run_dir(run_id): {
                    'bind': '/home/',
                    'mode': 'rw',
                    'chmod': '777'
                }
            },
            working_dir='/home/',
            detach=True,
        )

        # Install appworld
        result = container.exec_run("pip install appworld", stream=True)
        for line in result.output:
            print(line.decode(), end='')

        # Run appworld commands 
        result = container.exec_run("appworld install", stream=True)
        for line in result.output:
            print(line.decode(), end='')

        result = container.exec_run("appworld download data", stream=True)
        for line in result.output:
            print(line.decode(), end='')

        # Create experiments directory
        result = container.exec_run("mkdir -p /home/experiments/outputs/output/tasks/", stream=True)
        for line in result.output:
            print(line.decode(), end='')

        for task_id in self.splits[self.split]:
            result = container.exec_run(f"cp -r {task_id}/experiments/outputs/output/tasks/{task_id} /home/experiments/outputs/output/tasks/", stream=True)
            for line in result.output:
                print(line.decode(), end='')
        
        # Run evaluation
        try:
            result = container.exec_run(f"appworld evaluate output {self.split}", stream=True)
            for line in result.output:
                print(line.decode(), end='')
        except Exception as e:
            print_warning("Likely because not all tasks have been finished. Please continue the run to complete all tasks.")
            
        # set permissions of the results folder
        result = container.exec_run("chmod -R 777 .", stream=True)
        for line in result.output:
            print(line.decode(), end='')

        # Clean up
        container.stop()
        container.remove()
        client.images.remove('python:3.11', force=True)
        
        # Read in evaluation results
        run_dir = self.get_run_dir(run_id)
        with open(os.path.join(run_dir, 'experiments', 'outputs', 'output', 'evaluations', f'{self.split}.json'), 'r') as file:
            eval_results = json.load(file)
        
        return eval_results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.
        
        Args:
            eval_results: Dictionary containing evaluation results
            
        Returns:
            Dictionary with calculated metrics and task lists
        """
        metrics = {}
        
        # Get aggregate scores if available
        if "aggregate" in eval_results:
            aggregate = eval_results["aggregate"]
            for metric_name, value in aggregate.items():
                metrics[f"aggregate_{metric_name}"] = value
                
        # if "aggregate_task_goal_completion" in metrics rename to "accuracy"
        if "aggregate_task_goal_completion" in metrics:
            metrics["accuracy"] = metrics.pop("aggregate_task_goal_completion")
        
        # Process individual results to get successful and failed task IDs
        if "individual" in eval_results:
            individual = eval_results["individual"]
            
            successful_tasks = []
            failed_tasks = []
            
            for task_id, result in individual.items():
                if result.get("success", False):
                    successful_tasks.append(task_id)
                else:
                    failed_tasks.append(task_id)
            
            # Add task lists to metrics
            metrics["successful_tasks"] = successful_tasks
            metrics["failed_tasks"] = failed_tasks
            
        return metrics

    def mount_benchmark(self):
        """Mount AppWorld benchmark environment - not needed since we use pip install"""
        pass


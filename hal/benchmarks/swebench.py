import json
import os
from typing import Dict, Any, List
from .base_benchmark import BaseBenchmark
from datasets import load_dataset
import subprocess

class SWEBenchBenchmark(BaseBenchmark):
    """SWEBench benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], mini: bool = False):
        self.benchmark_name = 'swebench_verified_mini' if mini else 'swebench_verified'
        self.vm_only = False
        self.mini = mini
        super().__init__(agent_dir, config, vm_only=self.vm_only)
        
        # Read mini instance ids
        with open('hal/benchmarks/swebench_verified_mini_task_ids.txt', 'r') as f:
            self.mini_instance_ids = [line.strip() for line in f.readlines()]
        
        # download swebench verified from huggingface
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        self.benchmark = {}
        
        # Load benchmark dataset
        if mini:
            for task in ds:
                if task['instance_id'] in self.mini_instance_ids:
                    self.benchmark[task['instance_id']] = task
        else:
            for task in ds:
                self.benchmark[task['instance_id']] = task
                

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate SWEBench submissions"""
        run_dir = self.get_run_dir(run_id)
        
        results = []
        for task_id, result in agent_output.items():
            results.append({
                'instance_id': task_id,
                'model_patch': result,
                'model_name_or_path': self.benchmark_name
            })
        
        # Save submissions to file
        submissions_path = os.path.join(run_dir, f"{run_id}_SWE_BENCH_SUBMISSIONS.jsonl")
        with open(submissions_path, 'w') as f:
            for result in results:
                json.dump(result, f)
                f.write('\n')
        
        
        command = ['conda', 'run', '-n', 'swebench_hal', 'python', '-m', 'swebench.harness.run_evaluation',
           '--dataset_name', 'princeton-nlp/SWE-bench_Verified',
           '--predictions_path', submissions_path,
           '--max_workers', '6',
           '--run_id', run_id]

        try:
            subprocess.run(['conda', 'create', '-n', 'swebench_hal', 'python=3.11', '-y', '--force'], check=True)
            subprocess.run([
                'conda', 'run', 
                '-n', 'swebench_hal', 
                'pip', 'install', 
                '-e', 'git+https://github.com/benediktstroebl/SWE-bench.git#egg=swebench'], check=True)

            subprocess.run(command, check=True)
            
            # Load the evaluation results
            with open(f"{self.benchmark_name}.{run_id}.json", 'r') as f:
                results = json.load(f)

            # delete file
            os.remove(f"{self.benchmark_name}.{run_id}.json")
            
            # remove conda environment
            # subprocess.run(['conda', 'env', 'remove', '-n', 'swebench_hal', '--yes', '--all'], check=True)

            return results


        except subprocess.CalledProcessError as e:
            print(f"Error running SWE-bench evaluation harness: {e}")
            print(f"Stdout: {e.output}")
            print(f"Stderr: {e.stderr}")
            raise

        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        total_instances = 50 if self.mini else eval_results['total_instances']
        return {
            "accuracy": eval_results['resolved_instances']/total_instances,
            'successful_tasks': list(set(eval_results['resolved_ids'])),
            'failed_tasks': list(set(eval_results['unresolved_ids'] + eval_results['error_ids']))
        } 
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
        self.requirements_file = 'swebench'
        self.vm_only = False
        super().__init__(agent_dir, config, vm_only=self.vm_only)
        
        # Define mini dataset instance IDs
        self.mini_instance_ids = ['django__django-11206',
            'django__django-12262',
            'django__django-12419',
            'django__django-13128',
            'django__django-13346',
            'django__django-13809',
            'django__django-14725',
            'django__django-15569',
            'django__django-16493',
            'django__django-7530',
            'pydata__xarray-3151',
            'pylint-dev__pylint-4551',
            'pytest-dev__pytest-5631',
            'sphinx-doc__sphinx-9367',
            'sympy__sympy-15017',
            'sympy__sympy-15599',
            'sympy__sympy-17318',
            'sympy__sympy-18763',
            'sympy__sympy-22914',
            'sympy__sympy-23824',
            'django__django-14855',
            'django__django-15128',
            'django__django-11333',
            'django__django-14053',
            'django__django-11099',
            'django__django-13658',
            'astropy__astropy-7336',
            'django__django-12304',
            'django__django-14011',
            'django__django-16901',
            'django__django-15930',
            'matplotlib__matplotlib-24970',
            'django__django-15629',
            'scikit-learn__scikit-learn-11310',
            'psf__requests-1766',
            'pallets__flask-5014',
            'matplotlib__matplotlib-23476',
            'sympy__sympy-13551',
            'sympy__sympy-13480',
            'django__django-16950',
            'sympy__sympy-15349',
            'sphinx-doc__sphinx-9281',
            'scikit-learn__scikit-learn-10908',
            'sympy__sympy-12481',
            'sympy__sympy-15875',
            'sphinx-doc__sphinx-9461',
            'sympy__sympy-20428',
            'sympy__sympy-16450',
            'scikit-learn__scikit-learn-25747',
            'scikit-learn__scikit-learn-25973'
        ]
        
        # download swebench verified from huggingface
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        self.benchmark = {}

        
        # Set instance IDs based on mini flag
        self.instance_ids = self.mini_instance_ids if mini else None
        
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
            'successful_tasks': set(eval_results['resolved_ids']),
            'failed_tasks': set(eval_results['unresolved_ids'] + eval_results['error_ids'])
        } 
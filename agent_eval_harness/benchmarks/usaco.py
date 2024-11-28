import subprocess
import json
import shutil
import os
import tempfile
import sys
import time
from typing import Dict, Any
from typing_extensions import NotRequired, TypedDict
from datetime import datetime

from .base_benchmark import BaseBenchmark
from ..utils.weave_utils import get_total_cost, assert_task_id_logging



class USACOBenchmark(BaseBenchmark):
    """USACO benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = 'usaco'
        self.requirements_file = 'usaco'
        self.vm_only = False
        super().__init__(agent_dir, config, vm_only=self.vm_only)
        
        # Load benchmark dataset
        dataset_path = os.path.join(
            os.path.dirname(__file__), 
            'USACO/data/datasets/usaco_subset307_dict.json'
        )
        with open(dataset_path) as f:
            self.benchmark = json.load(f)
            
        # For testing, limit to 1 task
        self.benchmark = {
            k: v for k, v in self.benchmark.items() 
            if k in list(self.benchmark.keys())[:1]
        }
        
        # Set benchmark directory
        self.benchmark_dir = os.path.join(os.path.dirname(__file__), 'USACO')


    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Run USACO evaluation harness on agent outputs"""
        try:
            # Mount benchmark environment
            self.mount_benchmark()
            
            # Write outputs to temp file
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                json.dump(agent_output, temp_file)
                temp_file_path = temp_file.name
            
            # Run evaluation command
            cmd = f"cd {self.benchmark_dir} && poetry run python harness.py --problem_dict_with_responses {temp_file_path} --run_id {run_id}"
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output
            while True:
                stdout = process.stdout.readline()
                stderr = process.stderr.readline()
                
                if stdout:
                    print(stdout.strip())
                if stderr:
                    print(stderr.strip(), file=sys.stderr)
                    
                if stdout == '' and stderr == '' and process.poll() is not None:
                    break
                    
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, 
                    cmd
                )
                
            # Parse results
            rdict, sdict, rs, ss = self._parse_evaluation_result(run_id)
            return {
                "rdict": rdict,
                "sdict": sdict,
                "rs": rs,
                "ss": ss
            }

        finally:
            # Cleanup
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            self.unmount_benchmark()

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        sdict = eval_results["sdict"]
        
        return {
            "accuracy": sum(
                1 if float(sdict[key][0]['result']['fraction_passed']) == 1 
                else 0 for key in sdict
            ) / len(sdict),
            'successful_tasks': [
                key for key in sdict 
                if float(sdict[key][0]['result']['fraction_passed']) == 1
            ],
            'failed_tasks': [
                key for key in sdict 
                if float(sdict[key][0]['result']['fraction_passed']) < 1
            ]
        }

    def mount_benchmark(self):
        """Mount USACO benchmark environment"""
        try:
            requirements_path = f'agent_eval_harness/benchmarks/requirements/{self.requirements_file}.toml'
            with open(requirements_path) as f:
                requirements = f.read()
        except FileNotFoundError:
            raise ValueError(f"Requirements file for {self.benchmark_name} not found")

        with open('pyproject.toml', 'w') as f:
            f.write(f"""[tool.poetry]\npackage-mode = false\n\n{requirements}""")

        print(f"Mounting {self.benchmark_name} benchmark environment...")
        subprocess.run(
            'poetry env use python3 && poetry lock && poetry install --no-root',
            shell=True,
            check=True
        )
        print("Benchmark environment mounted")

    def unmount_benchmark(self):
        """Cleanup benchmark environment"""
        if os.path.exists('pyproject.toml'):
            os.remove('pyproject.toml')
        if os.path.exists('poetry.lock'):
            os.remove('poetry.lock')
            
        # Cleanup benchmark directories
        shutil.rmtree(f'{self.benchmark_dir}/results', ignore_errors=True)
        shutil.rmtree(f'{self.benchmark_dir}/judge_sandbox', ignore_errors=True)
        shutil.rmtree(f'{self.benchmark_dir}/code_sandbox', ignore_errors=True)

    def _parse_evaluation_result(self, run_id: str):
        """Parse evaluation results from files"""
        results_dir = f'{self.benchmark_dir}/results'
        
        with open(f'{results_dir}/sdict_{run_id}.json') as f:
            sdict = json.load(f)
        with open(f'{results_dir}/rdict_{run_id}.json') as f:
            rdict = json.load(f)
            
        # Cleanup result files
        os.remove(f'{results_dir}/sdict_{run_id}.json')
        os.remove(f'{results_dir}/rdict_{run_id}.json')
        
        return rdict, sdict, list(rdict.values()), list(sdict.values())

        
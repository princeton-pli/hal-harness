# benchmark_manager.py

import importlib
from typing import Dict, Any, Optional
from .benchmarks.base_benchmark import BaseBenchmark
from .benchmarks.swe_bench import SWEBenchBenchmark
from .benchmarks.usaco import USACOBenchmark
import subprocess
import os


class BenchmarkManager:
    def __init__(self, config: Optional[Dict[str, Any]] = {}):
        self.config = config
        self.benchmarks = {
            "swebench_lite": SWEBenchBenchmark(config),
            "usaco": USACOBenchmark(config)
        }

    def get_benchmark(self, benchmark_name: str) -> BaseBenchmark:
    
        benchmark = self.benchmarks.get(benchmark_name)
        if benchmark is None:
            raise ValueError(f"Benchmark '{benchmark_name}' not found. Available benchmarks: {', '.join(self.benchmarks.keys())}")
        
        return benchmark

    def list_benchmarks(self) -> list[str]:
        return list(self.benchmarks.keys())

    
    def mount_benchmark(self, benchmark_name: str):
        try:
            with open(f'agent_eval_harness/benchmarks/requirements/{benchmark_name}.toml', 'r') as f:
                requirements_toml = f.read()
        except FileNotFoundError:
            raise ValueError(f"Requirements file for benchmark '{benchmark_name}' not found. Available benchmarks: {', '.join(self.benchmarks.keys())}")
        with open('pyproject.toml', 'w') as f:
            f.write(f"""[tool.poetry]
                        package-mode = false\n\n{requirements_toml}""")

        # install dependencies
        try:
            print(f"Installing dependencies for benchmark '{benchmark_name}'")
            # this command install the benchmark dependencies in a virtual environment separate from the agent_eval_harness and agent dependencies to run the evals
            subprocess.run(['poetry env use python3 && poetry install --no-root'], shell=True)
            print(f"Done!")
        except Exception as e:
            print(e)
            raise ValueError(f"Failed to install dependencies for benchmark '{benchmark_name}'")

    def unmount_benchmark(self, benchmark_name: str):
        if os.path.exists('pyproject.toml'):
            os.remove('pyproject.toml')
        if os.path.exists('poetry.lock'):
            os.remove('poetry.lock')



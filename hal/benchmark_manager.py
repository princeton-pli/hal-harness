# benchmark_manager.py

import importlib
from typing import Dict, Any, Optional
from .benchmarks.base_benchmark import BaseBenchmark
from .benchmarks.inspect_benchmark import InspectBenchmark
# from .benchmarks.swe_bench import SWEBenchBenchmark
# from .benchmarks.usaco import USACOBenchmark
# from .benchmarks.mlagentbench import MLAgentBenchBenchmark
import subprocess
import os


class BenchmarkManager:
    def __init__(self, agent_dir: str = 'agent/', config: Optional[Dict[str, Any]] = {}, agent_args: Optional[Dict[str, Any]] = {}):
        self.config = config
        self.agent_dir = agent_dir
        self.agent_args = agent_args
        self.benchmarks = ['usaco', 
                           'swebench_verified', 
                           'swebench_verified_mini', 
                           'appworld_test_normal',
                           'appworld_test_challenge',
                           'taubench_retail',
                           'taubench_airline',
                           'inspect_evals/gaia',
                           'inspect_evals/cybench',
                           'inspect_evals/appworld',
                           'inspect_evals/agentharm',
                           'inspect_evals/agentharm_benign']

    def get_benchmark(self, benchmark_name: str) -> BaseBenchmark:
        """Get benchmark instance for given name"""
        if benchmark_name.startswith("inspect:") or benchmark_name.startswith("inspect_evals/"):
            return InspectBenchmark(self.agent_dir, self.config, benchmark_name, self.agent_args)
        elif benchmark_name == "usaco":
            from .benchmarks.usaco import USACOBenchmark
            benchmark = USACOBenchmark(self.agent_dir, self.config)
        elif benchmark_name == "mlagentbench":
            from .benchmarks.mlagentbench import MLAgentBenchBenchmark
            benchmark = MLAgentBenchBenchmark(self.agent_dir, self.config)
        elif benchmark_name in ['swebench_verified', 'swebench_verified_mini']:
            from .benchmarks.swebench import SWEBenchBenchmark
            if benchmark_name == 'swebench_verified_mini':
                benchmark = SWEBenchBenchmark(self.agent_dir, self.config, mini=True)
            else:    
                benchmark = SWEBenchBenchmark(self.agent_dir, self.config, mini=False)
        elif benchmark_name in ['appworld_test_normal', 'appworld_test_challenge']:
            from .benchmarks.appworld import AppWorldBenchmark
            benchmark = AppWorldBenchmark(self.agent_dir, self.config, benchmark_name)
        elif benchmark_name in ['taubench_retail', 'taubench_airline']:
            from .benchmarks.taubench import TauBenchBenchmark
            benchmark = TauBenchBenchmark(self.agent_dir, self.config, benchmark_name, self.agent_args)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        return benchmark

    def list_benchmarks(self) -> list[str]:
        return self.benchmarks

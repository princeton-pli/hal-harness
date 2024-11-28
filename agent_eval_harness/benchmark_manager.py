# benchmark_manager.py

import importlib
from typing import Dict, Any, Optional
from .benchmarks.base_benchmark import BaseBenchmark
# from .benchmarks.swe_bench import SWEBenchBenchmark
# from .benchmarks.usaco import USACOBenchmark
# from .benchmarks.mlagentbench import MLAgentBenchBenchmark
import subprocess
import os


class BenchmarkManager:
    def __init__(self, agent_dir: str = 'agent/', config: Optional[Dict[str, Any]] = {}):
        self.config = config
        self.agent_dir = agent_dir
        self.benchmarks = ['usaco', 'mlagentbench', 'swebench_lite', 'swebench_verified', 'swebench_verified_mini', 'appworld']

    def get_benchmark(self, benchmark_name: str) -> BaseBenchmark:
        if benchmark_name == 'usaco':
            from .benchmarks.usaco import USACOBenchmark
            benchmark = USACOBenchmark(self.agent_dir, self.config)
        elif benchmark_name == 'mlagentbench':
            from .benchmarks.mlagentbench import MLAgentBenchBenchmark
            benchmark = MLAgentBenchBenchmark(self.agent_dir, self.config)
        elif benchmark_name in ['swebench_lite', 'swebench_verified', 'swebench_verified_mini']:
            from .benchmarks.swebench import SWEBenchBenchmark
            benchmark = SWEBenchBenchmark(self.agent_dir, self.config, benchmark_name)
        elif benchmark_name == 'appworld':
            from .benchmarks.appworld import AppWorldBenchmark
            benchmark = AppWorldBenchmark(self.agent_dir, self.config)
        else:
            raise ValueError(f"Unknown benchmark '{benchmark_name}'")
        
        return benchmark

    def list_benchmarks(self) -> list[str]:
        return self.benchmarks

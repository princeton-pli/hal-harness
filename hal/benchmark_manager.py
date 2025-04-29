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
        self.benchmarks = ['scicode',
                           'scicode_easy',
                           'scicode_hard',
                           'usaco', 
                           'swebench_verified', 
                           'swebench_verified_mini', 
                           'appworld_test_normal',
                           'appworld_test_challenge',
                           'taubench_retail',
                           'taubench_airline',
                           'gaia',
                           'inspect_evals/gaia',
                           'inspect_evals/cybench',
                           'inspect_evals/appworld',
                           'inspect_evals/agentharm',
                           'inspect_evals/agentharm_benign',
                           'corebench_easy',
                           'corebench_medium',
                           'corebench_hard',
                           'scienceagentbench',
                           'assistantbench',
                           ]

    def get_benchmark(self, benchmark_name: str) -> BaseBenchmark:
        """Get benchmark instance for given name"""
        if benchmark_name.startswith("inspect:") or benchmark_name.startswith("inspect_evals/"):
            return InspectBenchmark(self.agent_dir, self.config, benchmark_name, self.agent_args)
        elif benchmark_name in ["scicode", "scicode_easy", "scicode_hard"]:
            from .benchmarks.scicode import SciCodeBenchmark
            benchmark = SciCodeBenchmark(self.agent_dir, self.config, benchmark_name)
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
            benchmark = TauBenchBenchmark(self.agent_dir, self.config, benchmark_name)
        elif benchmark_name == 'gaia':
            from .benchmarks.gaia import GaiaBenchmark
            benchmark = GaiaBenchmark(self.agent_dir, self.config, benchmark_name)
        elif benchmark_name == 'corebench_easy':
            from .benchmarks.corebench import CoreBenchEasy
            benchmark = CoreBenchEasy(self.agent_dir, self.config)
        elif benchmark_name == 'corebench_medium':
            from .benchmarks.corebench import CoreBenchMedium
            benchmark = CoreBenchMedium(self.agent_dir, self.config)
        elif benchmark_name == 'corebench_hard':
            from .benchmarks.corebench import CoreBenchHard
            benchmark = CoreBenchHard(self.agent_dir, self.config)
        elif benchmark_name == 'scienceagentbench':
            from .benchmarks.scienceagentbench import ScienceAgentBench
            benchmark = ScienceAgentBench(self.agent_dir, self.config)
        elif benchmark_name == 'assistantbench':
            from .benchmarks.assistantbench import AssistantBenchBenchmark
            benchmark = AssistantBenchBenchmark(self.agent_dir, self.config)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        return benchmark

    def list_benchmarks(self) -> list[str]:
        return self.benchmarks

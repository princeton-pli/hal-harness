# benchmark_manager.py

import importlib
from typing import Dict, Any
from .benchmarks.base_benchmark import BaseBenchmark
from .benchmarks.swe_bench import SWEBenchBenchmark

class BenchmarkManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmarks = {
            "swebench_lite": SWEBenchBenchmark(config)
        }

    def get_benchmark(self, benchmark_name: str) -> BaseBenchmark:
    
        benchmark = self.benchmarks.get(benchmark_name)
        if benchmark is None:
            raise ValueError(f"Benchmark '{benchmark_name}' not found. Available benchmarks: {', '.join(self.benchmarks.keys())}")
        return benchmark

    def list_benchmarks(self) -> list[str]:
        return list(self.benchmarks.keys())


# Function to be used in other modules
def load_benchmark(config: Dict[str, Any], benchmark_name: str) -> BaseBenchmark:
    """
    Load a specific benchmark.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        benchmark_name (str): The name of the benchmark to load.
    
    Returns:
        BaseBenchmark: The loaded benchmark instance.
    """
    manager = BenchmarkManager(config)
    return manager.get_benchmark(benchmark_name)
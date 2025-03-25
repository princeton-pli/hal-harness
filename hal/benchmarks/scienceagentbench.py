from typing import Dict, Any, Optional
from ..base_benchmark import BaseBenchmark

class YourBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        super().__init__(agent_dir, config)
        self.benchmark_name = "your_benchmark"
        # Benchmark dataset
        self.benchmark: Dict[str, Any]

        # Optional: Set if benchmark requires VM execution
        self.vm_only = False
        # Optional: Path to VM setup script
        self.setup_script = "hal/benchmarks/your_benchmark/setup.sh"
        
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        pass
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        pass
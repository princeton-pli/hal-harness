from typing import Dict, Any, Optional
from .base_benchmark import BaseBenchmark

from datasets import load_dataset

class ScienceAgentBench(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "scienceagentbench"

        ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
        self.benchmark = {}

        for task in ds:
            self.benchmark[task['instance_id']] = {}
            for key in task:
                self.benchmark[task['instance_id']][key] = task[key]

        # Optional: Set if benchmark requires VM execution
        self.vm_only = False
        # Optional: Path to VM setup script
        self.setup_script = "hal/benchmarks/your_benchmark/setup.sh"
        super().__init__(agent_dir, config)
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        pass
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        pass
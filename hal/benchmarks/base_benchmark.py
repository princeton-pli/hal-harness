from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
from pydantic import BaseModel, TypeAdapter
import json
import os
from inspect_ai.log import EvalLog, write_eval_log
from datetime import datetime
from ..utils.weave_utils import get_total_cost, get_weave_calls
from ..utils.logging_utils import print_warning
from ..utils.utils import make_json_serializable


class BaseBenchmark(ABC):
    """Base class for all benchmarks"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], vm_only: bool = False, setup_script: Optional[str] = None):
        self.agent_dir = agent_dir
        self.config = config
        self.benchmark_name: str
        self.requirements_file: str
        self.setup_script = setup_script # Path to setup script relative to benchmark dir
        self.base_results_dir = "results"
        self.benchmark_results_dir = os.path.join(self.base_results_dir, self.benchmark_name)
        self.agent_args: Dict[str, Any] = {}  # Store agent args
        self.vm_only = vm_only # Whether benchmark requires VM execution
        

    @abstractmethod
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs"""
        raise NotImplementedError("Benchmark must implement evaluate_output")
        
    def get_dataset(self) -> Dict[str, Any]:
        """Get the benchmark dataset. Override if needed."""
        return self.benchmark

    def get_run_dir(self, run_id: str) -> str:
        """Get the results directory for a specific run"""
        run_dir = os.path.join(self.benchmark_results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def process_results(self, 
                       agent_name: str,
                       run_id: str, 
                       agent_args: Dict[str, Any],
                       run_command: str,
                       eval_results: Dict[str, Any],
                       weave_client,
                       upload: bool = False) -> Dict[str, Any]:
        """Process evaluation results and optionally upload"""
        
        # Get run directory
        run_dir = self.get_run_dir(run_id)
        
        # Store raw results
        if isinstance(eval_results, EvalLog):
            # read json file that does not contain "SUBMISSION" and is most recently created (workaround because we cant set the filename and not convert the eval_results to json serializable variable from inspect output)
            json_files = [f for f in os.listdir(run_dir) if f.endswith('.json') and "SUBMISSION" not in f]
            latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(run_dir, x)))
            with open(os.path.join(run_dir, latest_file), 'r') as f:
                inspect_eval_results = json.load(f)
        else:
            results_path = os.path.join(run_dir, f"{run_id}.json")
            with open(results_path, 'w') as f:
                json.dump(eval_results, f)

        # Get cost and usage metrics
        total_cost, total_usage = get_total_cost(weave_client)
        raw_logging, latency_dict = get_weave_calls(weave_client)

        # Prepare results summary
        results_summary = {
            "config": {
                'agent_name': agent_name,
                'benchmark_name': self.benchmark_name,
                'date': datetime.now().strftime("%Y-%m-%d"),
                'run_id': run_id,
                'agent_args': agent_args,
                'run_command': run_command
            },
            "results": {**self.get_metrics(eval_results), 
                        'total_cost': total_cost, 
                        'latencies': latency_dict
            },
            "raw_eval_results": inspect_eval_results if isinstance(eval_results, EvalLog) else eval_results,
            "raw_logging_results": raw_logging,
            "total_usage": total_usage,
            'total_cost': total_cost,
        }
        
        # Save full results
        upload_path = os.path.join(run_dir, f"{run_id}_UPLOAD.json")
        try:
            with open(upload_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
        except TypeError as e:
            print_warning(f"Error serializing results summary: {e}. Converting to json serializable.")
            with open(upload_path, 'w') as f:
                json.dump(make_json_serializable(results_summary), f, indent=2)

        if upload:
            self.upload_results(run_id, results_summary)

        return results_summary["results"]

    @abstractmethod
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        pass

    def upload_results(self, run_id: str, results: Dict[str, Any]):
        """Upload results to storage. Override if needed."""
        pass




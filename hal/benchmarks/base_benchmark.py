from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os

from datetime import datetime
from ..utils.weave_utils import get_total_cost, get_weave_calls
from ..utils.utils import make_json_serializable, get_git_info, compute_agent_dir_hash
import logging

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for all benchmarks"""

    def __init__(
        self,
        agent_dir: str,
        config: Dict[str, Any],
        requires_sandbox: bool = False,
        setup_script: Optional[str] = None,
    ):
        self.agent_dir = agent_dir
        self.config = config
        self.benchmark_name: str
        self.setup_script = (
            setup_script  # Path to setup script relative to benchmark dir
        )
        self.base_results_dir = "results"
        self.benchmark_results_dir = os.path.join(
            self.base_results_dir, self.benchmark_name
        )
        self.agent_args: Dict[str, Any] = {}  # Store agent args
        self.requires_sandbox = (
            requires_sandbox  # Whether benchmark requires VM execution
        )

    def _normalize_agent_output(self, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize agent output to handle both old and new formats:
        - Old format: {task_id: response}
        - New format: {task_id: {answer: response, metrics: metrics}}
        Returns normalized format: {task_id: response}
        """
        normalized = {}
        for task_id, task_data in agent_output.items():
            if isinstance(task_data, dict) and "answer" in task_data:
                # New format: extract the answer
                normalized[task_id] = task_data["answer"]
            else:
                # Old format: use as-is
                normalized[task_id] = task_data
        return normalized

    @abstractmethod
    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
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

    def get_task_prompts(self) -> Dict[str, str]:
        """Extract task prompts from the benchmark dataset."""
        prompt_keys = ["prompt", "problem_statement", "problem_description", "task"]
        prompts = {}
        for task_id, task_data in self.benchmark.items():
            if isinstance(task_data, dict):
                for key in prompt_keys:
                    if key in task_data:
                        prompts[task_id] = str(task_data[key])
                        break
        return prompts

    def process_results(
        self,
        agent_name: str,
        run_id: str,
        agent_args: Dict[str, Any],
        run_command: str,
        eval_results: Dict[str, Any],
        weave_client,
        agent_output: Dict[str, Any] = None,
        upload: bool = False,
        agent_dir: Optional[str] = None,
        agent_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process evaluation results and optionally upload"""

        # Get run directory
        run_dir = self.get_run_dir(run_id)

        # Store raw results
        results_path = os.path.join(run_dir, f"{run_id}.json")
        with open(results_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        # Extract task metrics from agent output if available
        task_metrics = {}
        if agent_output:
            for task_id, task_data in agent_output.items():
                if isinstance(task_data, dict) and "metrics" in task_data:
                    task_metrics[task_id] = task_data["metrics"]

        # Extract step counts from task metrics
        task_step_counts = {}
        for task_id, metrics in task_metrics.items():
            if "step_count" in metrics:
                task_step_counts[task_id] = metrics["step_count"]

        # Get cost and usage metrics
        total_cost, total_usage = get_total_cost(weave_client)
        raw_logging, latency_dict = get_weave_calls(weave_client)

        # Build config with optional agent scaffold info
        config = {
            "agent_name": agent_name,
            "benchmark_name": self.benchmark_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "run_id": run_id,
            "agent_args": agent_args,
            "run_command": run_command,
        }
        if agent_version:
            config["agent_version"] = agent_version

        # Compute agent hash
        agent_hash = compute_agent_dir_hash(agent_dir) if agent_dir else None

        # Read wall-clock times if available
        wall_clock_times = {}
        timings_file = os.path.join(run_dir, f"{run_id}_WALL_CLOCK_TIMES.jsonl")
        if os.path.exists(timings_file):
            with open(timings_file) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        wall_clock_times[entry["task_id"]] = entry["wall_clock_time"]

        # Prepare results summary
        results_summary = {
            "config": config,
            "results": {
                **self.get_metrics(eval_results),
                "total_cost": total_cost,
                "latencies": latency_dict,
            },
            "raw_eval_results": eval_results,
            "raw_logging_results": raw_logging,
            "task_prompts": {
                task_id: prompt
                for task_id, prompt in self.get_task_prompts().items()
                if task_id in eval_results
            },
            "total_usage": total_usage,
            "total_cost": total_cost,
            "git_info": get_git_info(),
            "agent_hash": agent_hash,
            "wall_clock_times": wall_clock_times,
            "task_step_counts": task_step_counts,
        }

        # Include task metrics if available from agent output
        if task_metrics:
            results_summary["task_metrics"] = task_metrics

        # Save full results
        upload_path = os.path.join(run_dir, f"{run_id}_UPLOAD.json")
        try:
            with open(upload_path, "w") as f:
                json.dump(results_summary, f, indent=2)
        except TypeError as e:
            logger.warning(
                f"Error serializing results summary: {e}. Converting to json serializable."
            )
            with open(upload_path, "w") as f:
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

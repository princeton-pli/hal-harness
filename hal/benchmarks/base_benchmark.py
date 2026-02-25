from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os

from datetime import datetime
from ..utils.weave_utils import get_total_cost, get_weave_calls
from ..utils.utils import make_json_serializable, get_git_info
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
        base_results_dir: str = "results",
    ):
        self.agent_dir = agent_dir
        self.config = config
        self.benchmark_name: str
        self.setup_script = (
            setup_script  # Path to setup script relative to benchmark dir
        )
        self.base_results_dir = base_results_dir
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
        prompt_sensitivity: bool = False,
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

        # Get cost and usage metrics
        total_cost, total_usage = get_total_cost(weave_client)
        raw_logging, latency_dict = get_weave_calls(weave_client)

        # Calculate prompt sensitivity metrics if enabled
        sensitivity_metrics = None
        if prompt_sensitivity:
            sensitivity_metrics = self._calculate_sensitivity_metrics(
                eval_results, agent_output
            )

            # Create flattened eval_results for get_metrics (use mean score across variations)
            flattened_eval_results = {}
            for task_id, variations in eval_results.items():
                if isinstance(variations, list) and len(variations) > 0:
                    # Calculate mean score across all variations
                    scores = [
                        v.get("score", 0) for v in variations if isinstance(v, dict)
                    ]
                    if scores:
                        mean_score = sum(scores) / len(scores)
                        # Create a dict in the format expected by the benchmark
                        # For TauBench, use 'reward'; for others, might be 'score'
                        flattened_eval_results[task_id] = {
                            "reward": mean_score,
                            "score": mean_score,
                        }
                    else:
                        flattened_eval_results[task_id] = {"reward": 0, "score": 0}
                else:
                    # Shouldn't happen, but handle gracefully
                    flattened_eval_results[task_id] = variations
            eval_results_for_metrics = flattened_eval_results
        else:
            eval_results_for_metrics = eval_results

        # Prepare results summary
        results_summary = {
            "config": {
                "agent_name": agent_name,
                "benchmark_name": self.benchmark_name,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "run_id": run_id,
                "agent_args": agent_args,
                "run_command": run_command,
                "prompt_sensitivity": prompt_sensitivity,
            },
            "results": {
                **self.get_metrics(eval_results_for_metrics),
                "total_cost": total_cost,
                "latencies": latency_dict,
            },
            "raw_eval_results": eval_results,
            "raw_logging_results": raw_logging,
            "total_usage": total_usage,
            "total_cost": total_cost,
            "git_info": get_git_info(),
        }

        # Add sensitivity metrics if available
        if sensitivity_metrics:
            results_summary["prompt_sensitivity_metrics"] = sensitivity_metrics

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

    def _calculate_sensitivity_metrics(
        self, eval_results: Dict[str, Any], agent_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate prompt sensitivity metrics from evaluation results.

        Args:
            eval_results: Dictionary containing evaluation results per task per variation
            agent_output: Dictionary containing agent outputs per task per variation

        Returns:
            Dictionary with sensitivity metrics including variance and per-task statistics
        """
        import numpy as np

        # Structure: eval_results should be {task_id: {variation_id: score}}
        # or {task_id: [scores]} depending on implementation

        task_variances = {}
        task_means = {}
        task_min_max_gaps = {}

        for task_id, variation_results in eval_results.items():
            # Extract scores for each variation
            scores = []

            if isinstance(variation_results, list):
                # List of scores per variation
                for result in variation_results:
                    if isinstance(result, dict):
                        # Support both 'score' and 'reward' keys (different benchmarks use different names)
                        score_value = result.get("score", result.get("reward", None))
                        if score_value is not None:
                            try:
                                scores.append(float(score_value))
                            except (ValueError, TypeError):
                                # Skip invalid scores
                                print(
                                    f"Warning: Could not convert score to float for task {task_id}: {score_value}"
                                )
                                continue
                    elif isinstance(result, (int, float)):
                        scores.append(float(result))
            elif isinstance(variation_results, dict):
                # Dict mapping variation_id to results
                for var_id, result in variation_results.items():
                    if isinstance(result, dict):
                        # Support both 'score' and 'reward' keys (different benchmarks use different names)
                        score_value = result.get("score", result.get("reward", None))
                        if score_value is not None:
                            try:
                                scores.append(float(score_value))
                            except (ValueError, TypeError):
                                # Skip invalid scores
                                print(
                                    f"Warning: Could not convert score to float for task {task_id}: {score_value}"
                                )
                                continue
                    elif isinstance(result, (int, float)):
                        scores.append(float(result))

            if len(scores) > 1:
                # Calculate metrics for this task
                task_means[task_id] = float(np.mean(scores))
                task_variances[task_id] = float(np.var(scores))
                task_min_max_gaps[task_id] = float(np.max(scores) - np.min(scores))

        # Calculate overall metrics
        if task_variances:
            overall_metrics = {
                "mean_variance": float(np.mean(list(task_variances.values()))),
                "std_variance": float(np.std(list(task_variances.values()))),
                "mean_min_max_gap": float(np.mean(list(task_min_max_gaps.values()))),
                "max_min_max_gap": float(np.max(list(task_min_max_gaps.values()))),
                "task_variances": task_variances,
                "task_means": task_means,
                "task_min_max_gaps": task_min_max_gaps,
                "num_tasks": len(task_variances),
            }
        else:
            overall_metrics = {
                "mean_variance": 0.0,
                "std_variance": 0.0,
                "mean_min_max_gap": 0.0,
                "max_min_max_gap": 0.0,
                "task_variances": {},
                "task_means": {},
                "task_min_max_gaps": {},
                "num_tasks": 0,
            }

        return overall_metrics

    def upload_results(self, run_id: str, results: Dict[str, Any]):
        """Upload results to storage. Override if needed."""
        pass

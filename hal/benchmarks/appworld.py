import os
from typing import Any, Dict, List

from .base_benchmark import BaseBenchmark


class AppWorldBenchmark(BaseBenchmark):
    """AppWorld benchmark implementation"""

    def __init__(
        self,
        agent_dir: str,
        config: Dict[str, Any],
        benchmark_name: str = "appworld_test_normal",
    ):
        self.benchmark_name = benchmark_name
        if benchmark_name not in ["appworld_test_normal", "appworld_test_challenge"]:
            raise ValueError(
                "Invalid benchmark name. Use 'appworld_test_normal' or 'appworld_test_challenge'."
            )
        self.split = benchmark_name.removeprefix("appworld_")
        self.setup_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "appworld", "setup.sh"
        )
        self.requires_sandbox = False
        super().__init__(
            agent_dir,
            config,
            requires_sandbox=self.requires_sandbox,
            setup_script=self.setup_script,
        )
        # Load dataset splits
        self.splits = {
            "train": self._read_task_ids("train.txt"),
            "dev": self._read_task_ids("dev.txt"),
            "test_normal": self._read_task_ids("test_normal.txt"),
            "test_challenge": self._read_task_ids("test_challenge.txt"),
        }
        # Create benchmark dictionary
        self.benchmark = {}
        benchmark_directory = os.path.join(os.getcwd(), "hal", "benchmarks", "appworld")
        data_directory = os.path.join(benchmark_directory, "data")
        if not os.path.exists(data_directory):
            raise FileNotFoundError(
                f"Data directory {data_directory} does not exist. "
                "Please download AppWorld data: "
                f"`pip install appworld && appworld download data --root {benchmark_directory}`"
            )
        for task_id in self.splits[self.split]:
            self.benchmark[task_id] = {
                "task_id": task_id,
                "files": {"data": data_directory},
            }

    def _read_task_ids(self, filename: str) -> List[str]:
        """Read task IDs from a given file"""
        file_path = os.path.join(os.getcwd(), "hal", "benchmarks", "appworld", filename)
        with open(file_path) as file:
            return [line.strip() for line in file.readlines()]

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        # NOTE: world.evaluate().to_dict() expected to to be called in the agent code itself.
        return agent_output

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.

        Args:
            eval_results: Dictionary containing evaluation results

        Returns:
            Dictionary with calculated metrics and task lists
        """
        metrics = {}
        if "aggregate" in eval_results:
            aggregate = eval_results["aggregate"]
            for metric_name, value in aggregate.items():
                metrics[f"aggregate_{metric_name}"] = value
        if "aggregate_task_goal_completion" in metrics:
            metrics["accuracy"] = metrics.pop("aggregate_task_goal_completion")
        if "individual" in eval_results:
            individual = eval_results["individual"]
            successful_tasks = []
            failed_tasks = []
            for task_id, result in individual.items():
                if result.get("success", False):
                    successful_tasks.append(task_id)
                else:
                    failed_tasks.append(task_id)
            metrics["successful_tasks"] = successful_tasks
            metrics["failed_tasks"] = failed_tasks
        return metrics

    def mount_benchmark(self):
        """Mount AppWorld benchmark environment - not needed since we use pip install"""
        pass

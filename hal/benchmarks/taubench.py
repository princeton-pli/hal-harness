from typing import Dict, Any
from .base_benchmark import BaseBenchmark


class TauBenchBenchmark(BaseBenchmark):
    """TauBench benchmark implementation"""

    def __init__(
        self,
        agent_dir: str,
        config: Dict[str, Any],
        benchmark_name: str = "taubench_retail",
    ):
        self.benchmark_name = benchmark_name
        self.split = "retail" if benchmark_name == "taubench_retail" else "airline"
        self.setup_script = "hal/benchmarks/taubench/taubench_setup.sh"
        self.requires_sandbox = False
        super().__init__(
            agent_dir,
            config,
            requires_sandbox=self.requires_sandbox,
            setup_script=self.setup_script,
        )

        # Create benchmark dictionary
        self.benchmark = {}
        if self.split == "retail":
            self.benchmark = {
                str(task_index): {
                    "env": "retail",
                    "user_strategy": "llm",
                    "user_model": "gpt-4o",
                    "task_split": "test",
                    "user_provider": "openai",
                    "task_index": task_index,
                }
                for task_index in range(115)
            }
        elif self.split == "airline":
            self.benchmark = {
                str(task_index): {
                    "env": "airline",
                    "user_strategy": "llm",
                    "user_model": "gpt-4o",
                    "task_split": "test",
                    "user_provider": "openai",
                    "task_index": task_index,
                }
                for task_index in range(50)
            }

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent outputs using AppWorld evaluation"""
        return agent_output

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.

        Args:
            eval_results: Dictionary containing evaluation results

        Returns:
            Dictionary with calculated metrics and task lists
        """
        reward = 0
        for task_id, task_output in eval_results.items():
            try:
                reward += task_output["reward"]
            except TypeError:
                print(f"Task {task_id} does not have a reward. Skipping...")

        number_of_tasks = len(self.benchmark)

        successful_tasks = []
        for task_id, task_output in eval_results.items():
            if not isinstance(task_output, str):
                if task_output["reward"] > 0:
                    successful_tasks.append(task_id)

        failed_tasks = []
        for task_id, task_output in eval_results.items():
            if isinstance(task_output, str):
                failed_tasks.append(task_id)
            elif task_output["reward"] == 0:
                failed_tasks.append(task_id)

        results = {
            "accuracy": reward / number_of_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }
        return results

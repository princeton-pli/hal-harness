import logging
from typing import Any, Dict

from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class TerminalBenchBenchmark(BaseBenchmark):
    """Terminal Bench 2.0 benchmark executed via Harbor."""

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "terminal_bench"
        self.setup_script = None
        self.requires_sandbox = False
        self.benchmark: Dict[str, Any] = {}
        super().__init__(
            agent_dir,
            config,
            requires_sandbox=self.requires_sandbox,
            setup_script=self.setup_script,
        )

    def uses_external_runner(self) -> bool:
        return True

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "terminal_bench is executed through Harbor and does not use AgentRunner."
        )

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        total_tasks = len(eval_results)
        if total_tasks == 0:
            return {
                "accuracy": 0.0,
                "successful_tasks": [],
                "failed_tasks": [],
            }

        successful_tasks = []
        failed_tasks = []

        for task_id, task_output in eval_results.items():
            reward = 0.0
            if isinstance(task_output, dict):
                try:
                    reward = float(task_output.get("reward", 0.0))
                except (TypeError, ValueError):
                    reward = 0.0

            if reward > 0:
                successful_tasks.append(task_id)
            else:
                failed_tasks.append(task_id)

        return {
            "accuracy": len(successful_tasks) / total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }

    def run_external_evaluation(self, **kwargs) -> Dict[str, Any]:
        from hal.terminal_bench_harbor import run_terminal_bench_harbor

        logger.info("Running terminal_bench through Harbor.")
        return run_terminal_bench_harbor(benchmark=self, **kwargs)

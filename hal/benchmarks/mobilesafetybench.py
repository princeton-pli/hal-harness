import os
import json
import dotenv

from typing import Dict, Any, Optional
from .base_benchmark import BaseBenchmark

from mobile_safety.eval import eval

dotenv.load_dotenv()
_WORK_PATH = os.getenv("MOBILE_SAFETY_HOME")


class MobileSafetyBench(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "mobilesafetybench"
        self.requirements_file = "mobilesafetybench"
        self.requires_sandbox = False
        super().__init__(agent_dir, config)

        self.benchmark_category = config.get("benchmark_category", "daily")
        self.agent_config = config.get(
            "agent_config",
            {
                "model": "gpt-4o-2024-08-06",
                "prompt_mode": "scot",
            },
        )

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        tasks = self._load_tasks(self.benchmark_category)

        results = []
        for task in tasks:
            result = eval(
                task_category=task["task_category"],
                task_id=task["task_id"],
                model=self.agent_config["model"],
                prompt_mode=self.agent_config["prompt_mode"],
                is_emu_already_open=True,
            )
            results.append(result)

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        metrics = {}

        goal_achievement = []
        harm_prevention = []

        for result in eval_results:
            goal_achievement.append(1 if result["goal_achievement"] else 0)
            harm_prevention.append(1 if result["harm_prevention"] else 0)

        metrics["goal_achievement_rate"] = sum(goal_achievement) / len(goal_achievement)
        metrics["harm_prevention_rate"] = sum(harm_prevention) / len(harm_prevention)

        return metrics

    def _load_tasks(self, benchmark_category: str):
        if benchmark_category == "daily":
            task_path = f"{_WORK_PATH}/asset/tasks/tasks.json"
        elif benchmark_category == "robustness":
            task_path = f"{_WORK_PATH}/asset/tasks/tasks_robustness_aug.json"
        else:
            raise ValueError(f"Invalid task category: {benchmark_category}")

        with open(task_path, "r") as f:
            tasks = json.load(f)

        return tasks

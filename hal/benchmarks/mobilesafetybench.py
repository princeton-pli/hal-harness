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
        self.benchmark = {}
        super().__init__(agent_dir, config)

        self.benchmark_category = "daily"

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        tasks = self._load_tasks(self.benchmark_category)
        
        model_name = agent_output.get("model_name", "gpt-4o-2024-05-13")
        prompt_mode = agent_output.get("prompt_mode", "scot")
        base_prompt_module = agent_output.get("base_prompt_module", None)

        results = []
        for task in tasks:
            result = eval(
                task_category=task["task_category"],
                task_id=task["task_id"],
                model=model_name,
                prompt_mode=prompt_mode,
                gui=False,
                base_prompt_module=base_prompt_module,
                is_emu_already_open=False,
            )
            results.append(result)

        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        metrics = {}

        goal_achievement = []
        harm_prevention = []

        for result in eval_results:
            goal_achievement.append(1 if result[0] else 0)
            harm_prevention.append(1 if result[1] else 0)

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

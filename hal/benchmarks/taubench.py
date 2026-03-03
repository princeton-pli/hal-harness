import logging
from typing import Dict, Any, Optional
from .base_benchmark import BaseBenchmark
from hal.utils.compliance_checkers import ComplianceMonitor
from hal.utils.structural_perturbations import (
    StructuralPerturbator,
    PerturbationStrength,
    PerturbationConfig,
)

logger = logging.getLogger(__name__)


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

        # Load actual tasks from tau-bench to get instructions
        try:
            if self.split == "retail":
                from tau_bench.envs.retail.tasks_test import TASKS
            elif self.split == "airline":
                from tau_bench.envs.airline.tasks_test import TASKS
            else:
                TASKS = []
        except ImportError:
            logger.warning(
                "Could not import tau-bench tasks. Prompt sensitivity will not be available."
            )
            TASKS = []

        # Create benchmark dictionary with instructions
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
                    "instruction": TASKS[task_index].instruction
                    if task_index < len(TASKS)
                    else None,
                }
                for task_index in range(min(115, len(TASKS)) if TASKS else 115)
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
                    "instruction": TASKS[task_index].instruction
                    if task_index < len(TASKS)
                    else None,
                }
                for task_index in range(min(50, len(TASKS)) if TASKS else 50)
            }

        # Initialize compliance monitor if enabled
        self.compliance_monitor = None
        if hasattr(self, "agent_args") and self.agent_args:
            if self.agent_args.get("enable_compliance_monitoring") == "true":
                constraints_str = self.agent_args.get("compliance_constraints", "")
                constraints = [
                    c.strip() for c in constraints_str.split(",") if c.strip()
                ]
                if constraints:
                    self.compliance_monitor = ComplianceMonitor(constraints=constraints)
                    logger.info(
                        f"Compliance monitoring enabled with {len(constraints)} constraints: {', '.join(constraints)}"
                    )

        # Initialize structural perturbator if enabled
        self.perturbator: Optional[StructuralPerturbator] = None
        if hasattr(self, "agent_args") and self.agent_args:
            if self.agent_args.get("enable_structural_perturbations") == "true":
                perturbation_type = self.agent_args.get("perturbation_type", "all")
                perturbation_strength = self.agent_args.get(
                    "perturbation_strength", "medium"
                )
                try:
                    strength_enum = PerturbationStrength(perturbation_strength)
                    config = PerturbationConfig.get_preset(strength_enum)
                except ValueError:
                    config = PerturbationConfig()
                self.perturbator = StructuralPerturbator(
                    perturbation_type=perturbation_type,
                    config=config,
                )
                logger.info(
                    f"Structural perturbations enabled: type={perturbation_type}, strength={perturbation_strength}"
                )

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
                logger.warning(f"Task {task_id} does not have a reward. Skipping...")

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

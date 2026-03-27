import os
import json
import weave
import time
import logging
from typing import Dict, Any, Optional
from .benchmark_manager import BenchmarkManager
from .utils.runner import Runner
from .utils.environment import Environment
from .utils.logging_utils import create_progress
from .utils.weave_utils import delete_calls
from .utils.fault_injection import FaultInjector

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs agent evaluations on benchmarks"""

    def __init__(
        self,
        agent_function: str,
        agent_dir: str,
        agent_args: Dict[str, Any],
        benchmark_name: str,
        config: Dict[str, Any],
        task_timeout: int,
        run_id: Optional[str] = None,
        max_concurrent: int = 1,
        conda_env: Optional[str] = None,
        continue_run: bool = False,
        run_command: str = "",
        ignore_errors: bool = False,
        max_tasks: Optional[int] = None,
        prompt_sensitivity: bool = False,
        num_variations: int = 3,
        variation_strength: str = "mild",
        variation_index: Optional[int] = None,
        results_dir: str = "results",
        task_ids: Optional[str] = None,
    ):
        # Validate agent_function format
        if not isinstance(agent_function, str) or "." not in agent_function:
            raise ValueError(
                "Invalid agent_function format. Must be in format 'module.function' (e.g., 'my_agent.run_agent')"
            )

        module_name, func_name = agent_function.rsplit(".", 1)
        if not module_name or not func_name:
            raise ValueError(
                "Invalid agent_function format. Both module and function names must be non-empty"
            )

        # Check for requirements.txt
        requirements_path = os.path.join(agent_dir, "requirements.txt")
        if not os.path.exists(requirements_path) and not conda_env:
            raise ValueError(
                f"No requirements.txt found in agent directory: {agent_dir}"
            )

        # Initialize benchmark
        self.benchmark_manager = BenchmarkManager()
        self.benchmark = self.benchmark_manager.get_benchmark(benchmark_name, agent_dir, config)
        self.benchmark.agent_args = agent_args

        # Override results directory if non-default
        if results_dir != "results":
            self.benchmark.base_results_dir = results_dir
            self.benchmark.benchmark_results_dir = os.path.join(
                results_dir, self.benchmark.benchmark_name
            )

        # Check if any task requires GPU
        has_gpu_task = False
        if hasattr(self.benchmark, "benchmark") and isinstance(
            self.benchmark.benchmark, dict
        ):
            for task_id, task_data in self.benchmark.benchmark.items():
                if isinstance(task_data, dict) and task_data.get("gpu", False):
                    has_gpu_task = True
                    break

        if has_gpu_task:
            logger.warning(
                "Warning: This benchmark contains tasks that require GPU. "
                "GPU tasks may not work correctly without dedicated GPU hardware."
            )

        self.run_command = run_command

        # Set run ID
        self.run_id = run_id or f"{benchmark_name}_{int(time.time())}"

        # Environment config
        self.environment = Environment(
            task_timeout=task_timeout,
            max_concurrent=max_concurrent,
        )

        # Initialize runner
        self.runner = Runner(
            max_concurrent=max_concurrent,
            conda_env=conda_env,
            log_dir=self.benchmark.get_run_dir(self.run_id),
            task_timeout=task_timeout,
        )

        self.agent_function = agent_function
        self.agent_dir = agent_dir
        self.agent_args = agent_args
        self.config = config
        self.max_concurrent = max_concurrent
        self.conda_env = conda_env
        self.continue_run = continue_run
        self.ignore_errors = ignore_errors
        self.max_tasks = max_tasks
        self.prompt_sensitivity = prompt_sensitivity
        self.num_variations = num_variations
        self.variation_strength = variation_strength
        self.variation_index = variation_index
        self.task_ids = task_ids

        # Initialize fault injector if enabled
        self.fault_injector = None
        if agent_args.get("enable_fault_injection") == "true":
            fault_rate = float(agent_args.get("fault_rate", "0.2"))
            max_recovery_attempts = int(agent_args.get("max_recovery_attempts", "3"))
            self.fault_injector = FaultInjector(
                fault_rate=fault_rate,
                config={"max_recovery_attempts": max_recovery_attempts},
            )
            logger.info(
                f"⚠️  Fault injection enabled (rate: {fault_rate * 100:.1f}%, max recoveries: {max_recovery_attempts})"
            )

    def get_remaining_tasks(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Get tasks that haven't been completed in previous runs"""
        run_dir = self.benchmark.get_run_dir(self.run_id)
        submissions_file = os.path.join(run_dir, f"{self.run_id}_RAW_SUBMISSIONS.jsonl")

        if not os.path.exists(submissions_file):
            logger.info("No previous submissions found, running all tasks")
            return dataset

        try:
            completed_tasks = set()
            previous_output = {}
            with open(submissions_file) as f:
                for line in f:
                    try:
                        submission = json.loads(line.strip())
                        task_id = list(submission.keys())[0]
                        result = submission[task_id]
                        if not isinstance(result, str) or not result.startswith("ERROR"):
                            completed_tasks.add(task_id)
                        previous_output.update(submission)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed line in submissions file: {e}"
                        )
                        continue

            return {
                task_id: data
                for task_id, data in dataset.items()
                if task_id not in completed_tasks
            }

        except Exception as e:
            logger.error(f"Error loading previous submissions: {e}")
            return dataset

    def _load_submissions(self, path: str) -> Dict[str, Any]:
        """Load all submissions from a JSONL file. Returns empty dict if file absent."""
        results: Dict[str, Any] = {}
        if not os.path.exists(path):
            return results
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.update(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed submission line: %.80s (%s)", line, e)
        return results

    async def _run_one_variation(
        self,
        dataset: Dict[str, Any],
        progress_label: str = "Running agents...",
    ) -> Dict[str, Any]:
        """Run one agent pass over a dataset."""
        with create_progress() as progress:
            task = progress.add_task(progress_label, total=len(dataset))
            return await self.runner.run_agent(
                dataset=dataset,
                agent_function=self.agent_function,
                agent_dir=self.agent_dir,
                agent_args=self.agent_args,
                run_id=self.run_id,
                benchmark=self.benchmark,
                task=task,
                progress=progress,
            )

    async def run(self, agent_name: str, upload: bool = False) -> Dict[str, Any]:
        """Run the full agent evaluation pipeline"""
        self.environment.validate()

        logger.info("Initializing logging with W&B Weave...")
        weave_client = weave.init(self.run_id)

        dataset = self.benchmark.get_dataset()
        if self.continue_run and not self.ignore_errors:
            dataset = self.get_remaining_tasks(dataset)
        elif self.continue_run and self.ignore_errors:
            dataset = {}

        if self.task_ids:
            requested_ids = set(tid.strip() for tid in self.task_ids.split(","))
            available_ids = set(dataset.keys())
            valid_ids = requested_ids & available_ids
            missing_ids = requested_ids - available_ids
            if missing_ids:
                logger.warning(
                    f"Task IDs not found in benchmark: {sorted(missing_ids)}"
                )
            if valid_ids:
                logger.info(f"Filtering to {len(valid_ids)} specific task IDs")
                dataset = {
                    task_id: dataset[task_id]
                    for task_id in dataset
                    if task_id in valid_ids
                }
            else:
                logger.error("No valid task IDs found. Exiting.")
                return {}

        if self.max_tasks and self.max_tasks > 0 and self.max_tasks < len(dataset):
            logger.info(f"Limiting to the first {self.max_tasks} tasks as requested")
            task_ids_list = list(dataset.keys())[: self.max_tasks]
            dataset = {task_id: dataset[task_id] for task_id in task_ids_list}

        # Handle prompt sensitivity if enabled
        prompt_variations_map = None
        single_variation_dataset = None
        if self.prompt_sensitivity:
            from .utils.prompt_variation import (
                PromptVariationGenerator,
                get_prompt_field_for_benchmark,
            )

            prompt_field = get_prompt_field_for_benchmark(self.benchmark.benchmark_name)
            generator = PromptVariationGenerator(
                num_variations=self.num_variations, strength=self.variation_strength
            )

            if self.variation_index is not None:
                logger.info(
                    f"Generating {self.variation_strength} variation {self.variation_index} for sensitivity testing..."
                )
                single_variation_dataset = (
                    generator.generate_single_variation_for_dataset(
                        dataset, prompt_field, self.variation_index
                    )
                )
                logger.info(
                    f"Generated variation {self.variation_index} for {len(single_variation_dataset)} tasks"
                )
            else:
                logger.info(
                    f"Generating {self.num_variations} {self.variation_strength} prompt variations for sensitivity testing..."
                )
                prompt_variations_map = generator.apply_variations_to_dataset(
                    dataset, prompt_field
                )
                logger.info(
                    f"Generated {self.variation_strength} prompt variations for {len(prompt_variations_map)} tasks"
                )

        if self.continue_run and not self.ignore_errors and dataset:
            logger.info("Cleaning up calls from previous run...")
            all_calls = weave_client.get_calls()
            calls_to_delete = {}
            for call in all_calls:
                task_id = call.attributes.get("weave_task_id")
                if task_id in dataset:
                    calls_to_delete.setdefault(task_id, []).append(call.id)
            for task_id, call_ids in calls_to_delete.items():
                if call_ids:
                    delete_calls(call_ids, weave_client)
            logger.info(f"Cleaned up calls for {len(calls_to_delete)} tasks")

        if not dataset:
            logger.warning("No remaining tasks to run")
            results_path = os.path.join(
                self.benchmark.get_run_dir(self.run_id), f"{self.run_id}_UPLOAD.json"
            )
            if os.path.exists(results_path):
                logger.info("Loading previous results...")
                with open(results_path) as f:
                    previous_results = json.load(f)
                return previous_results["results"]
            else:
                logger.info(
                    "No previous results found. Running evaluation harness on previous raw submissions..."
                )
                submissions_path = os.path.join(
                    self.benchmark.get_run_dir(self.run_id),
                    f"{self.run_id}_RAW_SUBMISSIONS.jsonl",
                )
                agent_output = self._load_submissions(submissions_path)

        else:
            # Determine execution mode without mutating self.prompt_sensitivity
            running_single_variation = (
                self.prompt_sensitivity and self.variation_index is not None
            )
            running_multi_variation = (
                self.prompt_sensitivity and self.variation_index is None
            )

            if running_single_variation:
                var_idx = self.variation_index
                logger.info(
                    f"Running variation {var_idx} ({self.variation_strength}) on {len(single_variation_dataset)} tasks..."
                )
                agent_output = await self._run_one_variation(
                    single_variation_dataset,
                    progress_label=f"Running agents on variation {var_idx}...",
                )

            elif running_multi_variation:
                all_variations_output = {}
                num_vars = self.num_variations + 1  # +1 for original

                for var_idx in range(num_vars):
                    logger.info(f"Running variation {var_idx + 1}/{num_vars}...")
                    var_dataset = {
                        task_id: variations_list[var_idx]
                        for task_id, variations_list in prompt_variations_map.items()
                        if var_idx < len(variations_list)
                    }
                    var_output = await self._run_one_variation(
                        var_dataset,
                        progress_label=f"Running agents on variation {var_idx + 1}...",
                    )
                    for task_id, output in var_output.items():
                        if task_id not in all_variations_output:
                            all_variations_output[task_id] = []
                        all_variations_output[task_id].append(
                            {"variation_id": var_idx, "output": output}
                        )

                agent_output = all_variations_output

            else:
                # Normal mode
                agent_output = await self._run_one_variation(
                    dataset,
                    progress_label="Running agents... (check logs in results directory for more details)",
                )

                if self.continue_run:
                    submissions_path = os.path.join(
                        self.benchmark.get_run_dir(self.run_id),
                        f"{self.run_id}_RAW_SUBMISSIONS.jsonl",
                    )
                    previous_output = self._load_submissions(submissions_path)
                    agent_output.update(previous_output)

        logger.info("Evaluating results...")

        # Use local variable to track which evaluation path we're in
        # (avoids mutating self.prompt_sensitivity mid-run)
        is_multi_variation_eval = self.prompt_sensitivity and self.variation_index is None

        if is_multi_variation_eval:
            weave.finish()
            eval_results = {}
            for task_id, variations in agent_output.items():
                eval_results[task_id] = []
                for var_data in variations:
                    var_id = var_data["variation_id"]
                    var_output = var_data["output"]
                    single_output = {task_id: var_output}
                    var_eval = self.benchmark.evaluate_output(single_output, self.run_id)
                    if task_id in var_eval:
                        eval_result = var_eval[task_id]
                        if isinstance(eval_result, dict):
                            score = eval_result.get(
                                "score",
                                eval_result.get("reward", eval_result.get("accuracy", 0)),
                            )
                            if isinstance(score, dict):
                                for v in eval_result.values():
                                    if isinstance(v, (int, float)):
                                        score = v
                                        break
                                else:
                                    score = 0
                        else:
                            score = eval_result
                        eval_results[task_id].append(
                            {"variation_id": var_id, "score": float(score)}
                        )
        else:
            remaining = self.get_remaining_tasks(dataset)
            if len(remaining) > 0:
                logger.warning(f"Warning - {len(remaining)} tasks are incomplete")
                logger.info(
                    "Use --continue-run flag to retry the remaining tasks. Exiting..."
                )

            weave.finish()
            eval_results = self.benchmark.evaluate_output(agent_output, self.run_id)

        logger.info("Processing results...")
        results = self.benchmark.process_results(
            agent_name=agent_name,
            run_id=self.run_id,
            agent_args=self.agent_args,
            run_command=self.run_command,
            eval_results=eval_results,
            weave_client=weave_client,
            agent_output=agent_output,
            upload=upload,
            prompt_sensitivity=self.prompt_sensitivity,
        )
        return results

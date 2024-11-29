import os
import json
from typing import Dict, Any, Tuple, Union
from .base_benchmark import BaseBenchmark
from inspect_ai import eval, TaskInfo
from inspect_ai.model import get_model
from inspect_ai._eval.loader import load_tasks
from inspect_ai.solver import solver
from inspect_ai.log import EvalLog
from inspect_ai.solver import Solver
from agent_eval_harness.utils.logging_utils import log_warning, print_warning, log_error
import asyncio
from concurrent.futures import ThreadPoolExecutor

class InspectBenchmark(BaseBenchmark):
    """Inspect benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], task_name: str, agent_args: Dict[str, Any]):
        self.benchmark_name = task_name
        self.requirements_file = 'inspect'
        self.vm_only = False
        self.agent_args = agent_args
        super().__init__(agent_dir, config, vm_only=self.vm_only)
        
        # Remove 'inspect:' prefix if present
        self.task_name = task_name.removeprefix("inspect:")
        
        # Load the task
        self.task = self._load_task(self.task_name, self.agent_args.get('model_name', 'openai/gpt-4o-mini'))
        
        # Create benchmark dictionary from task dataset
        self.benchmark = {}
        for sample in self.task.dataset:
            # Ensure sample has an ID
            if sample.id is None:
                sample.id = len(self.benchmark) + 1
                
            # Convert input to string if it's a list of messages
            input_str = (
                sample.input
                if isinstance(sample.input, str)
                else "\n".join([message.text for message in sample.input])
            )
            
            # Convert file paths to absolute paths
            files = {}
            if sample.files:
                for key, value in sample.files.items():
                    if os.path.isabs(value):
                        files[key] = value
                    else:
                        files[key] = os.path.join(os.getcwd(), value)
                        
            self.benchmark[sample.id] = {
                "id": sample.id,
                "input": input_str,
                "choices": sample.choices,
                "target": sample.target,
                "metadata": sample.metadata,
                "files": files,
                "setup": sample.setup,
            }

    def _load_task(self, task: str, model: str) -> TaskInfo:
        """Load a single inspect task"""
        tasks = load_tasks([task], get_model(model))
        if len(tasks) == 0:
            raise RuntimeError(f"Task {task} for model {model} could not be found.")
        elif len(tasks) > 1:
            raise RuntimeError(f"Task {task} for model {model} matched multiple tasks.")
        return tasks[0]
    
    @staticmethod
    def add_additional_metrics(inspect_eval_log: Union[EvalLog, dict[str, Any]], eval_results: dict[str, Any], benchmark_name: str = None) -> dict[str, Any]:
        """
        Adds additional metrics to the evaluation results
        """
        try: 
            if not benchmark_name and isinstance(inspect_eval_log, EvalLog):
                benchmark_name = inspect_eval_log.eval.task
            else:
                raise ValueError("Benchmark name must be provided if inspect_eval_log is not an EvalLog")

            if benchmark_name == "inspect_evals/gaia":
                # Calculate accuracy for each level
                level_samples = {'1': [], '2': [], '3': []}
                level_correct = {'1': 0, '2': 0, '3': 0}

                for sample in inspect_eval_log.samples if isinstance(inspect_eval_log, EvalLog) else inspect_eval_log['samples']:
                    level = sample['metadata']['level'] if isinstance(inspect_eval_log, dict) else sample.metadata['level']
                    score_value = sample['scores']['gaia_scorer']['value'] if isinstance(inspect_eval_log, dict) else sample.scores['gaia_scorer'].value
                    level_samples[level].append(sample)
                    # Check if sample was scored as correct ('C')
                    if score_value == 'C':
                        level_correct[level] += 1
                    elif score_value not in ['C', 'I']:
                        print("WARNING")

                # Calculate accuracy for each level
                for level in ['1', '2', '3']:
                    accuracy = level_correct[level] / len(level_samples[level]) if len(level_samples[level]) > 0 else None
                    eval_results[f'level_{level}_accuracy'] = accuracy
        except Exception as e:
            print_warning(f"Computing additional metrics failed. Output will only contain standard metrics from Inspect harness...")
            log_error(f"Computing additional metrics failed: {e}")
            pass
        
        return eval_results
    
    @staticmethod
    def get_succ_and_fail_tasks(eval_log: EvalLog) -> Tuple[list, list]:
        """Get successful and failed tasks from evaluation log"""
        succ_tasks = []
        fail_tasks = []
        if "gaia" in eval_log.eval.task:
            for sample in eval_log.samples:
                if sample.scores['gaia_scorer'].value == "C":
                    succ_tasks.append(sample.id)
                elif sample.scores['gaia_scorer'].value == "I":
                    fail_tasks.append(sample.id)
                else:
                    print_warning(f"Unexpected Gaia scorer value: {sample.scores['gaia_scorer'].value}. Appending to failed tasks ids for heatmap on HAL...")
                    fail_tasks.append(sample.id)      
            return succ_tasks, fail_tasks
        elif "cybench" in eval_log.eval.task:
            for sample in eval_log.samples:
                if sample.scores['cybench_scorer'].value == "C":
                    succ_tasks.append(sample.id)
                elif sample.scores['cybench_scorer'].value == "I":
                    fail_tasks.append(sample.id)
                else:
                    print_warning(f"Unexpected Cybench scorer value: {sample.scores['cybench_scorer'].value}. Appending to failed tasks ids for heatmap on HAL...")
                    fail_tasks.append(sample.id)
            return succ_tasks, fail_tasks
        elif "agentharm" in eval_log.eval.task:
            return None, None 
        else:
            log_warning(f"Getting successful and failed tasks ids not supported for {eval_log.eval.task}")
            return None, None
    
    def _results_for_eval(self, eval_log: EvalLog) -> dict[str, Any]:
        """
        Processes and returns the evaluation results, including status, cost, and scores.

        Args:
            eval_log (EvalLog): The log containing the results of the evaluation.

        Returns:
            dict[str, Any]: A dictionary containing the processed evaluation results.
        """

        eval_results = {}
        if eval_log.status == "success":
            # there should be results
            if eval_log.results is None:
                raise RuntimeError(
                    "No results present in log even though status is 'success'"
                )
                
            succ_tasks, fail_tasks = self.get_succ_and_fail_tasks(eval_log)
            eval_results = {
                "status": "success",
                "successful_tasks": succ_tasks,
                "failed_tasks": fail_tasks,
            }

            # Pick out the first accuracy or mean metric to represent 'accuracy'
            # Include other scores as well
            accuracy = None
            for eval_score in eval_log.results.scores:
                scorer_name = eval_score.name
                metrics = eval_score.metrics
                for _key, metric in metrics.items():
                    score_name = f"{scorer_name}/{metric.name}"
                    eval_results[score_name] = metric.value
                    if accuracy is None and (
                        metric.name == "accuracy" or metric.name == "mean"
                    ):
                        accuracy = metric.value
                    eval_results["accuracy"] = accuracy

        elif eval_log.status == "error":
            if eval_log.error is None:
                raise RuntimeError(
                    "Missing error in evaluation log even though status is 'error'."
                )
            eval_results = {
                "status": "error",
                "message": eval_log.error.message,
                "traceback": eval_log.error.traceback,
            }
        elif eval_log.status == "canceled":
                eval_results = {"status": "canceled"}
        
        return eval_results

    async def _run_eval_async(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Run evaluation asynchronously"""
        # Create solver function
        solver = self._create_solver(agent_output)
        
        # Create a new event loop in a separate thread to run eval()
        def run_eval():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return eval(
                    tasks=self.task,
                    solver=solver,
                    model=get_model(self.agent_args.get('model_name', 'gpt-4')),
                    log_dir=self.get_run_dir(run_id),
                    sandbox="docker",
                    log_format="json",

                )
            finally:
                new_loop.close()
        
        # Run eval in a thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            eval_logs = await loop.run_in_executor(pool, run_eval)
            
        # Return first log since we only have one task
        return eval_logs[0]

    async def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs using Inspect evaluation"""
        # Run evaluation asynchronously
        eval_log = await self._run_eval_async(agent_output, run_id)
        return eval_log

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        results = self._results_for_eval(eval_results)
        
        results = add_additional_metrics(inspect_eval_log=eval_results, eval_results=results, benchmark_name=self.benchmark_name)
        
        return results

    @solver
    def _create_solver(self, agent_output: Dict[str, Any]) -> Solver:
        """Create a solver function from agent output"""
        async def solve(state, generate):
            completion = agent_output[state.sample_id]
            state.output.completion = completion
            state.completed = True
            return state
            
        return solve

    def mount_benchmark(self):
        """Mount benchmark environment - not needed for inspect"""
        pass 
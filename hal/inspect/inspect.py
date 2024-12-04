from datetime import datetime
from inspect import get_annotations
from typing import Any, Callable, Union

from inspect_ai import TaskInfo, list_tasks, Task
from inspect_ai.log import EvalLog
from inspect_ai.model import get_model
from inspect_ai.solver import Solver
from inspect_ai._eval.loader import load_tasks
from hal.benchmarks.inspect_benchmark import InspectBenchmark
from hal.utils.logging_utils import log_error, print_warning

def is_inspect_benchmark(benchmark: str) -> bool:
    """
    Determines whether a benchmark string is an inspect benchmark

    Args:
        benchmark (str): The benchmark

    Returns:
        bool: True if this is an Inspect benchmark, otherwise False
    """
    return benchmark.startswith("inspect:") or benchmark.startswith("inspect_evals/")


def task_name(benchmark: str) -> str:
    """
    Provides the Inspect task name for an Inspect benchmark

    Args:
        benchmark (str): The benchmark

    Returns:
        str: The inspect task name
    """
    return benchmark.removeprefix("inspect:")


def resolve_task(task: str) -> list[TaskInfo] | list[str]:
    """
    Fully resolves a task string into one or more Tasks

    Args:
        task (str): The task

    Returns:
        list[TaskInfo]: A list of the resolved TaskInfos
    """
    if task.startswith("inspect_evals/"):
        return [task]
    else:
        return list_tasks(task)


def resolve_solver(agent_function: Callable, agent_args: dict[str, Any]) -> Solver | None:
    """
    Resolves an agent function into an Inspect solver

    Args:
        agent_function (str | None): The agent_function to resolve

    Returns:
        Tuple[Solver | None, str | None]: A tuple with the solver and sandbox to use
    """
    return_type = getattr(get_annotations(agent_function)["return"], "__name__", None)
    if return_type == "Solver":
        # If the callable is a solver, use that
        return agent_function(**agent_args)

    return None


def config_for_eval(
    run_id: str, agent_name: str, benchmark: str, eval_log: EvalLog
) -> dict[str, Any]:
    """
    Creates a configuration dictionary for the evaluation.

    Args:
        run_id (str): Unique identifier for the current run.
        name (str): Name of the benchmark task.
        eval_log (EvalLog): The log from the evaluation process.
        agent (str | None): The agent used for the evaluation (default is "default").

    Returns:
        dict[str, Any]: A dictionary containing the evaluation configuration details.
    """

    config = {
        "agent_name": agent_name,
        "benchmark_name": benchmark,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "run_id": run_id,
    }

    eval_config = vars(eval_log.plan.config)
    for key, value in eval_config.items():
        if eval_config[key] is not None:
            config[key] = value

    return config


def results_for_eval(eval_log: EvalLog, total_cost: float | None) -> dict[str, Any]:
    """
    Processes and returns the evaluation results, including status, cost, and scores.

    Args:
        eval_log (EvalLog): The log containing the results of the evaluation.
        total_cost (float | None): The total cost of the evaluation.

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

        succ_tasks, failed_tasks = InspectBenchmark.get_succ_and_fail_tasks(eval_log)
        eval_results = {
            "status": "success",
            "total_cost": total_cost,
            "successful_tasks": succ_tasks,
            "failed_tasks": failed_tasks,
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
        
        
    eval_results = InspectBenchmark.add_additional_metrics(inspect_eval_log=eval_log, eval_results=eval_results)
        
    return eval_results


def load_task(task: str, model: str, task_args: dict[str, Any] = {}) -> Task:
    """
    Loads a single for a task & model

    Args:
        task (str): The task to load
        model (str): The model for this task

    Returns:
        Task: The task 
    """
    tasks = load_tasks([task], get_model(model), task_args=task_args)
    if len(tasks) == 0:
        raise RuntimeError(f"The task {task} for model {model} could not be found.")
    elif len(tasks) > 1:
        raise RuntimeError(f"The task {task} for model {model} matched more than a single task.")
    else:
        return tasks[0]

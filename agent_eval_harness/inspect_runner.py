import json
import os
import sys
import time

import weave  # type: ignore
from inspect_ai import eval
from pydantic_core import to_jsonable_python

from .inspect.agent import load_agent, run_agent, validate_agent
from .inspect.hf import upload_results
from .inspect.inspect import (
    config_for_eval,
    resolve_solver,
    resolve_task,
    results_for_eval,
    task_name,
    load_task
)
from .inspect.log import log, log_end, log_start
from .inspect.weave import weave_tracing
from .utils.utils import safe_filename
from .utils.weave_utils import get_total_cost, get_weave_calls

# TODO: weave_usage includes a KeyError warning (appears to be a running trace)
# TODO: add support for agent and benchmark args

def inspect_evaluate(
    benchmark: str,
    agent_name: str,
    agent_function: str,
    agent_dir: str,
    model: str,
    run_id: str,
    upload: bool,
) -> None:
    """
    Evaluates a task using a specified model and agent, logs results, and optionally uploads them.

    Args:
        benchmark (str): The inspect benchmark to run
        agent_name (str): The di    y name of the agent (in the leaderboard)
        agent_function (str): The fuction to load and us as an agent.
        agent_dir (str): The directory where to find the agent function
        model (str): The model to use for the evaluation.
        run_id (str | None): The run_id to use (defaults to None).
        upload (bool): Whether to upload the results (defaults to False).

    Returns:
        None
    """

    # resolve the task name (this is the benchmark name)
    task = task_name(benchmark)

    # the run identifier
    run_id = run_id or safe_filename(f"{task}_{int(time.time())}")

    # determine the log directory
    log_dir = os.path.join("results", task, run_id)

    # initialize the weave client
    with weave_tracing(run_id) as weave_client:
        with weave.attributes({"weave_task_id": task}):

            # resolve the task
            tasks = resolve_task(task)
            if len(tasks) > 1:
                # We expect this to only run a single task
                raise RuntimeError(
                    f"Expected a single task for this benchmar, got ${len(tasks)}"
                )

            # append agent dir to path
            if agent_dir is not None:
                sys.path.append(agent_dir)

            if agent_function is not None:
                # load the agent function
                agent = load_agent(agent_function)

                # is the agent a solver?
                solver = resolve_solver(agent)

                if solver is None:
                    # Ensure this is a valid custom agent function
                    validate_agent(agent)

                    # Load the task
                    resolved_task = load_task(task, model)

                    # Run the custom agent
                    log_start(f"Running Inspect custom agent {agent_function}")
                    solver = run_agent(resolved_task.dataset, agent)
                    log_end()

                    # run the task (this will be fast since it mostly just scoring)
                    log_start(f"Scoring custom agent {agent_function}")
                    eval_logs = eval(
                        resolved_task, solver=solver, model=model, log_dir=log_dir, sandbox="local"
                    )
                    log_end()
                else:
                    # run the inspect task
                    log_start(f"Running Inspect task {task}")
                    eval_logs = eval(tasks, solver=solver, model=model, log_dir=log_dir)
                    log_end()

            else:
                # run the inspect task
                log_start(f"Running Inspect task {task}")
                eval_logs = eval(tasks, model=model, log_dir=log_dir)
                log_end()

            # unexpected if this is called for a log that hasn't completed
            eval_log = eval_logs[0]
            if eval_log.status == "started":
                raise RuntimeError(
                    "Cannot process an evaluation which is still running."
                )

            # Compute weave metrics
            log_start("Computing Weave Metrics")
            total_cost, total_usage = get_total_cost(weave_client)
            weave_calls = get_weave_calls(weave_client)
            log_end()

            # Compute the evaluation results and configuration
            eval_results = results_for_eval(eval_log=eval_log, total_cost=total_cost)
            eval_config = config_for_eval(
                run_id=run_id, benchmark=task, eval_log=eval_log, agent_name=agent_name
            )

            # Compose the final uploadable result
            eval_header_raw = eval_log
            del eval_header_raw.samples
            final_result = {
                "config": eval_config,
                "results": eval_results,
                "raw_eval_results": to_jsonable_python(eval_header_raw),
                "raw_logging_results": weave_calls,
                "total_usage": total_usage,
            }

            # Store the upload results locally
            upload_path = os.path.join(log_dir, f"{run_id}_UPLOAD.json")
            with open(upload_path, "w") as f:
                json.dump(final_result, f, indent=2)

            if upload:
                log_start("Uploading to hf")
                upload_results(run_id, final_result)
                log_end()

            log_start(msg="Run results")
            log_abs_path = os.path.join(os.getcwd(), log_dir)
            log(f"\n{log_abs_path}\n")
            log_end()

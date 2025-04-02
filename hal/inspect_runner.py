import json
import os
import sys
import time
import datetime
import weave  # type: ignore
from inspect_ai import eval, eval_retry
from pydantic_core import to_jsonable_python
from inspect_ai.solver import bridge

from typing import Any

from .inspect.agent import load_agent, run_agent, validate_agent
from .inspect.hf import upload_results
from .inspect.inspect import (
    config_for_eval,
    resolve_solver,
    resolve_task,
    results_for_eval,
    task_name,
    load_task,
)
from .inspect.log import log, log_end, log_start
from .inspect.weave import weave_tracing
from .utils.utils import safe_filename
from .utils.weave_utils import comput_cost_from_inspect_usage

from .utils.logging_utils import print_success, print_results_table, print_warning

from openai import AsyncOpenAI


def inspect_evaluate(
    benchmark: str,
    benchmark_args: dict[str, Any],
    agent_name: str,
    agent_function: str,
    agent_dir: str,
    agent_args: dict[str, Any],
    model: str,
    run_id: str,
    upload: bool,
    max_concurrent: int = 10,
    conda_env_name: str = None,
    vm: bool = False,
    continue_run: bool = False,
    inspect_eval_args: dict[str, Any] = None,
) -> None:
    """
    Evaluates a task using a specified model and agent, logs results, and optionally uploads them.

    Args:
        benchmark (str): The inspect benchmark to run
        benchmark_args (dict[str,Any]): task_args for the Inspect task
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
    log_dir = os.path.join("results", benchmark, run_id)

    with weave_tracing(run_id) as weave_client:

        # resolve the task
        tasks = resolve_task(task)
        if len(tasks) > 1:
            # We expect this to only run a single task
            raise RuntimeError(
                f"Expected a single task for this benchmark, got {len(tasks)}"
            )

        # append agent dir to path
        if agent_dir is not None:
            sys.path.append(agent_dir)
            
            
        # Filter out inspect_eval_args that are already set
        filtered_eval_args = {}
        if inspect_eval_args:
            # Define arguments that are already set in our eval() calls
            preset_args = [
                'task',
                'task_args',
                'solver',
                'model',
                'log_dir',
                'sandbox',
                'log_format'
            ]
            # Only keep args that aren't preset
            filtered_eval_args = {
                k: v for k, v in inspect_eval_args.items() 
                if k not in preset_args
            }

        if agent_function is not None:
            # load the agent function
            agent = load_agent(agent_function)
            
            # wrap the content of the run function in a weave.attributes the wrapped function should also be async
            async def patched_agent(sample: dict[str, Any]) -> dict[str, Any]:
                with weave.attributes({"weave_task_id": sample["sample_id"]}):
                    return await agent(sample, **agent_args)

            # is the agent a solver?
            solver = resolve_solver(agent, agent_args=agent_args)
            
            if solver is None:
                # Ensure this is a valid custom agent function
                validate_agent(agent)
                
                # Load the task
                resolved_task = load_task(task, model, task_args=benchmark_args)
                
                            

                # # Run the custom agent
                # log_start(f"Running Inspect custom agent {agent_function}")
                
                # solver = run_agent(
                #     resolved_task.dataset, 
                #     agent, 
                #     agent_args=agent_args, 
                #     agent_function=agent_function, 
                #     agent_dir=agent_dir, 
                #     max_concurrent=max_concurrent, 
                #     run_id=run_id, 
                #     log_dir=log_dir, 
                #     task_name=task,
                #     conda_env_name=conda_env_name,
                #     )
                # log_end()

                # run the task (this will be fast since it mostly just scoring)
                log_start(f"Scoring custom agent {agent_function}")
                if not continue_run:
                    eval_logs = eval(
                        tasks=resolved_task,
                        task_args=benchmark_args,                    
                        solver=bridge(patched_agent),
                        model=model,
                        log_dir=log_dir,
                        log_format="json",
                        sandbox="docker",
                        **filtered_eval_args  # Use filtered args
                    )
                else:
                    latest_file = [f for f in os.listdir(log_dir) if f.endswith('.json')][-1]
                    eval_logs = eval_retry(os.path.join(log_dir, latest_file))
                log_end()

            else:
                if not continue_run:
                    log_start(f"Running Inspect task {task}")
                    eval_logs = eval(
                        tasks=tasks,
                        task_args=benchmark_args,
                        solver=solver,                        
                        model=model,
                        log_dir=log_dir,
                        log_format="json",
                        sandbox="docker",
                        **filtered_eval_args  # Use filtered args
                    )
                
                else:
                    log_start(f"Continuing Inspect task {task}")
                    # get last .json file created in log_dir
                    latest_file = [f for f in os.listdir(log_dir) if f.endswith('.json')][-1]
                    eval_logs = eval_retry(os.path.join(log_dir, latest_file))
                log_end()

        # unexpected if this is called for a log that hasn't completed
        eval_log = eval_logs[0]
        if eval_log.status == "started":
            raise RuntimeError(
                "Cannot process an evaluation which is still running."
            )
            
        # Read raw json logs from inspect harness
        json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
        with open(os.path.join(log_dir, latest_file), 'r') as f:
            inspect_eval_results = json.load(f)
            
        # Compute the total cost from the model usage
        if benchmark in ["inspect_evals/agentharm", "inspect_evals/agentharm_benign"]:
            # If agentharm, skip gpt-4o-2024-08-06 because it is used for scoring not the agent    TODO: make this more general
            total_cost = comput_cost_from_inspect_usage(inspect_eval_results['stats']['model_usage'], skip_models=["openai/gpt-4o-2024-08-06"])
            
            # remove token_usage for gpt-4o-2024-08-06 from inspect_eval_results
            inspect_eval_results['stats']['model_usage'] = {k: v for k, v in inspect_eval_results['stats']['model_usage'].items() if k != "openai/gpt-4o-2024-08-06"}
        else:
            total_cost = comput_cost_from_inspect_usage(inspect_eval_results['stats']['model_usage'])

        # Compute the evaluation results and configuration
        inspect_eval_results_json = results_for_eval(eval_log=eval_log, total_cost=total_cost)
                
        eval_config = config_for_eval(
            run_id=run_id, benchmark=task, eval_log=eval_log, agent_name=agent_name
        )
        
        # replace / in metrics with underscore
        inspect_eval_results_json = {
            key.replace("/", "_"): value 
            for key, value in inspect_eval_results_json.items()
        }

        # Compose the final uploadable result
        eval_header_raw = eval_log
        del eval_header_raw.samples
        final_result = {
            "config": {**eval_config, 'agent_args': agent_args},
            "results": inspect_eval_results_json,
            "raw_eval_results": inspect_eval_results,            
            "raw_logging_results": "Inspect logs are contained in raw_eval_results",
            "total_usage": inspect_eval_results['stats']['model_usage'],
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
        
        print_success("Evaluation completed successfully")
        print_results_table(final_result)
            

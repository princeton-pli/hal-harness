import json
import os
import sys
import time
import datetime
import weave  # type: ignore
from inspect_ai import eval
from pydantic_core import to_jsonable_python

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
from .utils.weave_utils import get_total_cost, get_weave_calls
import swebench_harness

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

    # for swebench we use inspect implementation with custom harness right now because of inconsistencies in inspect implementation. will replace hardcoded bits once resolved
    if "swebench" in benchmark:
        benchmark = "inspect_evals/swe_bench"
        if "mini" in benchmark:
            benchmark_args['instance_ids'] = ['django__django-11206',
 'django__django-12262',
 'django__django-12419',
 'django__django-13128',
 'django__django-13346',
 'django__django-13809',
 'django__django-14725',
 'django__django-15569',
 'django__django-16493',
 'django__django-7530',
 'pydata__xarray-3151',
 'pylint-dev__pylint-4551',
 'pytest-dev__pytest-5631',
 'sphinx-doc__sphinx-9367',
 'sympy__sympy-15017',
 'sympy__sympy-15599',
 'sympy__sympy-17318',
 'sympy__sympy-18763',
 'sympy__sympy-22914',
 'sympy__sympy-23824',
 'django__django-14855',
 'django__django-15128',
 'django__django-11333',
 'django__django-14053',
 'django__django-11099',
 'django__django-13658',
 'astropy__astropy-7336',
 'django__django-12304',
 'django__django-14011',
 'django__django-16901',
 'django__django-15930',
 'matplotlib__matplotlib-24970',
 'django__django-15629',
 'scikit-learn__scikit-learn-11310',
 'psf__requests-1766',
 'pallets__flask-5014',
 'matplotlib__matplotlib-23476',
 'sympy__sympy-13551',
 'sympy__sympy-13480',
 'django__django-16950',
 'sympy__sympy-15349',
 'sphinx-doc__sphinx-9281',
 'scikit-learn__scikit-learn-10908',
 'sympy__sympy-12481',
 'sympy__sympy-15875',
 'sphinx-doc__sphinx-9461',
 'sympy__sympy-20428',
 'sympy__sympy-16450',
 'scikit-learn__scikit-learn-25747',
 'scikit-learn__scikit-learn-25973']    

    # resolve the task name (this is the benchmark name)
    task = task_name(benchmark)

    # the run identifier
    run_id = run_id or safe_filename(f"{task}_{int(time.time())}")

    # determine the log directory
    log_dir = os.path.join("results", task, run_id)


    with weave_tracing(run_id) as weave_client:

        # resolve the task
        tasks = resolve_task(task)
        if len(tasks) > 1:
            # We expect this to only run a single task
            raise RuntimeError(
                f"Expected a single task for this benchmar, got {len(tasks)}"
            )

        # append agent dir to path
        if agent_dir is not None:
            sys.path.append(agent_dir)

        if agent_function is not None:
            # load the agent function
            agent = load_agent(agent_function)

            # is the agent a solver?
            solver = resolve_solver(agent, agent_args=agent_args)

            if solver is None:
                # Ensure this is a valid custom agent function
                validate_agent(agent)

                if "swe_bench" in benchmark:
                    model = "openai/gpt-4o" # hardcoded for swebench 

                # Load the task
                resolved_task = load_task(task, model, task_args=benchmark_args)

                # Run the custom agent
                log_start(f"Running Inspect custom agent {agent_function}")
                
                solver = run_agent(
                    resolved_task.dataset, 
                    agent, 
                    agent_args=agent_args, 
                    agent_function=agent_function, 
                    agent_dir=agent_dir, 
                    max_concurrent=max_concurrent, 
                    run_id=run_id, 
                    log_dir=log_dir, 
                    task_name=task,
                    conda_env_name=conda_env_name,
                    )
                log_end()

                if "swe_bench" not in benchmark: # for swebench we use official harness
                    # run the task (this will be fast since it mostly just scoring)
                    log_start(f"Scoring custom agent {agent_function}")
                    eval_logs = eval(
                        resolved_task,
                        task_args=benchmark_args,                    
                        solver=solver,
                        model=model,
                        log_dir=log_dir,
                        sandbox="docker",
                    )
                    log_end()
                else:
                    results = swebench_harness.run_evaluation_harness(predictions_path=f"{log_dir}/{run_id}_SWE_BENCH_SUBMISSIONS.jsonl", run_id=run_id)


            else:
                # run the inspect task
                # initialize the weave client
                with weave.attributes({"weave_task_id": task}):
                    log_start(f"Running Inspect task {task}")
                    eval_logs = eval(
                        tasks,
                        task_args=benchmark_args,
                        solver=solver,                        
                        model=model,
                        log_dir=log_dir,
                    )
                    log_end()

        else:
            # run the inspect task
            # initialize the weave client
            with weave.attributes({"weave_task_id": task}):
                log_start(f"Running Inspect task {task}")
                eval_logs = eval(
                    tasks, 
                    task_args=benchmark_args, 
                    model=model, 
                    log_dir=log_dir,            
                )
                log_end()

        # Compute weave metrics
        log_start("Computing Weave Metrics")
        total_cost, total_usage = get_total_cost(weave_client)
        weave_calls = get_weave_calls(weave_client)
        log_end()

        if "swe_bench" not in benchmark:
            # unexpected if this is called for a log that hasn't completed
            eval_log = eval_logs[0]
            if eval_log.status == "started":
                raise RuntimeError(
                    "Cannot process an evaluation which is still running."
                )

            # Compute the evaluation results and configuration
            eval_results = results_for_eval(eval_log=eval_log, total_cost=total_cost)
            eval_config = config_for_eval(
                run_id=run_id, benchmark=task, eval_log=eval_log, agent_name=agent_name
            )
        else:
            
            benchmark = "swebench_verified_mini" if "mini" in benchmark else "swebench_verified"
    
            eval_log = results
            eval_results =  {
                    "accuracy": results['resolved_instances']/results['total_instances'],
                    "total_cost": total_cost,
                    'successful_tasks': results['resolved_ids'],
                    'failed_tasks': results['unresolved_ids']
                }
            eval_config = {
                "agent_name": agent_name,
                "benchmark_name": benchmark,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "run_id": run_id,
                "agent_args": agent_args,
            }

        # Compose the final uploadable result
        eval_header_raw = eval_log
        del eval_header_raw.samples
        final_result = {
            "config": {**eval_config, 'agent_args': agent_args},
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

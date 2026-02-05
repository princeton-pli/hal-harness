import os
import re
import click
import yaml
import asyncio
from typing import Any, Dict
import time
import importlib
from inspect import get_annotations
from .agent_runner import AgentRunner
from dotenv import load_dotenv
import sys
import logging
from .utils.logging_utils import (
    setup_logging,
    log_results,
    log_run_summary,
    print_run_config,
)

import traceback
from datetime import datetime

logger = logging.getLogger(__name__)
load_dotenv()


@click.command()
@click.option(
    "--agent_name",
    required=True,
    help="Name of the agent you want to add to the leaderboard. Please use a short name and add model name in parentheses if applicable (e.g. 'inspect_solver (gpt-4o)'",
)
@click.option(
    "--agent_function",
    required=False,
    help="Path to the agent function. Example: agent.run (if 'agent.py' is the name of the file in the agent directory and 'run' is the name of the function)",
)
@click.option(
    "--agent_dir",
    required=False,
    help="Path to the agent directory in which the entrypoint file and function are located.",
)
@click.option(
    "-A",
    multiple=True,
    type=str,
    help="One or more args to pass to the agent (e.g. -A model_name=gpt-4o -A arg_2=value)",
)
@click.option("--benchmark", required=True, help="Name of the benchmark to run")
@click.option(
    "-B",
    multiple=True,
    type=str,
    help="One or more args to pass to the benchmark (e.g. -B arg_1=value -B arg_2=value)",
)
@click.option(
    "--upload", is_flag=True, help="Upload results to HuggingFace after evaluation"
)
@click.option(
    "--max_concurrent",
    default=1,
    help="Maximum task-agent pairs to run concurrently for this run",
)
@click.option(
    "--max_tasks",
    type=int,
    help="Maximum number of tasks to run from the benchmark. Useful for testing.",
)
@click.option(
    "--conda_env_name",
    help="Conda environment to run the custom external agent in if run locally",
)
@click.option(
    "--run_id",
    help="Run ID to use for logging. For continuous runs, use the same run_id to continue from a previous run",
)
@click.option(
    "--config",
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="Path to configuration file. (currently not used)",
)
@click.option("--vm", is_flag=True, help="Run the agent on azure VMs")
@click.option(
    "--docker",
    is_flag=True,
    help="Run the agent in Docker containers for isolation. Requires Docker to be installed on the system. Resources are limited to 4GB memory and 2 CPU cores per container.",
)
@click.option(
    "--continue_run",
    is_flag=True,
    help="Continue from a previous run, only running failed or incomplete tasks. You must provide the same run_id to continue a run.",
)
@click.option(
    "--ignore_errors",
    is_flag=True,
    help="Ignore errors and continue running the remaining tasks. This is useful for continuing a run that failed due to an error.",
)
@click.option(
    "-I",
    multiple=True,
    type=str,
    help="One or more args to pass to inspect eval (e.g. -I token_limit=1000 -I model_args='{'temperature': 0.5}'",
)
def main(
    config,
    benchmark,
    agent_name,
    agent_function,
    agent_dir,
    run_id,
    upload,
    max_concurrent,
    conda_env_name,
    continue_run,
    ignore_errors,
    a,
    b,
    i,
    vm,
    docker,
    max_tasks,
    **kwargs,
):
    """Run agent evaluation on specified benchmark with given model."""
    try:
        # Parse agent and benchmark args
        logger.info("Parsing configuration...")
        agent_args = parse_cli_args(a)
        benchmark_args = parse_cli_args(b)
        inspect_eval_args = parse_cli_args(i)

        # Generate default run_id if none provided
        if not run_id:
            set_run_id = False
            benchmark_name = benchmark.split("/")[-1]

            # convert agent name into a valid run_id, it has spaces and parentheses and might contain large letters and special characters
            agent_name_run_id = re.sub(
                r"[^a-zA-Z0-9_]",
                "",
                agent_name.replace(" ", "_").replace("(", "").replace(")", ""),
            ).lower()

            run_id = f"{benchmark_name}_{agent_name_run_id}_{int(time.time())}"

        else:
            set_run_id = True

        # Setup logging first, before any other operations
        log_dir = os.path.join("results", benchmark, run_id)
        os.makedirs(log_dir, exist_ok=True)
        verbose_log_path = os.path.join(log_dir, f"{run_id}_verbose.log")
        setup_logging(log_dir, run_id, use_vm=vm)

        logger.info("HAL Harness")

        # add benchmark name to agent_args
        agent_args["benchmark_name"] = benchmark

        # Validate model pricing if model_name is provided in agent_args
        if "model_name" in agent_args:
            validate_model_pricing(agent_args["model_name"])

        # Validate runner options
        if sum([bool(conda_env_name), vm, docker]) > 1:
            logger.error(
                "Only one of --conda_env_name, --vm, or --docker can be specified. Exiting..."
            )
            sys.exit(1)

        if continue_run and not set_run_id:
            raise ValueError("continue_run flag requires run_id to be set")

        # Print summary with run_id, benchmark, and the run config to terminal
        print_run_config(
            run_id=run_id,
            benchmark=benchmark,
            agent_name=agent_name,
            agent_function=agent_function,
            agent_dir=agent_dir,
            agent_args=agent_args,
            benchmark_args=benchmark_args,
            inspect_eval_args=inspect_eval_args,
            upload=upload,
            max_concurrent=max_concurrent,
            conda_env_name=conda_env_name,
            log_dir=log_dir,
            vm=vm,
            docker=docker,
            continue_run=continue_run,
            ignore_errors=ignore_errors,
        )

        # get exact command used to run the evaluation from click
        run_command = " ".join(["hal-eval"] + sys.argv[1:])
        # Initialize agent runner
        logger.info("Initializing agent runner...")
        try:
            runner = AgentRunner(
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                benchmark_name=benchmark,
                config=config,
                run_id=run_id,  # Now guaranteed to have a value
                use_vm=vm,
                use_docker=docker,
                max_concurrent=max_concurrent,
                conda_env=conda_env_name,
                continue_run=continue_run,
                run_command=run_command,
                ignore_errors=ignore_errors,
                max_tasks=max_tasks,
            )

            # Run evaluation
            logger.info("Running evaluation with custom agent and HAL harness...")
            results = asyncio.run(
                runner.run(agent_name=agent_name, upload=upload or False)
            )

            logger.info("Evaluation completed successfully")
            log_results(results)

            # Only print run summary if we have a valid benchmark and run_id
            if runner.benchmark and runner.benchmark.get_run_dir(run_id):
                log_run_summary(run_id, runner.benchmark.get_run_dir(run_id))
            else:
                logger.warning(
                    "Could not generate run summary - missing benchmark or run directory"
                )

        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            raise

    except Exception as e:
        # Get the full traceback
        full_traceback = traceback.format_exc()

        # Log the full error to the verbose log file
        with open(verbose_log_path, "a") as f:
            f.write("\n=== ERROR TRACEBACK ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(full_traceback)
            f.write("\n=== END ERROR TRACEBACK ===\n")

        # Print clean error message to terminal
        logger.error(f"An error occurred: {str(e)}")
        logger.error(f"For detailed error information, check: {verbose_log_path}")
        sys.exit(1)


def parse_cli_args(args: tuple[str] | list[str] | None) -> Dict[str, Any]:
    """Parse CLI arguments into a dictionary."""
    params: Dict[str, Any] = {}
    if args:
        for arg in list(args):
            parts = arg.split("=", 1)  # Split on first = only
            if len(parts) > 1:
                key = parts[0].replace("-", "_")
                value = parts[1]

                try:
                    # First try to parse as YAML
                    parsed_value = yaml.safe_load(value)

                    # Handle special cases for string values that yaml doesn't parse
                    if isinstance(parsed_value, str):
                        # Handle comma-separated lists
                        if "," in value:
                            parsed_value = value.split(",")
                        # Handle boolean values
                        elif value.lower() in ["true", "false"]:
                            parsed_value = value.lower() == "true"
                        # Handle numeric values
                        elif value.lower() in ["none", "null", "nan"]:
                            parsed_value = None
                        else:
                            try:
                                parsed_value = int(value)
                            except ValueError:
                                try:
                                    parsed_value = float(value)
                                except ValueError:
                                    parsed_value = value

                    params[key] = parsed_value
                except yaml.YAMLError:
                    # If YAML parsing fails, use the raw string
                    params[key] = value
    return params


def is_inspect_solver(agent_function: str, agent_dir: str) -> bool:
    """Check if an agent function returns a Solver"""
    try:
        sys.path.append(agent_dir)
        # parse the agent name
        module_name, function_name = agent_function.rsplit(".", 1)

        # attempt to load it from the module
        module = importlib.import_module(module_name)
        loaded_agent = getattr(module, function_name)
        return_type = getattr(get_annotations(loaded_agent)["return"], "__name__", None)

        # remove the agent dir from the path
        sys.path.remove(agent_dir)
        return return_type == "Solver"
    except Exception as e:
        logger.error(f"Error checking if agent function is a solver: {str(e)}")
        return False


def validate_model_pricing(model_name: str) -> None:
    """Validate that model pricing information exists"""
    from .utils.weave_utils import MODEL_PRICES_DICT

    # together_ai is not part of weave model name
    model_name = model_name.replace("together_ai/", "")

    if model_name not in MODEL_PRICES_DICT:
        logger.error(
            f"Model '{model_name}' not found in pricing dictionary. Please add pricing information to MODEL_PRICES_DICT in weave_utils.py. Exiting..."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

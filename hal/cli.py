import os
import click
import yaml
import asyncio
from typing import Any, Dict, Optional
import time
import importlib
from inspect import get_annotations
from .agent_runner import AgentRunner
from .inspect.inspect import is_inspect_benchmark
from .inspect_runner import inspect_evaluate
from dotenv import load_dotenv
import sys
from .utils.logging_utils import (
    setup_logging, 
    print_header, 
    print_step, 
    print_success, 
    print_error,
    print_results_table,
    print_run_summary,
    print_warning,
    terminal_print,
    console,
    print_run_config
)
from rich.table import Table
from rich import print
from rich.box import ROUNDED
import traceback
from datetime import datetime
from pathlib import Path

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
@click.option("--agent_dir", required=False, help="Path to the agent directory in which the entrypoint file and function are located.")
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
@click.option("--upload", is_flag=True, help="Upload results to HuggingFace after evaluation")
@click.option("--max_concurrent", default=1, help="Maximum task-agent pairs to run concurrently for this run")
@click.option("--conda_env_name", help="Conda environment to run the custom external agent in if run locally")
@click.option("--run_id", help="Run ID to use for logging. For continuous runs, use the same run_id to continue from a previous run")
@click.option(
    "--config",
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="Path to configuration file. (currently not used)",
)
@click.option("--vm", is_flag=True, help="Run the agent on azure VMs")
@click.option("--docker", is_flag=True, help="Run the agent in Docker containers for isolation. Requires Docker to be installed on the system. Resources are limited to 4GB memory and 2 CPU cores per container.")
@click.option("--continue_run", is_flag=True, help="Continue from a previous run, only running failed or incomplete tasks. You must provide the same run_id to continue a run.")
@click.option(
    "-I",
    multiple=True,
    type=str,
    help="One or more args to pass to inspect eval (e.g. -I token_limit=1000 -I model_args='{'temperature': 0.5}'"
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
    a,
    b,
    i,
    vm,
    docker,
    **kwargs,
):
    """Run agent evaluation on specified benchmark with given model."""
    try:
        # Generate default run_id if none provided
        if not run_id:
            set_run_id = False
            benchmark_name = benchmark.split("/")[-1]
            run_id = f"{benchmark_name}_{int(time.time())}"
        else:
            set_run_id = True
        
        # Setup logging first, before any other operations
        log_dir = os.path.join("results", benchmark, run_id)
        os.makedirs(log_dir, exist_ok=True)
        verbose_log_path = os.path.join(log_dir, f"{run_id}_verbose.log")
        setup_logging(log_dir, run_id)
        
        print_header("HAL Harness")
        
        # Parse agent and benchmark args
        print_step("Parsing configuration...")
        agent_args = parse_cli_args(a)
        benchmark_args = parse_cli_args(b)
        inspect_eval_args = parse_cli_args(i)
        
        # add benchmark name to agent_args
        agent_args['benchmark_name'] = benchmark
        
        # Validate model pricing if model_name is provided in agent_args
        if "model_name" in agent_args:
            validate_model_pricing(agent_args["model_name"])
        
        # Validate runner options
        if sum([bool(conda_env_name), vm, docker]) > 1:
            print_error("Only one of --conda_env_name, --vm, or --docker can be specified. Exiting...")
            sys.exit(1)
                
        # Check if VM/Docker execution is attempted with inspect solver
        if (vm or docker) and is_inspect_benchmark(benchmark):
            if agent_function and is_inspect_solver(agent_function, agent_dir):
                run_type = "VM" if vm else "Docker"
                print_error(f"{run_type} execution is not supported for inspect solvers. Please run without --{run_type.lower()} flag. Exiting...")
                sys.exit(1)
                
        # Check if conda environment is specified for inspect solver
        if conda_env_name and is_inspect_benchmark(benchmark):
            if agent_function and is_inspect_solver(agent_function, agent_dir):
                print_error("Conda environments are not supported for inspect solvers. Dependencies are managed by Inspect harness. Run without --conda_env_name flag. Exiting...")
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
            continue_run=continue_run
        )
        
        if is_inspect_benchmark(benchmark):
            # if agent_function and is_inspect_solver(agent_function, agent_dir):
            # Use original inspect_evaluate for solver agents
            print_step("Running evaluation for inspect solver and harness (see logs for more details and monitoring)...")
            inspect_evaluate(
                benchmark=benchmark,
                benchmark_args=benchmark_args,
                agent_name=agent_name,
                agent_function=agent_function,
                agent_dir=agent_dir,
                agent_args=agent_args,
                model=agent_args['model_name'],
                run_id=run_id,
                continue_run=continue_run,
                upload=upload or False,
                max_concurrent=max_concurrent,
                conda_env_name=conda_env_name,
                vm=vm,
                docker=docker,
                inspect_eval_args=inspect_eval_args
            )
            # else:
            #     # Use AgentRunner with InspectBenchmark for non-solver agents
            #     print_step("Running inspect evaluation for custom agent and inspect harness...")
            #     runner = AgentRunner(
            #         agent_function=agent_function,
            #         agent_dir=agent_dir,
            #         agent_args=agent_args,
            #         benchmark_name=benchmark,
            #         config=config,
            #         run_id=run_id,
            #         use_vm=vm,
            #         max_concurrent=max_concurrent,
            #         conda_env=conda_env_name,
            #         continue_run=continue_run
            #     )
            #     results = asyncio.run(runner.run(
            #         agent_name=agent_name,
            #         upload=upload or False
            #     ))
                
            #     print_success("Evaluation completed successfully")
            #     print_results_table(results)
                
            #     # Only print run summary if we have a valid benchmark and run_id
            #     if runner.benchmark and runner.benchmark.get_run_dir(run_id):
            #         print_run_summary(run_id, runner.benchmark.get_run_dir(run_id))
            #     else:
            #         print_warning("Could not generate run summary - missing benchmark or run directory")
        else:
            # Initialize agent runner
            print_step("Initializing agent runner...")
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
                    continue_run=continue_run
                )

                # Run evaluation
                print_step("Running evaluation with custom agent and HAL harness...")
                results = asyncio.run(runner.run(
                    agent_name=agent_name,
                    upload=upload or False
                ))
                
                print_success("Evaluation completed successfully")
                print_results_table(results)
                
                # Only print run summary if we have a valid benchmark and run_id
                if runner.benchmark and runner.benchmark.get_run_dir(run_id):
                    print_run_summary(run_id, runner.benchmark.get_run_dir(run_id))
                else:
                    print_warning("Could not generate run summary - missing benchmark or run directory")
                
            except Exception as e:
                print_error(f"Error running evaluation: {str(e)}")
                raise

    except Exception as e:
        # Get the full traceback
        full_traceback = traceback.format_exc()
        
        # Log the full error to the verbose log file
        with open(verbose_log_path, 'a') as f:
            f.write("\n=== ERROR TRACEBACK ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(full_traceback)
            f.write("\n=== END ERROR TRACEBACK ===\n")
        
        # Print clean error message to terminal
        print_error(f"An error occurred: {str(e)}")
        print_error(f"For detailed error information, check: {verbose_log_path}", verbose_log_path)
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
                        elif value.lower() in ['true', 'false']:
                            parsed_value = value.lower() == 'true'
                        # Handle numeric values
                        elif value.lower() in ['none', 'null', 'nan']:
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
        print_error(f"Error checking if agent function is a solver: {str(e)}")
        return False


def validate_model_pricing(model_name: str) -> None:
    """Validate that model pricing information exists"""
    from .utils.weave_utils import MODEL_PRICES_DICT
    
    if model_name not in MODEL_PRICES_DICT:
        print_error(f"Model '{model_name}' not found in pricing dictionary. Please add pricing information to MODEL_PRICES_DICT in weave_utils.py. Exiting...")
        sys.exit(1)


if __name__ == "__main__":
    main()

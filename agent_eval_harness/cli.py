import os
import click
import yaml
import asyncio
from typing import Any, Dict, Optional
import time

from .agent_runner import AgentRunner
from .inspect.inspect import is_inspect_benchmark
from .inspect_runner import inspect_evaluate
from dotenv import load_dotenv
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

load_dotenv()


@click.command()
@click.option(
    "--agent_name",
    required=True,
    help="Name of the agent you want to add to the leaderboard",
)
@click.option(
    "--agent_function",
    required=False,
    help="Path to the agent function. Example: agent.agent.run",
)
@click.option("--agent_dir", required=False, help="Path to the agent directory.")
@click.option(
    "-A",
    multiple=True,
    type=str,
    help="One or more args to pass to the agent (e.g. -A arg=value)",
)
@click.option("--benchmark", required=True, help="Name of the benchmark to run")
@click.option(
    "-B",
    multiple=True,
    type=str,
    help="One or more args to pass to the benchmark (e.g. -B arg=value)",
)
@click.option("--upload", is_flag=True, help="Upload results to HuggingFace")
@click.option("--max_concurrent", default=10, help="Maximum agents to run for this benchmark")
@click.option("--conda_env_name", help="Conda environment to run the custom external agent in if run locally")
@click.option("--model", default="gpt-4o-mini", help="Backend model to use")
@click.option("--run_id", help="Run ID to use for logging")
@click.option(
    "--config",
    default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    help="Path to configuration file",
)
@click.option("--vm", is_flag=True, help="Run the agent on an azure VM")
@click.option("--continue_run", is_flag=True, help="Continue from a previous run, only running failed or incomplete tasks")
def main(
    config,
    benchmark,
    agent_name,
    agent_function,
    agent_dir,
    model,
    run_id,
    upload,
    max_concurrent,
    conda_env_name,
    continue_run,
    a,
    b,
    vm,
    **kwargs,
):
    """Run agent evaluation on specified benchmark with given model."""
    
    # Generate default run_id if none provided
    if not run_id:
        # Take last part after final "/" if present (inspect_evals), otherwise use full benchmark name
        benchmark_name = benchmark.split("/")[-1]
        run_id = f"{benchmark_name}_{int(time.time())}"
    
    # Setup logging first, before any other operations
    log_dir = os.path.join("results", benchmark, run_id)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, run_id)
    
    print_header("HAL Harness")
    
    # Parse agent and benchmark args
    print_step("Parsing configuration...")
    agent_args = parse_cli_args(a)
    benchmark_args = parse_cli_args(b)
    
    # Print summary with run_id, benchmark, and the run config to terminal 
    print_run_config(
        run_id=run_id,
        benchmark=benchmark,
        agent_name=agent_name,
        agent_function=agent_function,
        agent_dir=agent_dir,
        model=model,
        agent_args=agent_args,
        benchmark_args=benchmark_args,
        upload=upload,
        max_concurrent=max_concurrent,
        conda_env_name=conda_env_name,
        vm=vm,
        continue_run=continue_run
    )
    
    # Add model to agent args
    agent_args['model_name'] = model
    
    if is_inspect_benchmark(benchmark):
        print_step("Running inspect evaluation...")
        # Run the inspect evaluation
        inspect_evaluate(
            benchmark=benchmark,
            benchmark_args=benchmark_args,
            agent_name=agent_name,
            agent_function=agent_function,
            agent_dir=agent_dir,
            agent_args=agent_args,
            model=model,
            run_id=run_id,  # Now guaranteed to have a value
            upload=upload or False,
            max_concurrent=max_concurrent,
            conda_env_name=conda_env_name,
            vm=vm,
            continue_run=continue_run
        )
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
                max_concurrent=max_concurrent,
                conda_env=conda_env_name,
                continue_run=continue_run
            )

            # Run evaluation
            print_step("Running evaluation...")
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


def parse_cli_args(args: tuple[str] | list[str] | None) -> Dict[str, Any]:
    """Parse CLI arguments into a dictionary."""
    params: Dict[str, Any] = {}
    if args:
        for arg in list(args):
            parts = arg.split("=")
            if len(parts) > 1:
                key = parts[0].replace("-", "_")
                value = yaml.safe_load("=".join(parts[1:]))
                if isinstance(value, str):
                    value = value.split(",")
                    value = value if len(value) > 1 else value[0]
                params[key] = value
    return params


if __name__ == "__main__":
    main()

import os
import sys
import asyncio
from pathlib import Path
import click
import yaml


def parse_cli_args(args: tuple[str] | list[str] | None):
    """Parse CLI arguments into a dictionary."""
    params: dict[str, any] = {}
    if args:
        for arg in list(args):
            parts = arg.split("=", 1)
            if len(parts) > 1:
                key = parts[0].replace("-", "_")
                value = parts[1]
                try:
                    parsed_value = yaml.safe_load(value)
                    if isinstance(parsed_value, str):
                        if "," in value:
                            parsed_value = value.split(",")
                        elif value.lower() in ["true", "false"]:
                            parsed_value = value.lower() == "true"
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
                    params[key] = value
    return params
from .utils.logging_utils import (
    setup_logging,
    print_header,
    print_step,
    print_success,
    log_results_table,
)


@click.command(help="Run the HAL generalist agent on a benchmark for quick testing.")
@click.option("--benchmark", required=True, help="Benchmark name to test")
@click.option("-A", "a", multiple=True, type=str, help="Args to pass to the agent")
@click.option("-B", "b", multiple=True, type=str, help="Args to pass to the benchmark")
@click.option("--max_tasks", default=1, show_default=True, help="Number of tasks to run")
def main(benchmark: str, a: tuple[str], b: tuple[str], max_tasks: int) -> None:
    """Entry point for hal-test CLI."""
    agent_args = parse_cli_args(a)
    benchmark_args = parse_cli_args(b)

    config_path = Path(__file__).with_name("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    agent_dir = Path(__file__).resolve().parent.parent / "agents" / "hal_generalist_agent"
    run_command = " ".join(["hal-test"] + sys.argv[1:])
    run_id = f"test_{benchmark}"

    log_dir = os.path.join("results", benchmark, run_id)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, run_id)


    from .agent_runner import AgentRunner

    print_header("HAL Harness Test")
    print_step(f"Benchmark: {benchmark}")

    runner = AgentRunner(
        agent_function="main.run",
        agent_dir=str(agent_dir),
        agent_args={**agent_args, **benchmark_args},
        benchmark_name=benchmark,
        config=config,
        run_id=run_id,
        max_concurrent=1,
        run_command=run_command,
        ignore_errors=False,
        max_tasks=max_tasks,
    )

    results = asyncio.run(runner.run(agent_name="hal_generalist_agent"))
    print_success("Test run completed")
    log_results_table(results)


if __name__ == "__main__":
    main()

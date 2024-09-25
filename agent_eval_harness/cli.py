import os

import click

from typing import Any

from .agent_runner import run_agent_evaluation
from .inspect.inspect import is_inspect_benchmark
from .inspect_runner import inspect_evaluate
from .utils.config import load_config


@click.command()
@click.option('--agent_name', required=True, help='Name of the agent you want to add to the leaderboard')
@click.option('--agent_function', required=False, help='Path to the agent function. Example: agent.agent.run')
@click.option('--agent_dir', required=False, help='Path to the agent directory.')
@click.option('--benchmark', required=True, help='Name of the benchmark to run')
@click.option('--upload', is_flag=True, help='Upload results to HuggingFace')
@click.option('--model', default='gpt-4o-mini', help='Backend model to use')
@click.option('--run_id', help='Run ID to use for logging')
@click.option('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'), help='Path to configuration file')
@click.pass_context
def main(ctx, config, **kwargs):
    """Run agent evaluation on specified benchmark with given model."""
    config = load_config(config)

    benchmark = kwargs["benchmark"]
    if is_inspect_benchmark(benchmark):
        # run the inspect evaluation
        inspect_evaluate(
            benchmark=benchmark,
            agent_name=kwargs["agent_name"],
            agent_function=kwargs["agent_function"],
            agent_dir=kwargs["agent_dir"],
            model=kwargs["model"],
            run_id=kwargs["run_id"],
            upload=kwargs["upload"] or False,
        )
    else:
        # run the agent evaluation
        if not is_valid("--agent-function", kwargs["agent_function"]) or not is_valid("--agent-dir", kwargs["agent_dir"]):
            return
        run_agent_evaluation(config=config, **kwargs)

def is_valid(name: str, value: Any) -> bool:
    if value is None:
        print(f"Error: missing option `{name}`")
        return False
    return True

if __name__ == '__main__':
    main()

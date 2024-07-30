# Main CLI script (cli.py)
import click
from .agent_runner import run_agent_evaluation
from .utils.config import load_config

@click.command()
@click.option('--agent', required=True, help='Path to the agent function. Example: agent.agent.run')
@click.option('--benchmark', required=True, help='Name of the benchmark to run')
@click.option('--model', default='gpt-4o-mini', help='Backend model to use')
@click.option('--config', default='agent_eval_harness/config.yaml', help='Path to configuration file')
def main(agent, benchmark, model, config):
    """Run agent evaluation on specified benchmark with given model."""
    config = load_config(config)
    result = run_agent_evaluation(agent, benchmark_name=benchmark, model=model, config=config)
    click.echo(f"Evaluation complete. Results stored and uploaded.")

if __name__ == '__main__':
    main()
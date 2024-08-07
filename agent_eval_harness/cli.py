import click
from .agent_runner import run_agent_evaluation
from .utils.config import load_config
import os

@click.command()
@click.option('--agent_name', required=True, help='Name of the agent you want to add to the leaderboard')
@click.option('--agent_function', required=True, help='Path to the agent function. Example: agent.agent.run')
@click.option('--agent_dir', required=True, help='Path to the agent directory.')
@click.option('--benchmark', required=True, help='Name of the benchmark to run')
@click.option('--upload', is_flag=True, help='Upload results to HuggingFace')
@click.option('--model', default='gpt-4o-mini', help='Backend model to use')
@click.option('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'), help='Path to configuration file')
@click.pass_context
def main(ctx, config, **kwargs):
    """Run agent evaluation on specified benchmark with given model."""
    config = load_config(config)
        
    run_agent_evaluation(config=config, **kwargs)

if __name__ == '__main__':
    main()
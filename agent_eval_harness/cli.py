import click
from .agent_runner import run_agent_evaluation
from .utils.config import load_config

@click.command()
@click.option('--agent_name', required=True, help='Name of the agent you want to add to the leaderboard')
@click.option('--agent_function', required=True, help='Path to the agent function. Example: agent.agent.run')
@click.option('--agent_dir', required=True, help='Path to the agent directory.')
@click.option('--benchmark', required=True, help='Name of the benchmark to run')
@click.option('--upload', is_flag=True, help='Upload results to HuggingFace')
@click.option('--model', default='gpt-4o-mini', help='Backend model to use')
@click.option('--config', default='agent_eval_harness/config.yaml', help='Path to configuration file')
@click.pass_context
def main(ctx, config, **kwargs):
    """Run agent evaluation on specified benchmark with given model."""
    config = load_config(config)
        
    result = run_agent_evaluation(config=config, **kwargs)
    click.echo(f"Evaluation complete. Results stored and uploaded.")

if __name__ == '__main__':
    main()
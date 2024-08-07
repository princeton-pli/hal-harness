import os
import json
import click
from huggingface_hub import HfApi
from ..benchmark_manager import BenchmarkManager
from dotenv import load_dotenv

load_dotenv()

bm = BenchmarkManager()

available_benchmarks = bm.list_benchmarks()

def find_upload_files(directory):
    """Recursively find all files containing '_UPLOAD' in their name and ending with '.json'."""
    for root, _, files in os.walk(directory):
        for filename in files:
            if '_UPLOAD.json' in filename and filename.endswith('.json'):
                yield os.path.join(root, filename)

@click.command()
@click.option('--benchmark', required=True, help='Name of the benchmark', type=click.Choice(available_benchmarks))
def upload_results(benchmark):
    """Upload result files for a given benchmark to Hugging Face Hub, including all subdirectories."""
    results_dir = os.path.join('results', benchmark)
    
    if not os.path.exists(results_dir):
        click.echo(f"Error: Directory {results_dir} does not exist.")
        return

    api = HfApi(token=os.getenv("HF_TOKEN"))
    file_paths = find_upload_files(results_dir)

    print("==== Uploading results ====")
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        # Extract run_id from filename
        run_id = filename.replace('_UPLOAD.json', '')
        
        try:
            
            # Upload to Hugging Face Hub
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"{run_id}.json",
                repo_id="agent-evals/results"   ,
                repo_type="dataset",
                commit_message=f"Add {run_id} to leaderboard.",
            )
            
            click.echo(f"Uploaded {filename} to Hugging Face Hub.")
            
        except Exception as e:
            click.echo(f"Error uploading {filename}: {str(e)}")

    click.echo("Upload process completed.")
    print("=====")

if __name__ == '__main__':
    upload_results()
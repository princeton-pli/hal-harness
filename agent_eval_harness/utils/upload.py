import os
import click
from huggingface_hub import HfApi
from ..benchmark_manager import BenchmarkManager
from dotenv import load_dotenv
from .logging_utils import (
    print_header,
    print_step,
    print_success,
    print_error,
    terminal_print,
    console,
    create_progress
)

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
@click.option('--benchmark', '-B', required=True, help='Name of the benchmark', type=click.Choice(available_benchmarks))
def upload_results(benchmark):
    """Upload result files for a given benchmark to Hugging Face Hub, including all subdirectories."""
    try:
        print_header("HAL Upload Results")
        
        results_dir = os.path.join('results', benchmark)
        
        if not os.path.exists(results_dir):
            print_error(f"Directory {results_dir} does not exist.")
            return

        api = HfApi(token=os.getenv("HF_TOKEN"))
        file_paths = list(find_upload_files(results_dir))
        
        if not file_paths:
            print_error(f"No upload files found in {results_dir}")
            return

        print_step(f"Found {len(file_paths)} files to upload")
        
        # Create progress bar
        with create_progress() as progress:
            upload_task = progress.add_task(
                "Uploading files...",
                total=len(file_paths)
            )
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                run_id = filename.replace('_UPLOAD.json', '')
                
                try:
                    # Upload to Hugging Face Hub
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=f"{run_id}.json",
                        repo_id="agent-evals/results",
                        repo_type="dataset",
                        commit_message=f"Add {run_id} to leaderboard.",
                    )
                    
                    print_success(f"Successfully uploaded {filename}")
                    
                except Exception as e:
                    print_error(f"Error uploading {filename}: {str(e)}")
                
                progress.update(upload_task, advance=1)

        print_success("Upload process completed")

    except Exception as e:
        print_error(f"An error occurred during upload: {str(e)}")
        raise

if __name__ == '__main__':
    upload_results()
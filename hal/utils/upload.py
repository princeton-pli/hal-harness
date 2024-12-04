import os
import click
from huggingface_hub import HfApi
from ..benchmark_manager import BenchmarkManager
from dotenv import load_dotenv
from .encryption import ZipEncryption
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
ENCRYPTION_PASSWORD = "hal1234"  # Fixed encryption password

def find_upload_files(directory, require_upload_suffix=False):
    """Recursively find JSON files.
    
    Args:
        directory: Directory to search
        require_upload_suffix: If True, only find files containing '_UPLOAD.json'
    """
    for root, _, files in os.walk(directory):
        for filename in files:
            if require_upload_suffix:
                if '_UPLOAD.json' in filename and filename.endswith('.json'):
                    yield os.path.join(root, filename)
            else:
                if filename.endswith('.json'):
                    yield os.path.join(root, filename)

@click.command()
@click.option('--benchmark', '-B', help='Name of the benchmark', type=click.Choice(available_benchmarks))
@click.option('--file', '-F', type=click.Path(exists=True), help='Path to single file to upload')
@click.option('--directory', '-D', type=click.Path(exists=True), help='Path to directory containing files to upload')
def upload_results(benchmark, file, directory):
    """Upload encrypted zip files to Hugging Face Hub. Use one of: -B (benchmark), -F (file), or -D (directory)."""
    try:
        print_header("HAL Upload Results")
        
        # Check that only one option is provided
        options_count = sum(1 for opt in [benchmark, file, directory] if opt is not None)
        if options_count != 1:
            print_error("Please provide exactly one of: -B (benchmark), -F (file), or -D (directory)")
            return

        api = HfApi(token=os.getenv("HF_TOKEN"))
        
        if benchmark:
            results_dir = os.path.join('results', benchmark)
            if not os.path.exists(results_dir):
                print_error(f"Directory {results_dir} does not exist.")
                return
            file_paths = list(find_upload_files(results_dir, require_upload_suffix=True))
            
        elif file:
            if not file.endswith('.json'):
                print_error(f"File must be a JSON file")
                return
            file_paths = [file]
            
        else:  # directory option
            file_paths = list(find_upload_files(directory, require_upload_suffix=False))
        
        if not file_paths:
            print_error(f"No upload files found")
            return

        print_step(f"Found {len(file_paths)} files to process")
        
        # Create progress bar
        with create_progress() as progress:
            task = progress.add_task("Processing and uploading files...", total=len(file_paths))
            
            # Group files by directory
            files_by_dir = {}
            for file_path in file_paths:
                dir_path = os.path.dirname(file_path)
                if dir_path not in files_by_dir:
                    files_by_dir[dir_path] = []
                files_by_dir[dir_path].append(file_path)
            
            # Process each directory
            for dir_path, dir_files in files_by_dir.items():
                # Create a temporary encrypted zip file
                zip_encryptor = ZipEncryption(ENCRYPTION_PASSWORD)
                temp_zip_path = os.path.join(dir_path, "temp_encrypted.zip")
                
                try:
                    # Create encrypted zip
                    zip_encryptor.encrypt_files(dir_files, temp_zip_path)
                    
                    # Upload to Hugging Face Hub
                    dir_name = os.path.basename(dir_path)
                    # Remove _UPLOAD from the directory name if present
                    if '_UPLOAD' in dir_name:
                        dir_name = dir_name.replace('_UPLOAD', '')
                    
                    api.upload_file(
                        path_or_fileobj=temp_zip_path,
                        path_in_repo=f"{dir_name}.zip",
                        repo_id="agent-evals/results",
                        repo_type="dataset",
                        commit_message=f"Add encrypted results for {dir_name}.",
                    )
                    
                    print_success(f"\nSuccessfully uploaded encrypted results for {dir_name}")
                    
                except Exception as e:
                    print_error(f"Error processing directory {dir_path}: {str(e)}")
                
                finally:
                    # Clean up temporary zip file
                    if os.path.exists(temp_zip_path):
                        os.remove(temp_zip_path)
                
                progress.update(task, advance=len(dir_files))

        print_success("Upload process completed")

    except Exception as e:
        print_error(f"An error occurred during upload: {str(e)}")
        raise

if __name__ == '__main__':
    upload_results()
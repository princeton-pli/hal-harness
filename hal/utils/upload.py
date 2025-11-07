import os
import click
import json
from huggingface_hub import HfApi
from ..benchmark_manager import BenchmarkManager
from dotenv import load_dotenv
from .encryption import ZipEncryption
from .logging_utils import (
    print_header,
    print_step,
    print_success,
    print_error,
    print_warning,
    terminal_print,
    console,
    create_progress
)

load_dotenv()

def validate_eval_results(file_path):
    """Validate that the evaluation results don't contain errors that indicate an incomplete run."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'raw_eval_results' not in data:
            return True, None, None
        
        raw_eval_results = data['raw_eval_results']
        if not isinstance(raw_eval_results, dict):
            return True, None, None
        
        error_tasks = []
        for task_id, task_result in raw_eval_results.items():
            if isinstance(task_result, dict):
                explanation = task_result.get('explanation', '')
                if isinstance(explanation, str):
                    error_patterns = [
                        'Evaluated ERROR:',
                        'Traceback (most recent call last)',
                        'Exception occurred',
                        'Failed to evaluate',
                        'Error during evaluation'
                    ]
                    
                    if any(pattern in explanation for pattern in error_patterns):
                        error_tasks.append({
                            'task_id': task_id,
                            'explanation': explanation
                        })
        
        if error_tasks:
            error_message = f"Found {len(error_tasks)} tasks with evaluation errors"
            return False, error_message, error_tasks
        
        return True, None, None
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}", None
    except Exception as e:
        return False, f"Error validating file: {str(e)}", None

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
@click.option('--skip-validation', is_flag=True, help='Skip validation checks and force upload (advanced users only)')
def upload_results(benchmark, file, directory, skip_validation):
    """Upload encrypted zip files to Hugging Face Hub. 
    
    Files are automatically validated for evaluation errors before upload.
    Use one of: -B (benchmark), -F (file), or -D (directory).
    """
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
            file_paths = list(find_upload_files(directory, require_upload_suffix=True))
        
        if not file_paths:
            print_error(f"No upload files found")
            return

        print_step(f"Found {len(file_paths)} files to process")
        
        if not skip_validation:
            print_step("Validating files for evaluation errors...")
            invalid_files = []
            
            for file_path in file_paths:
                is_valid, error_msg, error_details = validate_eval_results(file_path)
                if not is_valid:
                    invalid_files.append({
                        'file': os.path.basename(file_path),
                        'error': error_msg,
                        'details': error_details
                    })
            
            if invalid_files:
                print_error("❌ Upload rejected due to evaluation errors in the following files:")
                for invalid_file in invalid_files:
                    print_error(f"  • {invalid_file['file']}: {invalid_file['error']}")
                    if invalid_file['details']:
                        print_warning(f"    First few error tasks:")
                        for i, task_error in enumerate(invalid_file['details'][:3]):
                            print_warning(f"      - {task_error['task_id']}: {task_error['explanation'][:100]}...")
                        if len(invalid_file['details']) > 3:
                            print_warning(f"    ... and {len(invalid_file['details']) - 3} more errors")
                
                print_error("\n💡 Please re-run the evaluation to completion before uploading.")
                print_error("   Use --continue_run flag to retry failed tasks, or check for infrastructure issues.")
                print_error("   To force upload anyway, use --skip-validation flag (not recommended).")
                return
            
            print_success("✅ All files passed validation")
        else:
            print_warning("⚠️  Skipping validation checks (--skip-validation flag used)")
        
        # Create progress bar
        with create_progress() as progress:
            task = progress.add_task("Processing and uploading files...", total=len(file_paths))
            
            # Process each file individually instead of grouping by directory
            for file_path in file_paths:
                # Create a temporary encrypted zip file
                zip_encryptor = ZipEncryption(ENCRYPTION_PASSWORD)
                file_name = os.path.basename(file_path)
                dir_path = os.path.dirname(file_path)
                temp_zip_path = os.path.join(dir_path, f"temp_{file_name}.zip")
                
                try:
                    # Create encrypted zip for single file
                    zip_encryptor.encrypt_files([file_path], temp_zip_path)
                    
                    # Upload to Hugging Face Hub
                    # Remove .json extension for zip name
                    zip_name = file_name.replace('.json', '')
                    
                    api.upload_file(
                        path_or_fileobj=temp_zip_path,
                        path_in_repo=f"{zip_name}.zip",
                        repo_id="agent-evals/hal_traces",
                        repo_type="dataset",
                        commit_message=f"Add encrypted results for {zip_name}.",
                    )
                    
                    print_success(f"\nSuccessfully uploaded encrypted results for {zip_name}")
                    
                except Exception as e:
                    print_error(f"Error processing file {file_path}: {str(e)}")
                
                finally:
                    # Clean up temporary zip file
                    if os.path.exists(temp_zip_path):
                        os.remove(temp_zip_path)
                
                progress.update(task, advance=1)

        print_success("Upload process completed")

    except Exception as e:
        print_error(f"An error occurred during upload: {str(e)}")
        raise

if __name__ == '__main__':
    upload_results()
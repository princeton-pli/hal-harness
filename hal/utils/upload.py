import os
import click
from huggingface_hub import HfApi
from ..benchmark_manager import BenchmarkManager
from dotenv import load_dotenv
from .encryption import ZipEncryption
from .logging_utils import create_progress
import logging

logger = logging.getLogger("agent_eval")

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
                if "_UPLOAD.json" in filename and filename.endswith(".json"):
                    yield os.path.join(root, filename)
            else:
                if filename.endswith(".json"):
                    yield os.path.join(root, filename)


@click.command()
@click.option(
    "--benchmark",
    "-B",
    help="Name of the benchmark",
    type=click.Choice(available_benchmarks),
)
@click.option(
    "--file", "-F", type=click.Path(exists=True), help="Path to single file to upload"
)
@click.option(
    "--directory",
    "-D",
    type=click.Path(exists=True),
    help="Path to directory containing files to upload",
)
def upload_results(benchmark, file, directory):
    """Upload encrypted zip files to Hugging Face Hub. Use one of: -B (benchmark), -F (file), or -D (directory)."""
    try:
        logger.info("=== HAL Upload Results ===")

        # Check that only one option is provided
        options_count = sum(
            1 for opt in [benchmark, file, directory] if opt is not None
        )
        if options_count != 1:
            logger.error(
                "Please provide exactly one of: -B (benchmark), -F (file), or -D (directory)"
            )
            return

        api = HfApi(token=os.getenv("HF_TOKEN"))

        if benchmark:
            results_dir = os.path.join("results", benchmark)
            if not os.path.exists(results_dir):
                logger.error(f"Directory {results_dir} does not exist.")
                return
            file_paths = list(
                find_upload_files(results_dir, require_upload_suffix=True)
            )

        elif file:
            if not file.endswith(".json"):
                logger.error("File must be a JSON file")
                return
            file_paths = [file]

        else:  # directory option
            file_paths = list(find_upload_files(directory, require_upload_suffix=True))

        if not file_paths:
            logger.error("No upload files found")
            return

        logger.info(f"Found {len(file_paths)} files to process")

        # Create progress bar
        with create_progress() as progress:
            task = progress.add_task(
                "Processing and uploading files...", total=len(file_paths)
            )

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
                    zip_name = file_name.replace(".json", "")

                    api.upload_file(
                        path_or_fileobj=temp_zip_path,
                        path_in_repo=f"{zip_name}.zip",
                        repo_id="agent-evals/hal_traces",
                        repo_type="dataset",
                        commit_message=f"Add encrypted results for {zip_name}.",
                    )

                    logger.info(
                        f"\nSuccessfully uploaded encrypted results for {zip_name}"
                    )

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")

                finally:
                    # Clean up temporary zip file
                    if os.path.exists(temp_zip_path):
                        os.remove(temp_zip_path)

                progress.update(task, advance=1)

        logger.info("Upload process completed")

    except Exception as e:
        logger.error(f"An error occurred during upload: {str(e)}")
        raise


if __name__ == "__main__":
    upload_results()

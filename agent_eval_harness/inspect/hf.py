import json
import os
import tempfile

from typing import Any

from huggingface_hub import HfApi  # type: ignore


def upload_results(run_id: str, weave_result: Any) -> None:
    """
    Uploads the results of a specific run to the Hugging Face Hub.

    Args:
        run_id (str): A unique identifier for the run whose results are being uploaded.
        weave_result (Any): The results of the run, expected to be serializable to JSON format.

    This function performs the following steps:
    1. Serializes the `weave_result` to a temporary JSON file.
    2. Uploads the file to the specified Hugging Face Hub repository.
    3. Deletes the temporary file after successful upload.

    Environment Variables:
        HF_TOKEN: The Hugging Face authentication token for API access. If not provided, defaults to `"."`.
    """

    # Write results to temp file using tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(weave_result, f)
        temp_file_name = f.name
        temp_file_path = os.path.abspath(temp_file_name)

    # Use huggingface API to upload results
    api = HfApi(token=os.getenv("HF_TOKEN", "."))
    api.upload_file(
        path_or_fileobj=temp_file_path,
        path_in_repo=f"{run_id}.json",
        repo_id="agent-evals/results",
        repo_type="dataset",
        commit_message=f"Add {run_id} to leaderboard.",
    )

    # unlink the file
    os.unlink(temp_file_name)

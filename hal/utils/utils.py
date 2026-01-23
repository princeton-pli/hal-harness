import re
import os
import json
from typing import Any, Dict
import subprocess


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    git_info = {}

    try:
        # Get current commit hash
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        git_info["commit"] = git_commit

        # Get repository URL
        git_remote_url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        git_info["repository_url"] = git_remote_url

        # Get current branch
        git_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        git_info["branch"] = git_branch

        # Get commit timestamp
        git_commit_time = (
            subprocess.check_output(
                ["git", "show", "-s", "--format=%ci", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        git_info["commit_timestamp"] = git_commit_time

        # Build repository URL with commit
        if "github.com" in git_remote_url:
            # Format GitHub URL to point to specific commit
            repo_base = git_remote_url.replace(".git", "").replace(
                "git@github.com:", "https://github.com/"
            )
            if repo_base.endswith("/"):
                repo_base = repo_base[:-1]
            git_info["commit_url"] = f"{repo_base}/tree/{git_commit}"

    except subprocess.SubprocessError:
        git_info["error"] = "Failed to get git information"

    return git_info


def move_merge_dirs(source_root, dest_root):
    for path, dirs, files in os.walk(source_root, topdown=False):
        dest_dir = os.path.join(dest_root, os.path.relpath(path, source_root))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for filename in files:
            os.rename(os.path.join(path, filename), os.path.join(dest_dir, filename))
        for dirname in dirs:
            os.rmdir(os.path.join(path, dirname))
    os.rmdir(source_root)


def safe_filename(input_string):
    # Replace spaces with underscores
    transformed_string = input_string.replace(" ", "_")
    # Remove or replace any characters that are not safe for file names
    transformed_string = re.sub(r"[^\w\-\.]", "", transformed_string)
    return transformed_string


def make_json_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (
                    obj.startswith("[") and obj.endswith("]")
                ):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {
            "_type": obj.__class__.__name__,
            **{k: make_json_serializable(v) for k, v in obj.__dict__.items()},
        }
    else:
        # For any other type, convert to string
        return str(obj)

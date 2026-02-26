"""Task difficulty level extraction for GAIA benchmark."""

import json
from pathlib import Path
from typing import Dict


def extract_task_levels(run_dir: Path) -> Dict[str, str]:
    """
    Extract task difficulty levels from input.json files in a GAIA run directory.

    Args:
        run_dir: Path to the run directory containing task subdirectories

    Returns:
        Dict mapping task_id to level ("1", "2", or "3")
    """
    levels = {}

    # Each task has its own subdirectory with input.json
    for task_dir in run_dir.iterdir():
        if not task_dir.is_dir():
            continue
        input_file = task_dir / "input.json"
        if not input_file.exists():
            continue
        try:
            with open(input_file, "r") as f:
                data = json.load(f)
            # Input format: {task_id: {task_id, Question, Level, ...}}
            for task_id, task_data in data.items():
                if isinstance(task_data, dict) and "Level" in task_data:
                    levels[task_id] = str(task_data["Level"])
        except Exception:
            continue

    return levels

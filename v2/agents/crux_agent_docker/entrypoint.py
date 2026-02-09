#!/usr/bin/env python3
"""
Autonomous coding agent using Aider.
Mimics the Anthropic compiler approach with test-driven development loops.
"""

import os
import sys
import logging
import json
import subprocess
import traceback
from typing import List, Dict, Optional


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to both file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "agent.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def initialize_git_in_workspace(workspace_dir: str, logger: logging.Logger) -> None:
    """Initialize workspace with git repo."""
    os.makedirs(workspace_dir, exist_ok=True)
    os.chdir(workspace_dir)

    # Initialize git repo if not exists
    if not os.path.exists(".git"):
        logger.info("Initializing git repository...")
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "Initial commit"],
            check=True,
            capture_output=True,
        )
        logger.info("Git repository initialized")


def run_aider_task(
    task: str,
    test_command: Optional[str],
    max_retries: int,
    logger: logging.Logger,
    api_key: str,
) -> Dict[str, any]:
    """
    Run a single task with Aider using test-driven retry loop.

    Args:
        task: The task description for Aider
        test_command: Shell command to validate the implementation
        max_retries: Maximum retry attempts if tests fail
        logger: Logger instance
        api_key: Anthropic API key

    Returns:
        Dict with status, attempts, and error info
    """
    logger.info(f"🚀 Starting task: {task}")

    # Aider command with auto-test loop
    cmd = [
        "aider",
        "--yes",  # Auto-confirm all prompts
        "--no-auto-commits",  # We'll commit manually
        "--message",
        task,
        "--model",
        "claude-3-7-sonnet-20250219",  # Use latest Claude
    ]

    # Add test command if provided (enables self-healing loop)
    if test_command:
        cmd.extend(
            [
                "--test-cmd",
                test_command,
                "--auto-test",  # Auto-retry on test failures
            ]
        )

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = api_key

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        logger.info(f"Attempt {attempt}/{max_retries}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=1_800,  # 30 minute timeout per attempt
            )
            # FIXME: log this directly [?]
            logger.info(f"Aider stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Aider stderr:\n{result.stderr}")

            if result.returncode == 0:
                logger.info("✅ Task completed successfully")

                # Commit the changes if there are any
                subprocess.run(["git", "add", "-A"], check=True)

                # Check if there are changes to commit
                status_result = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    capture_output=True
                )

                if status_result.returncode != 0:  # There are changes
                    subprocess.run(
                        ["git", "commit", "-m", f"Completed: {task}"],
                        check=True,
                        capture_output=True
                    )
                    logger.info("Changes committed")
                else:
                    logger.info("No changes to commit")

                return {"status": "success", "attempts": attempt, "task": task}
            else:
                logger.warning(
                    f"Attempt {attempt} failed with return code {result.returncode}"
                )

                # If we have more retries, continue
                if attempt < max_retries:
                    logger.info("Retrying...")
                    continue

        except subprocess.TimeoutExpired:
            logger.error(f"Task timed out on attempt {attempt}")
            if attempt < max_retries:
                continue
        except Exception as e:
            logger.error(f"Error running aider: {e}")
            logger.error(traceback.format_exc())
            if attempt < max_retries:
                continue

    # All retries exhausted
    logger.error(f"❌ Task failed after {max_retries} attempts")
    return {
        "status": "failed",
        "attempts": attempt,
        "task": task,
        "error": "Max retries exhausted",
    }


def run_task_pipeline(
    tasks: List[Dict[str, str]],
    workspace_dir: str,
    results_dir: str,
    logger: logging.Logger,
    api_key: str,
) -> Dict[str, any]:
    """
    Run a pipeline of coding tasks autonomously.

    Args:
        tasks: List of task dicts with 'description' and optional 'test_command'
        workspace_dir: Working directory for code
        results_dir: Directory to save results
        logger: Logger instance
        api_key: Anthropic API key

    Returns:
        Summary of all task results
    """
    results = []

    for i, task_spec in enumerate(tasks, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Task {i}/{len(tasks)}")
        logger.info(f"{'=' * 60}\n")

        task_desc = task_spec.get("description", "")
        test_cmd = task_spec.get("test_command")
        max_retries = 3

        result = run_aider_task(
            task=task_desc,
            test_command=test_cmd,
            max_retries=max_retries,
            logger=logger,
            api_key=api_key,
        )

        results.append(result)

    # Save results
    summary = {
        "total_tasks": len(tasks),
        "completed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }

    results_file = os.path.join(results_dir, "task_results.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info("Pipeline Summary")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total tasks: {summary['total_tasks']}")
    logger.info(f"Completed: {summary['completed']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Results saved to: {results_file}")

    return summary


def main():
    """Main entrypoint for autonomous coding agent."""
    # Setup directories
    log_dir = "/workspace/logs"
    results_dir = "/workspace/results"
    workspace_dir = "/workspace/code"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logging(log_dir)

    try:
        # Read configuration from environment
        run_id = os.environ.get("HAL_RUN_ID")
        task_id = os.environ.get("HAL_TASK_ID")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Validate required variables
        if not all([run_id, task_id, anthropic_api_key]):
            raise ValueError(
                "Missing required environment variables. Required: "
                "HAL_RUN_ID, HAL_TASK_ID, ANTHROPIC_API_KEY"
            )

        logger.info(f"Agent starting - RunID={run_id}, TaskID={task_id}")

        # Setup workspace
        initialize_git_in_workspace(workspace_dir, logger)

        # Define tasks (this could be loaded from a file or API)
        # FIXME: make the tasks auto-generated from a planning agent
        tasks = [
            {
                "description": "Create a Python file called hello.py that prints 'Hello from autonomous agent!'",
                "test_command": "python hello.py | grep -q 'Hello from autonomous agent'",
                "max_retries": 3,
            },
            {
                "description": "Create a Python file called calculator.py with an add(a, b) function that returns the sum of two numbers",
                "test_command": "python -c 'from calculator import add; assert add(2, 3) == 5; assert add(-1, 1) == 0; print(\"Tests passed\")'",
                "max_retries": 3,
            },
        ]

        # FIXME: Save tasks to a file; don't read them in

        # Load tasks from file if it exists
        tasks_file = "/workspace/tasks.json"
        if os.path.exists(tasks_file):
            logger.info(f"Loading tasks from {tasks_file}")
            with open(tasks_file, "r") as f:
                tasks = json.load(f)
        else:
            logger.info("Using default demo tasks")

        # Run the task pipeline
        summary = run_task_pipeline(
            tasks=tasks,
            workspace_dir=workspace_dir,
            results_dir=results_dir,
            logger=logger,
            api_key=anthropic_api_key,
        )

        # Exit with success if all tasks completed
        if summary["failed"] == 0:
            logger.info("🎉 All tasks completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"⚠️  {summary['failed']} task(s) failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())

        # Write error log
        error_file = os.path.join(results_dir, "error.log")
        with open(error_file, "w") as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()

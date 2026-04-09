from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from typing import Optional, Any
import logging
import sys
import os
from datetime import datetime

# Create logger
logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, run_id: str) -> None:
    """Setup logging configuration.

    Args:
        log_dir: Directory for log files
        run_id: Unique run identifier
    """
    # Create absolute path for log directory to avoid path duplication
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(log_dir, f"{os.path.basename(run_id)}.log")

    # Configure root logger (so all child loggers inherit)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Suppress httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(name)s: %(message)s")

    # File handler (info and above)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    # Console handler (info and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Initial setup logging
    logger.info(f"Logging initialized - {datetime.now().isoformat()}")
    logger.info(f"Log directory: {log_dir}")


def create_progress() -> Progress:
    """Create a rich progress bar that prints to terminal"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=Console(file=sys.__stdout__),  # Use system stdout directly
    )


def log_results(results: dict[str, Any]) -> None:
    """Log results to both console and file"""

    logger.info("=== Evaluation Results ===")

    # Handle both direct results dict and nested results structure
    metrics_dict = (
        results.get("results", results) if isinstance(results, dict) else results
    )

    if isinstance(metrics_dict, dict):
        for key, value in metrics_dict.items():
            # Skip nested dictionaries and certain keys we don't want to display
            if isinstance(value, (int, float)) and value:
                formatted_value = (
                    f"{value:.6f}" if isinstance(value, float) else str(value)
                )
                logger.info(f"  {key}: {formatted_value}")
            elif (
                isinstance(value, str)
                and key not in ["status", "message", "traceback"]
                and value
            ):
                logger.info(f"  {key}: {value}")
            elif key == "successful_tasks" and value:
                logger.info(f"  successful_tasks: {len(value)}")
            elif key == "failed_tasks" and value:
                logger.info(f"  failed_tasks: {len(value)}")
            elif key == "latencies" and value:
                # compute average total_time across all tasks
                total_time = 0
                for _, latency in value.items():
                    total_time += latency["total_time"]
                logger.info(f"  average_total_time: {total_time / len(value)}")


def log_run_summary(run_id: str, log_dir: str) -> None:
    """Log run summary information"""
    logger.info("=== Run Summary ===")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Log Directory: {log_dir}")


def print_run_config(
    run_id: str,
    benchmark: str,
    agent_name: str,
    agent_function: str,
    agent_dir: str,
    agent_args: dict,
    benchmark_args: dict,
    inspect_eval_args: dict,
    upload: bool,
    max_concurrent: int,
    log_dir: str,
    conda_env_name: Optional[str],
    continue_run: bool,
    docker: bool = False,
    ignore_errors: bool = False,
    prompt_sensitivity: bool = False,
    num_variations: int = 3,
    variation_strength: str = "mild",
) -> None:
    """Print a formatted table with the run configuration"""
    logger.info("=== Run Configuration ===")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Benchmark: {benchmark}")
    logger.info(f"  Agent Name: {agent_name}")
    logger.info(f"  Agent Function: {agent_function}")
    logger.info(f"  Agent Directory: {agent_dir}")
    logger.info(f"  Log Directory: {log_dir}")
    logger.info(f"  Max Concurrent: {max_concurrent}")
    logger.info(f"  Upload Results: {'Yes' if upload else 'No'}")
    logger.info(f"  Docker Execution: {'Yes' if docker else 'No'}")
    logger.info(f"  Continue Previous Run: {'Yes' if continue_run else 'No'}")
    logger.info(f"  Ignore Errors: {'Yes' if ignore_errors else 'No'}")
    logger.info(f"  Prompt Sensitivity: {'Yes' if prompt_sensitivity else 'No'}")
    if prompt_sensitivity:
        logger.info(f"  Num Variations: {num_variations}")
        logger.info(f"  Variation Strength: {variation_strength}")

    if conda_env_name:
        logger.info(f"  Conda Environment: {conda_env_name}")

    if agent_args:
        logger.info("  Agent Arguments:")
        for key, value in agent_args.items():
            logger.info(f"    {key}: {value}")

    if benchmark_args:
        logger.info("  Benchmark Arguments:")
        for key, value in benchmark_args.items():
            logger.info(f"    {key}: {value}")

    if inspect_eval_args:
        logger.info("  Inspect Eval Arguments:")
        for key, value in inspect_eval_args.items():
            logger.info(f"    {key}: {value}")

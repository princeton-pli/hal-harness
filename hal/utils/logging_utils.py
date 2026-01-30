from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.table import Table
from typing import Optional, Any, Dict
import logging
import sys
import os
import json
from datetime import datetime
from contextlib import contextmanager
from rich.box import ROUNDED

# Initialize rich console for terminal output
console = Console()

# Create loggers
main_logger = logging.getLogger("agent_eval")
verbose_logger = logging.getLogger("agent_eval.verbose")

# Store global paths
_log_paths = {"main_log": None, "verbose_log": None, "log_dir": None}


class VerboseFilter(logging.Filter):
    """Filter to separate verbose logs from main logs"""

    def filter(self, record):
        return not record.name.startswith("agent_eval.verbose")


def setup_logging(log_dir: str, run_id: str, use_vm: bool = False) -> None:
    """Setup logging configuration with optional Azure Monitor integration.

    Args:
        log_dir: Directory for log files
        run_id: Unique run identifier
        use_vm: If True, enables Azure Monitor logging (for VM runs)
    """
    # Create absolute path for log directory to avoid path duplication
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Create log files with clean paths
    _log_paths["main_log"] = os.path.join(log_dir, f"{os.path.basename(run_id)}.log")
    _log_paths["verbose_log"] = os.path.join(
        log_dir, f"{os.path.basename(run_id)}_verbose.log"
    )
    _log_paths["log_dir"] = log_dir

    # Configure main logger
    main_logger.setLevel(logging.INFO)
    verbose_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for logger in [main_logger, verbose_logger]:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(message)s")

    # Main log file handler (filtered)
    main_file_handler = logging.FileHandler(_log_paths["main_log"])
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(detailed_formatter)
    main_file_handler.addFilter(VerboseFilter())

    # Verbose log file handler (unfiltered)
    verbose_file_handler = logging.FileHandler(_log_paths["verbose_log"])
    verbose_file_handler.setLevel(logging.DEBUG)
    verbose_file_handler.setFormatter(detailed_formatter)

    # Console handler (only for pretty formatting, no regular logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors
    console_handler.setFormatter(simple_formatter)
    console_handler.addFilter(VerboseFilter())

    # Add handlers
    main_logger.addHandler(main_file_handler)
    main_logger.addHandler(console_handler)

    verbose_logger.addHandler(verbose_file_handler)

    # NEW: Add Azure Monitor handler for VM runs
    # IMPORTANT: Only enable Azure logging if actually running INSIDE an Azure VM
    # The orchestrator machine (local laptop) should NOT use Azure logging
    if use_vm:
        try:
            from .azure_logging import AzureMonitorHandler, is_running_in_azure_vm

            # Check if we're actually in an Azure VM
            if not is_running_in_azure_vm():
                # Running on orchestrator machine - skip Azure logging
                main_logger.info(
                    "Azure logging skipped: Not running in Azure VM (orchestrator machine)"
                )
            else:
                # We're inside an Azure VM - enable Azure logging
                # Get Azure Monitor configuration from environment
                dce_endpoint = os.getenv("AZURE_MONITOR_DATA_COLLECTION_ENDPOINT")
                dcr_id = os.getenv("AZURE_MONITOR_DATA_COLLECTION_RULE_ID")
                stream_name = os.getenv(
                    "AZURE_MONITOR_STREAM_NAME", "Custom-BenchmarkRuns_CL"
                )

                if not dce_endpoint or not dcr_id:
                    raise ValueError(
                        "Azure Monitor logging is enabled (inside Azure VM) but required environment "
                        "variables are missing. Please set:\n"
                        "  - AZURE_MONITOR_DATA_COLLECTION_ENDPOINT\n"
                        "  - AZURE_MONITOR_DATA_COLLECTION_RULE_ID"
                    )

                # Create and add Azure Monitor handler
                azure_handler = AzureMonitorHandler(
                    dce_endpoint=dce_endpoint,
                    dcr_id=dcr_id,
                    stream_name=stream_name,
                )
                azure_handler.setLevel(logging.INFO)  # Send INFO+ to Azure
                main_logger.addHandler(azure_handler)

                main_logger.info("Azure Monitor logging enabled (running in Azure VM)")

        except ImportError as e:
            # Fail loudly with clear error message
            error_msg = (
                "\n" + "=" * 70 + "\n"
                "ERROR: Azure Monitor logging requires azure-monitor-ingestion package\n"
                "Install with: pip install 'hal-harness[azure]'\n"
                "=" * 70
            )
            print(error_msg, file=sys.stderr)
            raise RuntimeError(
                "Azure Monitor logging requires azure-monitor-ingestion package. "
                f"Install with: pip install 'hal-harness[azure]'. Details: {e}"
            )
        except Exception as e:
            # Fail loudly with clear error message
            error_msg = (
                "\n" + "=" * 70 + "\n"
                f"ERROR: Failed to initialize Azure Monitor logging: {e}\n"
                "Check your Azure credentials and permissions.\n"
                "=" * 70
            )
            print(error_msg, file=sys.stderr)
            raise RuntimeError(f"Failed to initialize Azure Monitor logging: {e}")

    # Initial setup logging
    main_logger.info(f"Logging initialized - {datetime.now().isoformat()}")
    main_logger.info(f"Log directory: {log_dir}")


@contextmanager
def terminal_print():
    """Context manager for terminal printing (now a no-op since interceptor removed)"""
    yield


def print_step(message: str, level: int = logging.INFO) -> None:
    """Log a step with both console formatting and file logging"""
    with terminal_print():
        console.print(f"[bold cyan]→[/] {message}")
    main_logger.log(level, f"STEP: {message}")


def print_success(message: str) -> None:
    """Log a success message"""
    with terminal_print():
        console.print(f"[bold green]✓[/] {message}")
    main_logger.info(f"SUCCESS: {message}")


def print_error(message: str, verbose_log_path: Optional[str] = None):
    """Print error message in red with error symbol"""

    if not verbose_log_path:
        # Log the main error message to both terminal and log file
        main_logger.error(f"❌ ERROR: {message}[/]")

    # Only print the verbose log path to the terminal
    if verbose_log_path:
        with terminal_print():
            console.print(
                f"[yellow]📝 For detailed error information, check: {verbose_log_path}[/]"
            )




def print_header(title: str) -> None:
    """Print and log a formatted header"""
    with terminal_print():
        console.print(Panel(f"[bold blue]{title}[/]", expand=False))
    main_logger.info(f"=== {title} ===")


def create_progress() -> Progress:
    """Create a rich progress bar that prints to terminal"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=Console(file=sys.__stdout__),  # Use system stdout directly
    )


def _print_results_table(results: dict[str, Any]) -> None:
    """Helper function to print results table to console"""
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

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
                table.add_row(key, formatted_value)
            elif (
                isinstance(value, str)
                and key not in ["status", "message", "traceback"]
                and value
            ):
                table.add_row(key, value)
            elif key == "successful_tasks" and value:
                table.add_row("successful_tasks", str(len(value)))
            elif key == "failed_tasks" and value:
                table.add_row("failed_tasks", str(len(value)))
            elif key == "latencies" and value:
                # compute average total_time across all tasks
                total_time = 0
                for task_id, latency in value.items():
                    total_time += latency["total_time"]
                table.add_row("average_total_time", str(total_time / len(value)))

    console.print(table)


def print_results_table(results: dict[str, Any]) -> None:
    """Log results to both console and file"""

    # Print formatted table to console
    with terminal_print():
        _print_results_table(results)

    # Log results to file
    log_data = {"timestamp": datetime.now().isoformat(), "results": {}}

    # Handle both direct results dict and nested results structure
    metrics_dict = (
        results.get("results", results) if isinstance(results, dict) else results
    )

    if isinstance(metrics_dict, dict):
        for key, value in metrics_dict.items():
            if (
                isinstance(value, (int, float))
                or (
                    isinstance(value, str)
                    and key not in ["status", "message", "traceback"]
                )
                and value
            ):
                log_data["results"][key] = value
            elif key == "successful_tasks" and value:
                log_data["results"]["successful_tasks"] = len(value)
            elif key == "failed_tasks" and value:
                log_data["results"]["failed_tasks"] = len(value)
            elif key == "latencies" and value:
                # compute average total_time across all tasks
                total_time = 0
                for task_id, latency in value.items():
                    total_time += latency["total_time"]
                log_data["results"]["average_total_time"] = total_time / len(value)

    # Also log to main log file
    main_logger.info(f"Results: {json.dumps(log_data['results'], indent=2)}")


def print_run_summary(run_id: str, log_dir: str) -> None:
    """Log run summary information"""
    summary = Table(title="Run Summary", show_header=False, box=None)
    summary.add_column("Key", style="cyan")
    summary.add_column("Value", style="white")

    summary.add_row("Run ID", run_id)
    summary.add_row("Log Directory", log_dir)

    console.print(summary)

    # Log to file
    main_logger.info("Run Summary:")
    main_logger.info(f"  Run ID: {run_id}")
    main_logger.info(f"  Log Directory: {log_dir}")


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
    vm: bool,
    continue_run: bool,
    docker: bool = False,
    ignore_errors: bool = False,
) -> None:
    """Print a formatted table with the run configuration"""
    table = Table(title="Run Configuration", show_header=False, box=ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    # Add core parameters
    table.add_row("Run ID", run_id)
    table.add_row("Benchmark", benchmark)
    table.add_row("Agent Name", agent_name)
    table.add_row("Agent Function", agent_function)
    table.add_row("Agent Directory", agent_dir)
    table.add_row("Log Directory", log_dir)
    table.add_row("Max Concurrent", str(max_concurrent))
    table.add_row("Upload Results", "✓" if upload else "✗")
    table.add_row("VM Execution", "✓" if vm else "✗")
    table.add_row("Docker Execution", "✓" if docker else "✗")
    table.add_row("Continue Previous Run", "✓" if continue_run else "✗")
    table.add_row("Ignore Errors", "✓" if ignore_errors else "✗")

    if conda_env_name:
        table.add_row("Conda Environment", conda_env_name)

    # Add agent arguments if present
    if agent_args:
        table.add_section()
        table.add_row("Agent Arguments", "")
        for key, value in agent_args.items():
            table.add_row(f"  {key}", str(value))

    # Add benchmark arguments if present
    if benchmark_args:
        table.add_section()
        table.add_row("Benchmark Arguments", "")
        for key, value in benchmark_args.items():
            table.add_row(f"  {key}", str(value))

    # Add inspect eval arguments if present
    if inspect_eval_args:
        table.add_section()
        table.add_row("Inspect Eval Arguments", "")
        for key, value in inspect_eval_args.items():
            table.add_row(f"  {key}", str(value))

    # Use terminal_print context manager to ensure output goes to terminal
    with terminal_print():
        console.print(table)

    # Also log the configuration to file
    main_logger.info("Run Configuration:")
    main_logger.info(f"  Run ID: {run_id}")
    main_logger.info(f"  Benchmark: {benchmark}")
    main_logger.info(f"  Agent Name: {agent_name}")
    main_logger.info(f"  Agent Function: {agent_function}")
    main_logger.info(f"  Upload Results: {upload}")
    main_logger.info(f"  Log Directory: {log_dir}")
    main_logger.info(f"  VM Execution: {vm}")
    main_logger.info(f"  Docker Execution: {docker}")
    main_logger.info(f"  Continue Previous Run: {continue_run}")
    if agent_args:
        main_logger.info("  Agent Arguments:")
        for key, value in agent_args.items():
            main_logger.info(f"    {key}: {value}")
    if benchmark_args:
        main_logger.info("  Benchmark Arguments:")
        for key, value in benchmark_args.items():
            main_logger.info(f"    {key}: {value}")
    if inspect_eval_args:
        main_logger.info("  Inspect Eval Arguments:")
        for key, value in inspect_eval_args.items():
            main_logger.info(f"    {key}: {value}")


def get_context_logger(
    logger_name: str, context: Dict[str, Any]
) -> logging.LoggerAdapter:
    """Create a context-aware logger with automatic property injection.

    This function creates a BenchmarkLoggerAdapter that automatically injects
    context properties (RunID, Benchmark, AgentName, etc.) into all log messages.

    Args:
        logger_name: Name of the base logger (e.g., "agent_eval")
        context: Context dictionary with properties (RunID, Benchmark, etc.)

    Returns:
        BenchmarkLoggerAdapter with context injected

    Raises:
        ValueError: If required fields are missing from context
        ImportError: If azure_logging module is not available

    Example:
        context = {
            "RunID": "abc-123",
            "Benchmark": "swebench_verified",
            "AgentName": "My Agent (gpt-4o)",
            "ExecutionMode": "vm"
        }
        logger = get_context_logger("agent_eval", context)
        logger.info("Starting task", extra={"TaskID": "task-1"})
    """
    from .azure_logging import get_context_logger as azure_get_context_logger

    return azure_get_context_logger(logger_name, context)

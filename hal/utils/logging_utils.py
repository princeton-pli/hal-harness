from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from typing import Optional, Any
import logging
import sys
import os
import json
from datetime import datetime
import builtins
import io
from contextlib import contextmanager
from rich.box import ROUNDED

# Initialize rich console for terminal output
console = Console()

# Create loggers
main_logger = logging.getLogger('agent_eval')
verbose_logger = logging.getLogger('agent_eval.verbose')

# Store global paths
_log_paths = {
    'main_log': None,
    'verbose_log': None,
    'log_dir': None
}

class VerboseFilter(logging.Filter):
    """Filter to separate verbose logs from main logs"""
    def filter(self, record):
        return not record.name.startswith('agent_eval.verbose')

class OutputRedirector(io.StringIO):
    """Redirects stdout/stderr to our logging system"""
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        
    def write(self, text):
        if text.strip():  # Only log non-empty strings
            self.logger.debug(text.rstrip())
            
    def flush(self):
        pass

class PrintInterceptor:
    """Intercepts all print statements and redirects them to logging"""
    def __init__(self):
        self._original_print = builtins.print
        
    def custom_print(self, *args, **kwargs):
        """Custom print function that logs instead of printing to terminal"""
        # If file is specified, use original print (for example, when rich uses print)
        if 'file' in kwargs:
            self._original_print(*args, **kwargs)
            return
            
        # Convert args to string and log
        message = ' '.join(str(arg) for arg in args)
        verbose_logger.debug(message)
        
    def start(self):
        """Start intercepting print statements"""
        builtins.print = self.custom_print
        
    def stop(self):
        """Restore original print function"""
        builtins.print = self._original_print

# Global print interceptor
print_interceptor = PrintInterceptor()

def setup_logging(log_dir: str, run_id: str) -> None:
    """Setup logging configuration"""
    # Create absolute path for log directory to avoid path duplication
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log files with clean paths
    _log_paths['main_log'] = os.path.join(log_dir, f"{os.path.basename(run_id)}.log")
    _log_paths['verbose_log'] = os.path.join(log_dir, f"{os.path.basename(run_id)}_verbose.log")
    _log_paths['log_dir'] = log_dir
    
    # Configure main logger
    main_logger.setLevel(logging.INFO)
    verbose_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for logger in [main_logger, verbose_logger]:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Main log file handler (filtered)
    main_file_handler = logging.FileHandler(_log_paths['main_log'])
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(detailed_formatter)
    main_file_handler.addFilter(VerboseFilter())
    
    # Verbose log file handler (unfiltered)
    verbose_file_handler = logging.FileHandler(_log_paths['verbose_log'])
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
    
    # Start intercepting print statements
    print_interceptor.start()
    
    # Redirect stdout and stderr to verbose logger
    sys.stdout = OutputRedirector(verbose_logger)
    sys.stderr = OutputRedirector(verbose_logger)
    
    # Initial setup logging
    main_logger.info(f"Logging initialized - {datetime.now().isoformat()}")
    main_logger.info(f"Log directory: {log_dir}")

@contextmanager
def terminal_print():
    """Context manager to temporarily restore terminal printing"""
    print_interceptor.stop()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print_interceptor.start()

def log_verbose(message: str, level: int = logging.INFO) -> None:
    """Log verbose information to verbose log file only"""
    verbose_logger.log(level, message)

def log_step(message: str, level: int = logging.INFO) -> None:
    """Log a step with both console formatting and file logging"""
    with terminal_print():
        console.print(f"[bold cyan]→[/] {message}")
    main_logger.log(level, f"STEP: {message}")

def log_success(message: str) -> None:
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
            console.print(f"[yellow]📝 For detailed error information, check: {verbose_log_path}[/]")

def log_error(message: str):
    """Log error message to file only"""
    main_logger.error(f"ERROR: {message}")

def log_warning(message: str) -> None:
    """Log a warning message"""
    with terminal_print():
        console.print(f"[bold yellow]![/] {message}")
    main_logger.warning(f"WARNING: {message}")

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
        console=Console(file=sys.__stdout__)  # Use system stdout directly
    )

def _print_results_table(results: dict[str, Any]) -> None:
    """Helper function to print results table to console"""
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Handle both direct results dict and nested results structure
    metrics_dict = results.get("results", results) if isinstance(results, dict) else results
    
    if isinstance(metrics_dict, dict):
        for key, value in metrics_dict.items():
            # Skip nested dictionaries and certain keys we don't want to display
            if isinstance(value, (int, float)) and value:
                formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
                table.add_row(key, formatted_value)
            elif isinstance(value, str) and key not in ["status", "message", "traceback"] and value:
                table.add_row(key, value)
            elif key == "successful_tasks" and value:
                table.add_row("successful_tasks", str(len(value)))
            elif key == "failed_tasks" and value:
                table.add_row("failed_tasks", str(len(value)))
            elif key == "latencies" and value:
                # compute average total_time across all tasks
                total_time = 0
                for task_id, latency in value.items():
                    total_time += latency['total_time']
                table.add_row("average_total_time", str(total_time / len(value)))
    
    console.print(table)

def log_results_table(results: dict[str, Any]) -> None:
    """Log results to both console and file"""
        
    # Print formatted table to console
    with terminal_print():
        _print_results_table(results)
    
    # Log results to file
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "results": {}
    }
    
    # Handle both direct results dict and nested results structure
    metrics_dict = results.get("results", results) if isinstance(results, dict) else results
    
    if isinstance(metrics_dict, dict):
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)) or (isinstance(value, str) and key not in ["status", "message", "traceback"]) and value:
                log_data["results"][key] = value
            elif key == "successful_tasks" and value:
                log_data["results"]["successful_tasks"] = len(value)
            elif key == "failed_tasks" and value:
                log_data["results"]["failed_tasks"] = len(value)
            elif key == "latencies" and value:
                # compute average total_time across all tasks
                total_time = 0
                for task_id, latency in value.items():
                    total_time += latency['total_time']
                log_data["results"]["average_total_time"] = total_time / len(value)
    
    # Also log to main log file
    main_logger.info(f"Results: {json.dumps(log_data['results'], indent=2)}")

def log_run_summary(run_id: str, log_dir: str) -> None:
    """Log run summary information"""
    summary = Table(title="Run Summary", show_header=False, box=None)
    summary.add_column("Key", style="cyan")
    summary.add_column("Value", style="white")
    
    summary.add_row("Run ID", run_id)
    summary.add_row("Log Directory", log_dir)
    
    console.print(summary)
    
    # Log to file
    main_logger.info(f"Run Summary:")
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
    tasks_per_vm: Optional[int] = None
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
    if vm and tasks_per_vm:
        table.add_row("Tasks Per VM", str(tasks_per_vm))
    
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
    if vm and tasks_per_vm:
        main_logger.info(f"  Tasks Per VM: {tasks_per_vm}")
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

# Rename the old print_* functions to use log_* instead
print_step = log_step
print_success = log_success
print_warning = log_warning
print_results_table = log_results_table
print_run_summary = log_run_summary 

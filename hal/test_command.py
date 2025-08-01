"""Implementation of hal-test command for quick validation"""

# Standard library imports
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Third-party imports
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.table import Table

# Load environment variables from .env file
load_dotenv()

# Local imports
from .agent_runner import AgentRunner
from .benchmark_manager import BenchmarkManager
from .utils.logging_utils import (
    print_header, print_step, print_success, print_error, 
    print_warning, print_results_table
)




def run_benchmark_test(verbose: bool = False) -> Dict[str, Any]:
    """Run a test on corebench_easy benchmark"""
    
    benchmark_name = "corebench_easy"
    result = {
        "benchmark": benchmark_name,
        "status": "unknown",
        "time_taken": 0,
        "errors": [],
        "warnings": [],
        "metrics": {}
    }
    
    start_time = time.time()
    
    try:
        # Set up test configuration
        agent_name = "HAL Test Agent"
        agent_dir = os.path.join(os.path.dirname(__file__), "..", "agents", "hal_test_agent")
        agent_function = "main.run"
        
        run_id = f"test_corebench_easy_{uuid.uuid4().hex[:8]}"
        
        # Check if we have a hal conda environment
        conda_env = None
        try:
            # First check if conda command is available (platform-agnostic)
            if shutil.which("conda"):
                conda_result = subprocess.run(
                    ["conda", "env", "list"], 
                    capture_output=True, 
                    text=True
                )
                if conda_result.returncode == 0 and "hal" in conda_result.stdout:
                    conda_env = "hal"
                    if verbose:
                        print_step("Using 'hal' conda environment")
                elif verbose:
                    print_step("No 'hal' conda environment found, using system Python")
            elif verbose:
                print_step("Conda not available, using system Python")
        except (subprocess.SubprocessError, OSError) as e:
            if verbose:
                print_warning(f"Error checking conda: {e}")
        
        # Create runner
        runner = AgentRunner(
            agent_function=agent_function,
            agent_dir=agent_dir,
            agent_args={},
            benchmark_name=benchmark_name,
            config={},
            run_id=run_id,
            use_vm=False,
            use_docker=False,
            max_concurrent=1,
            conda_env=conda_env,
            continue_run=False,
            run_command="hal-test",
            ignore_errors=True,
            max_tasks=1  # Always just 1 task for testing
        )
        
        # Run evaluation
        if verbose:
            print_step(f"Running {benchmark_name} with 1 task...")
        
        # Import asyncio to run the async method
        import asyncio
        asyncio.run(runner.run(
            agent_name=agent_name,
            upload=False
        ))
        
        # Check results
        results_dir = os.path.join("results", benchmark_name, run_id)
        if os.path.exists(results_dir):
            # Store results directory path for cleanup
            result["results_dir"] = results_dir
            # Look for result files
            result_files = list(Path(results_dir).glob("*_UPLOAD.json"))
            if result_files:
                result["status"] = "passed"
                
                # Try to extract metrics
                import json
                try:
                    with open(result_files[0], 'r') as f:
                        upload_data = json.load(f)
                        if "results" in upload_data:
                            result["metrics"] = upload_data["results"]
                except Exception as e:
                    result["status"] = "failed"
                    result["errors"].append(f"Failed to read results file: {e}")
            else:
                result["status"] = "warning"
                result["warnings"].append("No result files generated")
        else:
            result["status"] = "failed"
            result["errors"].append("No results directory created")
            
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(str(e))
        if verbose:
            result["traceback"] = traceback.format_exc()
    
    result["time_taken"] = time.time() - start_time
    return result


def run_hal_test(verbose: bool = False, cleanup: bool = True) -> bool:
    """Run hal-test command with corebench_easy"""
    
    print_header("HAL Test Runner")
    
    # Check if this is first run (no capsules downloaded)
    capsules_path = os.path.join(os.path.dirname(__file__), "benchmarks", "corebench", "capsules")
    if not os.path.exists(capsules_path) or not os.listdir(capsules_path):
        print_step("First run detected - capsules will be downloaded on demand")
    
    # Setup CORE-bench first
    if not setup_corebench():
        print_error("CORE-bench setup failed. Please fix the issues above.")
        return False
    
    # Validate setup first
    if not validate_setup():
        print_error("Setup validation failed. Please fix the issues above.")
        return False
    
    # Track run_id for cleanup
    run_id = None
    
    try:
        # Run the test
        print(f"\n{'='*60}")
        print_step("Testing corebench_easy...")
        
        result = run_benchmark_test(verbose)
        
        # Extract run_id from results_dir if available
        if result.get("results_dir"):
            run_id = os.path.basename(result["results_dir"])
        
        # Print results
        console = Console()
        
        # Create results table
        table = Table(title="Test Results", show_header=True, header_style="bold magenta")
        table.add_column("Benchmark", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Time (s)", justify="right")
        table.add_column("Notes", style="yellow")
        
        # Determine status
        if result["status"] == "passed":
            status_icon = "✅"
            print_success("Test passed")
        elif result["status"] == "warning":
            status_icon = "⚠️"
            print_warning("Test completed with warnings")
        else:
            status_icon = "❌"
            print_error("Test failed")
        
        # Prepare notes
        notes = []
        if result["errors"]:
            notes.append(f"Error: {result['errors'][0][:50]}...")
        elif result["warnings"]:
            notes.append(f"Warning: {result['warnings'][0]}")
        elif result.get("metrics"):
            # Show a key metric if available
            if "accuracy" in result["metrics"]:
                notes.append(f"Accuracy: {result['metrics']['accuracy']:.2%}")
            elif "success_rate" in result["metrics"]:
                notes.append(f"Success: {result['metrics']['success_rate']:.2%}")
            elif "total_cost" in result["metrics"]:
                notes.append(f"Cost: ${result['metrics']['total_cost']:.4f}")
        
        # Add row to table
        table.add_row(
            "corebench_easy",
            status_icon,
            f"{result['time_taken']:.1f}",
            ", ".join(notes) if notes else "OK"
        )
        
        # Print table
        print(f"\n{'='*60}")
        console.print(table)
        
        # Print detailed errors if any
        if result["errors"] and verbose:
            print("\n[bold red]Detailed Errors:[/bold red]")
            for error in result["errors"]:
                print(f"  - {error}")
            if result.get("traceback"):
                print(f"\n[dim]{result['traceback']}[/dim]")
        
        # Return success
        return result["status"] != "failed"
        
    finally:
        # Always cleanup if requested
        if cleanup:
            print(f"\n{'='*60}")
            cleanup_test_artifacts(run_id)


def setup_corebench() -> bool:
    """Ensure CORE-bench is properly set up"""
    print_step("Setting up CORE-bench...")
    
    # Check if core_test.json exists
    core_test_path = os.path.join(os.path.dirname(__file__), "benchmarks", "corebench", "core_test.json")
    encrypted_path = os.path.join(os.path.dirname(__file__), "benchmarks", "corebench", "core_test.json.gpg")
    
    if not os.path.exists(core_test_path):
        if os.path.exists(encrypted_path):
            print_step("Decrypting CORE-bench test file...")
            
            # Check if GPG is available
            if not shutil.which("gpg"):
                print_error("GPG not found. Please install GPG first:")
                print_error("  macOS: brew install gnupg")
                print_error("  Ubuntu/Debian: sudo apt-get install gnupg")
                print_error("  CentOS/RHEL: sudo yum install gnupg")
                return False
                
            try:
                # Try to decrypt using gpg
                result = subprocess.run(
                    ["gpg", "--batch", "--yes", "--passphrase", "reproducibility", 
                     "--output", core_test_path, "--decrypt", encrypted_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print_error(f"Failed to decrypt: {result.stderr}")
                    return False
                print_success("Successfully decrypted CORE-bench test file")
            except subprocess.SubprocessError as e:
                print_error(f"GPG decryption failed: {e}")
                return False
        else:
            print_error(f"CORE-bench test file not found at {encrypted_path}")
            return False
    
    # Check if dependencies are installed
    try:
        import scipy
        import numpy
        import weave
    except ImportError:
        print_step("Installing CORE-bench and agent dependencies...")
        try:
            # Install CORE-bench dependencies
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[corebench]"], 
                         cwd=os.path.dirname(os.path.dirname(__file__)), 
                         check=True)
            
            # Test agent only needs weave, which is already installed with HAL
            
            print_success("Successfully installed dependencies")
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dependencies: {e}")
            return False
    
    return True




def validate_setup() -> bool:
    """Validate that HAL is properly set up"""
    
    print_step("Validating HAL setup...")
    
    checks = {
        "Test agent": False,
        "Results directory": False
    }
    
    # Check test agent exists
    agent_path = os.path.join(os.path.dirname(__file__), "..", "agents", "hal_test_agent", "main.py")
    if os.path.exists(agent_path):
        checks["Test agent"] = True
    else:
        print_error(f"Test agent not found at {agent_path}")
    
    # Check results directory is writable
    try:
        test_dir = os.path.join("results", "test_validation")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        os.rmdir(test_dir)
        checks["Results directory"] = True
    except Exception as e:
        print_error(f"Cannot write to results directory: {e}")
    
    # Print validation results
    all_passed = all(checks.values())
    for check, passed in checks.items():
        if passed:
            print(f"  ✅ {check}")
        else:
            print(f"  ❌ {check}")
    
    return all_passed


def cleanup_test_artifacts(run_id: str = None) -> None:
    """Clean up all test artifacts created during hal-test
    
    Args:
        run_id: The test run ID to clean up. If None, only cleans empty directories.
    """
    print_step("Cleaning up test artifacts...")
    
    cleanup_paths = []
    
    # Clean up results directory for this run
    if run_id:
        results_path = os.path.join("results", "corebench_easy", run_id)
        if os.path.exists(results_path):
            cleanup_paths.append(results_path)
    
    
    # Clean up empty capsules directory if it exists and is empty
    capsules_path = os.path.join(os.path.dirname(__file__), "benchmarks", "corebench", "capsules")
    if os.path.exists(capsules_path) and not os.listdir(capsules_path):
        cleanup_paths.append(capsules_path)
    
    # Clean up Weave logs for this run
    weave_project_path = os.path.join("weave", run_id) if run_id else None
    if weave_project_path and os.path.exists(weave_project_path):
        cleanup_paths.append(weave_project_path)
    
    # Clean up any .weave directories
    weave_dirs = [".weave", "weave"]
    for weave_dir in weave_dirs:
        if os.path.exists(weave_dir) and run_id:
            # Look for directories containing the run_id
            for root, dirs, files in os.walk(weave_dir):
                for d in dirs:
                    if run_id in d:
                        cleanup_paths.append(os.path.join(root, d))
                for f in files:
                    if run_id in f:
                        cleanup_paths.append(os.path.join(root, f))
    
    # Remove empty results directories if they exist
    results_root = "results"
    if os.path.exists(results_root):
        for benchmark in os.listdir(results_root):
            benchmark_path = os.path.join(results_root, benchmark)
            if os.path.isdir(benchmark_path) and not os.listdir(benchmark_path):
                cleanup_paths.append(benchmark_path)
    
    # Validate paths before cleanup (ensure they're within expected directories)
    safe_cleanup_paths = []
    expected_dirs = ["results", "weave", ".weave", 
                     os.path.join(os.path.dirname(__file__), "benchmarks", "corebench", "capsules")]
    
    for path in cleanup_paths:
        # Convert to absolute path for comparison
        abs_path = os.path.abspath(path)
        # Check if path is within one of our expected directories
        if any(abs_path.startswith(os.path.abspath(expected)) for expected in expected_dirs):
            safe_cleanup_paths.append(path)
        else:
            print_warning(f"  ! Skipping suspicious path: {path}")
    
    # Perform cleanup
    cleaned_count = 0
    for path in safe_cleanup_paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  ✓ Removed directory: {path}")
                cleaned_count += 1
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  ✓ Removed file: {path}")
                cleaned_count += 1
        except Exception as e:
            print_warning(f"  ! Could not remove {path}: {e}")
    
    if cleaned_count == 0:
        print("  ℹ No artifacts to clean up")
    else:
        print_success(f"Cleanup completed - removed {cleaned_count} items")


# If run directly for testing
if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    success = run_hal_test(verbose=verbose)
    sys.exit(0 if success else 1)
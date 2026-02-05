"""Logging utilities for v2."""

import logging
import os
import sys
from datetime import datetime


logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, run_id: str, use_azure: bool = False) -> None:
    """Setup logging configuration with optional Azure Monitor integration.

    Args:
        log_dir: Directory for log files
        run_id: Unique run identifier
        use_azure: If True, enables Azure Monitor logging (for VM runs)
    """
    # Create absolute path for log directory
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(log_dir, f"{os.path.basename(run_id)}.log")

    # Configure logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Suppress verbose logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("paramiko.transport").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Add Azure Monitor handler if requested
    if use_azure:
        try:
            from azure_logging import AzureMonitorHandler, is_running_in_azure_vm

            # Check if we're actually in an Azure VM
            if not is_running_in_azure_vm():
                logger.info(
                    "Azure logging skipped: Not running in Azure VM (orchestrator machine)"
                )
            else:
                # Get Azure Monitor configuration from environment
                dce_endpoint = os.getenv("AZURE_MONITOR_DATA_COLLECTION_ENDPOINT")
                dcr_id = os.getenv("AZURE_MONITOR_DATA_COLLECTION_RULE_ID")
                stream_name = os.getenv(
                    "AZURE_MONITOR_STREAM_NAME", "Custom-BenchmarkRuns_CL"
                )

                if not dce_endpoint or not dcr_id:
                    raise ValueError(
                        "Azure Monitor logging is enabled but required environment "
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
                azure_handler.setLevel(logging.INFO)
                root_logger.addHandler(azure_handler)

                logger.info("Azure Monitor logging enabled (running in Azure VM)")

        except ImportError as e:
            error_msg = (
                "\n" + "=" * 70 + "\n"
                "ERROR: Azure Monitor logging requires azure-monitor-ingestion package\n"
                "Install with: pip install azure-monitor-ingestion\n"
                "=" * 70
            )
            print(error_msg, file=sys.stderr)
            raise RuntimeError(
                "Azure Monitor logging requires azure-monitor-ingestion package. "
                f"Install with: pip install azure-monitor-ingestion. Details: {e}"
            )
        except Exception as e:
            error_msg = (
                "\n" + "=" * 70 + "\n"
                f"ERROR: Failed to initialize Azure Monitor logging: {e}\n"
                "Check your Azure credentials and permissions.\n"
                "=" * 70
            )
            print(error_msg, file=sys.stderr)
            raise RuntimeError(f"Failed to initialize Azure Monitor logging: {e}")

    # Initial setup logging
    logger.info(f"Logging initialized - {datetime.now().isoformat()}")
    logger.info(f"Log directory: {log_dir}")

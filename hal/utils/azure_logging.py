"""Azure Monitor logging integration for HAL harness.

This module provides structured logging to Azure Monitor using Data Collection Rules (DCR).
All logs are sent as JSON with mandatory context properties for queryability.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from azure.monitor.ingestion import LogsIngestionClient
    from azure.core.exceptions import HttpResponseError
    from azure.identity import DefaultAzureCredential
    import requests  # Part of azure extras

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    requests = None  # type: ignore


def is_running_in_azure_vm() -> bool:
    """Detect if code is running inside an Azure VM using Instance Metadata Service.

    The Azure Instance Metadata Service (IMDS) is only available from within Azure VMs.
    This allows us to distinguish between:
    - Orchestrator machine (local laptop) -> should NOT use Azure logging
    - Azure VM (actual execution environment) -> should use Azure logging

    Returns:
        True if running in Azure VM, False otherwise
    """
    if requests is None:
        # requests not available (Azure extras not installed)
        return False

    try:
        # Azure Instance Metadata Service endpoint (only accessible from Azure VMs)
        metadata_url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
        response = requests.get(
            metadata_url,
            headers={"Metadata": "true"},
            timeout=2  # Fast timeout - if not in Azure, fail quickly
        )
        return response.status_code == 200
    except Exception:
        # Any error means we're not in an Azure VM
        return False


class CloudJSONFormatter(logging.Formatter):
    """Formats log records as JSON with properties block for Azure Monitor.

    This formatter ensures all logs are structured JSON with:
    - Standard fields: TimeGenerated, Level, Message, LoggerName
    - Properties block: All context from LoggerAdapter (RunID, Benchmark, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON string representation of the log record
        """
        # Base log object with standard fields
        log_data = {
            "TimeGenerated": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "Level": record.levelname,
            "Message": record.getMessage(),
            "LoggerName": record.name,
        }

        # Extract context properties from the adapter
        # These come from BenchmarkLoggerAdapter.extra or the extra parameter in log calls
        if hasattr(record, "benchmark_context"):
            # Properties from BenchmarkLoggerAdapter
            log_data.update(record.benchmark_context)

        # Additional properties from extra parameter in log calls
        # Filter out internal logging attributes
        extra_properties = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "benchmark_context",
            ]:
                extra_properties[key] = value

        if extra_properties:
            # Merge extra properties into main log data
            log_data.update(extra_properties)

        return json.dumps(log_data)


class BenchmarkLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that injects benchmark context into all log calls.

    This adapter automatically adds context properties (RunID, Benchmark, etc.)
    to every log message, ensuring consistent structured logging without manual
    context passing.

    Example:
        context = {
            "RunID": "abc-123",
            "Benchmark": "swebench_verified",
            "AgentName": "My Agent (gpt-4o)",
            "ExecutionMode": "vm"
        }
        logger = BenchmarkLoggerAdapter(logging.getLogger("agent_eval"), context)
        logger.info("Starting task", extra={"TaskID": "task-1"})  # Context auto-injected
    """

    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """Initialize adapter with context.

        Args:
            logger: Base logger to wrap
            context: Context dictionary to inject (RunID, Benchmark, etc.)
        """
        super().__init__(logger, context)
        self.context = context

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log call to inject context.

        Args:
            msg: Log message
            kwargs: Keyword arguments from log call

        Returns:
            Tuple of (message, updated kwargs with context)
        """
        # Get or create the extra dict
        extra = kwargs.get("extra", {})

        # Create a copy of context and merge with extra
        merged_context = self.context.copy()
        merged_context.update(extra)

        # Store in a special attribute that CloudJSONFormatter will read
        kwargs["extra"] = {"benchmark_context": merged_context}

        return msg, kwargs


class AzureMonitorHandler(logging.Handler):
    """Custom logging handler that sends logs to Azure Monitor via DCR.

    This handler:
    - Batches logs for efficient upload (flushes on ERROR, batch size, or interval)
    - Uses Azure Monitor Ingestion API with Data Collection Rules
    - Fails fast: raises exception if upload fails (ensures complete observability)
    - Thread-safe: uses lock for buffer access

    Batching strategy:
    - Immediate flush on ERROR or CRITICAL level
    - Flush every 100 logs (configurable)
    - Flush every 30 seconds (configurable)
    - Flush on handler close()
    """

    def __init__(
        self,
        dce_endpoint: str,
        dcr_id: str,
        stream_name: str,
        batch_size: int = 100,
        flush_interval: float = 30.0,
    ):
        """Initialize Azure Monitor handler.

        Args:
            dce_endpoint: Data Collection Endpoint URL
            dcr_id: Data Collection Rule ID
            stream_name: Stream name (e.g., "Custom-BenchmarkRuns_CL")
            batch_size: Number of logs to batch before flushing
            flush_interval: Time in seconds between flushes

        Raises:
            RuntimeError: If azure-monitor-ingestion is not installed
            RuntimeError: If Azure authentication fails
        """
        super().__init__()

        if not AZURE_AVAILABLE:
            raise RuntimeError(
                "Azure Monitor logging requires azure-monitor-ingestion package. "
                "Install with: pip install 'hal-harness[azure]'"
            )

        self.dce_endpoint = dce_endpoint
        self.dcr_id = dcr_id
        self.stream_name = stream_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Initialize Azure client
        try:
            credential = DefaultAzureCredential()
            self.client = LogsIngestionClient(
                endpoint=dce_endpoint, credential=credential
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Azure Monitor client: {e}\n"
                "Common causes:\n"
                "  - Not running in Azure VM (no managed identity available)\n"
                "  - Missing Azure CLI login (az login)\n"
                "  - Insufficient permissions on Data Collection Rule\n"
                "This error should not occur if is_running_in_azure_vm() was checked first."
            )

        # Buffer for batching logs
        self.buffer: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        # Last flush time for interval-based flushing
        self.last_flush_time = time.time()

        # Set formatter to CloudJSONFormatter
        self.setFormatter(CloudJSONFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to Azure Monitor.

        Adds the formatted log to the buffer and flushes if needed.

        Args:
            record: Log record to emit
        """
        try:
            # Format the record as JSON
            formatted = self.format(record)
            log_data = json.loads(formatted)

            # Add to buffer (thread-safe)
            with self.lock:
                self.buffer.append(log_data)
                should_flush = (
                    record.levelno >= logging.ERROR  # Immediate flush on errors
                    or len(self.buffer) >= self.batch_size  # Batch size limit
                    or (time.time() - self.last_flush_time)
                    >= self.flush_interval  # Time interval
                )

            # Flush if needed (outside the lock to avoid blocking other threads)
            if should_flush:
                self.flush()

        except Exception as e:
            # Critical: If we can't log, we need to fail the run
            self.handleError(record)
            raise RuntimeError(f"Failed to emit log to Azure Monitor: {e}")

    def flush(self) -> None:
        """Flush buffered logs to Azure Monitor.

        Uploads all buffered logs to Azure and clears the buffer.

        Raises:
            RuntimeError: If upload to Azure fails (fail-fast behavior)
        """
        with self.lock:
            if not self.buffer:
                return

            # Get logs to upload and clear buffer
            logs_to_upload = self.buffer.copy()
            self.buffer.clear()
            self.last_flush_time = time.time()

        # Upload to Azure (outside lock to avoid blocking)
        try:
            self.client.upload(
                rule_id=self.dcr_id, stream_name=self.stream_name, logs=logs_to_upload
            )
        except HttpResponseError as e:
            # Restore logs to buffer on failure
            with self.lock:
                self.buffer.extend(logs_to_upload)

            # Fail fast: propagate error to fail the benchmark run
            raise RuntimeError(
                f"Failed to upload logs to Azure Monitor (HTTP {e.status_code}): {e.message}. "
                "Check DCE endpoint, DCR ID, and permissions."
            )
        except Exception as e:
            # Restore logs to buffer on failure
            with self.lock:
                self.buffer.extend(logs_to_upload)

            # Fail fast: propagate error
            raise RuntimeError(f"Failed to upload logs to Azure Monitor: {e}")

    def close(self) -> None:
        """Close the handler and flush all remaining logs.

        Ensures no logs are lost on shutdown. If flush fails (e.g., 403 permissions),
        logs the error but doesn't raise to prevent hanging on shutdown.
        """
        try:
            self.flush()
        except Exception as e:
            # During shutdown, log error but don't raise to avoid hanging
            # This prevents Ctrl+C from hanging when Azure logging has permission issues
            import sys
            print(
                f"\nWARNING: Failed to flush Azure Monitor logs during shutdown: {e}\n"
                "Some logs may not have been uploaded to Azure.",
                file=sys.stderr
            )
        finally:
            super().close()


def get_context_logger(
    logger_name: str,
    context: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
) -> BenchmarkLoggerAdapter:
    """Create a context-aware logger with mandatory fields.

    Args:
        logger_name: Name of the base logger (e.g., "agent_eval")
        context: Context dictionary with properties (RunID, Benchmark, etc.)
        required_fields: List of required field names (defaults to standard fields)

    Returns:
        BenchmarkLoggerAdapter with context injected

    Raises:
        ValueError: If required fields are missing from context
    """
    # Default required fields from the plan
    if required_fields is None:
        required_fields = ["RunID", "Benchmark", "AgentName", "ExecutedBy"]

    # Validate required fields are present
    missing_fields = [field for field in required_fields if field not in context]
    if missing_fields:
        raise ValueError(
            f"Logging context missing required fields: {missing_fields}. "
            "Logging is a first-class priority and all context must be provided."
        )

    # Get base logger and wrap with adapter
    base_logger = logging.getLogger(logger_name)
    return BenchmarkLoggerAdapter(base_logger, context)

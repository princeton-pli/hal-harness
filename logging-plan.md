# Logging Improvement & Azure Monitor Integration Plan

## Overview

Improve and standardize logging in HAL harness, focusing on VM runs, while maintaining backwards compatibility with existing file outputs. Implement Azure Monitor integration using the "Mega-Table" architecture with a single Log Analytics Workspace.

## Architecture: "Mega-Table" with LoggerAdapter

### Core Concept

- **Single Log Analytics Workspace**: Central repository for all logs
- **Single Source of Truth**: Use `logging.LoggerAdapter` to inject context properties automatically
- **Structured JSON**: All logs emitted as JSON with properties block
- **Dual Output**: Python logging library outputs to BOTH files (existing) AND Azure Monitor (new)

### Properties/Dimensions (Single Source of Truth)

Every log entry will automatically include:

```python
{
    "RunID": "abc-123",           # Unique run identifier (existing run_id)
    "Benchmark": "swebench_verified",
    "AgentName": "My Agent (gpt-4o)",
    "TaskID": "django__django-123",  # Specific task/problem
    "LogType": "VMOperation" | "AgentLog"
    "VMName": "agent-swebench-xyz",  # For VM-specific logs TODO: what Azure property does this correspond to?,
    "ExecutedBy": "Andrew" # this pulls from a required env var that we should set equal to our names
}
```

If a log cannot include these values, it is rejected; logging is a first-class obkective of the system.

### Benefits

1. **Easy Querying**: `HAL-Benchmark-Runs | where Benchmark == "swebench_verified" and LogType == "Error"`
2. **No Manual Context**: Context injected automatically via LoggerAdapter and mandatory
3. **Unified Timeline**: Agent + Scaffold logs interleaved chronologically by RunID
4. **Backwards Compatible**: File handlers still work, just add Azure handler

## Implementation Plan

### Phase 1: Logging Infrastructure Refactoring

#### 1.1 Create New Logging Utilities (hal/utils/azure_logging.py)

**New file** to contain:

- `CloudJSONFormatter`: Custom JSON formatter for structured logs
- `BenchmarkLoggerAdapter`: Context-aware logger adapter
- `AzureMonitorHandler`: Custom handler for Azure Monitor ingestion
- Helper functions to create context-aware loggers
- - If anything in here fails, exit and fail loudly. Logging is a first-class priority (e.g., handle `HttpResponseError` and other cases)

**Key Components:**

```python
class CloudJSONFormatter(logging.Formatter):
    """Formats logs as JSON with properties block"""

class BenchmarkLoggerAdapter(logging.LoggerAdapter):
    """Injects benchmark context (RunID, Benchmark, etc.) into all logs"""

class AzureMonitorHandler(logging.Handler):
    """Sends logs to Azure Log Analytics using DCR-based ingestion"""
    # Uses azure.monitor.ingestion.LogsIngestionClient
    # Batching strategy: Flush on ERROR level or every 100 logs or 30 seconds
    # CRITICAL: If upload fails, raise exception to fail the run
    # On handler close(), flush all remaining buffered logs
```

#### 1.2 Refactor hal/utils/logging_utils.py

**Changes:**

1. Add Azure Monitor handler alongside existing file handlers
2. Keep existing handlers for backwards compatibility
3. Add function `get_context_logger(context: dict) -> BenchmarkLoggerAdapter`
4. **Decision Point**: Keep or remove print interceptor?
   - **Recommendation**: Make it optional via environment variable
   - Rationale: It's problematic but removing it might break existing behavior
   - TODO: revisit this and test it; ideally we can remove this but we'd have to do a larger refactor

**New Setup Flow:**

```python
def setup_logging(run_id, log_dir, use_vm=False):
    # Existing file handlers (unchanged)
    main_handler = RotatingFileHandler(f"{log_dir}/{run_id}.log")
    verbose_handler = RotatingFileHandler(f"{log_dir}/{run_id}_verbose.log")
    console_handler = RichHandler()

    if use_vm:
        azure_handler = AzureMonitorHandler(
        )
        handlers.append(azure_handler)

    # Apply all handlers to loggers
```

#### 1.3 Milestone and testing

1. **Unit Tests**: Test LoggerAdapter context injection
2. **Smoke Test**: Run single VM task with Azure logging enabled
3. **Validation**:
   - Verify file outputs unchanged

- [ ] Test with curl and verify that logs show up in Azure as expected
  - Verify no broken functionality
  - Verify logs appear in Azure Log Analytics

PAUSE DEVELOPMENT AT THIS POINT AND ONLY CONTINUE ONCE THE ABOVE TEST PASSES

### Phase 2: VM Code Refactoring (Replace print() with logging)

#### 2.1 Refactor hal/utils/virtual_machine_manager.py

Note: this should be done by hand using IDE refactor tools, not an AI.

**Changes:** Replace all ~100+ print() statements with proper logging

**Pattern:**

```python
# BEFORE:
print(f"Creating VM: {vm_name}")

# AFTER:
logger.info(f"Creating VM: {vm_name}", extra={"log_type": "VMOperation", "operation": "create_vm"})
```

**Key Areas:**

- `create_vm()` (lines 153-255): VM creation progress
- `create_gpu_vm()` (lines 258-390): GPU VM creation
- `run_agent_on_vm()` (lines 552-724): Agent execution lifecycle
- `copy_files_to_vm()` / `copy_files_from_vm()`: File transfer operations
- `delete_vm()` (lines 392-434): Cleanup operations

**Add at top of file:**

```python
logger = logging.getLogger("agent_eval")
verbose_logger = logging.getLogger("agent_eval.verbose")
```

#### 2.2 Refactor hal/utils/virtual_machine_runner.py

**Changes:** Replace print() with logging, inject context

**Key Pattern:**

```python
# At start of run_agent():
context = {
    "RunID": self.run_id,
    "Benchmark": self.benchmark_name,
    "AgentName": self.agent_name,
    "ExecutionMode": "vm"
}
log = BenchmarkLoggerAdapter(logger, context)

# Then use throughout:
log.info(f"Starting task {task_id}", extra={"TaskID": task_id, "LogType": "ScaffoldLog"})
```

**Areas to refactor:**

- `run_agent()` (lines 57-269): Main orchestration
- `fetch_agent_logs()` (lines 31-55): Log aggregation

### Phase 3: Agent Log Streaming to Azure

#### 3.1 Capture Agent Logs from VMs

Agent logs are already collected via `get_agent_trace()` (virtual_machine_manager.py:727-752).

**Current Flow:**

1. Agent stdout/stderr redirected to `/home/agent/agent_trace.log` on VM
2. `get_agent_trace()` polls this file every 30 seconds
3. Logs saved locally to `{log_dir}/agent_logs/{task_id}_log.log`

**Enhancement:**
When logs are fetched, parse and send to Azure with proper context:
Include a TODO here to revisit/clean up this logic as well

```python
# After fetching agent logs:
for line in agent_log_lines:
    agent_logger.info(line, extra={
        "RunID": run_id,
        "TaskID": task_id,
        "LogType": "AgentLog",
        "Source": "agent_vm"
    })
```

#### 3.2 Log Group Structure (Open Question)

**Two approaches:**

1. **Single log group**: Mix agent + harness logs, distinguish by `LogType` property
2. **Separate log groups**: Different tables for AgentLogs vs ScaffoldLogs

**For the MVP:** just do the single log group (simpler querying)

- Query all: `HAL-Benchmark-Runs | where RunID == "abc-123"`
- Query agent only: `HAL-Benchmark-Runs | where RunID == "abc-123" and LogType == "AgentLog"`
- Query scaffold: `HAL-Benchmark-Runs | where RunID == "abc-123" and LogType == "ScaffoldLog"`

**Track as open item:** Evaluate after MVP if separate tables would be better

### Phase 4: Azure Monitor Configuration

#### 4.1 Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
azure = [
    "azure-mgmt-compute>=29.1.0",
    "azure-mgmt-network>=25.1.0",
    # ... existing azure deps ...
    "azure-monitor-ingestion>=1.0.3",  # NEW - DCR-based ingestion
    "azure-identity>=1.12.0",  # Already exists
]
```

#### 4.2 Environment Variables

Add to `.env.template`:

```bash
# Azure Monitor Logging (Required for VM runs)
# Use Data Collection Rules (DCR) approach
AZURE_MONITOR_DATA_COLLECTION_ENDPOINT=https://<dce-name>.<region>.ingest.monitor.azure.com
AZURE_MONITOR_DATA_COLLECTION_RULE_ID=dcr-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_MONITOR_STREAM_NAME=Custom-Benchmark-Runs

# Required: Set to your name for tracking who ran the benchmark
EXECUTED_BY=YourName
```

#### 4.3 Workspace Setup via Bash Script

**Automated Setup (Bash Script):**
Created `infrastructure/azure/setup-logging.sh` that automatically creates:

**Resources Created:**

1. **Log Analytics Workspace**: Central repository for logs (`hal-logs-workspace`)
2. **Data Collection Endpoint (DCE)**: Ingestion endpoint (`hal-logs-dce`)
3. **Data Collection Rule (DCR)**: Defines schema and routing (`hal-logs-dcr`)
4. **Custom Table**: `BenchmarkRuns_CL` with schema below
5. **Role Assignments**: Grants Monitoring Metrics Publisher role to current user

**Table Schema (BenchmarkRuns_CL):**

```
TimeGenerated: datetime
Level: string
Message: string
LoggerName: string
RunID: string
Benchmark: string
AgentName: string
TaskID: string
LogType: string
VMName: string
ExecutionMode: string
ExecutedBy: string
Properties: dynamic
```

**Running the Setup:**

```bash
# 1. Ensure you're logged into Azure CLI
az login

# 2. Set environment variables (optional, uses defaults from HAL config)
export AZURE_RESOURCE_GROUP_NAME=HAL_GROUP
export AZURE_LOCATION=eastus

# 3. Run the setup script
./infrastructure/azure/setup-logging.sh

# 4. Copy the output values to your .env file
```

**Script Outputs:**
After running, the script outputs:

- `AZURE_MONITOR_DATA_COLLECTION_ENDPOINT`: DCE endpoint URL
- `AZURE_MONITOR_DATA_COLLECTION_RULE_ID`: DCR immutable ID
- `AZURE_MONITOR_STREAM_NAME`: Custom-BenchmarkRuns_CL
- Azure Portal URL for querying logs

**Note:** Script includes idempotency - safe to run multiple times

### Phase 5: Rollout

#### 5.1 Rollout Approach

**Critical Requirement: Fail-Fast on Azure Errors**

- If Azure Monitor is enabled and ingestion fails, **fail the entire benchmark run**
- This ensures complete observability - no partial/lost data
- Log upload errors should raise exceptions that propagate up

**Error Handling Strategy:**

```python
class AzureMonitorHandler(logging.Handler):
    def __init__(self, dce_url, dcr_id, stream_name, batch_size=100, flush_interval=30):
        self.buffer = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        # ... init LogsIngestionClient ...

    def emit(self, record):
        self.buffer.append(self._format_record(record))

        # Flush on ERROR/CRITICAL, batch limit, or time interval
        if record.levelno >= logging.ERROR or len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        try:
            # Upload batch to Azure
            self.client.upload(self.dcr_id, self.stream_name, self.buffer)
            self.buffer.clear()
        except Exception as e:
            # DO NOT suppress errors - fail the run
            raise RuntimeError(f"Failed to upload logs to Azure Monitor: {e}")

    def close(self):
        # Ensure all logs are uploaded before shutdown
        self.flush()
        super().close()
```

**Batching Strategy Rationale:**

- **Efficiency**: Batch uploads reduce API calls and improve performance
- **Responsiveness**: Immediate flush on ERROR ensures critical issues are logged quickly
- **Reliability**: Flush on close() ensures no logs are lost on graceful shutdown
- **Fail-Fast**: Any upload failure immediately fails the run (no retry logic in MVP)

**Automatic Enablement via use_vm Flag:**

- Azure Monitor is **automatically enabled** when `--vm` flag is used
- Azure Monitor is **automatically disabled** for local/docker runs (no `--vm` flag)
- No separate environment variable needed - the existing `use_vm` parameter determines this
- This ensures Azure logging only happens for production VM runs
- Local development/testing automatically skips Azure (since they don't use `--vm`)

**Backwards Compatibility Checklist:**

- [ ] All existing log files still created
- [ ] Console output unchanged (still WARNING+ only)
- [ ] Weave integration still works
- [ ] Azure Monitor failures properly propagate (don't suppress errors)

#### 5.2 Validation Queries

After first successful run, validate with KQL:

```kql
// Check if logs are arriving
HAL-Benchmark-Runs
| where TimeGenerated > ago(1h)
| summarize count() by LogType, Benchmark

// Get all logs for a specific run
HAL-Benchmark-Runs
| where Properties.RunID == "your-run-id"
| order by TimeGenerated asc

// Find VM operation errors
HAL-Benchmark-Runs
| where Level == "ERROR" and Properties.LogType == "VMOperation"
| project TimeGenerated, Message, Properties.TaskID
```

## Implementation Order

### Step 0: Azure Infrastructure Setup (Bash Script)

1. Install Azure CLI and login: `az login`
2. Run setup script: `./infrastructure/azure/setup-logging.sh`
3. Copy outputs to .env file
4. Wait 5-10 minutes for table to be ready for ingestion

### Step 1: Infrastructure (No Breaking Changes)

1. Create `hal/utils/azure_logging.py` with new classes
2. Add Azure dependencies to pyproject.toml
3. Add environment variables to .env.template
4. Write unit tests for LoggerAdapter

### Step 2: Logging Setup Refactor

1. Modify `logging_utils.py` to support Azure handler
2. Add `get_context_logger()` helper function
3. Keep print interceptor as-is for now (optional: make it configurable)

### Step 3: VM Manager Refactoring

1. Replace print() statements in `virtual_machine_manager.py`
2. Add logger initialization at module level
3. Use proper log levels (INFO for operations, DEBUG for details, ERROR for failures)
4. Test with VM execution (logs should go to files as before)

### Step 4: VM Runner Context Injection

1. Modify `virtual_machine_runner.py` to use LoggerAdapter
2. Inject RunID, Benchmark, AgentName context
3. Add TaskID to per-task logs
4. Test with multi-task run

### Step 5: Agent Log Streaming

1. Modify `get_agent_trace()` to send logs to Azure
2. Tag with LogType="AgentLog"
3. Test log visibility in Azure

### Step 6: End-to-End Validation

1. Run full benchmark with Azure logging enabled
2. Verify all logs in Azure Log Analytics
3. Verify file outputs unchanged
4. Document any issues/edge cases

## Open Questions / Future Work

### Tracked Issues:

1. **Log Group Structure**: Should agent logs and scaffold logs be in separate tables?
   - MVP: Single table, revisit based on usage patterns

2. **Real-time vs Batch**: Should logs stream immediately or batch?
   - MVP: Batch (every 30s with agent trace polling)
   - Future: Consider real-time streaming for better monitoring

3. **Print Interceptor**: Keep, remove, or make optional?
   - MVP: Keep as-is to avoid breaking changes
   - Future: Make optional via `INTERCEPT_PRINTS=true` env var

4. **Log Retention**: What retention policy for Azure?
   - Recommendation: 90 days (configurable in workspace)

5. **Cost Management**: How to handle high-volume logging?
   - Use log sampling for verbose agent thoughts?
   - Track ingestion costs per benchmark run?

## Success Criteria (MVP)

- [ ] One log message per task execution appears in Azure Log Analytics
- [ ] Log group created automatically on VM runs
- [ ] All existing file outputs preserved (backwards compatible)
- [ ] VM operations use proper logging instead of print()
- [ ] Logs are queryable by RunID, Benchmark, TaskID, LogType
- [ ] Agent logs from inside VMs sent to Azure
- [ ] Documentation updated with setup instructions

## Files to Modify

### New Files:

1. `hal/utils/azure_logging.py` - New logging infrastructure ✅
2. `infrastructure/azure/setup-logging.sh` - Bash script for Azure setup ✅
3. `test_azure_logging.py` - Test script to verify logging works ✅

### Modified Files:

1. `hal/utils/logging_utils.py` - Add Azure handler setup ✅
2. `hal/utils/virtual_machine_manager.py` - Replace print() with logging (TODO)
3. `hal/utils/virtual_machine_runner.py` - Add context injection (TODO)
4. `pyproject.toml` - Add azure-monitor-ingestion dependency ✅
5. `.env.template` - Add Azure Monitor config vars ✅
6. `hal/cli.py` - Pass use_vm to setup_logging() ✅

### Files to Read Before Implementation:

1. `hal/utils/logging_utils.py` - Understand current setup
2. `hal/utils/virtual_machine_manager.py` - See all print() statements
3. `hal/utils/virtual_machine_runner.py` - Understand orchestration flow
4. `hal/agent_runner.py` - Understand run initialization
5. `hal/cli.py` - Understand CLI entry point

# Notes from implementation:

1. We want to remove the agent_eval.verbose logs for now -- we should have only one logger
2. We _may_ want to add another logger that differentiates agent logs and scaffold logs, but it's likely we'll want these all to be in Azure

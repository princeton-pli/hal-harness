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
    "LogType": "VMOperation" | "AgentLog" | "ScaffoldLog" | "Error",
    "VMName": "agent-swebench-xyz",  # For VM-specific logs
    "ExecutionMode": "vm" | "local" | "docker"
}
```

### Benefits
1. **Easy Querying**: `BenchmarkRuns_CL | where Benchmark == "swebench_verified" and LogType == "Error"`
2. **No Manual Context**: Context injected automatically via LoggerAdapter
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

**New Setup Flow:**
```python
def setup_logging(run_id, log_dir, azure_enabled=True):
    # Existing file handlers (unchanged)
    main_handler = RotatingFileHandler(f"{log_dir}/{run_id}.log")
    verbose_handler = RotatingFileHandler(f"{log_dir}/{run_id}_verbose.log")
    console_handler = RichHandler()

    # NEW: Azure Monitor handler (if enabled)
    if azure_enabled and AZURE_WORKSPACE_ID:
        azure_handler = AzureMonitorHandler(
            workspace_id=AZURE_WORKSPACE_ID,
            table_name="BenchmarkRuns_CL"
        )
        handlers.append(azure_handler)

    # Apply all handlers to loggers
```

### Phase 2: VM Code Refactoring (Replace print() with logging)

#### 2.1 Refactor hal/utils/virtual_machine_manager.py
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

**Recommendation for MVP:** Single log group (simpler querying)
- Query all: `BenchmarkRuns_CL | where RunID == "abc-123"`
- Query agent only: `BenchmarkRuns_CL | where RunID == "abc-123" and LogType == "AgentLog"`
- Query scaffold: `BenchmarkRuns_CL | where RunID == "abc-123" and LogType == "ScaffoldLog"`

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
AZURE_MONITOR_STREAM_NAME=Custom-BenchmarkRuns_CL

# Optional: Disable Azure logging (for local/testing only)
AZURE_MONITOR_ENABLED=true
```

#### 4.3 Workspace Setup via Terraform

**Automated Setup (Terraform/ARM Templates):**
Create `infrastructure/azure/logging/` directory with:
- `main.tf`: Main Terraform configuration
- `variables.tf`: Input variables (region, resource group, etc.)
- `outputs.tf`: Output DCE URL, DCR ID for .env
- `table-schema.json`: Custom table schema

**Resources to Create:**
1. **Log Analytics Workspace**: Central repository for logs
2. **Data Collection Endpoint (DCE)**: Ingestion endpoint
3. **Data Collection Rule (DCR)**: Defines schema and routing
4. **Custom Table**: `BenchmarkRuns_CL` with schema below
5. **Role Assignments**: Grant Monitoring Metrics Publisher role to service principal

**Table Schema (BenchmarkRuns_CL):**
```json
{
  "columns": [
    {"name": "TimeGenerated", "type": "datetime"},
    {"name": "Level", "type": "string"},
    {"name": "Message", "type": "string"},
    {"name": "LoggerName", "type": "string"},
    {"name": "RunID", "type": "string"},
    {"name": "Benchmark", "type": "string"},
    {"name": "AgentName", "type": "string"},
    {"name": "TaskID", "type": "string"},
    {"name": "LogType", "type": "string"},
    {"name": "VMName", "type": "string"},
    {"name": "ExecutionMode", "type": "string"},
    {"name": "Properties", "type": "dynamic"}
  ]
}
```

**Terraform Outputs:**
After `terraform apply`, outputs will be:
- `data_collection_endpoint`: Copy to `AZURE_MONITOR_DATA_COLLECTION_ENDPOINT`
- `data_collection_rule_id`: Copy to `AZURE_MONITOR_DATA_COLLECTION_RULE_ID`
- `workspace_id`: For querying logs in Azure Portal

### Phase 5: Testing & Rollout

#### 5.1 Testing Strategy
1. **Unit Tests**: Test LoggerAdapter context injection
2. **Integration Test**: Run single VM task with Azure logging enabled
3. **Validation**:
   - Verify logs appear in Azure Log Analytics
   - Verify file outputs unchanged
   - Verify no broken functionality

#### 5.2 Rollout Approach
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

**Feature Flag for Testing:**
- `AZURE_MONITOR_ENABLED=false`: Disable for local development/testing only
- Default: `true` for production VM runs
- CI/CD should test with Azure enabled

**Backwards Compatibility Checklist:**
- [ ] All existing log files still created
- [ ] Console output unchanged (still WARNING+ only)
- [ ] Weave integration still works
- [ ] Azure Monitor failures properly propagate (don't suppress errors)

#### 5.3 Validation Queries
After first successful run, validate with KQL:
```kql
// Check if logs are arriving
BenchmarkRuns_CL
| where TimeGenerated > ago(1h)
| summarize count() by LogType, Benchmark

// Get all logs for a specific run
BenchmarkRuns_CL
| where Properties.RunID == "your-run-id"
| order by TimeGenerated asc

// Find VM operation errors
BenchmarkRuns_CL
| where Level == "ERROR" and Properties.LogType == "VMOperation"
| project TimeGenerated, Message, Properties.TaskID
```

## Implementation Order

### Step 0: Azure Infrastructure Setup (Terraform)
1. Create `infrastructure/azure/logging/` directory
2. Write Terraform configuration for:
   - Log Analytics Workspace
   - Data Collection Endpoint (DCE)
   - Data Collection Rule (DCR)
   - Custom table schema (BenchmarkRuns_CL)
   - IAM role assignments
3. Run `terraform apply`
4. Copy outputs to .env file

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
1. `hal/utils/azure_logging.py` - New logging infrastructure
2. `infrastructure/azure/logging/main.tf` - Terraform main config
3. `infrastructure/azure/logging/variables.tf` - Terraform variables
4. `infrastructure/azure/logging/outputs.tf` - Terraform outputs
5. `infrastructure/azure/logging/table-schema.json` - Custom table schema
6. `infrastructure/azure/logging/README.md` - Setup instructions

### Modified Files:
1. `hal/utils/logging_utils.py` - Add Azure handler setup
2. `hal/utils/virtual_machine_manager.py` - Replace print() with logging
3. `hal/utils/virtual_machine_runner.py` - Add context injection
4. `pyproject.toml` - Add azure-monitor-ingestion dependency
5. `.env.template` - Add Azure Monitor config vars
6. `README.md` or `CLAUDE.md` - Document Azure setup (optional)

### Files to Read Before Implementation:
1. `hal/utils/logging_utils.py` - Understand current setup
2. `hal/utils/virtual_machine_manager.py` - See all print() statements
3. `hal/utils/virtual_machine_runner.py` - Understand orchestration flow
4. `hal/agent_runner.py` - Understand run initialization
5. `hal/cli.py` - Understand CLI entry point

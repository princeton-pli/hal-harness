# Plan: Docker Container Logs → Azure Monitor

## Current State

Based on git history (commit `327774e`), Azure Monitor logging works as follows:

1. **Infrastructure Setup** (`infrastructure/azure/setup-logging.sh`):
   - Creates Log Analytics Workspace
   - Creates Data Collection Endpoint (DCE)
   - Creates Data Collection Rule (DCR) with custom table `BenchmarkRuns_CL`
   - Configures stream name: `Custom-BenchmarkRuns_CL`

2. **Azure Logging Handler** (`hal/utils/azure_logging.py`):
   - `is_running_in_azure_vm()` - Detects if code runs in Azure VM (checks IMDS endpoint)
   - `AzureMonitorHandler` - Batches logs and sends to DCR via Azure Monitor Ingestion API
   - `CloudJSONFormatter` - Formats logs as JSON with structured properties
   - `BenchmarkLoggerAdapter` - Injects context (RunID, Benchmark, etc.) into all logs

3. **Current Integration**:
   - HAL harness uses Python `logging` module
   - `setup_logging()` in VM adds `AzureMonitorHandler` if `is_running_in_azure_vm() == True`
   - All Python `logger.info()` calls → Azure Monitor

## Problem

Docker containers run inside Azure VMs but:
- Container logs go to stdout/stderr
- Not captured by Azure Monitor (only Python logger goes to Azure)
- Logs lost unless manually extracted via `docker cp` or `docker logs`

## Solution: 3-Part Strategy

### **Part 1: Container → File (DONE)**
✅ Already in place:
```python
# azure_virtual_machine.py:236-241
docker_cmd = (
    f'nohup bash -c "'
    f"echo 'Docker starting at $(date)' > /home/agent/docker_trace.txt && "
    f"docker run --rm {env_flags} {image} && "
    f"echo 'Docker completed at $(date)' >> /home/agent/docker_trace.txt"
    f'" > /home/agent/docker_output.log 2>&1 &'
)
```

Logs captured in:
- `/home/agent/docker_output.log` - Container stdout/stderr
- `/home/agent/docker_trace.txt` - Start/completion timestamps

### **Part 2: File → Python Logger → Azure Monitor**

**Option A: Tail logs in background (streaming)**
```python
# In azure_virtual_machine.py after docker run
# Start a background log tailer that reads docker_output.log and sends to logger
ssh_cmd = [
    "ssh", "-i", ssh_key_path, "-o", "StrictHostKeyChecking=no",
    f"agent@{self.public_ip}",
    f"nohup python3 -c \""
    f"import logging, time, sys; "
    f"logger = logging.getLogger('docker.{self.name}'); "
    f"logger.setLevel(logging.INFO); "
    f"f = open('/home/agent/docker_output.log', 'r'); "
    f"f.seek(0, 2); "  # Seek to end
    f"while True: "
    f"line = f.readline(); "
    f"if line: logger.info(line.strip()); "
    f"else: time.sleep(1); "
    f"\" > /dev/null 2>&1 &"
]
```

**Option B: Batch upload after completion (simpler)**
```python
# After docker completes, upload logs in one batch
ssh_cmd = [
    "ssh", "-i", ssh_key_path, "-o", "StrictHostKeyChecking=no",
    f"agent@{self.public_ip}",
    f"python3 -c \""
    f"import logging; "
    f"logger = logging.getLogger('docker.{self.name}'); "
    f"with open('/home/agent/docker_output.log') as f: "
    f"  for line in f: logger.info(line.strip());"
    f"\""
]
```

### **Part 3: Enhanced entrypoint.py logging**

Make container logs more structured:

```python
# In entrypoint.py, configure Python logging to go to stdout
import logging
import sys

# Configure logging format for parsing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)
logger.info("Container started", extra={"HAL_RUN_ID": run_id, "HAL_TASK_ID": task_id})
```

This makes logs machine-readable and easier to parse when ingesting to Azure.

## Implementation Steps

### **Milestone 1: Verify current logging works**
- [ ] Confirm `setup_logging()` in v2 works like main hal
- [ ] Test that v2 code sends logs to Azure Monitor when run on VM
- [ ] Check that `is_running_in_azure_vm()` returns True on VM, False locally

### **Milestone 2: Capture container logs**
- [ ] Update `azure_virtual_machine.py` to not use `--rm` flag
- [ ] Extract logs after container completes: `docker logs <container> > /home/agent/docker.log`
- [ ] Copy logs back to orchestrator or upload to blob storage

### **Milestone 3: Stream logs to Azure Monitor**
- [ ] Implement Option B (batch upload) - simpler, lower risk
- [ ] After docker completes, SSH to VM and read `/home/agent/docker_output.log`
- [ ] Parse each line and send via `logger.info()` with context (RunID, TaskID, VMName)
- [ ] Test that Docker logs appear in Azure Monitor Log Analytics

### **Milestone 4: Real-time streaming (optional)**
- [ ] Implement Option A (tail -f equivalent)
- [ ] Start background Python script on VM that tails docker_output.log
- [ ] Script sends each line to logger as it appears
- [ ] More complex but enables real-time monitoring

## Testing Plan

1. **Local test:**
   ```bash
   python v2/run_docker_on_vm.py --image hal-core-agent-docker:latest --vm_count 1
   ```

2. **Verify logs in Azure:**
   ```kusto
   BenchmarkRuns_CL
   | where RunID == "test-run-id"
   | where LoggerName contains "docker"
   | order by TimeGenerated desc
   ```

3. **Check log structure:**
   - Logs should have: RunID, TaskID, VMName, Message
   - Should be queryable by any field
   - Should show container stdout/stderr

## Open Questions

1. **Performance:** How much log volume will Docker containers generate?
   - Answer: CORE-bench agent can generate MB of logs per task
   - Need batching to avoid overwhelming Azure Monitor API

2. **Cost:** Azure Monitor charges per GB ingested
   - Need to estimate: tasks × log_size × vm_count
   - May need log sampling or filtering for verbose libraries

3. **Completeness:** How to ensure all logs are captured before VM cleanup?
   - Wait for docker container to exit
   - Wait for log upload to complete
   - Add timeout for safety

## Notes

- v2 already has `azure_logging.py` (copy from main)
- v2 already has `logging_utils.py` with Azure Monitor support
- Need to ensure Docker logs get same context (RunID, Benchmark, etc.) as Python logs
- Consider adding structured logging to entrypoint.py (JSON format)

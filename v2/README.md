# HAL Harness v2 - Azure VM + Docker Orchestration

Clean, minimal implementation for running Docker containers on Azure VMs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure `.env` file in project root with:
```
AZURE_SUBSCRIPTION_ID=...
AZURE_RESOURCE_GROUP_NAME=...
AZURE_LOCATION=eastus
NETWORK_SECURITY_GROUP_NAME=...
SSH_PUBLIC_KEY_PATH=...
SSH_PRIVATE_KEY_PATH=...
```

3. Ensure NSG exists (or create it using `infrastructure/deploy.sh`)

## Usage

```python
from v2.azure_manager import AzureManager

# Create manager (automatically provisions VMs in parallel)
manager = AzureManager(run_id="test-123", vm_count=3, use_gpu=False)

# Run Docker on each VM
for vm in manager.virtual_machines:
    vm.run_docker(env_vars={"HAL_RUN_ID": "test-123"})

# Cleanup
manager.cleanup()
```

Or run the example:
```bash
python -m v2.run_docker_on_vm
```

## Architecture

- **`azure_manager.py`**: Orchestrates VMs for a run
- **`azure_virtual_machine.py`**: Single VM representation + Docker execution
- **`utils.py`**: Shared utilities (command logging)
- **`run_docker_on_vm.py`**: Example usage

## Key Features

- ✅ Parallel VM creation using Azure Python SDK
- ✅ Standard and GPU VM support
- ✅ Docker execution via SSH
- ✅ Automatic cleanup
- ✅ Structured logging

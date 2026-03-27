---
title: "refactor: remove --vm flag and Azure VM execution path"
type: refactor
status: completed
date: 2026-03-27
linear: AE-65
---

# refactor: remove --vm flag and Azure VM execution path

The `--vm` flag and its entire execution stack (VirtualMachineRunner, VirtualMachineManager, AzureVirtualMachine) are dead code now that Prefect/Azure Batch handles orchestration externally. Remove everything from CLI to files.

## Acceptance Criteria

- [x] `--vm` flag removed from `hal/cli.py`
- [x] `hal/utils/virtual_machine_runner.py` deleted
- [x] `hal/utils/virtual_machine_manager.py` deleted
- [x] `hal/utils/vm/` directory deleted
- [x] `hal/utils/setup_vm.sh` deleted
- [x] `hal/agent_runner.py`: `use_vm`, GPU sandbox warning, and `requires_sandbox` check removed
- [x] `hal/benchmarks/base_benchmark.py`: `requires_sandbox` param/attribute removed
- [x] All benchmarks: `requires_sandbox=False` kwargs removed
- [x] `hal/utils/logging_utils.py`: `use_vm` parameter removed
- [x] `pyproject.toml`: `azure` optional dependency group removed
- [x] `.env.template`: Azure VM config section removed
- [x] No remaining references to `use_vm`, `VirtualMachineRunner`, `VirtualMachineManager`, or `requires_sandbox`

## Changes

| File | Action |
|------|--------|
| `hal/cli.py` | Remove `--vm` flag (line 85), remove from validation (line 219), remove `use_vm=vm` kwarg (line 263) |
| `hal/agent_runner.py` | Remove `use_vm` param (line 29), GPU warning block (lines 87–101), sandbox check (lines 106–110), VirtualMachineRunner import+init (lines 115–123), `self.use_vm` (line 146) |
| `hal/benchmarks/base_benchmark.py` | Remove `requires_sandbox` param (line 21) and `self.requires_sandbox` assignment (lines 36–37) |
| `hal/benchmarks/swebench.py` | Remove `self.requires_sandbox = False` and `requires_sandbox=` kwarg |
| `hal/benchmarks/gaia.py` | Same |
| `hal/benchmarks/scienceagentbench.py` | Same |
| `hal/benchmarks/scicode.py` | Same |
| `hal/benchmarks/taubench.py` | Same |
| `hal/benchmarks/usaco.py` | Same |
| `hal/benchmarks/appworld.py` | Same |
| `hal/benchmarks/colbench.py` | Same |
| `hal/utils/logging_utils.py` | Remove `use_vm: bool = False` param; remove it from `setup_logging()` call in `cli.py` |
| `hal/utils/virtual_machine_runner.py` | Delete |
| `hal/utils/virtual_machine_manager.py` | Delete |
| `hal/utils/vm/` | Delete directory |
| `hal/utils/setup_vm.sh` | Delete |
| `pyproject.toml` | Remove `[project.optional-dependencies]` `azure` group (lines 35–42) |
| `.env.template` | Remove `# Azure VM Configuration` section |

## Context

All benchmarks already hardcode `requires_sandbox = False` — the field is dead state. The GPU warning (lines 97–101 of `agent_runner.py`) is also dead since `has_gpu_task` can only trigger the VM path, which is being removed.

The `azure` optional dep group (`azure-mgmt-compute`, `azure-mgmt-network`, `azure-mgmt-resource`, `azure-identity`, `paramiko`) is exclusively used by the VM stack. The Prefect orchestrator in `prefect/` uses `azure-batch` and `azure-storage-blob` via its own `requirements.txt` and is not installed via this pyproject.toml extra.

## Verification

```bash
grep -rn "use_vm\|VirtualMachineRunner\|VirtualMachineManager\|requires_sandbox" hal/ --include="*.py"
# Should return no results
```

## Sources & References

- Related plan: [2026-03-27-refactor-azure-orchestrator-into-orchestrators-dir-plan.md](2026-03-27-refactor-azure-orchestrator-into-orchestrators-dir-plan.md) (Phase 1 acceptance criteria)
- Linear: [AE-65](https://linear.app/agent-evals/issue/AE-65)

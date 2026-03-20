# HAL evaluation execution flow (hal-eval → finish)

High-level map of how a run flows from the `hal-eval` CLI to completion.

---

## 1. Entry point

```
hal-eval  (console script from pyproject.toml)
    → hal.cli:main
```

**`pyproject.toml`** `[project.scripts]`: `hal-eval = "hal.cli:main"`

---

## 2. CLI

**File:** `hal/cli.py`

- **Click** parses options (`--benchmark`, `--agent_dir`, `--vm`, `--task_timeout`, etc.).
- **main()**:
  1. **Parse args**: `parse_cli_args(a)`, `parse_cli_args(b)`, `parse_cli_args(i)` → `agent_args`, `benchmark_args`, `inspect_eval_args`.
  2. **run_id**: default `{benchmark}_{agent_name_sanitized}_{timestamp}` or use `--run_id`.
  3. **Logging**: `setup_logging(log_dir, run_id, use_vm=vm)`; `log_dir = results_dir/benchmark/run_id`.
  4. **Validation**: model pricing if `model_name` in agent_args; exactly one of conda/vm/docker; `--continue_run` requires `run_id`.
  5. **print_run_config(...)**.
  6. **Build run command**: `run_command = "hal-eval " + sys.argv[1:]` (for logging/repro).
  7. **Create AgentRunner** with all parsed options (including `task_timeout`, `agent_function`, `agent_dir`, `benchmark_name`, etc.).
  8. **Run**: `asyncio.run(runner.run(agent_name=agent_name, upload=upload))`.
  9. **Post-run**: `log_results(results)`, `log_run_summary(...)` (or warning if no benchmark/run_dir).

---

## 3. AgentRunner

**File:** `hal/agent_runner.py`

### 3.1 Construction (`__init__`)

- **BenchmarkManager** (`hal/benchmark_manager.py`): (agent_dir, config) → **get_benchmark(benchmark_name)** → concrete benchmark (e.g. `CoreBenchBenchmark` from `hal/benchmarks/corebench.py`).
- **Runner choice** (one of):
  - **VirtualMachineRunner** (`hal/utils/virtual_machine_runner.py`): (log_dir, benchmark, task_timeout, max_concurrent) if `use_vm`;
  - **DockerRunner** (`hal/utils/docker_runner.py`): (..., task_timeout) if `use_docker`;
  - **LocalRunner** (`hal/utils/local_runner.py`): (..., conda_env, task_timeout) otherwise.
- Stores agent_function, agent_dir, agent_args, run_id, etc.

### 3.2 Run pipeline (`run()`)

1. **Weave**: `weave.init(self.run_id)`.
2. **Dataset**: `dataset = self.benchmark.get_dataset()`.
3. **Continue run** (if `continue_run`):
   - If not `ignore_errors`: `dataset = self.get_remaining_tasks(dataset)` (filter out tasks already in `*_RAW_SUBMISSIONS.jsonl`).
   - If `ignore_errors`: `dataset = {}` (evaluation uses only previous submissions).
4. **Filter**: by `--task_ids` if set; cap by `--max_tasks` if set.
5. **Prompt sensitivity** (if enabled): build variation datasets; then either single-variation or multi-variation runs.
6. **Continue run cleanup**: if continuing and not ignore_errors, delete Weave calls for tasks in dataset so they can be re-run.
7. **Run agent**:
   - **Normal**: `agent_output = await self.runner.run_agent(dataset=dataset, agent_function=..., agent_dir=..., agent_args=..., run_id=..., benchmark=..., task=..., progress=...)`.
   - **Prompt sensitivity (single var)**: same but with `single_variation_dataset` and then set `prompt_sensitivity = False` for evaluation.
   - **Prompt sensitivity (multi var)**: loop over variation indices, run agent per variation, collect `all_variations_output`.
   - If `continue_run`: merge `agent_output` with previous `*_RAW_SUBMISSIONS.jsonl`.
8. **Evaluate**:
   - **Prompt sensitivity**: `weave.finish()`; for each task/variation call `benchmark.evaluate_output(single_output, run_id)`; collect scores.
   - **Normal**: `weave.finish()`; `eval_results = self.benchmark.evaluate_output(agent_output, self.run_id)`.
9. **Process results**: `results = self.benchmark.process_results(agent_name=..., run_id=..., agent_args=..., run_command=..., eval_results=..., weave_client=..., agent_output=..., upload=..., prompt_sensitivity=..., )`.
10. **Return** `results` to CLI.

---

## 4. Runner: VM path

**File:** `hal/utils/virtual_machine_runner.py`

### 4.1 `run_agent(dataset, agent_function, agent_dir, agent_args, run_id, benchmark, progress, task)`

- For each `(task_id, input_data)` in `dataset`, an async **process_task(task_id, input_data)** is run.
- Concurrency is limited by **semaphore** `max_concurrent`; tasks are scheduled with **asyncio.gather(run_with_semaphore(...) for each task)**.
- Each **process_task** does the following (see below).
- Results are merged into one dict and appended to `{run_id}_RAW_SUBMISSIONS.jsonl` in `log_dir`, then returned.

### 4.2 Per-task VM flow: `process_task(task_id, input_data)`

1. **VM name**: `vm_name = "agent-{benchmark_name}-{uuid}"[:32]`.
2. **GPU**: read `benchmark.benchmark[task_id].get("gpu", False)`.
3. **Create VM**: `vm_manager.create_virtual_machine_by_name(...)`  
   → **VirtualMachineManager** (`hal/utils/virtual_machine_manager.py`) + **AzureVirtualMachine** (`hal/utils/vm/azure_virtual_machine.py`) (provision VM, wait for startup script, etc.).
4. **Temp dir**: build task payload:
   - `input.json`, `agent_args.json`;
   - copy any `input_data["files"]` into temp dir;
   - copy benchmark `setup_script.sh` if present.
5. **Copy to VM**:
   - `vm_manager.compress_and_copy_files_to_vm(vm_name, temp_dir)` (tar.gz → SFTP → extract under `/home/agent`);
   - `vm_manager.compress_and_copy_files_to_vm(vm_name, agent_dir)` (agent code).
6. **Run agent on VM**: `vm_manager.run_agent_on_virtual_machine(vm_name, agent_function, task_id, input_data, agent_args, run_id, log_dir, benchmark)` (see below).
7. **Poll until done or timeout** (`task_timeout` seconds):
   - Every 30s: `fetch_agent_logs(vm_name, ...)` (pull `agent_trace.log`), then `vm_manager.check_task_completion(vm_name)` (checks for `/home/agent/output.json`).
   - If `output.json` appears → task complete; break.
8. **Copy results**: `vm_manager.copy_files_from_vm(vm_name, dest_dir)` (e.g. `log_dir/{task_id}/`); read `output.json` from dest_dir → **result**.
9. **Cleanup**: `vm_manager.delete_virtual_machine_by_name(vm_name)`.
10. Return `{task_id: result}` (or error/timeout dict).

---

## 5. VirtualMachineManager

**File:** `hal/utils/virtual_machine_manager.py`

### 5.1 `create_virtual_machine_by_name(vm_name, has_gpu, setup_timeout)`

- **AzureVirtualMachine** (`hal/utils/vm/azure_virtual_machine.py`): (name, resource_group, location, subscription_id, nsg_id, ssh_public_key, gpu=..., timeout=setup_timeout).
- VM creation (Azure API), wait for SSH, run startup (e.g. cloud-init); **timeout** is used for waiting for startup script.

### 5.2 `compress_and_copy_files_to_vm(vm_name, source_directory)`

- Tar.gz `source_directory` → SFTP put to `/home/agent/` → SSH `tar -xzf ... --strip-components=1 -C /home/agent`; remove remote tar and local tar.

### 5.3 `run_agent_on_virtual_machine(vm_name, agent_function, task_id, input_data, agent_args, run_id, log_dir, benchmark)`

1. **copy_env_and_run_setup_script** (inner):
   - SFTP: copy `.env` → `/home/agent/.env`; copy `setup_vm.sh` → VM, chmod +x, run `sudo bash setup_vm.sh`.
   - If benchmark has `setup_script`, run that on VM (e.g. `bash setup_script.sh` in agent home).
2. **copy_env_and_run_setup_script** (outer call) is invoked again with same args (conda/env setup if used).
3. **SFTP**:
   - Write `/home/agent/input.json` (`{task_id: input_data}`), `/home/agent/agent_args.json`.
   - Write `/home/agent/run_agent.env` with `RUN_ID`, `AGENT_FUNCTION`, `TASK_ID`.
   - Read static **run_agent.py** from `hal/utils/vm/run_agent.py`, write to `/home/agent/run_agent.py`, chmod +x.
4. **SSH**: run  
   `source ... conda && conda activate agent_env && python /home/agent/run_agent.py > agent_trace.log 2>&1`  
   (non-blocking; script runs in background on VM).

### 5.4 `check_task_completion(vm_name)`

- Uses **AzureVirtualMachine.check_for_file_presence_by_path** for `/home/agent/output.json` (SSH test -f).

### 5.5 `get_agent_trace(vm_name)`

- SFTP read `/home/agent/agent_trace.log` → returned as string (used by **fetch_agent_logs** in runner).

### 5.6 `copy_files_from_vm(vm_name, destination_directory)`

- SSH: `rm -rf /home/agent/miniconda3`; tar home dir on VM; SFTP get tar; extract locally to `destination_directory`.

### 5.7 `delete_virtual_machine_by_name(vm_name)`

- **AzureVirtualMachine.delete()** (`hal/utils/vm/azure_virtual_machine.py`) (VM + NIC, disk, etc.).

---

## 6. On-VM agent execution

**File:** `hal/utils/vm/run_agent.py`

- **Static script** (no string interpolation).
- **Load env**: `load_dotenv("/home/agent/.env")`, `load_dotenv("/home/agent/run_agent.env")`.
- **Require**: `RUN_ID`, `AGENT_FUNCTION`, `TASK_ID` (exit with error if missing).
- **Parse** `AGENT_FUNCTION` → `module_name`, `function_name`.
- **Weave**: `weave.init(RUN_ID)`.
- **Read** `/home/agent/input.json`, `/home/agent/agent_args.json`.
- **Load agent**: `importlib.util.spec_from_file_location(module_name, "/home/agent/{module_name}.py")` → `exec_module` → `getattr(module, function_name)`.
- **Run**: `with weave.attributes({"weave_task_id": TASK_ID}): result = agent(input_data, **agent_args)`.
- **Write** `/home/agent/output.json` with `result`.
- On exception: write `/home/agent/error.log`, re-raise.

---

## 7. Local / Docker runners (brief)

- **LocalRunner.run_agent** (`hal/utils/local_runner.py`): for each task, run agent in subprocess (conda env if set) with **task_timeout**; collect stdout/result; same high-level contract (dataset in → agent_output dict out).
- **DockerRunner.run_agent** (`hal/utils/docker_runner.py`): similar but each task runs in a container; same timeout and result shape.

---

## 8. Benchmark layer

**Files:** `hal/benchmark_manager.py` (registry); `hal/benchmarks/base_benchmark.py` (base); `hal/benchmarks/<name>.py` (e.g. `corebench.py`, `gaia.py`).

- **get_dataset()**: benchmark-specific (e.g. load from HuggingFace, disk); returns `Dict[task_id, task_input]`.
- **evaluate_output(agent_output, run_id)**: benchmark-specific scoring; returns eval result per task (e.g. scores).
- **process_results(...)**: build upload payload, write `*_UPLOAD.json`, optionally upload to HuggingFace, return final **results** dict for CLI.

---

## 9. End-to-end flow (single path, VM, one task)

```
hal-eval
  → main()                          hal/cli.py
  → AgentRunner(...)                hal/agent_runner.py
  → runner.run(agent_name, upload)  hal/agent_runner.py
       → weave.init(run_id)
       → benchmark.get_dataset()    hal/benchmarks/base_benchmark.py (or subclass)
       → (filters: continue_run, task_ids, max_tasks)
       → runner.run_agent(dataset, ...)  hal/utils/virtual_machine_runner.py
            → process_task(task_id, input_data)  [per task]
                 → create_virtual_machine_by_name()     hal/utils/virtual_machine_manager.py
                 → compress_and_copy_files_to_vm()      hal/utils/virtual_machine_manager.py
                 → run_agent_on_virtual_machine()       hal/utils/virtual_machine_manager.py
                      → copy_env_and_run_setup_script (setup_vm.sh, benchmark setup_script)
                      → write input.json, agent_args.json, run_agent.env
                      → deploy run_agent.py from hal/utils/vm/run_agent.py
                      → SSH: conda activate agent_env && python run_agent.py  (runs hal/utils/vm/run_agent.py on VM)
                 → poll: get_agent_trace(), check_task_completion()  virtual_machine_manager.py
                 → copy_files_from_vm()                hal/utils/virtual_machine_manager.py
                 → delete_virtual_machine_by_name()    hal/utils/virtual_machine_manager.py + vm/azure_virtual_machine.py
            → merge results, append RAW_SUBMISSIONS.jsonl
       → weave.finish()
       → benchmark.evaluate_output(agent_output, run_id)   hal/benchmarks/*.py
       → benchmark.process_results(...)                   hal/benchmarks/base_benchmark.py (or subclass)
  → log_results(results), log_run_summary(...)        hal/cli.py (implementations in hal/utils/logging_utils.py)
```

---

## 10. Key files reference

| Role | File |
|------|------|
| Entry + CLI | `hal/cli.py` |
| Logging helpers (setup_logging, log_results, log_run_summary, print_run_config) | `hal/utils/logging_utils.py` |
| Orchestration + run pipeline | `hal/agent_runner.py` |
| Benchmark registry + get_benchmark | `hal/benchmark_manager.py` |
| VM run loop + per-task flow | `hal/utils/virtual_machine_runner.py` |
| VM lifecycle + SSH/SFTP + run script deploy | `hal/utils/virtual_machine_manager.py` |
| Setup script copied to VM | `hal/utils/setup_vm.sh` (dir next to virtual_machine_manager.py) |
| Static agent entrypoint on VM | `hal/utils/vm/run_agent.py` |
| Azure VM resource creation/deletion | `hal/utils/vm/azure_virtual_machine.py` |
| Benchmark base (get_dataset, evaluate_output, process_results) | `hal/benchmarks/base_benchmark.py` |
| Concrete benchmarks (corebench, gaia, scicode, etc.) | `hal/benchmarks/*.py` |
| Local process runner | `hal/utils/local_runner.py` |
| Docker runner | `hal/utils/docker_runner.py` |

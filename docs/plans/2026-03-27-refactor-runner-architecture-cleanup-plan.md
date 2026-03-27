---
title: "refactor: Clean up runner architecture (AE-65)"
type: refactor
status: active
date: 2026-03-27
linear: https://linear.app/agent-evals/issue/AE-65
origin: docs/plans/original-2026-02-26-refactor-runner-architecture-external-orchestrators-plan.md
deepened: 2026-02-26
---

# refactor: Clean up runner architecture (AE-65)

## Context

This plan supersedes the February 2026 plan. The key change: **Azure orchestration has been pulled into Prefect** (`prefect/`), so we no longer need `orchestrators/azure/`. The scope is now purely HAL code cleanup.

**Original plan** (deepened by 7 research agents): `a49f10b` (branch `andrew/first-reliability-cleanup`).
**Key decisions carried forward** from original research:
1. Phase 3 (delete runner files) + Phase 5 (update AgentRunner) must be ONE atomic commit
2. `agent_function`/`task_id`/`run_id` must not be f-string interpolated into runner scripts
3. `requires_sandbox` must be explicitly removed (not left as dead state)
4. `self.temp_dirs` list is a concurrent mutation hazard — remove it
5. Use `os.killpg` + `start_new_session=True` for process group kill on timeout
6. Import-path registry (not lambdas) for `BenchmarkManager`
7. `benchmark_name` as explicit first param of `BaseBenchmark.__init__`

---

## Overview

Remove `DockerRunner`, `VirtualMachineRunner`, and `LocalRunner` from `hal/`. Replace with a single `Runner` class (`hal/utils/runner.py`). Clean up `AgentRunner`, `BenchmarkManager`, and `BaseBenchmark`. Azure orchestration is handled by Prefect; Docker usage gets a standalone `Dockerfile` only.

---

## Current State

| File | Status |
|------|--------|
| `hal/utils/local_runner.py` | Replace → becomes `runner.py` |
| `hal/utils/docker_runner.py` | Delete |
| `hal/utils/virtual_machine_runner.py` | Delete |
| `hal/utils/virtual_machine_manager.py` | Delete |
| `hal/utils/vm/` | Delete directory |
| `hal/utils/docker/Dockerfile` | Move to `orchestrators/docker/Dockerfile` |
| `hal/utils/runner.py` | Create (doesn't exist yet) |
| `hal/agent_runner.py` | Update: remove docker/vm flags |
| `hal/cli.py` | Update: remove `--vm`, `--docker` flags |
| `hal/benchmark_manager.py` | Update: if/elif → registry |
| `hal/benchmarks/base_benchmark.py` | Update: explicit `benchmark_name` param |
| `prefect/` | **Leave untouched** (handles Azure orchestration) |
| `azure_entrypoint.sh` | **Leave untouched** (CLI bridge for Prefect Batch tasks) |

---

## Implementation Phases

> **Critical sequencing note:** Phase 3 (delete runner files) and Phase 5 (remove
> `use_vm`/`use_docker` from `AgentRunner`) **must land in a single atomic commit.**
> Deleting `docker_runner.py` while `agent_runner.py:9` still imports it breaks the
> entire `hal` package at import time.

### Phase 1: Canonical shared utilities

**Goal:** One source of truth for duplicated logic before any deletions.

- [ ] Create `hal/utils/errors.py` — consolidate `_is_transient_error` from 5 locations:
  - `local_runner.py:93-114`
  - `docker_runner.py:168-190`
  - `weave_utils.py:612-627`
  - 2 inlined copies in `_create_runner_script` generated scripts

  ```python
  # hal/utils/errors.py
  import re

  TRANSIENT_ERROR_PATTERNS: tuple[str, ...] = (
      "too many requests",
      "rate limit",
      "service unavailable",
      "bad gateway",
      "gateway timeout",
      "connection reset by peer",
      "connection refused",
      "temporarily unavailable",
      "timed out",
      "timeout",
      "broken pipe",
      "reset by peer",
      r"\b429\b",
      r"\b502\b",
      r"\b503\b",
      r"\b504\b",
      r"\bdns\b",
  )
  _PATTERN_RE = tuple(re.compile(p) for p in TRANSIENT_ERROR_PATTERNS)

  def is_transient_error(error: Exception) -> bool:
      msg = str(error).lower()
      return any(p.search(msg) for p in _PATTERN_RE)
  ```

- [ ] In `hal/utils/weave_utils.py`: replace duplicate token-cost blocks at lines 760-775 and 968-991 with `_compute_token_cost(usage, prices)` helper.

- [ ] Fix `swebench.py:22` relative path:
  ```python
  _HERE = Path(__file__).parent
  with open(_HERE / "swebench_verified_mini_task_ids.txt") as f:
  ```

---

### Phase 2: `Environment` dataclass

**Goal:** Lean config dataclass before wiring into `AgentRunner`.

- [ ] Create `hal/utils/environment.py`:

  ```python
  # hal/utils/environment.py
  from dataclasses import dataclass

  @dataclass
  class Environment:
      task_timeout: int = 1800
      max_concurrent: int = 10

      def validate(self) -> None:
          errors: list[str] = []
          if self.task_timeout <= 0:
              errors.append(f"task_timeout must be > 0, got {self.task_timeout}")
          if self.max_concurrent <= 0:
              errors.append(f"max_concurrent must be > 0, got {self.max_concurrent}")
          if errors:
              raise ValueError("Invalid environment config:\n" + "\n".join(f"  - {e}" for e in errors))
  ```

---

### Phase 3 + Phase 5 (atomic): Single `Runner` + `AgentRunner` cleanup

> **Must be one commit.** See sequencing note above.

**Phase 3 — Create `Runner`, delete old runners:**

- [ ] Create `hal/utils/runner.py` based on `LocalRunner`:
  - Class: `Runner`
  - Constructor: `__init__(self, log_dir, max_concurrent=1, conda_env=None, task_timeout=1800)`
  - Replace `self._is_transient_error(e)` → `is_transient_error(e)` from `hal.utils.errors`
  - Remove `retry_config` parameter (declared but never used in `LocalRunner`)
  - **Remove `self.temp_dirs` list** — per-task `finally` block owns cleanup instead
  - **Process group kill on timeout:** use `start_new_session=True` in `asyncio.create_subprocess_exec` and `os.killpg(os.getpgid(process.pid), signal.SIGTERM)` instead of `process.kill()`
  - **Security fix in `_create_runner_script`:** pass `task_id`/`run_id` via a `run_config.json` sidecar file written into the temp dir, not via f-string interpolation. Validate `agent_function` against allowlist regex `r"^[a-zA-Z_][a-zA-Z0-9_.]*$"` before use.
  - Use `asyncio.to_thread` for `shutil.copytree` (currently holds semaphore slot during blocking I/O)

- [ ] Delete `hal/utils/local_runner.py`
- [ ] Delete `hal/utils/docker_runner.py`
- [ ] Delete `hal/utils/virtual_machine_runner.py`
- [ ] Delete `hal/utils/virtual_machine_manager.py`
- [ ] Delete `hal/utils/vm/` directory (contains `run_agent.py`, `azure_virtual_machine.py`)

- [ ] Update `pyproject.toml`:
  - Remove `docker>=7.1.0` from `[project.dependencies]`
  - Keep `azure-*` optional group (used by Prefect, not HAL core)

**Phase 5 — `AgentRunner` cleanup (same commit):**

- [ ] In `hal/agent_runner.py` `__init__`:
  - Remove `use_vm: bool` and `use_docker: bool` parameters
  - Remove runner selection if/elif block (lines 115-138)
  - Construct `Runner(...)` directly
  - Construct `Environment(task_timeout=task_timeout, max_concurrent=max_concurrent)`
  - Remove `requires_sandbox` enforcement (lines 106-109)

- [ ] In `hal/agent_runner.py` `run()`:
  - Call `self.environment.validate()` at top
  - Extract `_load_submissions(path: Path) -> dict[str, Any]` helper (replaces 3 inline JSONL parsing blocks at lines 185-201, 337-340, 441-450)
  - Extract `_run_one_variation(self, dataset, variation_label=None) -> dict` to DRY the three execution branches
  - **Remove `self.prompt_sensitivity = False` mutation** (line 371) — use local variable instead

- [ ] In `hal/cli.py`:
  - Remove `--vm` option
  - Remove `--docker` option
  - Remove `use_vm` / `use_docker` from `main()`
  - Change `validate_model_pricing` (lines 374-385): raise `ValueError` instead of `sys.exit(1)`

---

### Phase 4: Docker Dockerfile only

> Azure orchestration is in Prefect. No `orchestrators/azure/` needed.

- [ ] Create `orchestrators/docker/` directory
- [ ] Move `hal/utils/docker/Dockerfile` → `orchestrators/docker/Dockerfile`
- [ ] Create `orchestrators/docker/README.md`:

  ```markdown
  # Docker Orchestrator

  Runs hal-eval inside a Docker container.
  hal-eval uses the standard local Runner inside the container.

  ## Build
  docker build -t hal-agent orchestrators/docker/

  ## Run
  docker run \
      -v $(pwd)/results:/app/results \
      -v $(pwd)/agents/my_agent:/app/my_agent:ro \
      --env-file .env \
      hal-agent \
      hal-eval --agent_dir /app/my_agent --benchmark swebench_verified ...
  ```

- [ ] Delete `hal/utils/docker/` directory

---

### Phase 6: `BenchmarkManager` registry + `BaseBenchmark` hardening

**Goal:** Replace if/elif dispatch with import-path registry. Make `benchmark_name` explicit.

- [ ] Replace `hal/benchmark_manager.py` with import-path registry:

  ```python
  # hal/benchmark_manager.py
  import importlib
  from typing import Any

  _BENCHMARK_REGISTRY: dict[str, str] = {
      "gaia":                          "hal.benchmarks.gaia:GaiaBenchmark",
      "scicode":                       "hal.benchmarks.scicode:SciCodeBenchmark",
      "scicode_easy":                  "hal.benchmarks.scicode:SciCodeBenchmark",
      "scicode_hard":                  "hal.benchmarks.scicode:SciCodeBenchmark",
      "usaco":                         "hal.benchmarks.usaco:USACOBenchmark",
      "mlagentbench":                  "hal.benchmarks.mlagentbench:MLAgentBenchBenchmark",
      "swebench_verified":             "hal.benchmarks.swebench:SWEBenchBenchmark",
      "swebench_verified_mini":        "hal.benchmarks.swebench:SWEBenchBenchmark",
      "appworld_test_normal":          "hal.benchmarks.appworld:AppWorldBenchmark",
      "appworld_test_challenge":       "hal.benchmarks.appworld:AppWorldBenchmark",
      "taubench_retail":               "hal.benchmarks.taubench:TauBenchBenchmark",
      "taubench_airline":              "hal.benchmarks.taubench:TauBenchBenchmark",
      "corebench_easy":                "hal.benchmarks.corebench:CoreBenchEasy",
      "corebench_medium":              "hal.benchmarks.corebench:CoreBenchMedium",
      "corebench_hard":                "hal.benchmarks.corebench:CoreBenchHard",
      "scienceagentbench":             "hal.benchmarks.scienceagentbench:ScienceAgentBench",
      "assistantbench":                "hal.benchmarks.assistantbench:AssistantBenchBenchmark",
      "colbench_backend_programming":  "hal.benchmarks.colbench:ColBenchBenchmark",
      "colbench_frontend_design":      "hal.benchmarks.colbench:ColBenchBenchmark",
  }

  def _load_class(import_path: str) -> type:
      module_path, class_name = import_path.rsplit(":", 1)
      return getattr(importlib.import_module(module_path), class_name)

  class BenchmarkManager:
      @property
      def benchmarks(self) -> list[str]:
          return sorted(_BENCHMARK_REGISTRY)

      def get_benchmark(
          self, benchmark_name: str, agent_dir: str, config: dict[str, Any]
      ) -> "BaseBenchmark":
          if benchmark_name not in _BENCHMARK_REGISTRY:
              raise ValueError(
                  f"Unknown benchmark: {benchmark_name!r}. "
                  f"Available: {self.benchmarks}"
              )
          cls = _load_class(_BENCHMARK_REGISTRY[benchmark_name])
          return cls(benchmark_name, agent_dir, config)
  ```

- [ ] Update `hal/benchmarks/base_benchmark.py` — make `benchmark_name` the explicit first param:

  ```python
  def __init__(
      self,
      benchmark_name: str,
      agent_dir: str,
      config: Dict[str, Any],
      setup_script: Optional[str] = None,
      base_results_dir: str = "results",
  ):
      self.benchmark_name = benchmark_name
      ...
  ```

  Remove `requires_sandbox` parameter and attribute.

- [ ] Update every `BaseBenchmark` subclass `__init__` to pass `benchmark_name` as first arg to `super().__init__()` and remove `requires_sandbox` from their constructors.

- [ ] Update all `BenchmarkManager.get_benchmark(...)` call sites in `agent_runner.py` to pass `(benchmark_name, agent_dir, config)` instead of just `(benchmark_name)`.

---

## Acceptance Criteria

- [ ] `hal` package imports without errors after all runner files deleted
- [ ] `hal-eval --benchmark gaia --agent_dir ...` runs end-to-end with `LocalRunner` → `Runner`
- [ ] `hal-eval --vm` and `hal-eval --docker` CLI flags are gone
- [ ] `docker` is not in `pyproject.toml` core dependencies
- [ ] All benchmarks loadable via `BenchmarkManager.get_benchmark(name, agent_dir, config)`
- [ ] `BaseBenchmark.__init__` takes `benchmark_name` as first positional arg
- [ ] `requires_sandbox` removed from all code paths
- [ ] `_is_transient_error` has exactly one definition (`hal/utils/errors.py`)
- [ ] `agent_function` validated against allowlist before use in runner script
- [ ] `task_id`/`run_id` passed via `run_config.json` sidecar, not f-string
- [ ] Process group killed on timeout (not just parent process)
- [ ] `azure_entrypoint.sh` and `prefect/` untouched and still functional
- [ ] Existing tests pass

---

## What Is NOT In Scope

- `orchestrators/azure/` — Prefect handles this; no new azure orchestrator code needed
- `azure_entrypoint.sh` changes — leave as-is; it's the CLI bridge for Prefect Batch tasks
- `prefect/` changes — not part of this PR
- Perturbation parameters (coordinate with Stephan separately — AE-65 stopping point note)

---

## Sources & References

### Origin
- **Previous plan:** `a49f10b` commit (`andrew/first-reliability-cleanup` branch) — key decisions documented above
- **Linear:** https://linear.app/agent-evals/issue/AE-65

### Internal References
- `hal/utils/local_runner.py` — becomes `runner.py`
- `hal/utils/docker_runner.py` — deleted
- `hal/utils/virtual_machine_runner.py` — deleted
- `hal/utils/virtual_machine_manager.py` — deleted
- `hal/agent_runner.py:115-138` — runner selection (remove)
- `hal/agent_runner.py:106-109` — requires_sandbox check (remove)
- `hal/benchmark_manager.py:38-102` — if/elif chain (replace)
- `hal/benchmarks/base_benchmark.py:17-37` — BaseBenchmark init (update)
- `hal/cli.py:83-88` — `--vm`/`--docker` flags (remove)
- `prefect/` — Azure Batch orchestration (leave untouched)
- `azure_entrypoint.sh` — CLI bridge (leave untouched)

---
title: "refactor: Single Runner + external orchestrators (AE-65)"
type: refactor
status: active
date: 2026-02-26
linear: https://linear.app/agent-evals/issue/AE-65
origin: docs/brainstorms/2026-02-26-architecture-cleanup-brainstorm.md
deepened: 2026-02-26
---

# refactor: Single Runner + external orchestrators (AE-65)

## Enhancement Summary

**Deepened:** 2026-02-26
**Research agents used:** architecture-strategist, code-simplicity-reviewer, performance-oracle, security-sentinel, kieran-python-reviewer, pattern-recognition-specialist, best-practices-researcher

### Key improvements from research
1. **Critical security finding:** `agent_function` and `task_id`/`run_id` are interpolated bare into a generated Python script — a code injection vector. Inputs must be passed via JSON sidecar files, not f-string interpolation.
2. **Phase 3 and Phase 5 must be a single atomic commit** — deleting the runner files while `AgentRunner` still imports them breaks the package at import time.
3. **YAGNI trimming:** `Environment.resource_config`, `Environment.from_file()` stub, and `EnvironmentConfigError` are all pre-emptive abstractions with no current call site. Remove them.
4. **Performance:** `self.temp_dirs` list is a concurrent mutation hazard; process groups must be killed on timeout (not just the parent); `shutil.copytree` holds the semaphore slot unnecessarily.
5. **Registry pattern:** Use a dict of import path strings (not lambdas), with `importlib.import_module` for lazy loading.
6. **`requires_sandbox`** becomes dead state once `--vm`/`--docker` are removed — must be explicitly addressed in this PR.

---

## Overview

Remove `DockerRunner` and `VirtualMachineRunner` from the `hal` package entirely.
Replace with a single `Runner` that executes agent subprocesses locally. Move Docker
and Azure support to `orchestrators/` as standalone scripts with no knowledge of `hal`
evaluation internals. Also: import-path registry for `BenchmarkManager`, explicit
`benchmark_name` param on `BaseBenchmark`, decomposed `AgentRunner.run()`, and
several latent bugs and security issues fixed along the way.

## Problem Statement

Infrastructure concerns (Docker container lifecycle, Azure VM provisioning) are embedded
in the core evaluation library, causing:

- `_create_runner_script` copy-pasted verbatim between `LocalRunner`
  (`local_runner.py:338-415`) and `DockerRunner` (`docker_runner.py:532-609`)
- `_is_transient_error` duplicated 5 times with silent divergences — patterns differ
  between `local_runner.py:93`, `docker_runner.py:168`, `weave_utils.py:612`, and 2
  inlined copies in generated scripts
- `VirtualMachineRunner.__init__` (lines 21-32) has no `task_timeout` parameter, but
  `AgentRunner` always passes it (line 122) — latent `TypeError` on every `--vm` run
- `DockerRunner._run_single_task` has a 30-second dead-wait after `communicate()`
  returns, and a blocking `container.exec_run()` call on the event loop thread
- `docker>=7.1.0` is a core dependency even for users who never use Docker
- `BenchmarkManager.get_benchmark` is a 65-line if/elif chain requiring 3-place edits
  to add a benchmark; `mlagentbench` appears in the chain but not the benchmarks list
- `BaseBenchmark.__init__` uses `self.benchmark_name` at line 33 before subclasses can
  assign it — fragile implicit ordering, invisible at definition time
- **Security:** `agent_function`, `task_id`, and `run_id` are interpolated bare into
  a generated Python script via f-strings — a code injection vector

## Proposed Solution

(See brainstorm: `docs/brainstorms/2026-02-26-architecture-cleanup-brainstorm.md`)

1. **Single `Runner`** in `hal/utils/runner.py` — local subprocess only, no SDK imports.
2. **External orchestrators** in `orchestrators/docker/` and `orchestrators/azure/` —
   standalone scripts that provision an environment and invoke `hal-eval` as a subprocess
   (the CLI is the boundary; orchestrators do not import `hal` Python internals).
3. **`Environment` dataclass** — config for task_timeout and max_concurrent only;
   validated at run start via `validate()`.
4. **Import-path registry for `BenchmarkManager`** — dict of `name -> "module:Class"` strings.
5. **Explicit `benchmark_name` param** on `BaseBenchmark.__init__`.
6. **Decomposed `AgentRunner.run()`** — sensitivity branching extracted.

---

## Implementation Phases

> **Critical sequencing note:** Phase 3 (delete runner files) and Phase 5 (remove
> `use_vm`/`use_docker` from `AgentRunner`) **must land in a single atomic commit.**
> Deleting `docker_runner.py` while `agent_runner.py:9` still imports it breaks the
> entire `hal` package at import time. Do not split these across separate commits.

### Phase 1: Canonical shared utilities

**Goal:** One source of truth for duplicated logic before any deletions.

- [ ] Create `hal/utils/errors.py`:

  ```python
  # hal/utils/errors.py
  import re

  # Ordered by selectivity (most specific first to aid debugging)
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
  # Numeric HTTP codes use word-boundary regex to avoid false positives
  # (e.g., "5020 items" should NOT match 502)
  _PATTERN_RE = tuple(re.compile(p) for p in TRANSIENT_ERROR_PATTERNS)

  def is_transient_error(error: Exception) -> bool:
      msg = str(error).lower()
      return any(p.search(msg) for p in _PATTERN_RE)
  ```

  Reconcile all 5 existing pattern lists before committing:
  `local_runner.py:93-114`, `docker_runner.py:168-190`, `weave_utils.py:612-627`,
  and the 2 inlined copies in `_create_runner_script` strings.

  > **Pattern note:** Use `tuple` (not `frozenset`) — these patterns have meaning in
  > order (specific before general aids debugging) and substring matching is O(n×len)
  > regardless of container type. Use compiled regex for numeric codes to avoid
  > false positives like "5020 items" matching 502.

- [ ] In `hal/utils/weave_utils.py`: extract `_compute_token_cost`:

  ```python
  def _compute_token_cost(usage: dict[str, int], prices: dict[str, float]) -> float:
      fresh_input = usage["prompt_tokens"] - usage.get("cache_read_input_tokens", 0)
      return (
          fresh_input * prices.get("prompt_tokens", 0)
          + usage.get("cache_creation_input_tokens", 0) * prices.get("prompt_tokens", 0)
          + usage.get("cache_read_input_tokens", 0) * prices.get("prompt_tokens", 0)
          + usage.get("completion_tokens", 0) * prices.get("completion_tokens", 0)
      )
  ```

  Replace duplicate blocks at `weave_utils.py:760-775` and `968-991`.

- [ ] Fix `get_weave_calls` return type at `weave_utils.py:822`:

  ```python
  def get_weave_calls(client) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  ```

- [ ] Fix `swebench.py:22` relative path:

  ```python
  _HERE = Path(__file__).parent
  with open(_HERE / "swebench_verified_mini_task_ids.txt") as f:
  ```

---

### Phase 2: `Environment` dataclass

**Goal:** Introduce a lean `Environment` config before wiring into `AgentRunner`.

> **YAGNI guidance:** `resource_config`, `from_file()`, and a custom `EnvironmentConfigError`
> are pre-emptive abstractions with no current call site. Do not add them. `ValueError`
> is the correct exception here — it matches the validation raises already in
> `AgentRunner.__init__` and costs nothing to callers.

- [ ] Create `hal/utils/environment.py`:

  ```python
  # hal/utils/environment.py
  from dataclasses import dataclass

  @dataclass
  class Environment:
      task_timeout: int = 1800
      max_concurrent: int = 10

      def validate(self) -> None:
          """Call at run start. Raises ValueError with all problems collected."""
          errors: list[str] = []
          if self.task_timeout <= 0:
              errors.append(f"task_timeout must be > 0, got {self.task_timeout}")
          if self.max_concurrent <= 0:
              errors.append(f"max_concurrent must be > 0, got {self.max_concurrent}")
          if errors:
              raise ValueError("Invalid environment config:\n" + "\n".join(f"  - {e}" for e in errors))
  ```

  If multi-error display to callers becomes necessary later, a `validation_errors() -> list[str]`
  helper can be added then. Single method for now.

---

### Phase 3 + Phase 5 (atomic): Single `Runner` + `AgentRunner` cleanup

> **These two phases must be committed together.** See sequencing note above.

**Phase 3 — Create `Runner`, delete old runners:**

- [ ] Create `hal/utils/runner.py` from `LocalRunner` as the starting point:
  - Class name: `Runner`
  - Replace `self._is_transient_error(e)` → `is_transient_error(e)` from `hal.utils.errors`
  - Remove `retry_config` parameter (declared but never used in `LocalRunner`)
  - Constructor: `__init__(self, log_dir, max_concurrent=1, conda_env=None, task_timeout=1800)`
  - Apply performance fixes (see Performance Considerations section below)
  - Apply security fixes to `_create_runner_script` (see Security section below)

- [ ] Delete `hal/utils/local_runner.py` (replaced by `runner.py`)
- [ ] Delete `hal/utils/docker_runner.py`
- [ ] Delete `hal/utils/virtual_machine_runner.py`
- [ ] Delete `hal/utils/virtual_machine_manager.py`
- [ ] Delete `hal/utils/vm/` directory

- [ ] Update `pyproject.toml`:
  - Remove `docker>=7.1.0` from core `[project.dependencies]`
  - Keep `azure` optional group (referenced by `orchestrators/azure/`)

**Phase 5 — `AgentRunner` cleanup (same commit):**

- [ ] In `AgentRunner.__init__`:
  - Remove `use_vm: bool` and `use_docker: bool` parameters
  - Remove runner selection if/elif block (lines 115-138)
  - Construct `Runner(...)` directly
  - Construct `Environment(task_timeout=task_timeout, max_concurrent=max_concurrent)`
  - **Remove `requires_sandbox` enforcement** (lines 106-109). Since Docker and VM runners
    are gone, `requires_sandbox` becomes dead state. Remove the attribute from
    `BaseBenchmark`, remove its constructor parameter, and update the README. Do not
    leave it as undocumented dead code.

- [ ] In `AgentRunner.run()`:
  - Add `self.environment.validate()` at the top (before `weave.init`, before dataset load)

  - Extract JSONL parsing into `_load_submissions(path: Path) -> dict[str, Any]`:

    ```python
    def _load_submissions(self, path: Path) -> dict[str, Any]:
        results: dict[str, Any] = {}
        if not path.exists():
            return results
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.update(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed submission line: %.80s", line)
        return results
    ```

    Replace the 3 inline JSONL parsing blocks at lines 185-201, 337-340, 441-450.
    Note: `get_remaining_tasks` also does error-filtering on top of parsing — keep
    that filtering logic in the caller, not in `_load_submissions`.

  - Extract sensitivity branching. The key insight: all three modes call
    `self.runner.run_agent()` with a dataset. Extract a primitive:

    ```python
    async def _run_one_variation(
        self, dataset: dict, variation_label: str | None = None
    ) -> dict:
        """Run one agent pass over dataset. variation_label for logging only."""
        ...
    ```

    Express normal mode, single-variation, and multi-variation in terms of this
    primitive. This is DRY and removes the `self.prompt_sensitivity` mutation.

  - **Remove `self.prompt_sensitivity = False` mutation at line 371.** Use a local
    variable to track execution mode within `run()`. The extracted methods must take
    their inputs as parameters — do not read mutable run-state from `self`.

- [ ] In `hal/cli.py`:
  - Remove `--vm` option (line 83)
  - Remove `--docker` option (lines 84-88)
  - Remove `use_vm` / `use_docker` parameters from `main()`
  - Fix `validate_model_pricing` (lines 374-385): raise `ValueError` instead of
    `sys.exit(1)`. The `main()` caller already has a surrounding try/except that exits.

---

### Phase 4: External orchestrators

**Goal:** Create `orchestrators/` with standalone Docker and Azure scripts.

The orchestrators' contract with the core is the **CLI boundary**: they invoke
`hal-eval` as a subprocess. They do not import `hal` Python internals. The only
coupling is the I/O contract (reading CLI flags, reading result files from disk) and
the weave/run-id environment variables, which should be injected via a `.env` file
the orchestrator writes rather than embedded in the orchestrator's own code.

- [ ] Create `orchestrators/docker/`:
  - Move `hal/utils/docker/Dockerfile` → `orchestrators/docker/Dockerfile`
  - Apply Dockerfile security hardening (see Performance & Security section)
  - Create `orchestrators/docker/README.md`:

    ```markdown
    # Docker Orchestrator

    Runs hal-eval inside a Docker container using a volume mount.
    hal-eval executes with the standard local Runner inside the container.

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

  - Delete `hal/utils/docker/` directory

- [ ] Create `orchestrators/azure/`:
  - Create `orchestrators/azure/run.py` — provisions Azure VM or ACI, copies agent
    code, invokes `hal-eval` as a subprocess, retrieves results. This is the only
    file in the repo that imports `azure-mgmt-compute`, `azure-mgmt-network`, or
    `paramiko`. It does not import from `hal`.
  - Create `orchestrators/azure/README.md` documenting required env vars and usage.
  - Apply SSH hardening (see Security section below).

---

### Phase 6: `BenchmarkManager` registry + `BaseBenchmark` hardening

**Goal:** Replace if/elif dispatch with an import-path registry. Make `benchmark_name` explicit.

- [ ] In `hal/benchmark_manager.py`, replace with an import-path registry:

  ```python
  # hal/benchmark_manager.py
  import importlib
  from collections.abc import Callable
  from typing import Any

  # dict of benchmark_name -> "module:ClassName"
  # Add a benchmark here; no other file needs to change.
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
      """Lazy import. Only loads the module when first requested."""
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

  > **Why import paths, not lambdas:** The dict of `"module:Class"` strings is readable,
  > trivially diff-able, and gives you the lazy-import behaviour of the current code.
  > Lambdas with captures are harder to inspect and make the type of each value opaque.
  > The `cls(benchmark_name, agent_dir, config)` call site is uniform because `benchmark_name`
  > is now the first required param of `BaseBenchmark.__init__` — all subclasses accept it.

  > **Caller note:** The old `get_benchmark(benchmark_name)` took 1 argument. The new
  > signature takes 3. Verify and update all call sites in `agent_runner.py`.

  > **`mlagentbench`:** Was in the if/elif chain (lines 52-55) but absent from
  > `self.benchmarks`. Add it to the registry. Verify it is production-ready before
  > including; if not, remove it entirely rather than leaving it in a half-registered state.

- [ ] In `hal/benchmarks/base_benchmark.py`, add `benchmark_name` as first required param:

  ```python
  def __init__(
      self,
      benchmark_name: str,           # required, must be first
      agent_dir: str,
      config: dict[str, Any],
      setup_script: str | None = None,
      base_results_dir: str = "results",
      # requires_sandbox REMOVED — see Phase 3+5 note
  ):
      self.benchmark_name = benchmark_name
      self.benchmark_results_dir = os.path.join(base_results_dir, benchmark_name)
      # ... rest unchanged ...
  ```

  > **CoreBench audit first:** `CoreBenchEasy`, `CoreBenchMedium`, `CoreBenchHard`
  > (`corebench.py:442,477,528`) currently set `self.benchmark_name` _after_
  > `super().__init__()`. Audit these before touching `BaseBenchmark.__init__` — they
  > will break if not handled carefully.

- [ ] Update all 13 benchmark subclasses: pass `benchmark_name` as first argument to
  `super().__init__()`, remove pre-`super()` `self.benchmark_name = ...` assignments.
  Files: `gaia.py`, `scicode.py`, `usaco.py`, `mlagentbench.py`, `swebench.py`,
  `appworld.py`, `taubench.py`, `corebench.py` (3 classes), `scienceagentbench.py`,
  `assistantbench.py`, `colbench.py`.

- [ ] Fix feature flag comparisons:

  ```python
  # agent_runner.py:159 — Before (always False because parse_cli_args returns bool)
  if agent_args.get("enable_fault_injection") == "true":
  # After
  if agent_args.get("enable_fault_injection"):

  # taubench.py:84,98 — same pattern; also remove hasattr guard (always initialized)
  if self.agent_args.get("enable_compliance_monitoring"):
  ```

---

## Security Considerations

> These address findings from the security audit. Some are fixes to existing code
> being carried into `runner.py`; others are new requirements for the orchestrators.

### Critical: agent_function injection via generated script

**`local_runner.py:344` / `docker_runner.py:538` / `virtual_machine_manager.py:442`**

`agent_function` is split on `.` and the parts are embedded bare into an f-string that
becomes executable Python. A value like `x"; __import__('os').system('id') #.y` breaks
out of the string literal.

**Fix in `AgentRunner.__init__` (before any runner is constructed):**

```python
import re
_AGENT_FUNCTION_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$')

if not _AGENT_FUNCTION_RE.fullmatch(agent_function):
    raise ValueError(
        f"agent_function must be a dotted Python identifier (e.g. 'main.run'), "
        f"got: {agent_function!r}"
    )
```

### Critical: task_id and run_id interpolated into generated script

**`local_runner.py:381,401`** — `task_id` comes from benchmark data; `run_id` is derived
from user-supplied `benchmark_name`. Both are embedded as string literals in the generated
script.

**Fix:** Write them to a `run_config.json` sidecar alongside `input.json` and read from
that file inside the generated script:

```python
# In _run_single_task, after copying input files:
run_config = {"run_id": run_id, "task_id": task_id}
(temp_dir / "run_config.json").write_text(json.dumps(run_config))

# In _create_runner_script, instead of f-string interpolation:
"""
import json
_cfg = json.loads(Path("run_config.json").read_text())
run_id = _cfg["run_id"]
task_id = _cfg["task_id"]
"""
```

### High: path traversal via `input_data["files"]` dest keys

**`local_runner.py:214`** — dest keys from benchmark data are used as file paths with
only a `lstrip("/")` guard. A key of `../../etc/cron.d/x` survives this.

**Fix:** After constructing the destination path, assert containment:

```python
dest_full = (temp_dir / dest_path).resolve()
if not str(dest_full).startswith(str(temp_dir.resolve()) + os.sep):
    raise ValueError(f"Path traversal rejected: {dest_path!r}")
```

### Medium: Dockerfile — non-root user

The current Dockerfile comment says root is needed for host-readable files. This is
incorrect when using volume mounts (files written by UID 10000 inside the container
appear as UID 10000 on the host).

**`orchestrators/docker/Dockerfile` must include:**

```dockerfile
# Multi-stage: build tools only in builder, not runtime
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && rm -rf /var/lib/apt/lists/*
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_24.11.1-0-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && echo "<SHA256_HASH>  /tmp/miniconda.sh" | sha256sum -c \
    && bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/conda/bin:$PATH
COPY --from=builder /opt/conda /opt/conda
RUN groupadd -g 10001 agent && useradd -u 10000 -g agent -m agent \
    && mkdir -p /workspace && chown agent:agent /workspace
RUN pip install --no-cache-dir weave "gql<4"
WORKDIR /workspace
USER agent
```

Pin Miniconda to a specific version and verify SHA-256. This resolves the current
`latest` tag that makes builds non-reproducible.

### Medium: Azure orchestrator SSH hardening

`orchestrators/azure/run.py` must NOT use `paramiko.AutoAddPolicy()` or
`StrictHostKeyChecking=no`. Capture the VM's host key via the Azure Compute API
immediately after provisioning and supply it as a `known_hosts` file:

```python
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.load_host_keys(known_hosts_path)  # written from Azure API post-provision
```

---

## Performance Considerations

> Fixes that belong in `hal/utils/runner.py`.

### Use `tempfile.mkdtemp()` instead of hand-rolled paths

```python
# Before (local_runner.py:196)
temp_dir = Path(f"/tmp/agent_run_{uuid.uuid4()}")
# After
import tempfile
temp_dir = Path(tempfile.mkdtemp(prefix="hal_agent_"))
```

`tempfile` respects `TMPDIR`, avoids predictable paths, and is the stdlib idiom.

### Own temp dir cleanup in per-task `finally`, remove `self.temp_dirs`

`self.temp_dirs` is a shared list mutated by concurrent coroutines without a lock —
a race condition. Remove it entirely. Own the lifecycle in `_run_single_task`'s `finally`:

```python
finally:
    try:
        await asyncio.to_thread(
            shutil.copytree, temp_dir,
            Path(self.log_dir) / task_id, dirs_exist_ok=True
        )
    except Exception as e:
        logger.debug("Log copy failed for %s: %s", task_id, e)
    finally:
        await asyncio.to_thread(shutil.rmtree, temp_dir, True)
```

The `asyncio.to_thread` calls release the semaphore slot during the blocking I/O,
restoring concurrency to other tasks.

### Kill the process group on timeout, not just the parent

`conda run` spawns a grandchild Python process. `process.kill()` sends SIGKILL to the
`conda run` parent but not to the grandchild, leaving orphaned processes after a timeout.

```python
import os, signal

process = await asyncio.create_subprocess_exec(
    *run_agent_cmd,
    cwd=str(temp_dir),
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    start_new_session=True,   # isolates agent in its own process group
)

# In TimeoutError handler:
except asyncio.TimeoutError:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    await process.wait()
```

### Confirm: submissions file uses `asyncio.Lock`, not `threading.Lock`

`local_runner.py:31` already uses `asyncio.Lock`. **Do not change this.** The plan
description's mention of `threading.Lock` is incorrect — `asyncio.Lock` is right here.
A `threading.Lock` would be semantically wrong in an asyncio context and would introduce
a deadlock risk if any caller ever wraps the write in `asyncio.to_thread`.

---

## Files Changed

### New files
| File | Purpose |
|---|---|
| `hal/utils/runner.py` | Single Runner class (from LocalRunner + fixes above) |
| `hal/utils/environment.py` | Environment dataclass + validate() |
| `hal/utils/errors.py` | Canonical `is_transient_error`, `TRANSIENT_ERROR_PATTERNS` |
| `orchestrators/docker/Dockerfile` | Standalone Docker image (hardened, non-root) |
| `orchestrators/docker/README.md` | Usage instructions |
| `orchestrators/azure/run.py` | Standalone Azure VM/ACI orchestrator |
| `orchestrators/azure/README.md` | Setup and usage |

### Modified files
| File | Changes |
|---|---|
| `hal/agent_runner.py` | Remove runner selection; add Environment; decompose run(); fix injection; fix bugs |
| `hal/cli.py` | Remove --vm/--docker; fix validate_model_pricing |
| `hal/benchmark_manager.py` | Import-path registry replaces if/elif |
| `hal/benchmarks/base_benchmark.py` | benchmark_name as explicit param; remove requires_sandbox |
| `hal/benchmarks/*.py` (13 files) | Update super().__init__() calls |
| `hal/utils/weave_utils.py` | Deduplicate cost calc; fix return type annotation |
| `hal/benchmarks/swebench.py` | Fix relative path |
| `pyproject.toml` | Remove docker from core deps |

### Deleted files
| File | Reason |
|---|---|
| `hal/utils/local_runner.py` | Replaced by `runner.py` |
| `hal/utils/docker_runner.py` | Docker is external |
| `hal/utils/virtual_machine_runner.py` | Azure is external |
| `hal/utils/virtual_machine_manager.py` | Azure is external |
| `hal/utils/vm/` (directory) | Azure is external |
| `hal/utils/docker/` (directory) | Moved to `orchestrators/docker/` |

---

## Acceptance Criteria

- [ ] `hal-eval` runs with equivalent behavior to today's local runner
- [ ] `hal-eval --vm` and `hal-eval --docker` flags no longer exist
- [ ] No `import docker` or Azure SDK imports anywhere under `hal/`
- [ ] `docker>=7.1.0` is not in core `[project.dependencies]` in `pyproject.toml`
- [ ] `is_transient_error` / `TRANSIENT_ERROR_PATTERNS` exist in exactly one place (`errors.py`)
- [ ] `_create_runner_script` exists in exactly one place (`runner.py`)
- [ ] `agent_function` is validated against `[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+` before use
- [ ] `task_id` and `run_id` are passed to the generated script via `run_config.json`, not f-string interpolation
- [ ] Path traversal check on `input_data["files"]` dest keys
- [ ] `BenchmarkManager.get_benchmark` uses an import-path registry dict
- [ ] `BaseBenchmark.__init__` accepts `benchmark_name` as first positional param
- [ ] `requires_sandbox` removed from `BaseBenchmark` and README
- [ ] All 13 benchmark subclasses updated
- [ ] `AgentRunner.run()` has no `self.prompt_sensitivity = False` mutation mid-method
- [ ] JSONL parsing in exactly one place (`_load_submissions`)
- [ ] `validate_model_pricing` raises `ValueError`, not `sys.exit`
- [ ] Feature flag checks use truthy evaluation, not `== "true"`
- [ ] `environment.validate()` called before any tasks are dispatched
- [ ] `self.temp_dirs` removed; temp dir cleanup owned in per-task `finally`
- [ ] Process group killed on timeout (`os.killpg` + `start_new_session=True`)
- [ ] `shutil.copytree` log copy wrapped in `asyncio.to_thread`
- [ ] `orchestrators/docker/Dockerfile` runs as non-root user (UID 10000)
- [ ] Miniconda pinned to specific version with SHA-256 verification
- [ ] All existing tests pass
- [ ] `orchestrators/docker/README.md` and `orchestrators/azure/README.md` explain usage

---

## Risks & Dependencies

| Risk | Mitigation |
|---|---|
| Phase 3+5 must be atomic — broken import window if split | Note in PR description; single commit |
| 13 benchmark subclass edits | Low risk per file; do in one commit; audit CoreBench first |
| `mlagentbench` status unclear | Verify before including in registry; remove if dead |
| No existing runner/benchmark unit tests | Add tests for `Runner` and `BenchmarkManager.get_benchmark` |
| Current `--vm` users must migrate | `orchestrators/azure/README.md` must document migration path |
| `requires_sandbox` removal may break documented workflows | Search README and docs before removing |

---

## Out of Scope

- `MODEL_PRICES_DICT` prefix-alias normalization — separate PR
- `reliability_eval/` scripts — not part of `hal` package
- `Environment.from_file` / `--environment` CLI flag — add when there is a call site
- `Environment.resource_config` — add when there is a concrete use
- New benchmark adapters
- Changes to `input.json` / `output.json` agent protocol

---

## Resolved Questions

- **Orchestration approach:** External scripts in `orchestrators/`. Core `hal` has no
  Docker/Azure SDK imports. Orchestrators invoke `hal-eval` as a subprocess (CLI boundary).
- **Trajectory storage:** Leave Weave as source of truth for now.
- **`AgentRunner.run()` decomposition:** Extract `_run_one_variation()` primitive and
  express all three modes in terms of it. Extracted methods take inputs as parameters.
- **`EnvironmentConfigError`:** Omit (YAGNI). `ValueError` is sufficient and consistent
  with existing validation raises in `AgentRunner.__init__`.
- **`Environment.resource_config`:** Omit (YAGNI). Add when a concrete consumer exists.
- **Registry pattern:** Import-path strings dict (not lambdas, not `__init_subclass__`).
  Lazy loading via `importlib.import_module` at lookup time.

---

## Sources & References

### Origin
**Brainstorm:** `docs/brainstorms/2026-02-26-architecture-cleanup-brainstorm.md`

Key decisions carried forward:
- Orchestration fully external — no Docker/Azure SDK in `hal/` package
- Single `Runner`, no backends protocol
- `benchmark_name` as explicit `BaseBenchmark.__init__` parameter
- `AgentRunner.run()` decomposed in this pass

### Internal References
- `hal/utils/local_runner.py` — source for new `Runner`
- `hal/utils/docker_runner.py:532-609` — `_create_runner_script` (identical to local)
- `hal/benchmark_manager.py:38-102` — if/elif chain being replaced
- `hal/agent_runner.py:115-138` — runner selection block being removed
- `hal/agent_runner.py:344-398` — f-string injection vector being fixed
- `hal/benchmarks/base_benchmark.py:27-33` — fragile `benchmark_name` ordering

### Related
- Linear: [AE-65](https://linear.app/agent-evals/issue/AE-65/clean-up-runner-architecture-so-its-external-to-the-core-codebase)

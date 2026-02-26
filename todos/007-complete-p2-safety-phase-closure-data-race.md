---
status: complete
priority: p2
issue_id: "007"
tags: [code-review, concurrency, quality]
---

# Closure over loop variable in `phases/safety.py` causes data race under `ThreadPoolExecutor`

## Problem Statement

`analyze_task` is defined inside a `for upload_file` loop in `phases/safety.py` and
closes over `constraints`, `analyzer`, and `safety_model`. Because `ThreadPoolExecutor`
may still be executing tasks from iteration N when iteration N+1 reassigns `constraints`,
tasks submitted for one agent may be evaluated against a different agent's constraints.

## Findings

- **`phases/safety.py:168`** — `analyze_task` defined inside outer `for agent_config`
  loop; closes over `constraints` (line 66) which is reassigned each iteration
- `constraints` can differ per agent config (`benchmark_config.get("compliance_constraints")`)
- `ThreadPoolExecutor` submits tasks that may outlive one iteration of the outer loop

## Proposed Solutions

**Option A: Extract to module-level function with explicit parameters**
```python
def _analyze_task(task_info, constraints, analyzer, safety_model):
    ...
executor.submit(_analyze_task, task_info, constraints, analyzer, safety_model)
```

**Option B: Capture by value with default argument**
```python
def analyze_task(task_info, _constraints=constraints, _analyzer=analyzer, ...):
```
Idiomatic but less readable than Option A.

**Recommended:** Option A — explicit, testable, no hidden captures.

## Technical Details

- Affected file: `reliability_eval/phases/safety.py`

## Acceptance Criteria

- [ ] `analyze_task` is a module-level function with explicit parameters
- [ ] No closure over outer loop variables
- [ ] Safety phase tests pass

## Work Log

- 2026-02-26: Identified by kieran-python-reviewer on PR #152

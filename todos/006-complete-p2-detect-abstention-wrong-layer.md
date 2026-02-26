---
status: complete
priority: p2
issue_id: "006"
tags: [code-review, architecture]
---

# `detect_abstention` is a pure function in the wrong layer (`phases/` vs `metrics/`)

## Problem Statement

`phases/abstention.py` contains `detect_abstention()`, which is a pure metric
computation function (takes text, returns a score) that belongs in `metrics/`.
The `phases/` subpackage should contain only I/O side-effects: spawning subprocesses,
reading/writing files. Because `detect_abstention` is in `phases/`, the package's
public `__init__.py` imports a pure function from a side-effectful module.

## Findings

- `reliability_eval/__init__.py:14` — imports `detect_abstention` from `phases.abstention`
- `phases/abstention.py` contains both `detect_abstention` (pure function, ~150 lines)
  and `run_abstention_phase` (I/O side-effectful runner)
- This creates a cross-layer dependency in the public API

## Proposed Solutions

**Option A: Move `detect_abstention` to `metrics/abstention.py`**
`phases/abstention.py` retains only `run_abstention_phase`. `metrics/abstention.py`
gets `detect_abstention`. Update `__init__.py` import. One-line change at call sites.

**Option B: Keep as-is with a re-export**
Add `from reliability_eval.phases.abstention import detect_abstention` to
`metrics/abstention.py` as a re-export. Minimal change but doesn't fix the layering.

**Recommended:** Option A — correct layering, minimal churn.

## Technical Details

- Affected files: `reliability_eval/phases/abstention.py`,
  `reliability_eval/metrics/abstention.py`,
  `reliability_eval/__init__.py`

## Acceptance Criteria

- [ ] `detect_abstention` lives in `metrics/abstention.py`
- [ ] `phases/abstention.py` contains only `run_abstention_phase`
- [ ] Public `__init__.py` imports from `metrics` not `phases`
- [ ] All tests pass

## Work Log

- 2026-02-26: Identified by architecture-strategist on PR #152

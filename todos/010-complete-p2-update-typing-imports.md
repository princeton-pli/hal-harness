---
status: complete
priority: p2
issue_id: "010"
tags: [code-review, quality, python]
---

# Legacy `typing` imports across entire package — modernize to Python 3.10+ style

## Problem Statement

The project targets Python 3.11+ but every module uses legacy `typing` imports
(`Dict`, `List`, `Optional`, `Tuple`) instead of built-in generics and union syntax.
This is a systematic issue across all new files introduced in the PR.

## Findings

Affected files (all new in this PR):
- `reliability_eval/types.py:6-8`
- `reliability_eval/metrics/consistency.py:6`
- `reliability_eval/metrics/agent.py:6`
- `reliability_eval/metrics/robustness.py:4`
- `reliability_eval/metrics/safety.py:5`
- `reliability_eval/phases/runner.py:7`
- `reliability_eval/phases/safety.py:8`
- `reliability_eval/loaders/results.py:7`

Also: `metrics/safety.py:11` has `safety_lambda: float = None` which is a type error
(`None` is not `float`); should be `float | None = None` or use the constant default.

## Proposed Solutions

**Option A: Mechanical sed-style replacement**
```
List[X]     → list[X]
Dict[X, Y]  → dict[X, Y]
Optional[X] → X | None
Tuple[X, Y] → tuple[X, Y]
```
Remove unused `from typing import Dict, List, Optional, Tuple`; keep `Any`.

**Option B: Run `pyupgrade --py310-plus`**
Automates the above. Run on `reliability_eval/` and `tests/reliability_eval/`.

**Recommended:** Option B if `pyupgrade` is available; Option A otherwise.

## Technical Details

- Affects all new `.py` files in `reliability_eval/`
- Also fix: `safety_lambda: float = None` → `safety_lambda: float | None = None`

## Acceptance Criteria

- [ ] No `from typing import Dict, List, Optional, Tuple` in new files
- [ ] `X | None` used instead of `Optional[X]` throughout
- [ ] `safety_lambda` type annotation corrected
- [ ] All 218 tests pass

## Work Log

- 2026-02-26: Identified by kieran-python-reviewer on PR #152

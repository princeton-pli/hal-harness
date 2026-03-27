---
status: pending
priority: p3
issue_id: "013"
tags: [code-review, quality, python]
---

# `sys.path` mutation at import time and blanket `warnings.filterwarnings("ignore")`

## Problem Statement

Two issues in `analyze_reliability.py` that affect callers:

1. `sys.path.insert(0, ...)` runs unconditionally at module import time, mutating the
   interpreter's path for all subsequent code in the process.
2. `warnings.filterwarnings("ignore")` silences all warnings process-wide at module
   scope, hiding legitimate warnings from other parts of the codebase.

## Findings

- **`analyze_reliability.py:41-42`** — `sys.path.insert` at module level
- **`analyze_reliability.py:107`** — `warnings.filterwarnings("ignore")` at module scope
- Both run whenever the module is imported, not just when `main()` is called

## Proposed Solutions

**Option A: Guard with `if __name__ == "__main__"`**
Both calls moved inside the `if __name__ == "__main__"` block.

**Option B: Proper packaging**
Remove `sys.path` hack entirely by ensuring `reliability_eval` is installed as a
package (already in `pyproject.toml`). Scope `filterwarnings` to specific categories
via `warnings.catch_warnings()` context manager around the relevant numpy calls.

**Recommended:** Option B for `sys.path`; `warnings.catch_warnings()` for the filter.

## Acceptance Criteria

- [ ] `sys.path.insert` does not run at module import time
- [ ] `warnings.filterwarnings("ignore")` is scoped to specific warnings or removed

## Work Log

- 2026-02-26: Identified by kieran-python-reviewer on PR #152

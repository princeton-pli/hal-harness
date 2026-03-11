---
status: complete
priority: p1
issue_id: "003"
tags: [code-review, quality, dead-code]
---

# `harm_ref` is dead code threaded through 4 files

## Problem Statement

`compute_safety_metrics()` accepts a `harm_ref` parameter and it is threaded through
`analyze_agent`, `analyze_all_agents`, and exposed as a CLI flag — but the parameter
is never read inside the function. The old formula that used it is commented out.
Every call site passes a value that has zero effect.

## Findings

- **`metrics/safety.py:11`** — `harm_ref: float = HARM_REF` accepted but never used
  after line 33 (old exponential-decay formula commented out at lines 130–138)
- **`metrics/agent.py:649-650`** — `harm_ref=HARM_REF` passed to `analyze_agent`
- **`metrics/agent.py:848-849`** — `harm_ref=HARM_REF` passed to `analyze_all_agents`
- **`analyze_reliability.py:157`** — `--harm_ref` CLI argument exposed to users
- **`constants.py:4`** — `HARM_REF` constant imported solely to pass as dead argument

## Proposed Solutions

**Option A: Remove entirely**
Delete `harm_ref` from `compute_safety_metrics`, `analyze_agent`,
`analyze_all_agents`, the CLI parser, and the `HARM_REF` constant.

**Option B: Keep constant, remove parameter threading**
Retain `HARM_REF` in `constants.py` as documentation of the old formula. Remove it
from all function signatures and the CLI.

**Recommended:** Option A — YAGNI; the old formula is commented out.

## Technical Details

- Affected files: `reliability_eval/metrics/safety.py`,
  `reliability_eval/metrics/agent.py`,
  `reliability_eval/analyze_reliability.py`,
  `reliability_eval/constants.py`

## Acceptance Criteria

- [ ] `harm_ref` parameter removed from `compute_safety_metrics`, `analyze_agent`,
      `analyze_all_agents`
- [ ] `--harm_ref` CLI flag removed from `analyze_reliability.py`
- [ ] `HARM_REF` constant removed (or kept with a comment explaining its history)
- [ ] All tests pass after removal

## Work Log

- 2026-02-26: Identified by code-simplicity-reviewer on PR #152

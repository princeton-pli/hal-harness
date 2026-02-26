---
status: complete
priority: p1
issue_id: "004"
tags: [code-review, quality, dry]
---

# Duplicate trajectory algorithms and dead `_failure` branch

## Problem Statement

Two related issues introduced by the refactor:

1. The Levenshtein distance + JSD consistency algorithms are each implemented twice —
   once in `metrics/agent.py` (used by level-stratified functions) and once in
   `metrics/consistency.py` (used by the main computation path).

2. Both trajectory consistency functions return a `(success_score, failure_score)`
   tuple, but the `_failure` score is computed and immediately discarded at every
   call site. This runs dead computation on every eval.

## Findings

**Duplication:**
- `metrics/agent.py:20-70` — JSD distribution consistency duplicates
  `consistency.py:compute_trajectory_consistency_conditioned`
- `metrics/agent.py:73-119` — Levenshtein + pairwise similarity duplicates
  `consistency.py:compute_sequence_consistency`
- Estimated ~75 lines of duplicated logic

**Dead `_failure` branch:**
- `consistency.py:97` — returns `(consistency_trajectory_sequenceuccess, C_traj_failure)`
  (also has typo: `sequenceuccess` should be `sequence_success`)
- Failure score captured at `consistency.py:507-509,521` and never used

## Proposed Solutions

**Option A: Remove duplicates from `agent.py`**
Have `compute_level_stratified_metrics` call the existing functions in
`consistency.py` directly. Remove `_compute_trajectory_distribution_consistency` and
`_compute_trajectory_sequence_consistency` from `agent.py`.

**Option B: Remove dead failure branch**
If `C_traj_failure` is genuinely not needed, simplify both functions to return a
single score. If it may be needed later, add it to `ReliabilityMetrics` or `extra`.

**Recommended:** Both A and B — do A first (no behavior change), then decide on B.

## Technical Details

- Affected files: `reliability_eval/metrics/agent.py`,
  `reliability_eval/metrics/consistency.py`
- Also fix typo: `consistency_trajectory_sequenceuccess` → `consistency_trajectory_sequence_success`

## Acceptance Criteria

- [ ] `agent.py` does not contain duplicate Levenshtein or JSD implementations
- [ ] Typo `consistency_trajectory_sequenceuccess` fixed
- [ ] Either `_failure` scores are used in output or dead branches removed
- [ ] All 218 tests pass

## Work Log

- 2026-02-26: Identified by code-simplicity-reviewer and kieran-python-reviewer on PR #152

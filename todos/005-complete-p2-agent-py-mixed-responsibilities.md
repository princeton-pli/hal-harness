---
status: complete
priority: p2
issue_id: "005"
tags: [code-review, architecture, quality]
---

# `metrics/agent.py` mixes orchestration, serialization, and GAIA-specific logic

## Problem Statement

`metrics/agent.py` contains four distinct responsibilities that belong in separate
layers: (1) metric orchestration (`analyze_agent`, `analyze_all_agents`), (2) DataFrame
serialization (`metrics_to_dataframe`), (3) GAIA-specific level stratification
(hardcoded levels `"1"`, `"2"`, `"3"`), and (4) a duplicate ECE implementation
(`compute_ece_for_level`) that already exists in `metrics/predictability.py`.

## Findings

- `analyze_agent` / `analyze_all_agents` are orchestrators, not metric computations —
  they belong at the entry-point layer or in a dedicated `analyzer.py`
- `metrics_to_dataframe` belongs near the presentation layer, not `metrics/`
- `compute_ece_for_level` at line 457 reimplements ECE from `predictability.py`
- Level-stratified functions use hardcoded `"1"`, `"2"`, `"3"` — GAIA-specific logic
  leaking into the general computation layer
- `analyze_all_agents` uses `print()` for progress throughout (~30 print calls)

## Proposed Solutions

**Option A: Extract to `analyzer.py`**
Move `analyze_agent`, `analyze_all_agents`, and `metrics_to_dataframe` to a new
`reliability_eval/analyzer.py`. Move level-stratified code to `metrics/levels.py`.
Have `compute_ece_for_level` call `compute_ece_metrics` from `predictability.py`.

**Option B: Move orchestrators to entry point**
Pull `analyze_agent` and `analyze_all_agents` into `analyze_reliability.py` directly
(appropriate if they are only called from one place).

**Recommended:** Option A — cleaner separation, avoids bloating the entry point.

## Technical Details

- Affected files: `reliability_eval/metrics/agent.py`,
  `reliability_eval/metrics/predictability.py`,
  `reliability_eval/analyze_reliability.py`
- New file: `reliability_eval/analyzer.py` (or `reliability_eval/metrics/levels.py`)

## Acceptance Criteria

- [ ] `metrics/` contains only pure stateless computations
- [ ] `compute_ece_for_level` calls `compute_ece_metrics` instead of reimplementing ECE
- [ ] GAIA-specific level logic isolated from general metric computation
- [ ] All tests pass

## Work Log

- 2026-02-26: Identified by architecture-strategist on PR #152

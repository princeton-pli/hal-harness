---
status: complete
priority: p2
issue_id: "011"
tags: [code-review, quality, dead-code]
---

# `PHASE_SETTINGS` in `config.py` is never consumed — wire it up or remove it

## Problem Statement

`config.py` defines a `PHASE_SETTINGS` dict with defaults for all phases (k_runs,
fault_rate, num_variations, etc.), but none of the phase runners read from it. All
defaults come from CLI argument parser defaults in `run_reliability_eval.py`. The dict
gives a false impression of being the authoritative configuration source.

## Findings

- **`config.py:428-461`** — `PHASE_SETTINGS` defined with phase-level defaults
- Comments like `# Will be overridden by --k argument` reveal it is never authoritative
- `phases/baseline.py`, `phases/fault.py`, etc. do not import `PHASE_SETTINGS`
- `"abstention"` key has only a description and no settings
- Only reference: `run_reliability_eval.py:436` passes
  `PHASE_SETTINGS["safety"]["constraints"]` as a fallback — the one functional use

## Proposed Solutions

**Option A: Remove `PHASE_SETTINGS`**
Delete the dict. Move the `constraints` fallback inline or to a constant. Simplest.

**Option B: Wire it up as the source of defaults**
Have `argparse` defaults read from `PHASE_SETTINGS` so the dict is actually
authoritative: `default=PHASE_SETTINGS["fault"]["fault_rate"]`.

**Recommended:** Option A — YAGNI; the CLI args are the actual config mechanism.

## Technical Details

- Affected file: `reliability_eval/config.py`, `reliability_eval/run_reliability_eval.py`

## Acceptance Criteria

- [ ] `PHASE_SETTINGS` either removed or actually wired as the source of CLI defaults
- [ ] No misleading configuration-shaped dead code
- [ ] All tests pass

## Work Log

- 2026-02-26: Identified by code-simplicity-reviewer on PR #152

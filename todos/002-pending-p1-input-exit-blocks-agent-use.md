---
status: pending
priority: p1
issue_id: "002"
tags: [code-review, agent-native, quality]
---

# `input()` and `exit()` in library functions block agent/CI use

## Problem Statement

`phases/runner.py:check_api_keys()` calls `input()` to prompt the user interactively,
then calls `exit(1)` on a negative answer. Any non-interactive caller — CI pipeline,
agent, test suite — hangs indefinitely on the `input()` call. Multiple `exit()` calls
in `run_reliability_eval.py` also bypass exception handling in programmatic callers.

## Findings

- **`phases/runner.py:80`** — `input("   Continue anyway? (y/n): ")` halts any
  non-interactive process permanently
- **`phases/runner.py:82`** — `exit(1)` inside a library function bypasses cleanup
- **`run_reliability_eval.py:243,255,350`** — bare `exit(1)` instead of exceptions
- Both `main()` functions call `argparse.parse_args()` with no `argv` override, making
  them impossible to call programmatically without subprocess overhead
- **`run_reliability_eval.py` line 357** — log path hardcoded to
  `"reliability_eval/reliability_eval_log.json"` relative to cwd; breaks if called
  from a different directory

## Proposed Solutions

**Option A: Minimal fix**
- Move `input()` prompt to `main()` only; `check_api_keys()` raises `RuntimeError`
  when keys are missing
- Replace `exit(1)` with `raise SystemExit(1)` in `main()`, raise exceptions elsewhere
- Accept `argv: list[str] | None = None` in both `main()` functions; pass to
  `parser.parse_args(argv)`

**Option B: Full agent-native refactor**
Same as A, plus derive `log_path` from `--results_dir` or accept `--log_path`
argument; add `run_evaluation()` and `run_analysis()` wrappers to `__init__.py`.

**Recommended:** Option A now; Option B as a follow-up.

## Technical Details

- Affected files: `reliability_eval/phases/runner.py`,
  `reliability_eval/run_reliability_eval.py`,
  `reliability_eval/analyze_reliability.py`

## Acceptance Criteria

- [ ] `check_api_keys()` raises an exception instead of calling `input()` or `exit()`
- [ ] Both `main()` functions accept an optional `argv` parameter
- [ ] `check_api_keys()` is callable in tests without mocking stdin
- [ ] Existing tests pass

## Work Log

- 2026-02-26: Identified by agent-native-reviewer and kieran-python-reviewer on PR #152

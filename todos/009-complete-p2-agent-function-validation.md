---
status: complete
priority: p2
issue_id: "009"
tags: [code-review, security]
---

# `agent_function` and `agent_dir` config values not validated before subprocess use

## Problem Statement

`phases/runner.py:build_base_command()` passes `agent_function` and `agent_dir` from
`config.py` directly as CLI arguments to `hal-eval` without validation. While
`shell=False` prevents shell injection, unvalidated values can still cause path
traversal (via `agent_dir`) or module-path abuse (via `agent_function`). The MEMORY.md
for this project already flagged this as a required fix that was never implemented.

## Findings

- **`phases/runner.py:103-114`** — `agent_config["agent_dir"]` and
  `agent_config["agent_function"]` passed to cmd list without checks
- `config.py` is designed to be user-edited; malformed entries silently flow through
- MEMORY.md / CLAUDE.md note: "`agent_function` needs allowlist regex validation"

## Proposed Solutions

**Option A: Allowlist regex + path containment check**
```python
import re, pathlib
_AGENT_FUNCTION_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*$')
def _validate_agent_config(agent_config):
    if not _AGENT_FUNCTION_RE.match(agent_config["agent_function"]):
        raise ValueError(f"Invalid agent_function: {agent_config['agent_function']}")
    agent_dir = Path(agent_config["agent_dir"]).resolve()
    if not agent_dir.is_relative_to(Path.cwd()):
        raise ValueError(f"agent_dir outside repo: {agent_dir}")
```
Call in `build_base_command` or at startup in `get_valid_combinations`.

**Option B: Validate only `agent_function`**
Minimal fix: validate the function path format. Leave `agent_dir` for follow-up.

**Recommended:** Option A — both fields need validation per MEMORY.md.

## Technical Details

- Affected file: `reliability_eval/phases/runner.py`

## Acceptance Criteria

- [ ] `agent_function` validated against `^[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*$`
- [ ] `agent_dir` validated as relative to repo root
- [ ] Invalid configs raise `ValueError` with a clear message at startup

## Work Log

- 2026-02-26: Identified by security-sentinel on PR #152; flagged in MEMORY.md prior

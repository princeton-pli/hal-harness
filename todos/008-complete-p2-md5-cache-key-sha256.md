---
status: complete
priority: p2
issue_id: "008"
tags: [code-review, security]
---

# MD5 used as safety analysis cache key — replace with SHA-256

## Problem Statement

`hal/utils/llm_log_analyzer.py` uses MD5 to cache LLM safety analysis results in-memory.
MD5 has well-known chosen-prefix collision attacks. An adversarial agent could craft a
trace that produces the same MD5 digest as a previously-analyzed clean trace, causing
the cached "safe" verdict to be returned for a violating task.

## Findings

- **`hal/utils/llm_log_analyzer.py:405-408`**:
  ```python
  combined = f"{analysis_type}:{self.model}:{content}"
  return hashlib.md5(combined.encode()).hexdigest()
  ```
- Cache is in-memory per run; risk is within a single `run_safety_phase` call
  with concurrent `ThreadPoolExecutor` task processing

## Proposed Solutions

**Option A: Replace MD5 with SHA-256 (recommended)**
```python
return hashlib.sha256(combined.encode()).hexdigest()
```
One-line change, negligible performance difference, eliminates collision risk.

**Option B: Use a tuple key directly**
```python
self._cache: dict[tuple[str, str, str], dict] = {}
# key = (analysis_type, self.model, content)
```
No hashing at all; uses Python dict equality. More memory, simpler code.

**Recommended:** Option A — minimal change, no memory trade-off.

## Technical Details

- Affected file: `hal/utils/llm_log_analyzer.py`

## Acceptance Criteria

- [ ] `hashlib.md5` replaced with `hashlib.sha256` in `_get_cache_key`
- [ ] Existing tests pass

## Work Log

- 2026-02-26: Identified by security-sentinel on PR #152

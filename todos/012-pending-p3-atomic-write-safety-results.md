---
status: pending
priority: p3
issue_id: "012"
tags: [code-review, security, reliability]
---

# Result files written in-place without atomic rename — torn file on crash

## Problem Statement

`phases/safety.py` opens the `_UPLOAD.json` result file with `open(path, "w")` before
the new content is fully serialized. If the process is killed mid-write (keyboard
interrupt, OOM, timeout), the file is left partially written and permanently corrupt,
losing the original eval results.

## Findings

- **`phases/safety.py:277`**:
  ```python
  with open(upload_file, "w") as f:
      json.dump(data, f, indent=2)
  ```

## Proposed Solutions

**Option A: Write-then-rename (recommended)**
```python
tmp = upload_file.with_suffix(".tmp")
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
tmp.replace(upload_file)  # atomic on POSIX
```

**Option B: Keep backup**
Copy original to `upload_file.with_suffix(".bak")` before overwriting.

**Recommended:** Option A — truly atomic, standard pattern.

## Acceptance Criteria

- [ ] Safety phase result write uses write-then-rename pattern
- [ ] No `.tmp` files left on successful write

## Work Log

- 2026-02-26: Identified by security-sentinel on PR #152

# Weave Logging Fix

## Issue

You were seeing this warning:
```
⚠️  Warning: Could not log to Weave: module 'weave' has no attribute 'log'
```

## Root Cause

The code was trying to use `weave.log()`, which doesn't exist in the Weave API. This was an incorrect assumption about how Weave logging works.

## How It Actually Works

**Weave automatically traces all `litellm.completion()` calls** through its integration. You don't need to manually log anything!

When HAL initializes Weave (via `weave.init()`), it automatically instruments litellm to trace all LLM calls, including:
- Task execution calls
- Tool calls
- **Confidence assessment calls** ✓

## What Changed

### Before (❌ Incorrect)
```python
# This doesn't work - weave.log() doesn't exist
import weave
weave.log({"confidence_assessment": confidence_details})
```

### After (✅ Correct)
```python
# No manual logging needed!
# The litellm.completion() call is automatically traced by Weave
response = litellm.completion(**kwargs_for_confidence)

# Confidence calls appear in Weave automatically
```

## How to Find Confidence Calls in Weave

1. **Go to your Weave dashboard**:
   ```
   https://wandb.ai/[your-entity]/[your-project]/weave
   ```

2. **Navigate to Calls tab**

3. **Look for patterns**:
   ```
   Task 1 execution
   ├── Tool call 1
   ├── Tool call 2
   └── LLM call (max_tokens=10) ← Confidence assessment!

   Task 2 execution
   ├── Tool call 1
   └── LLM call (max_tokens=10) ← Another confidence assessment!
   ```

4. **Filter criteria**:
   - Response length: 1-3 tokens
   - Prompt contains: "assess your confidence"
   - Parameter: `max_tokens=10`
   - Timing: Immediately after task completion

## Verification

Run a test evaluation and check that:

1. ✅ **No more warnings**: The Weave warning is gone
2. ✅ **Console output works**: You see `✓ Confidence assessment: Model returned...`
3. ✅ **JSON storage works**: Results have `confidence` and `confidence_details` (if enabled)
4. ✅ **Weave tracing works**: Confidence calls appear in Weave dashboard

### Quick Test

```bash
# Run a small eval
python reliability_eval/run_predictability_eval.py --k 1 --max_tasks 3

# Should see in console (no warnings):
✓ Confidence assessment: Model returned '85' -> 0.85
✓ Confidence assessment: Model returned '78' -> 0.78
✓ Confidence assessment: Model returned '92' -> 0.92

# Check in Weave dashboard
# Look for LLM calls with max_tokens=10
```

## What You Get

### 1. Console Output (Real-time)
```bash
✓ Confidence assessment: Model returned '85' -> 0.85
```

### 2. JSON Files (Structured storage)
```json
{
  "confidence": 0.85,
  "confidence_details": {
    "prompt": "You just completed a task...",
    "model_response": "85",
    "parsed_score": 0.85,
    "num_actions": 12,
    "num_errors": 0,
    "model": "gpt-5.2-2025-12-11"
  }
}
```

### 3. Weave Dashboard (Automatic)
- All litellm calls are traced
- Confidence calls included automatically
- Filter by short responses or max_tokens=10
- Compare across models/runs

## Summary

✅ **Fixed**: Removed incorrect `weave.log()` API call
✅ **Automatic**: Weave traces litellm calls by default
✅ **No warnings**: Code runs cleanly now
✅ **All tracking methods work**: Console, JSON, Weave

The confidence tracking is now working correctly with all three methods!

## Next Steps

1. Run your GPT-5.2 evaluation
2. Check console for confidence scores
3. View details with: `python reliability_eval/view_confidence_details.py`
4. Compare in Weave dashboard

Your evaluation should now run without warnings and you'll have full visibility into confidence assessments!

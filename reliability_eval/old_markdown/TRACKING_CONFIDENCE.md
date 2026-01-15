# Tracking Confidence Assessment Calls

This guide explains the three ways to track and inspect confidence assessment interactions in HAL.

## Overview

When you enable confidence scoring (`compute_confidence=True`), the agent makes a separate LLM call to assess its confidence after completing each task. You can track these calls in three ways:

1. **Console Output** - Real-time logging to stdout/stderr
2. **Results JSON** - Full details stored in output files
3. **Weave Dashboard** - Logged to Weave for analysis

## Method 1: Console Output (Simplest)

The agent prints confidence assessments to the console in real-time.

### Usage

Just run your evaluation normally:

```bash
python reliability_eval/run_predictability_eval.py --k 3 --max_tasks 10
```

### Output

You'll see lines like:

```
✓ Confidence assessment: Model returned '85' -> 0.85
✓ Confidence assessment: Model returned '72' -> 0.72
✓ Confidence assessment: Model returned '90' -> 0.90
```

### Capturing to File

```bash
# Capture all output
python reliability_eval/run_predictability_eval.py --k 3 --max_tasks 10 2>&1 | tee eval_output.log

# Then grep for confidence
grep "Confidence assessment" eval_output.log
```

### Pros & Cons

✅ **Pros**: Immediate, no setup, shows what model actually returned
❌ **Cons**: Not structured, hard to analyze, scattered in logs

---

## Method 2: Results JSON (Best for Analysis)

Store full confidence details (prompt + response) in the output JSON files.

### Setup

Add `store_confidence_details=True` to your agent config:

```python
# In run_predictability_eval.py
"extra_agent_args": {
    "provider": "openai",
    "temperature": 0.0,
    "compute_confidence": True,
    "store_confidence_details": True  # ← Add this
}
```

### Output Structure

Each task's `output.json` will contain:

```json
{
  "task_0": {
    "reward": 1.0,
    "confidence": 0.85,
    "confidence_details": {
      "prompt": "You just completed a task. Please assess your confidence...",
      "model_response": "85",
      "parsed_score": 0.85,
      "num_actions": 12,
      "num_errors": 0,
      "task_reward": 1.0,
      "model": "gpt-5.2-2025-12-11"
    },
    "taken_actions": [...],
    "task": {...}
  }
}
```

### Viewing Details

Use the provided script:

```bash
# View last 10 tasks
python reliability_eval/view_confidence_details.py \
    --benchmark taubench_airline

# View all tasks with full prompts
python reliability_eval/view_confidence_details.py \
    --benchmark taubench_airline \
    --max_items 0 \
    --verbose

# View specific run
python reliability_eval/view_confidence_details.py \
    --run_id taubench_airline_gpt_5_2_1767472256

# Analyze response distribution
python reliability_eval/view_confidence_details.py \
    --benchmark taubench_airline \
    --analyze
```

### Example Output

```
════════════════════════════════════════════════════════════════════════════════
Run: taubench_airline_gpt_5_2_1767472256
Task: 0
Confidence: 0.850
Reward (Success): 0.0

📝 Prompt (truncated):
   You just completed a task. Please assess your confidence in the correctness of your solution.

   Task Summary:
   Task: Your user id is mia_li_3668. You want to fly from New York to Seattle...
   Number of actions taken: 19

🤖 Model Response:
   '85'

📊 Metadata:
   Model: gpt-5.2-2025-12-11
   Actions: 19
   Errors: 0
   Parsed Score: 0.850
```

### Analyzing Results

The script can show response distribution:

```bash
python reliability_eval/view_confidence_details.py \
    --benchmark taubench_airline \
    --analyze
```

Output:
```
Response Distribution:
Response             Count      Success    Failure    Success Rate
───────────────────────────────────────────────────────────────────────
85                   45         12         33         26.7%
75                   23         15         8          65.2%
90                   12         10         2          83.3%
...

⚠️  WARNING: Only 3 unique responses found
   Model has poor confidence discrimination.
```

### Pros & Cons

✅ **Pros**: Structured, easy to analyze, includes full context
✅ **Pros**: Persisted with results, reproducible
✅ **Pros**: Can query/filter programmatically
❌ **Cons**: Increases JSON file size (~1-2KB per task)

---

## Method 3: Weave Dashboard (Best for Exploration)

Confidence assessments are automatically logged to Weave if available.

### How It Works

Weave automatically traces all `litellm.completion()` calls, including the confidence assessment call. When you run evaluations with Weave initialized (via `weave.init()` in HAL), the confidence assessment appears as a separate LLM call in your trace:

```
Task Execution Trace
├── Tool Call 1 (get_user_details)
├── Tool Call 2 (search_flights)
├── Tool Call 3 (book_reservation)
└── Confidence Assessment ← Automatically traced!
    ├── Model: gpt-5.2-2025-12-11
    ├── Prompt: "You just completed a task..."
    ├── Response: "85"
    └── Tokens: 10
```

No manual logging needed - HAL's Weave integration captures everything.

### Accessing in Weave

1. **Find your project URL**:
   ```bash
   # Check your .env file or wandb config
   echo $WANDB_ENTITY
   echo $WANDB_PROJECT
   ```

2. **Navigate to Weave**:
   ```
   https://wandb.ai/[your-entity]/[your-project]/weave
   ```

3. **Filter LLM calls**:
   - Look for calls with short prompts (~500 tokens) containing "assess your confidence"
   - These will have very short responses (usually just 1-3 tokens: "85", "90", etc.)
   - Filter by: `model`, `input.messages[].content` (contains "confidence")
   - Look at the timestamp - confidence calls happen right after task completion

4. **Identifying confidence calls**:
   ```
   In the Calls table, look for:
   - Short responses (1-3 tokens)
   - Prompts mentioning "assess your confidence"
   - max_tokens=10 in the request
   - Calls that follow immediately after task completion
   ```

### Weave API (Advanced)

You can query confidence calls programmatically using the Weave API:

```python
import weave

# Initialize Weave client
client = weave.init("your-entity/your-project")

# Get all LLM calls from a run
calls = client.get_calls(
    filter={"trace_id": "your-run-id"}
)

# Filter for confidence assessment calls
confidence_calls = []
for call in calls:
    # Check if this is a confidence assessment call
    if call.inputs and 'messages' in call.inputs:
        messages = call.inputs['messages']
        if any('assess your confidence' in str(msg).lower() for msg in messages):
            confidence_calls.append({
                'response': call.output.get('choices', [{}])[0].get('message', {}).get('content', ''),
                'tokens': call.summary.get('usage', {}).get('total_tokens', 0),
                'timestamp': call.started_at
            })

# Analyze
for call in confidence_calls:
    print(f"Response: {call['response']}")
    print(f"Tokens: {call['tokens']}")
```

Note: The exact API may vary depending on your Weave version. Check the [Weave documentation](https://wandb.github.io/weave/) for details.

### Pros & Cons

✅ **Pros**: Powerful filtering/grouping, visual dashboard
✅ **Pros**: Integrated with other Weave metrics (cost, latency)
✅ **Pros**: Can compare across runs/models easily
❌ **Cons**: Requires Weave setup and internet access
❌ **Cons**: May have API rate limits

---

## Comparison Matrix

| Feature | Console | JSON | Weave |
|---------|---------|------|-------|
| **Setup** | None | `store_confidence_details=True` | Weave account |
| **Real-time** | ✅ Yes | ❌ No | ⚠️ Near real-time |
| **Structured** | ❌ No | ✅ Yes | ✅ Yes |
| **Queryable** | ❌ No | ⚠️ Manual | ✅ Yes |
| **Persisted** | ⚠️ If captured | ✅ Yes | ✅ Yes |
| **File size** | 0 | +1-2KB/task | 0 |
| **Analysis tools** | grep/awk | Python script | Dashboard |
| **Best for** | Debugging | Batch analysis | Exploration |

---

## Recommended Workflow

### During Development/Debugging

Use **Console Output** to see immediate feedback:

```bash
python reliability_eval/run_predictability_eval.py --k 1 --max_tasks 3
# Watch for: ✓ Confidence assessment lines
```

### For Production Evaluations

Use **JSON Storage** for reproducible analysis:

```python
# In run_predictability_eval.py
"extra_agent_args": {
    "compute_confidence": True,
    "store_confidence_details": True  # ← Enable this
}
```

Then analyze:

```bash
python reliability_eval/view_confidence_details.py \
    --benchmark taubench_airline \
    --analyze
```

### For Cross-Run Comparisons

Use **Weave Dashboard** to compare models:

1. Run evaluations for multiple models
2. Go to Weave dashboard
3. Filter: `confidence_assessment`
4. Group by: `model`
5. Compare: Distribution of `model_response` vs `task_reward`

---

## Troubleshooting

### No confidence output visible

**Problem**: You don't see `✓ Confidence assessment` lines

**Solutions**:
1. Check `compute_confidence=True` in agent config
2. Check stderr: `hal-eval ... 2>&1 | grep Confidence`
3. Look in verbose log: `results/*/run_id/*_verbose.log`

### No confidence_details in JSON

**Problem**: JSON has `confidence` but no `confidence_details`

**Solution**: Add `store_confidence_details=True` to agent config

### Weave not logging confidence calls

**Problem**: Can't find confidence assessments in Weave

**Solutions**:
1. **Check Weave is initialized**: Look for Weave initialization logs at the start of your run
2. **Check litellm integration**: Weave must be initialized before any `litellm` calls
3. **Look in the right place**: Search for LLM calls with:
   - Very short responses (1-3 tokens)
   - Prompts containing "assess your confidence"
   - `max_tokens=10` parameter
4. **Check timing**: Confidence calls appear right after task completion
5. **Verify API key**: Check `WANDB_API_KEY` is set in `.env`
6. **Wait for sync**: Weave may take 1-2 minutes to sync data to the dashboard

**Finding confidence calls in Weave**:
- Go to your project's Calls tab
- Sort by timestamp (most recent first)
- Look for patterns: normal task call → confidence call → next task
- Filter by `inputs.max_tokens = 10`

### All responses identical

**Problem**: Model returns same confidence every time (e.g., "85")

**This is NOT a bug** - it's poor model calibration:
- Smaller models (GPT-4o-mini) have weak metacognition
- Try larger models (GPT-5.2, Claude Sonnet)
- Or improve the prompt (add few-shot examples)

---

## Next Steps

1. **Run a test evaluation**:
   ```bash
   python reliability_eval/run_predictability_eval.py --k 1 --max_tasks 5
   ```

2. **Check console output** for confidence lines

3. **View details**:
   ```bash
   python reliability_eval/view_confidence_details.py
   ```

4. **Analyze distribution**:
   ```bash
   python reliability_eval/view_confidence_details.py --analyze
   ```

5. **If all identical**, try a different model or improve the prompt

---

## Additional Resources

- **Predictability Analysis**: [analyze_predictability.py](analyze_predictability.py)
- **Evaluation Script**: [run_predictability_eval.py](run_predictability_eval.py)
- **User Guide**: [README_PREDICTABILITY.md](README_PREDICTABILITY.md)
- **Implementation**: [agents/taubench_tool_calling/tool_calling.py](../agents/taubench_tool_calling/tool_calling.py)

---

## Questions?

If confidence tracking isn't working as expected:

1. Run the diagnostic tool: `python reliability_eval/test_confidence.py`
2. Check the verbose log: `cat results/*/run_id/*_verbose.log | grep -i confidence`
3. Verify agent config has both flags:
   ```python
   "compute_confidence": True,
   "store_confidence_details": True
   ```

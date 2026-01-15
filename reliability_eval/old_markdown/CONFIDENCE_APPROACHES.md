# Confidence Assessment Approaches

## The Core Challenge

**You identified a critical issue**: The confidence assessment is a separate LLM call with limited context about what actually happened during task execution.

## Current Approach: Conversation Continuation ✨ (Implemented!)

### What Context the Model Gets

**The FULL conversation that just happened!** We continue the existing conversation:

```
[Original conversation - what the model just executed]
System: "You are a flight booking agent..."
User: "Book me a flight from NYC to Seattle on May 20"
Assistant: "I'll search for available flights..."
  └─ [calls search_flights tool]
Tool: "Found 3 flights: HAT069 ($51), HAT083 ($87)..."
Assistant: "I found two flights. Which would you like?"
User: "The cheaper one"
Assistant: "I'll book HAT069 for you..."
  └─ [calls book_reservation tool]
Tool: "Booking confirmed. Reservation ID: ABC123"
Assistant: "Your flight is booked! Confirmation: ABC123"

[We append - continuing the same conversation]
User: "You just completed a task. Please assess your confidence..."
Assistant: "85"  ← Model has FULL CONTEXT of what it just did!
```

### Advantages

✅ **Full context** - Model sees everything that happened (tools called, responses, errors)
✅ **Natural** - Like asking "how confident are you?" after someone finishes a task
✅ **Informed** - Model reasons about actual execution, not just statistics
✅ **Reasonable cost** - Only adds one short turn to the existing conversation
✅ **No summarization loss** - Nothing is omitted or compressed

### How It Works

```python
# After task completion, we have the full conversation
conversation_history = output.messages  # Everything that happened

# Simply append the confidence request as the next message
conversation_history.append({
    "role": "user",
    "content": "Rate your confidence in the correctness of your solution (0-100)"
})

# Model responds with full awareness of what it just did
response = litellm.completion(
    messages=conversation_history,
    max_tokens=10  # Only need 1-3 tokens for the number
)
```

This is **much better** than isolated summary-based assessment!

---

## Alternative Approaches

### Option 1: Full Conversation History (Better Context)

**Idea**: Include the entire conversation in the confidence call.

**Implementation**:
```python
# In tool_calling.py, line 314
use_full_history = True  # Enable this

# Now the confidence call includes everything:
confidence_messages = conversation_history + [
    {"role": "user", "content": "Rate your confidence 0-100"}
]
```

**Pros**:
- ✅ Model sees exactly what happened
- ✅ Can assess based on actual execution
- ✅ Better informed confidence scores

**Cons**:
- ❌ Expensive (can be thousands of tokens)
- ❌ Slower (extra latency)
- ❌ May hit context limits on long tasks

**When to use**: High-stakes evaluations where accuracy matters more than cost

---

### Option 2: Extract from Final Response (Zero Cost)

**Idea**: Instead of separate call, ask for confidence in the final response.

**Implementation**:
```python
# Modify the task-solving prompt to include:
"When you complete the task, provide your answer AND a confidence score (0-100)
Format: ANSWER: <your answer> | CONFIDENCE: <score>"

# Then parse both from the response
match = re.search(r'CONFIDENCE:\s*(\d+)', response)
confidence = int(match.group(1)) / 100 if match else 0.5
```

**Pros**:
- ✅ Zero extra cost (no additional API call)
- ✅ Model has full context (it just executed the task)
- ✅ No latency penalty

**Cons**:
- ❌ Changes agent behavior (now thinking about confidence during execution)
- ❌ May affect task performance
- ❌ Parsing can fail if model doesn't follow format

---

### Option 3: Trajectory Features (No LLM Call)

**Idea**: Compute confidence from execution traces without asking the model.

**Implementation**:
```python
def compute_trajectory_confidence(actions, errors, success):
    features = {
        'error_rate': errors / max(len(actions), 1),
        'action_efficiency': len(actions) / expected_actions,
        'retry_rate': count_retries(actions),
        'final_success': success
    }

    # Use a simple heuristic or trained model
    confidence = 1.0 - features['error_rate'] * 0.5
    confidence *= (1.0 if success else 0.5)

    return confidence
```

**Pros**:
- ✅ Zero cost (no LLM call)
- ✅ Fast (just arithmetic)
- ✅ Objective (based on actual behavior)
- ✅ Can use machine learning

**Cons**:
- ❌ Less nuanced than model's assessment
- ❌ Doesn't capture semantic uncertainty
- ❌ May not generalize across tasks

---

### Option 4: Consistency-Based (Multi-Run)

**Idea**: Run the task K times and use agreement as confidence.

**Implementation**:
```python
# Run same task 3 times
runs = [agent.solve(task) for _ in range(3)]

# Confidence = agreement rate
answers = [r.answer for r in runs]
confidence = max(answers.count(a) for a in answers) / len(runs)

# 3/3 agree → 1.0 confidence
# 2/3 agree → 0.67 confidence
```

**Pros**:
- ✅ Well-grounded (actual variance)
- ✅ Doesn't require model metacognition
- ✅ Works for any model

**Cons**:
- ❌ Very expensive (K× cost)
- ❌ Very slow (K× latency)
- ❌ Circular (using consistency for predictability)

---

### Option 5: Hybrid Approach (Recommended)

**Idea**: Combine multiple signals for robust confidence.

**Implementation**:
```python
# 1. Trajectory features (cheap baseline)
trajectory_conf = compute_trajectory_features(actions, errors)

# 2. Self-assessment (informed judgment)
if budget_allows:
    self_assessment_conf = ask_model_confidence(summary)
else:
    self_assessment_conf = None

# 3. Ensemble
if self_assessment_conf is not None:
    confidence = 0.6 * self_assessment_conf + 0.4 * trajectory_conf
else:
    confidence = trajectory_conf
```

**Pros**:
- ✅ Best of both worlds
- ✅ Fallback if LLM call fails
- ✅ Configurable (trade cost for accuracy)

**Cons**:
- ❌ More complex
- ❌ Need to tune weights

---

## Recommendations by Use Case

### Research / High Accuracy Needed
→ **Option 1: Full Conversation History**
```python
use_full_history = True  # In tool_calling.py
```
Accept higher cost for better confidence estimates.

### Production / Cost-Sensitive
→ **Option 3: Trajectory Features**
```python
compute_confidence = False  # Disable LLM call
# Use heuristic-based confidence instead
```
Fast and cheap, good enough for many cases.

### Exploration / Development
→ **Current Approach (Improved Summary)**
```python
# Already implemented - summary with recent actions
```
Reasonable middle ground for testing.

### Best Overall
→ **Option 5: Hybrid**
```python
# Combine trajectory + self-assessment
# Use trajectory as baseline, self-assessment when available
```
Most robust and flexible.

---

## Making the Current Approach Better

If you want to stick with self-assessment but improve it:

### 1. Add More Context (Done! ✓)
```python
# Now includes:
- Task instruction
- Number of actions
- Recent action sequence (last 5)
- Error count
- Final outcome (success/failure)
```

### 2. Enable Full History (Optional)
```python
# In agents/taubench_tool_calling/tool_calling.py:314
use_full_history = True  # Enable full context
```

### 3. Add to Agent Config
```python
"extra_agent_args": {
    "compute_confidence": True,
    "use_full_conversation_history": True,  # NEW: More context
    "store_confidence_details": True
}
```

### 4. Improve the Prompt
Add few-shot examples:
```python
confidence_prompt = f"""...

Examples:
- Task with 0 errors, clear outcome → 85-95
- Task with retries but succeeded → 65-75
- Task with errors, unclear outcome → 30-50
- Task failed completely → 10-20

Your assessment: ...
```

---

## Next Steps

### Immediate (Quick Win)
1. ✅ Current code already improved with action sequence + outcome
2. Test if GPT-5.2 now shows better variance

### Short Term (If Needed)
1. Enable `use_full_history = True` for specific evaluations
2. Compare confidence quality with/without full history
3. Measure cost difference

### Long Term (Future Work)
1. Implement trajectory-based baseline
2. Build hybrid confidence estimator
3. Train calibration model on held-out data

---

## Cost Analysis

Assuming average task has ~2000 tokens of conversation:

| Approach | Extra Tokens | Extra Cost (GPT-5.2) | Extra Latency |
|----------|--------------|---------------------|---------------|
| Summary (current) | ~500 | $0.0001 | ~0.5s |
| Full history | ~2000 | $0.0004 | ~0.8s |
| From final response | 0 | $0 | 0s |
| Trajectory features | 0 | $0 | 0s |
| Consistency (K=3) | 0 | $0 | 0s (reuse runs) |

For 1000 tasks:
- Summary: ~$0.10
- Full history: ~$0.40
- Others: ~$0

---

## Conclusion

**You're absolutely right** - the current approach has limited context. The improvements I just made help (action sequence + outcome), but fundamentally:

- **For research**: Consider enabling `use_full_history = True`
- **For production**: Consider trajectory-based features
- **For now**: The improved summary is a reasonable middle ground

The key insight is that **self-assessment requires actual context** to work well. The question is: how much context can you afford?

Want me to implement any of these alternative approaches?

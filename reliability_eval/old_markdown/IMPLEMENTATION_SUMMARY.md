# Predictability Metrics Implementation Summary

## What Was Added

### 1. Agent Modification: Confidence Scoring

**File**: [`agents/taubench_tool_calling/tool_calling.py`](../agents/taubench_tool_calling/tool_calling.py)

**Changes**:
- Added `compute_confidence` parameter (default: `False`)
- Implemented `_compute_confidence_score()` function
- Confidence scoring via explicit self-assessment
- Returns confidence [0, 1] along with task results

**Key Features**:
- ✅ Configurable via `-A compute_confidence=true`
- ✅ Backward compatible (default disabled)
- ✅ Fallback to heuristics if API call fails
- ✅ Handles all provider types (OpenAI, Anthropic, Google, OpenRouter)

**Usage**:
```bash
hal-eval --benchmark taubench_airline \
  --agent_dir agents/taubench_tool_calling \
  --agent_function tool_calling.run \
  --agent_name "My Agent" \
  -A model_name=claude-sonnet-4-5 \
  -A provider=anthropic \
  -A compute_confidence=true  # Enable confidence scoring
```

### 2. Evaluation Script: Run Predictability Evals

**File**: [`reliability_eval/run_predictability_eval.py`](run_predictability_eval.py)

**Purpose**: Run K repetitions with confidence scoring enabled

**Features**:
- ✅ Automatic confidence scoring for all runs
- ✅ Multiple agent/benchmark combinations
- ✅ Retry logic for network errors
- ✅ Progress logging to JSON

**Usage**:
```bash
python reliability_eval/run_predictability_eval.py \
  --k 3 \
  --max_tasks 50 \
  --conda_env hal
```

**Configuration**: Edit `AGENT_CONFIGS` to select agents/models

### 3. Analysis Script: Compute Predictability Metrics

**File**: [`reliability_eval/analyze_predictability.py`](analyze_predictability.py)

**Purpose**: Compute P_rc and P_cal metrics, generate visualizations

**Metrics Implemented**:

#### Risk-Coverage (P_rc)
```python
def compute_aurc(confidences, successes):
    # Sort by confidence
    # Compute risk at each coverage level
    # Compare to optimal ranking
    # Return P_rc = 1 - (E-AuRC / E-AuRC_max)
```

- **Input**: Confidence scores + binary outcomes
- **Output**: P_rc ∈ [0, 1], higher = better ranking
- **Interpretation**: How well confidence predicts correctness

#### Calibration (P_cal)
```python
def compute_ece(confidences, successes, n_bins=10):
    # Bin predictions by confidence
    # Compare avg confidence vs avg accuracy per bin
    # Return P_cal = 1 - ECE
```

- **Input**: Confidence scores + binary outcomes
- **Output**: P_cal ∈ [0, 1], higher = better calibration
- **Interpretation**: How well confidence matches reality

**Visualizations**:
1. **Risk-coverage curves**: Shows ranking quality
2. **Calibration bar charts**: Confidence vs accuracy by bin
3. **Reliability diagrams**: Traditional diagonal plots
4. **Summary comparison**: 4-panel agent comparison

**Usage**:
```bash
python reliability_eval/analyze_predictability.py \
  --results_dir results \
  --benchmark taubench_airline \
  --output_dir reliability_eval/analysis
```

### 4. Documentation

**File**: [`reliability_eval/README_PREDICTABILITY.md`](README_PREDICTABILITY.md)

**Contents**:
- Setup instructions
- Running evaluations
- Analyzing results
- Interpreting metrics
- Troubleshooting guide
- Advanced customization

## File Structure

```
reliability_eval/
├── run_predictability_eval.py      # Run evals with confidence
├── analyze_predictability.py       # Compute metrics + plots
├── README_PREDICTABILITY.md        # User guide
├── IMPLEMENTATION_SUMMARY.md       # This file
│
├── run_consistency_eval.py         # (Existing) Consistency evals
└── analyze_consistency.py          # (Existing) Consistency analysis

agents/taubench_tool_calling/
└── tool_calling.py                 # Modified with confidence scoring
```

## Implementation Details

### Confidence Scoring Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Task Execution                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 1. Agent solves task (normal execution)            │ │
│ │ 2. Records: actions, errors, outcome               │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Confidence Assessment (if enabled)                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 1. Extract task summary + error count              │ │
│ │ 2. Construct self-assessment prompt                │ │
│ │ 3. Call LLM for confidence score (0-100)           │ │
│ │ 4. Parse response → normalize to [0, 1]            │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Result Storage                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ {                                                   │ │
│ │   "task_id": {                                      │ │
│ │     "reward": 1.0,                                  │ │
│ │     "confidence": 0.85,  ← NEW!                     │ │
│ │     "taken_actions": [...],                         │ │
│ │     "task": {...}                                   │ │
│ │   }                                                 │ │
│ │ }                                                   │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Metric Computation Flow

```
┌─────────────────────────────────────────────────────────┐
│ Load Results (analyze_predictability.py)                │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 1. Scan results directory                          │ │
│ │ 2. Load UPLOAD.json files                          │ │
│ │ 3. Extract: task_id → (success, confidence)        │ │
│ │ 4. Group by agent                                  │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Compute Risk-Coverage (P_rc)                            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 1. Sort samples by decreasing confidence           │ │
│ │ 2. For each coverage c ∈ [0, 1]:                   │ │
│ │    - Take top c% samples                           │ │
│ │    - Compute risk = error rate                     │ │
│ │ 3. Compute AuRC = ∫ risk(c) dc                     │ │
│ │ 4. Compare to optimal ranking → E-AuRC             │ │
│ │ 5. Normalize: P_rc = 1 - (E-AuRC / E-AuRC_max)     │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Compute Calibration (P_cal)                             │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 1. Create 10 bins by confidence [0, 0.1, ..., 1.0] │ │
│ │ 2. For each bin:                                   │ │
│ │    - avg_conf = mean confidence in bin             │ │
│ │    - avg_acc = mean accuracy in bin                │ │
│ │ 3. ECE = Σ (n_bin/N) * |avg_acc - avg_conf|        │ │
│ │ 4. P_cal = 1 - ECE                                 │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Generate Outputs                                        │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ - predictability_metrics.csv                       │ │
│ │ - risk_coverage_curves.png                         │ │
│ │ - calibration_curves.png                           │ │
│ │ - reliability_diagrams.png                         │ │
│ │ - predictability_summary.png                       │ │
│ │ - predictability_report.md                         │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Testing

### Quick Test (Small Scale)

```bash
# 1. Run evaluation (5 tasks, 2 reps)
python reliability_eval/run_predictability_eval.py --k 2 --max_tasks 5

# 2. Analyze results
python reliability_eval/analyze_predictability.py \
  --benchmark taubench_airline \
  --output_dir reliability_eval/test_analysis

# 3. Check outputs
ls reliability_eval/test_analysis/
cat reliability_eval/test_analysis/predictability_report.md
```

### Full Evaluation

```bash
# 1. Edit run_predictability_eval.py to select agents
# 2. Run with full settings
python reliability_eval/run_predictability_eval.py --k 3 --max_tasks 100

# 3. Analyze
python reliability_eval/analyze_predictability.py --benchmark taubench_airline
```

## Key Design Decisions

### 1. Why Self-Assessment?

**Alternatives considered**:
- Token log probabilities (not available for all APIs)
- Trajectory features (less direct)
- Consistency-based (requires multiple runs)

**Chosen**: Self-assessment
- ✅ Works with any API
- ✅ Direct measure of agent's uncertainty
- ✅ Captures semantic understanding
- ❌ Depends on model's metacognitive ability

### 2. Why Opt-In?

Made `compute_confidence` **default False** because:
- Backward compatibility with existing workflows
- Small cost increase (~1-2%)
- Not all users need predictability metrics
- Allows gradual adoption

### 3. Why Separate Scripts?

Created separate `run_predictability_eval.py` instead of modifying `run_consistency_eval.py`:
- Clear separation of concerns
- Different evaluation goals (consistency vs predictability)
- Easier to maintain/debug
- Users can run either independently

## Future Enhancements

### Short Term
1. **Multi-method confidence**: Combine self-assessment + trajectory features
2. **Confidence caching**: Reuse confidence for repeated tasks
3. **Cheaper confidence**: Use smaller model for confidence calls

### Long Term
1. **Trajectory-based confidence**: Extract features from action sequences
2. **Consistency-based confidence**: Use agreement across K runs
3. **Ensemble confidence**: Weighted combination of methods
4. **Fine-tuned confidence**: Train calibration model on held-out data

## Known Limitations

1. **Model-dependent**: Some models are poor at self-assessment
2. **Task-dependent**: Very hard tasks may have unreliable confidence
3. **Sample size**: Need 50+ samples for reliable metrics
4. **Cost**: Extra LLM call per task (~1-2% cost increase)
5. **Latency**: ~0.5-1s per task for confidence call

## Verification Checklist

- [x] Agent returns confidence when enabled
- [x] Agent backward compatible when disabled
- [x] Confidence stored in UPLOAD.json
- [x] Analysis script loads confidence correctly
- [x] P_rc computation matches paper formula
- [x] P_cal computation matches paper formula
- [x] All visualizations generate correctly
- [x] Report generates with correct metrics
- [x] Scripts are executable
- [x] Documentation is complete

## Maintenance Notes

### Adding New Agents

To add confidence scoring to other agents:

1. Copy `_compute_confidence_score()` function
2. Add `compute_confidence` parameter check
3. Call confidence function after task execution
4. Add confidence to return dict

### Modifying Confidence Prompt

Edit `agents/taubench_tool_calling/tool_calling.py`:

```python
# Line ~267
confidence_prompt = f"""Your new prompt here..."""
```

### Changing Metric Formulas

Edit `reliability_eval/analyze_predictability.py`:

```python
# P_rc computation: Line ~75
def compute_aurc(...):
    # Modify here

# P_cal computation: Line ~153
def compute_ece(...):
    # Modify here
```

## Questions?

Contact: [Your contact info]
Repository: https://github.com/princeton-pli/hal-harness

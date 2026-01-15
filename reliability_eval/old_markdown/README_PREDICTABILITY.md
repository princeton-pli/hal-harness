# Predictability Evaluation Guide

This guide explains how to run predictability evaluations with confidence scoring in HAL.

## Overview

Predictability metrics measure how well an agent knows when it will succeed or fail. We compute two key metrics:

1. **Risk-Coverage (P_rc)**: How well confidence scores rank predictions by correctness
2. **Calibration (P_cal)**: How well confidence estimates match empirical success rates

## Setup

### 1. Agent Modification

The taubench tool calling agent now supports optional confidence scoring via the `compute_confidence` parameter:

```python
# In your agent kwargs
-A compute_confidence=true
```

When enabled, the agent will:
- Complete the task normally
- Ask the model to self-assess its confidence (0-100)
- Return confidence score along with the answer

### 2. Confidence Scoring Method

The agent uses **explicit self-assessment**:
- After completing a task, asks the model: "How confident are you that your answer is correct?"
- Model considers: errors encountered, task clarity, information reliability
- Returns a score 0-100 (converted to 0-1)

## Running Evaluations

### Quick Start

Run predictability evaluation with default settings:

```bash
cd /path/to/hal-harness
python reliability_eval/run_predictability_eval.py --k 3 --max_tasks 50
```

### Configuration

Edit `run_predictability_eval.py` to configure:

```python
AGENT_CONFIGS = [
    {
        "name": "taubench_toolcalling_claude_sonnet_4_5",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "claude-sonnet-4-5",
        "benchmarks": ["taubench_airline"],
        "extra_agent_args": {
            "provider": "anthropic",
            "temperature": 0.0,
            "compute_confidence": True  # Enable confidence!
        }
    },
]
```

### Command Line Options

```bash
python reliability_eval/run_predictability_eval.py \
    --k 3 \                      # Number of repetitions per task
    --max_tasks 50 \              # Max tasks per benchmark
    --conda_env hal               # Optional conda environment
```

## Analyzing Results

### Basic Analysis

After running evaluations:

```bash
python reliability_eval/analyze_predictability.py \
    --results_dir results \
    --benchmark taubench_airline \
    --output_dir reliability_eval/analysis
```

### Output Files

The analysis generates:

1. **predictability_metrics.csv**: Agent-level summary table
2. **risk_coverage_curves.png**: Risk-coverage curves showing confidence ranking quality
3. **calibration_curves.png**: Bar charts comparing confidence vs accuracy by bin
4. **reliability_diagrams.png**: Traditional diagonal calibration plots
5. **predictability_summary.png**: 4-panel summary comparison
6. **predictability_report.md**: Detailed markdown report

### Interpreting Metrics

#### P_rc (Risk-Coverage Score)
- Range: [0, 1], higher is better
- Measures: How well confidence ranks predictions
- **P_rc = 1.0**: Perfect ranking (agent always knows when it's right/wrong)
- **P_rc = 0.5**: Moderate ranking ability
- **P_rc = 0.0**: Random ranking (confidence is useless)

**Use case**: Selective prediction - trust top 50% most confident predictions

#### P_cal (Calibration Score)
- Range: [0, 1], higher is better
- Measures: How well confidence matches reality
- **P_cal = 1.0**: Perfect calibration (80% confidence → 80% success)
- **P_cal = 0.5**: Moderate calibration error
- **P_cal = 0.0**: Terrible calibration

**Use case**: Risk estimation - when agent says "90% confident", is it really?

## Example Workflow

### 1. Run evaluation with confidence

```bash
# Edit AGENT_CONFIGS in run_predictability_eval.py
# Set compute_confidence=True for desired agents

python reliability_eval/run_predictability_eval.py --k 3 --max_tasks 100
```

### 2. Analyze results

```bash
python reliability_eval/analyze_predictability.py \
    --benchmark taubench_airline \
    --output_dir reliability_eval/analysis
```

### 3. Review outputs

```bash
# View summary metrics
cat reliability_eval/analysis/predictability_metrics.csv

# View detailed report
cat reliability_eval/analysis/predictability_report.md

# View visualizations
open reliability_eval/analysis/risk_coverage_curves.png
open reliability_eval/analysis/calibration_curves.png
```

## Troubleshooting

### No confidence scores in results

**Problem**: Analysis shows "NO CONFIDENCE SCORES"

**Solution**:
- Check that `compute_confidence=true` is set in agent kwargs
- Verify the agent code was updated with confidence scoring
- Check API logs for confidence assessment calls

### Low P_rc or P_cal scores

**Problem**: Agent has poor predictability metrics

**Possible causes**:
1. **Model limitations**: Some models are poor at self-assessment
2. **Task difficulty**: Very hard tasks may have unreliable confidence
3. **Sample size**: Need sufficient data (50+ samples minimum)

**Solutions**:
- Try different models (Claude models often have good self-assessment)
- Increase sample size with more tasks or repetitions
- Consider ensemble confidence (combine self-assessment + trajectory features)

### Confidence scores all similar

**Problem**: All confidence scores are 70-80, little variance

**Cause**: Model is overconfident or underconfident uniformly

**Solution**:
- Modify the confidence prompt to encourage more differentiation
- Consider alternative confidence methods (trajectory features, consistency-based)

## Advanced: Customizing Confidence Scoring

### Modify the prompt

Edit `agents/taubench_tool_calling/tool_calling.py`:

```python
confidence_prompt = f"""Your custom prompt here...

Consider:
1. Your custom criteria
2. More custom criteria

Rate 0-100."""
```

### Add trajectory features

Enhance confidence with trajectory analysis:

```python
# Count errors
num_errors = sum(1 for action in actions_taken if hasattr(action, 'error'))

# Adjust confidence based on errors
adjusted_confidence = base_confidence * (1 - 0.1 * num_errors)
```

### Use ensemble confidence

Combine multiple signals:

```python
confidence = (
    0.5 * self_assessment_score +
    0.3 * trajectory_feature_score +
    0.2 * consistency_score
)
```

## Integration with HAL

### Automatic confidence storage

Confidence scores are automatically stored in `UPLOAD.json`:

```json
{
  "raw_eval_results": {
    "task_1": {
      "reward": 1.0,
      "confidence": 0.85,
      ...
    }
  }
}
```

### Backward compatibility

Confidence scoring is **opt-in**:
- Default: `compute_confidence=false` (no confidence computed)
- Enable: `-A compute_confidence=true`
- Old results without confidence work fine with new code

## Cost Considerations

Confidence scoring adds one extra LLM call per task:
- **Tokens per confidence call**: ~10-20 output tokens
- **Cost impact**: Minimal (~1-2% increase)
- **Latency impact**: ~0.5-1 second per task

For cost-sensitive evaluations, consider:
1. Use confidence on subset of runs
2. Use cheaper model for confidence (e.g., GPT-4o-mini)
3. Cache confidence scores (same task → same confidence)

## Citation

If you use these predictability metrics, please cite:

```bibtex
@article{reliability-framework-2025,
  title={A Conceptual Framework for LLM Agent Reliability},
  author={Your Name},
  year={2025}
}
```

## Questions?

For issues or questions:
- Check [main README](../README.md)
- See [CLAUDE.md](../CLAUDE.md) for HAL usage
- Review example agents in `agents/`

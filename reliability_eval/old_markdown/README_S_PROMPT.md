# S_prompt: Prompt Sensitivity Metric

## Overview

**S_prompt (Prompt Robustness)** measures how consistent an agent's performance is across semantically-equivalent but differently-phrased prompts. Higher S_prompt indicates greater robustness to prompt variations.

**Formula**: `S_prompt = 1 - variance(performance across variations)`

**Range**: [0, 1] where:
- **1.0** = Perfect robustness (identical performance across all prompt variations)
- **0.0** = Maximum sensitivity (performance highly dependent on phrasing)

---

## Why S_prompt Matters

Prompt sensitivity is critical for real-world deployment because:

1. **User Diversity**: Real users phrase requests differently
2. **Robustness**: Production systems need consistent behavior regardless of wording
3. **Reliability**: Sensitive agents are unpredictable and harder to trust
4. **Fairness**: Performance shouldn't depend on how users happen to phrase queries

**Example**: An agent that solves a task when asked "Calculate the total" but fails when asked "Find the sum" has low prompt robustness.

---

## Implementation Status

### ✅ Fully Implemented

**Evaluation Pipeline**:
- [`run_prompt_sensitivity_eval.py`](run_prompt_sensitivity_eval.py) - Run evaluations with prompt variations
- [`analyze_prompt_sensitivity.py`](analyze_prompt_sensitivity.py) - Analyze results and compute S_prompt
- [`hal/utils/prompt_variation.py`](../hal/utils/prompt_variation.py) - Generate semantic-preserving variations

**Metrics Computed**:
- **S_prompt**: Agent-level robustness score
- **Task variance**: Per-task sensitivity
- **Min-max gap**: Performance range across variations
- **S_task**: Task-level robustness scores

**Visualizations**:
- Agent comparison plots (S_prompt, variance, min-max gap)
- Task-level distributions
- Most sensitive tasks identification
- Performance vs sensitivity scatter plots

---

## How It Works

### 1. Variation Generation

Uses GPT-4o-mini to generate semantic-preserving paraphrases:

```python
from hal.utils.prompt_variation import PromptVariationGenerator

generator = PromptVariationGenerator(num_variations=3)

# Original: "Calculate the total revenue for Q4"
variations = generator.generate_variations(original_prompt)

# Generated variations:
# 1. "Calculate the total revenue for Q4"  (original)
# 2. "Compute the overall income for the fourth quarter"
# 3. "What is the sum of revenues during Q4?"
# 4. "Determine total Q4 revenue"
```

**Variation Strategy**:
- Preserve exact semantic meaning
- Vary style (formal/casual, verbose/concise)
- Maintain all critical information
- Different sentence structures and word choices

### 2. Evaluation

Runs agent on all variations of each task:

```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/my_agent \
    --agent_function main.run \
    --agent_name "test_agent" \
    --prompt_sensitivity \
    --num_variations 3 \
    --max_tasks 20
```

**Process**:
1. For each task, generate N variations (+ original)
2. Run agent on all N+1 versions
3. Collect performance scores for each variation
4. Store in `prompt_sensitivity_metrics` in UPLOAD.json

### 3. Analysis

Computes sensitivity metrics from variation results:

```python
# Per task
variance_t = var(scores across variations for task t)
S_task_t = 1 - variance_t

# Per agent
S_prompt = 1 - mean(variance_t for all tasks)
```

**Additional Metrics**:
- **Mean variance**: Average performance variance across tasks
- **Std variance**: Standard deviation of variances
- **Mean min-max gap**: Average performance range
- **Max min-max gap**: Worst-case performance swing

---

## Usage Guide

### Step 1: Run Evaluation with Prompt Variations

```bash
# Run with prompt sensitivity enabled
python reliability_eval/run_prompt_sensitivity_eval.py \
    --num_variations 3 \
    --max_tasks 20
```

**Configuration** (edit `run_prompt_sensitivity_eval.py`):
- `AGENT_CONFIGS`: List of agents to evaluate
- `BENCHMARK_CONFIGS`: Supported benchmarks
- Number of variations per prompt
- Maximum tasks to evaluate

**Supported Benchmarks**:
- ✅ TauBench (airline, retail)
- ✅ GAIA
- ✅ SWE-bench
- ✅ USACO
- ✅ AssistantBench
- ✅ SciCode
- ✅ AppWorld

### Step 2: Analyze Results

```bash
# Analyze prompt sensitivity
python reliability_eval/analyze_prompt_sensitivity.py \
    --results_dir results/ \
    --benchmark taubench_airline \
    --output_dir reliability_eval/analysis
```

**Outputs**:
- `task_level_sensitivity.csv`: Per-task variance, S_task, mean score, min-max gap
- `agent_level_sensitivity.csv`: Agent-level S_prompt, aggregated statistics
- `sensitivity_comparison.png`: 4-panel agent comparison
- `task_sensitivity_distribution.png`: Task-level distributions
- `most_sensitive_tasks.png`: Top sensitive tasks
- `sensitivity_report.md`: Comprehensive analysis report

### Step 3: Interpret Results

**High S_prompt (≥ 0.9)**:
- Very robust to prompt variations
- Consistent performance regardless of phrasing
- Good: Low operational risk, predictable behavior

**Medium S_prompt (0.7 - 0.9)**:
- Moderately robust
- Some performance variation with phrasing
- Caution: May need prompt engineering

**Low S_prompt (< 0.7)**:
- Sensitive to prompt phrasing
- Significant performance swings
- Risk: Unpredictable, hard to deploy reliably

---

## Example Results

### Agent Comparison

```
| Agent                  | S_prompt | Mean Var | Mean Gap |
|------------------------|----------|----------|----------|
| gpt-4o                 | 0.947    | 0.053    | 0.12     |
| claude-sonnet-4        | 0.923    | 0.077    | 0.18     |
| gpt-4o-mini            | 0.891    | 0.109    | 0.24     |
| gemini-2.0-flash       | 0.857    | 0.143    | 0.31     |
```

**Interpretation**:
- GPT-4o is most robust (S_prompt = 0.947)
- Gemini Flash is most sensitive (S_prompt = 0.857)
- All agents show moderate to high robustness
- Mean gap indicates typical performance range

### Task-Level Analysis

**Most Sensitive Tasks** (high variance):
1. Task 23: variance = 0.421 (agent struggles with different phrasings)
2. Task 7: variance = 0.387 (performance highly dependent on wording)
3. Task 41: variance = 0.356 (inconsistent across variations)

**Most Robust Tasks** (low variance):
1. Task 12: variance = 0.003 (nearly identical performance)
2. Task 5: variance = 0.008 (very consistent)
3. Task 30: variance = 0.012 (minimal variation)

---

## Integration with Other Metrics

S_prompt complements other reliability metrics:

**vs C_out (Outcome Consistency)**:
- C_out: Consistency across identical prompts (random sampling)
- S_prompt: Consistency across varied prompts (semantic equivalence)
- Both needed: An agent can be consistent (high C_out) but sensitive (low S_prompt)

**vs R_fault (Fault Robustness)**:
- R_fault: Performance under API/tool failures
- S_prompt: Performance under prompt variations
- Together measure robustness to different perturbation types

**Relationship**:
```
High C_out + High S_prompt = Deterministic AND robust
High C_out + Low S_prompt  = Deterministic BUT brittle
Low C_out + High S_prompt  = Stochastic BUT robust
Low C_out + Low S_prompt   = Stochastic AND brittle (worst)
```

---

## Technical Details

### Variance Calculation

For each task with K variations:

```python
scores = [score_0, score_1, ..., score_K]  # Scores for each variation
variance_task = np.var(scores, ddof=1)      # Sample variance
S_task = 1 - variance_task                   # Task robustness
```

Agent-level:

```python
all_variances = [variance_task_1, variance_task_2, ...]
S_prompt = 1 - np.mean(all_variances)
```

### Prompt Variation Quality

Generated variations are:
- **Semantic-preserving**: Same meaning, different wording
- **Natural**: Fluent and realistic
- **Diverse**: Different styles (formal/casual, verbose/concise)
- **Complete**: All critical information preserved

**Quality Control**:
- Uses GPT-4o-mini with carefully crafted system prompt
- Filters out malformed or too-short variations
- Includes original prompt as baseline
- Typical generation success rate: >95%

### Limitations

1. **Computational Cost**: Requires N+1 evaluations per task
   - Mitigation: Use smaller N (3-5 variations typical)

2. **Variation Quality**: Depends on LLM generation
   - Mitigation: GPT-4o-mini is reliable for paraphrasing

3. **Binary Tasks**: Variance may not capture sensitivity well
   - Mitigation: Analyze min-max gap for binary outcomes

4. **Benchmark Support**: Not all benchmarks have extractable prompts
   - Limitation: Some benchmarks use environment-driven prompts

---

## Best Practices

### For Evaluation

1. **Use 3-5 variations**: Good balance of coverage vs cost
2. **Test on diverse benchmarks**: Different task types show different sensitivities
3. **Compare to baseline**: Run without variations to ensure overhead is acceptable
4. **Check variation quality**: Manually inspect a few to verify semantic preservation

### For Agent Development

1. **Target S_prompt ≥ 0.85**: Good robustness for production
2. **Identify sensitive tasks**: Focus improvement on high-variance tasks
3. **Test edge cases**: Add variations with different:
   - Formality levels
   - Verbosity
   - Terminology choices
   - Sentence structures

4. **Use diverse training data**: Expose agents to varied phrasings

### For Deployment

1. **Monitor S_prompt in production**: Track performance across user phrasings
2. **Provide prompt templates**: For low-S_prompt agents, guide users
3. **Set confidence thresholds**: Low-S_prompt agents may need human review
4. **A/B test prompts**: Find robust phrasings before deployment

---

## Troubleshooting

### No sensitivity metrics in results

**Problem**: `analyze_prompt_sensitivity.py` finds no sensitivity metrics

**Solutions**:
1. Ensure `--prompt_sensitivity` flag was used during evaluation
2. Check that `UPLOAD.json` contains `prompt_sensitivity_metrics` field
3. Verify benchmark supports prompt variations (has extractable prompt field)

### High variance for all agents

**Problem**: All agents show low S_prompt scores

**Solutions**:
1. Check variation quality - may not be semantic-preserving
2. Increase number of variations for better statistics
3. Verify task evaluation is deterministic (temperature=0)
4. Check if benchmark has inherently ambiguous tasks

### Some tasks have no variance

**Problem**: Some tasks show variance = 0

**Solutions**:
1. Normal if all variations produce identical results
2. For binary tasks, this indicates deterministic correct/incorrect
3. Check if task has extractable prompt (some may use defaults)

---

## Future Enhancements

**Planned**:
1. ✅ Support for all HAL benchmarks
2. ⏳ Automatic variation quality assessment
3. ⏳ Different variation strategies (synonyms, reordering, etc.)
4. ⏳ Correlation analysis with other reliability metrics
5. ⏳ Per-task variation suggestions for improvement

**Research Directions**:
- Optimal number of variations for robust estimation
- Relationship between prompt sensitivity and model size/architecture
- Transfer of robustness across benchmarks
- Adversarial prompt variations

---

## References

- [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) - Overall integration status
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - All implemented metrics
- [README_NEW_METRICS.md](README_NEW_METRICS.md) - Overview of new reliability metrics
- [Main README](../README.md) - HAL documentation

---

**Status**: Fully Operational ✅

**Last Updated**: 2026-01-09

**Contact**: See main README for support

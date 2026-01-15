# Prompt Sensitivity Evaluation

This directory contains scripts for evaluating agent robustness to prompt variations - a key reliability metric that measures how performance changes when the same task is phrased differently.

## Overview

**Prompt Sensitivity (S_prompt)** measures how consistent an agent's performance is across semantically equivalent prompt variations. A robust agent should perform similarly regardless of minor phrasing differences in the prompt.

### Metrics

- **S_prompt**: Prompt robustness score (0-1, higher is better)
  - `S_prompt = 1 - mean_variance`
  - Close to 1: Very robust, consistent across variations
  - Close to 0: Sensitive, performance varies with phrasing

- **Mean Variance**: Average variance in scores across variations per task (lower is better)

- **Min-Max Gap**: Difference between best and worst performance (lower is better)

## Quick Start

### Step 1: Run Evaluations with Prompt Variations

```bash
# Run evaluations with prompt sensitivity enabled
python reliability_eval/run_prompt_sensitivity_eval.py \
  --num_variations 3 \
  --max_tasks 20
```

This will:
1. Generate 3 paraphrases for each task prompt (4 total including original)
2. Run each agent on all variations
3. Save results to `results/<benchmark>/<run_id>/`

### Step 2: Analyze Results

```bash
# Analyze the results
python reliability_eval/analyze_prompt_sensitivity.py \
  --results_dir results/ \
  --benchmark taubench_airline \
  --output_dir reliability_eval/analysis
```

This will generate:
- CSV files with metrics
- Comparison visualizations
- Detailed markdown report

## Configuration

### Editing Agent Configurations

Edit `run_prompt_sensitivity_eval.py` to configure which agents to evaluate:

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
            "temperature": 0.0
        }
    },
]
```

### Editing Benchmark Configurations

```python
BENCHMARK_CONFIGS = [
    {
        "name": "taubench_airline",
        "benchmark_name": "taubench_airline",
        "requires_docker": False,
        "requires_vm": False,
        "extra_args": []
    },
]
```

## Command-Line Options

### run_prompt_sensitivity_eval.py

```bash
python reliability_eval/run_prompt_sensitivity_eval.py \
  --num_variations 3       # Number of variations to generate (default: 3)
  --max_tasks 20          # Max tasks per benchmark (default: 20)
  --conda_env hal         # Optional: specify conda environment
  --skip_swebench         # Skip SWE-bench (requires Docker)
```

### analyze_prompt_sensitivity.py

```bash
python reliability_eval/analyze_prompt_sensitivity.py \
  --results_dir results/            # Results directory (default: results)
  --benchmark taubench_airline      # Benchmark to analyze
  --output_dir reliability_eval/analysis  # Output directory
```

## Understanding the Results

### Agent-Level Metrics

The analysis produces a CSV file (`agent_level_sensitivity.csv`) with:

- `S_prompt`: Overall robustness score (higher = more robust)
- `mean_variance`: Average variance across tasks (lower = more stable)
- `mean_min_max_gap`: Average performance range (lower = more consistent)
- `max_min_max_gap`: Largest performance swing observed
- `num_tasks`: Number of tasks evaluated

### Task-Level Metrics

The analysis also produces task-level data (`task_level_sensitivity.csv`):

- `variance`: Performance variance for this task
- `mean_score`: Average score across variations
- `min_max_gap`: Performance range for this task
- `S_task`: Task-specific robustness score

### Visualizations

1. **sensitivity_comparison.png**: Compare S_prompt scores across agents
2. **task_sensitivity_distribution.png**: Distribution of task-level metrics
3. **most_sensitive_tasks.png**: Tasks most affected by prompt variations

### Report

The markdown report (`sensitivity_report.md`) provides:
- Summary statistics
- Ranking of agents by robustness
- Most prompt-sensitive tasks
- Detailed metric explanations

## How It Works

### Prompt Variation Generation

The system uses an LLM (GPT-4o-mini by default) to generate semantic-preserving paraphrases:

1. Each task prompt is sent to the LLM
2. The LLM generates N variations with different phrasing
3. Each variation maintains the exact same meaning
4. Variations differ in style (formal/casual, verbose/concise)

### Evaluation Process

1. **Generate Variations**: Create N+1 versions of each prompt (original + N variations)
2. **Run Agent**: Execute the agent on each variation independently
3. **Evaluate**: Score each response using the benchmark's evaluation function
4. **Compute Metrics**: Calculate variance and other statistics per task
5. **Aggregate**: Compute overall metrics across all tasks

### Metric Computation

For each task with scores `[s₀, s₁, ..., sₙ]` across variations:

- **Variance**: `σ² = mean((sᵢ - mean)²)`
- **Min-Max Gap**: `max(sᵢ) - min(sᵢ)`
- **S_task**: `1 - σ²` (task robustness)

For the agent overall:
- **S_prompt**: `1 - mean(variance across all tasks)`

## Supported Benchmarks

The prompt field is automatically detected for each benchmark:

- **GAIA**: `Question` field
- **SWE-bench**: `problem_statement` field
- **USACO**: `problem_statement` field
- **SciCode**: `problem_statement` field
- **AssistantBench**: `task` field
- **TauBench (airline, retail)**: `instruction` field

### How TauBench Support Works

TauBench is a special case because task prompts are normally generated by the tau-bench library from environment metadata. To enable prompt sensitivity:

1. The benchmark loads actual task instructions from tau-bench's task definitions
2. These instructions are added to the input data as an `instruction` field
3. The agent code overrides the environment's task instruction with the provided variation
4. The agent runs with the modified instruction

This allows prompt sensitivity evaluation while preserving TauBench's environment-based architecture.

### Adding Support for New Benchmarks

To add support for a new benchmark, edit `get_prompt_field_for_benchmark()` in [hal/utils/prompt_variation.py](../hal/utils/prompt_variation.py) and add the benchmark name and its prompt field name to the `prompt_field_map` dictionary.

## Example Workflow

```bash
# 1. Run evaluations (takes time, uses API calls)
python reliability_eval/run_prompt_sensitivity_eval.py \
  --num_variations 3 \
  --max_tasks 10

# 2. Analyze results
python reliability_eval/analyze_prompt_sensitivity.py \
  --benchmark taubench_airline \
  --output_dir reliability_eval/analysis

# 3. View results
cat reliability_eval/analysis/sensitivity_report.md
open reliability_eval/analysis/sensitivity_comparison.png
```

## Costs and Runtime

- **LLM Cost**: Generating variations uses GPT-4o-mini (~$0.15 per 1M input tokens)
- **Agent Cost**: Running N+1 variations multiplies agent evaluation costs by N+1
- **Runtime**: Approximately (N+1) × normal evaluation time

For 20 tasks with 3 variations:
- 20 tasks × 4 variations = 80 agent runs
- Variation generation: ~$0.01 (negligible)
- Agent runs: Depends on agent (e.g., 80 × Sonnet calls)

## Tips

1. **Start Small**: Use `--max_tasks 5` for testing before full runs
2. **Fewer Variations**: Start with `--num_variations 2` to reduce costs
3. **Temperature 0**: Use `temperature=0.0` for agents to reduce inherent variance
4. **Multiple Runs**: For very reliable agents, run multiple times to separate prompt sensitivity from sampling variance

## Troubleshooting

### No sensitivity metrics in results

Ensure you ran with `--prompt_sensitivity` flag. Check that:
```python
# In run command
--prompt_sensitivity --num_variations 3
```

### Analysis finds no results

Check that:
1. The benchmark name matches the results directory name
2. Results contain `prompt_sensitivity_metrics` in the UPLOAD.json files
3. You're using the correct `--results_dir` path

### High memory usage

For large benchmarks, the evaluation stores all variation outputs in memory. Consider:
- Using `--max_tasks` to limit evaluation size
- Running smaller batches and combining results

## Related

- [Consistency Evaluation](README_CONSISTENCY.md): Outcome consistency (C_out)
- [Predictability Evaluation](README_PREDICTABILITY.md): Risk-coverage and calibration
- [Main HAL Documentation](../README.md): General HAL harness usage

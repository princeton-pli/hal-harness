# Reliability Evaluation Framework

This directory contains scripts for running and analyzing reliability evaluations of LLM agents. The framework measures various dimensions of agent reliability beyond simple task accuracy.

## Overview

The reliability evaluation framework measures six key dimensions:

| Metric | Symbol | Description |
|--------|--------|-------------|
| **Outcome Consistency** | C_out | Reproducibility of results across repeated runs |
| **Prompt Sensitivity** | S_prompt | Robustness to semantically equivalent prompt variations |
| **Fault Robustness** | R_fault | Resilience to API/tool failures |
| **Predictability** | P_rc, P_cal | Quality of confidence estimates (risk-coverage, calibration) |
| **Structural Robustness** | R_struct | Robustness to environmental changes (schemas, formats) |
| **Compliance** | S_comp | Adherence to behavioral constraints and safety guardrails |

## Setup

### Prerequisites

1. Install the base HAL harness:
```bash
pip install -e .
```

2. Install reliability evaluation dependencies:
```bash
pip install -r reliability_eval/requirements.txt
```

3. Set up environment variables in `.env`:
```bash
OPENAI_API_KEY=your_key
WANDB_API_KEY=your_key
ANTHROPIC_API_KEY=your_key  # For Claude models
GEMINI_API_KEY=your_key     # For Gemini models
```

## Running Evaluations

### 1. Outcome Consistency (C_out)

Measures reproducibility by running K repetitions of the same evaluation.

```bash
python reliability_eval/run_consistency_eval.py \
    --k 5 \
    --max_tasks 20 \
    --conda_env hal
```

**Options:**
- `--k`: Number of repetitions per task (default: 5)
- `--max_tasks`: Maximum tasks per benchmark (default: 20)
- `--conda_env`: Conda environment name (optional)
- `--skip_swebench`: Skip SWE-bench evaluations (requires Docker)

**Understanding C_out:**
```
C_out(t) = 1 - Var_out(t) / (p_t(1 - p_t) + ε)
```
- C_out ≈ 1: Deterministic behavior (always succeeds or always fails)
- C_out ≈ 0: Maximum stochasticity given the success rate

### 2. Prompt Sensitivity (S_prompt)

Measures robustness to semantically equivalent prompt variations.

```bash
python reliability_eval/run_prompt_sensitivity_eval.py \
    --num_variations 3 \
    --max_tasks 20 \
    --conda_env hal
```

**Options:**
- `--num_variations`: Number of prompt variations to generate (default: 3)
- `--max_tasks`: Maximum tasks per benchmark (default: 20)
- `--conda_env`: Conda environment name (optional)
- `--skip_swebench`: Skip SWE-bench evaluations

**Note:** This uses the `--prompt_sensitivity` flag in `hal-eval` to generate LLM-based paraphrases of task instructions.

### 3. Fault Robustness (R_fault)

Measures resilience to injected faults (API errors, timeouts, rate limits).

```bash
python reliability_eval/run_fault_eval.py \
    --k 3 \
    --fault_rate 0.2 \
    --max_tasks 50 \
    --conda_env hal
```

**Options:**
- `--k`: Number of repetitions (default: 3)
- `--fault_rate`: Probability of fault injection, 0.0-1.0 (default: 0.2 = 20%)
- `--max_tasks`: Maximum tasks per benchmark (default: 50)
- `--conda_env`: Conda environment name (optional)

### 4. Predictability (P_rc, P_cal)

Measures quality of agent confidence estimates via self-assessment.

```bash
python reliability_eval/run_predictability_eval.py \
    --k 3 \
    --max_tasks 50 \
    --conda_env hal
```

**Options:**
- `--k`: Number of repetitions (default: 3)
- `--max_tasks`: Maximum tasks per benchmark (default: 50)
- `--conda_env`: Conda environment name (optional)

**Note:** Requires agents with `compute_confidence=True` in their configuration.

### 5. Structural Robustness (R_struct)

Measures robustness to environmental changes (API formats, schemas).

```bash
python reliability_eval/run_structural_robustness_eval.py \
    --perturbation_strength medium \
    --max_tasks 50
```

**Options:**
- `--perturbation_strength`: Level of perturbation (`low`, `medium`, `high`)
- `--max_tasks`: Maximum tasks per benchmark (default: 50)

### 6. Compliance (S_comp)

Measures adherence to behavioral constraints during execution.

```bash
python reliability_eval/run_compliance_eval.py \
    --k 3 \
    --max_tasks 50 \
    --conda_env hal
```

**Options:**
- `--k`: Number of repetitions (default: 3)
- `--max_tasks`: Maximum tasks per benchmark (default: 50)
- `--conda_env`: Conda environment name (optional)

**Default constraints monitored:**
- `no_pii_exposure`: Don't expose customer PII in logs
- `rate_limit_respect`: Respect API rate limits
- `no_destructive_ops`: Don't perform irreversible operations
- `data_minimization`: Only request necessary data

## Configuring Agents and Benchmarks

Each run script contains `AGENT_CONFIGS` and `BENCHMARK_CONFIGS` lists. Edit these to select which agents and benchmarks to evaluate.

### Agent Configuration Example

```python
AGENT_CONFIGS = [
    {
        "name": "taubench_toolcalling_gpt_4o",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "gpt-4o-2024-11-20",
        "benchmarks": ["taubench_airline", "taubench_retail"],
        "extra_agent_args": {
            "provider": "openai",
            "temperature": 0.0
        }
    },
]
```

### Supported Providers

| Provider | Model Name Prefix | Required API Key |
|----------|-------------------|------------------|
| OpenAI | `gpt-4o`, `gpt-4-turbo`, etc. | `OPENAI_API_KEY` |
| Anthropic | `claude-3-5-haiku`, `claude-sonnet-4-5`, etc. | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.5-flash`, etc. | `GEMINI_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-3-7-sonnet` | `OPENROUTER_API_KEY` |
| Together AI | `together_ai/meta-llama/...` | `TOGETHERAI_API_KEY` |

## Analyzing Results

After running evaluations, use the analysis scripts to compute metrics and generate visualizations.

### Analysis Scripts

| Script | Metrics Computed |
|--------|------------------|
| `analyze_consistency.py` | C_out per task, variance, success rates |
| `analyze_prompt_sensitivity.py` | S_prompt, per-task sensitivity scores |
| `analyze_fault_eval.py` | R_fault, recovery rates, time-to-recovery |
| `analyze_predictability.py` | P_rc (risk-coverage AUC), P_cal (calibration error) |
| `analyze_compliance.py` | S_comp, violation counts by constraint type |
| `analyze_structural_robustness.py` | R_struct, performance degradation under perturbation |
| `analyze_safety_metrics.py` | Combined safety and reliability analysis |

### Example Usage

```bash
# Analyze prompt sensitivity results
python reliability_eval/analyze_prompt_sensitivity.py \
    --results_dir results/ \
    --benchmark taubench_airline

# Analyze consistency results
python reliability_eval/analyze_consistency.py \
    --results_dir results/ \
    --benchmark taubench_airline
```

### Output

Analysis scripts generate:
- CSV files with detailed per-task metrics
- Aggregated summaries by agent/benchmark
- Visualizations (heatmaps, scatter plots, etc.)
- Markdown reports

Output is saved to `reliability_eval/analysis/`.

## Output Structure

Results are saved to `results/<benchmark>/<run_id>/`:

```
results/
└── taubench_airline/
    └── taubench_airline_agent_name_20250109123456/
        ├── <run_id>_UPLOAD.json          # Full results with metrics
        ├── <run_id>_RAW_SUBMISSIONS.jsonl # Raw agent outputs
        └── <run_id>_weave.json           # Weave trace data
```

The `*_UPLOAD.json` file contains:
- `config`: Run configuration
- `results`: Aggregated metrics (accuracy, cost, latencies)
- `raw_eval_results`: Per-task evaluation results
- `prompt_sensitivity_metrics`: (if enabled) Variance and sensitivity scores

## Tips

1. **Start small**: Use `--max_tasks 5` for initial testing
2. **Sequential runs**: Scripts run with `--max_concurrent 1` by default to avoid rate limits
3. **Resume on failure**: Check the `*_run_log.json` files for progress tracking
4. **Cost awareness**: Prompt sensitivity and predictability evaluations require additional LLM calls

## Troubleshooting

### Rate Limits

The scripts run evaluations sequentially with 5-second delays. If you still hit limits:
1. Reduce `--max_tasks`
2. Increase the delay in the run script (look for `time.sleep(5)`)

### Missing Results

If analysis shows "No results found":
1. Check evaluations completed: `ls results/<benchmark>/`
2. Verify UPLOAD.json files exist: `find results/ -name "*_UPLOAD.json"`
3. Check HAL logs for errors

### Docker Issues (SWE-bench)

If SWE-bench fails:
1. Ensure Docker is running: `docker ps`
2. Use `--skip_swebench` flag
3. SWE-bench doesn't support arm64 (M1/M2 Macs)

## Files

### Run Scripts
- `run_consistency_eval.py` - Outcome consistency evaluation
- `run_prompt_sensitivity_eval.py` - Prompt sensitivity evaluation
- `run_fault_eval.py` - Fault robustness evaluation
- `run_predictability_eval.py` - Predictability evaluation
- `run_structural_robustness_eval.py` - Structural robustness evaluation
- `run_compliance_eval.py` - Compliance evaluation

### Analysis Scripts
- `analyze_consistency.py` - Analyze consistency results
- `analyze_prompt_sensitivity.py` - Analyze prompt sensitivity results
- `analyze_fault_eval.py` - Analyze fault robustness results
- `analyze_predictability.py` - Analyze predictability results
- `analyze_structural_robustness.py` - Analyze structural robustness results
- `analyze_compliance.py` - Analyze compliance results
- `analyze_safety_metrics.py` - Combined safety analysis

### Other
- `requirements.txt` - Python dependencies for analysis
- `README.md` - This file

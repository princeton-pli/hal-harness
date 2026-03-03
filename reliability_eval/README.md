# Reliability Evaluation Framework

This directory contains scripts for running comprehensive reliability evaluations of AI agents and analyzing the results. The framework implements metrics from the reliability evaluation paper across four dimensions: **Consistency**, **Robustness**, **Predictability**, and **Safety**.

## Overview

The evaluation process consists of two main steps:

1. **`run_reliability_eval.py`** - Runs the evaluation experiments
2. **`analyze_reliability.py`** - Analyzes results and generates visualizations

## Supported Benchmarks

| Benchmark | Description | Key Constraints |
|-----------|-------------|-----------------|
| `taubench_airline` | Customer service (airline) | PII handling, destructive ops, data minimization, policy circumvention, financial accuracy, authentication, commitment overreach |
| `taubench_retail` | Customer service (retail) | PII handling, destructive ops, data minimization, policy circumvention, financial accuracy, authentication, commitment overreach |
| `gaia` | General Q&A tasks | PII (relaxed for Q&A), destructive ops |

## Quick Start

```bash
# 1. Run a quick test evaluation (2 repetitions, 5 tasks)
python reliability_eval/run_reliability_eval.py --n 2 --max_tasks 5 --phases baseline

# 2. Analyze the results
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark gaia
```

## Running Evaluations

### Basic Usage

```bash
python reliability_eval/run_reliability_eval.py --n <repetitions> --max_tasks <num_tasks> [OPTIONS]
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n` | 5 | Number of runs/variations for all multi-run metrics |
| `--k` | (uses --n) | Override: repetitions for baseline/fault phases |
| `--max_tasks` | (all) | Maximum tasks per benchmark |
| `--max_concurrent` | 5 | Maximum concurrent tasks per hal-eval run |
| `--phases` | all | Which phases to run (see below) |
| `--benchmark` | (all) | Run only on specific benchmark |
| `--conda_env` | (current) | Conda environment name |

### Evaluation Phases

The evaluation runs in phases, each measuring different reliability metrics:

| Phase | Metrics | Description |
|-------|---------|-------------|
| `baseline` | C_out, P_rc, P_cal, S_comp | K repetitions with confidence scoring and compliance monitoring |
| `fault` | R_fault | Fault injection robustness |
| `prompt` | R_prompt | Prompt variation robustness |
| `structural` | R_struct | Structural perturbation robustness |
| `safety` | S_harm, S_comp | LLM-based safety analysis on existing traces |
| `abstention` | abstention rate, calibration | Regex-based abstention/deferral detection on existing traces |

#### Phase Descriptions

**baseline** - Runs K identical repetitions of each task to measure consistency and predictability. Each run includes confidence scoring (agent self-reports confidence 0-1) and compliance monitoring. Used to compute outcome consistency (do results vary across runs?), confidence calibration (does confidence predict success?), and baseline accuracy.

**fault** - Injects simulated failures (e.g., tool errors, API timeouts) at a configurable rate during agent execution. Measures how well the agent recovers from and handles unexpected failures. R_fault = accuracy_with_faults / baseline_accuracy.

**prompt** - Runs each task with semantically-equivalent prompt variations (e.g., rephrased instructions, different formatting). Measures sensitivity to prompt wording. R_prompt = accuracy_on_variations / baseline_accuracy.

**structural** - Applies structural perturbations to task inputs (e.g., reordered fields, added whitespace, synonym substitution). Measures robustness to input formatting changes. R_struct = accuracy_with_perturbations / baseline_accuracy.

**safety** - Post-hoc LLM-based analysis of existing traces. Evaluates error severity (S_harm: how bad are failures on a 0-10 scale?) and compliance (S_comp: did the agent violate any constraints like exposing PII?). Does not run new experiments.

**abstention** - Post-hoc regex-based detection of abstention/deferral behavior in existing traces. Identifies when agents say "I can't do this", "I'm not sure", or ask for clarification. Measures abstention rate and whether abstentions correlate with actual failures (calibration).

### Examples

```bash
# Run all phases with 5 repetitions on 50 tasks
python reliability_eval/run_reliability_eval.py --n 5 --max_tasks 50

# Run only baseline phase (consistency + predictability)
python reliability_eval/run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline

# Run baseline and safety analysis
python reliability_eval/run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline safety

# Run only safety analysis on existing results (no new experiments)
python reliability_eval/run_reliability_eval.py --phases safety --results_dir results --n 5 --max_concurrent 5

# Run only abstention detection on existing results
python reliability_eval/run_reliability_eval.py --phases abstention --results_dir results --n 5 --max_concurrent 5

# Override specific phases (3 baseline reps, but use --n for other defaults)
python reliability_eval/run_reliability_eval.py --n 5 --k 3 --max_tasks 50

# Run on a specific benchmark only
python reliability_eval/run_reliability_eval.py --n 5 --max_tasks 50 --benchmark gaia

# Full GAIA evaluation (all phases)
python reliability_eval/run_reliability_eval.py \
    --phases baseline fault prompt structural safety abstention \
    --n 3 --max_concurrent 10 --benchmark gaia
```

### Phase-Specific Options

**Fault Injection:**
```bash
--fault_rate 0.5  # Fault injection rate (default: 0.5 = 50%)
```

**Prompt Sensitivity:**
```bash
--num_variations 5        # Number of prompt variations (default: uses --n)
--variation_strength strong  # mild, medium, strong, or naturalistic (default: strong)
```

**Structural Perturbations:**
```bash
--perturbation_strength severe  # mild, medium, or severe (default: severe)
```

**Safety Analysis:**
```bash
--safety_model gpt-4o-mini   # LLM model for safety analysis (default: gpt-4o-mini)
--results_dir results        # Directory containing results to analyze (default: results)
```

### Continuing Failed Runs

If runs fail, you can retry them:

```bash
# Retry all failed runs from the log
python reliability_eval/run_reliability_eval.py --retry_failed

# Continue a specific run by ID
python reliability_eval/run_reliability_eval.py --continue_run_id taubench_airline_agent_name_1234567890
```

### Configuring Agents and Benchmarks

Edit the `AGENT_CONFIGS` list in `run_reliability_eval.py` to specify which models to evaluate. Uncomment the agents you want to run:

```python
AGENT_CONFIGS = [
    # TauBench example (customer service)
    {
        "name": "taubench_toolcalling_gpt_4o_mini",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "gpt-4o-mini-2024-07-18",
        "provider": "openai",
        "benchmarks": ["taubench_airline"],
    },
    # GAIA example (Q&A tasks)
    {
        "name": "gaia_generalist_gemini_2_flash",
        "agent_dir": "agents/hal_generalist_agent",
        "agent_function": "main.run",
        "model_name": "gemini/gemini-2.0-flash",
        "benchmarks": ["gaia"],
        "extra_agent_args": {
            "provider": "google",
            "temperature": 0.0
        }
    },
    # Add more agents...
]
```

Supported providers: `openai`, `anthropic`, `google`

**Available GAIA models** (uncomment in `run_reliability_eval.py`):
- OpenAI: `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4o`, `o1`, `gpt-5.2`
- Anthropic: `claude-3-5-haiku`, `claude-sonnet-3-7`, `claude-sonnet-4-5`, `claude-opus-4-5`
- Google: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-pro`

## Analyzing Results

### Basic Usage

```bash
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark <benchmark_name>
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | results | Directory containing evaluation results |
| `--benchmark` | (required) | Benchmark to analyze (`gaia`, `taubench_airline`, etc.) |
| `--output_dir` | reliability_eval/analysis | Output directory for plots and reports |
| `--scaffold` | all | Filter to specific agent scaffold |
| `--use_llm_safety` | (flag) | Enable LLM-as-judge for safety analysis |
| `--llm_model` | gpt-4o-mini | LLM model for safety analysis |
| `--harm_ref` | 5.0 | Reference severity for S_harm saturation |

### Examples

```bash
# Basic analysis (GAIA)
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark gaia

# Basic analysis (TauBench)
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark taubench_airline

# With LLM-based safety analysis
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark gaia --use_llm_safety

# Custom output directory
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark gaia --output_dir my_analysis/

# Filter to specific agent type
python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark gaia --scaffold generalist
```

### Generated Outputs

The analysis script produces:

**Data Files:**
- `reliability_metrics.csv` - All computed metrics in tabular format

**Visualizations:**
- `reliability_dashboard.png` - Comprehensive dashboard with all metrics
- `reliability_heatmap.png` - Heatmap of all metrics across agents
- `reliability_radar.png` - Dimension-level radar chart (4 dimensions)
- `consistency_detailed.png` - Detailed consistency plots (C_out, C_traj_d, C_traj_s, C_conf, C_res)
- `predictability_detailed.png` - Detailed predictability plots (P_rc, P_cal, P_auroc, P_brier)
- `robustness_detailed.png` - Detailed robustness plots (R_fault, R_struct, R_prompt)
- `safety_detailed.png` - Detailed safety plots (S_harm, S_comp, S_safety)

**Reports:**
- `reliability_report.md` - Markdown report summarizing findings

## Metrics Reference

### Consistency (C)
| Metric | Description |
|--------|-------------|
| C_out | Outcome consistency - normalized by p(1-p) |
| C_traj_d | Trajectory distribution consistency (what actions) |
| C_traj_s | Trajectory sequence consistency (action order) |
| C_conf | Confidence consistency - CV of confidence scores |
| C_res | Resource consistency - conditioned on SUCCESS |

### Robustness (R)
| Metric | Description |
|--------|-------------|
| R_fault | Fault robustness - accuracy ratio under faults |
| R_struct | Structural robustness - accuracy ratio under perturbations |
| R_prompt | Prompt robustness - accuracy ratio under prompt variations |

### Predictability (P)
| Metric | Description |
|--------|-------------|
| P_rc | Risk-coverage score - excess AuRC over optimal |
| P_cal | Calibration score - 1 - ECE |
| P_auroc | Discrimination - AUC-ROC |
| P_brier | Overall quality - 1 - Brier Score |

### Safety (S)
| Metric | Description |
|--------|-------------|
| S_harm | Harm score - severity of errors (0-10 scale normalized) |
| S_comp | Compliance - constraint violation rate |
| S_safety | Aggregate safety = (S_harm + S_comp) / 2 |

### Abstention
| Metric | Description |
|--------|-------------|
| abstention_rate | Fraction of tasks where agent abstained/deferred |
| precision | P(fail \| abstain) - how often abstentions predict failure |
| recall | P(abstain \| fail) - how often failures are predicted by abstention |

## Environment Setup

### Required Environment Variables

The evaluation requires API keys for the models being tested:

```bash
# Always required
export WANDB_API_KEY=your_key

# Benchmark-specific
export HF_TOKEN=your_key            # For GAIA dataset access (Hugging Face)

# Provider-specific (based on configured agents)
export OPENAI_API_KEY=your_key      # For OpenAI models + safety analysis
export ANTHROPIC_API_KEY=your_key   # For Anthropic models
export GOOGLE_API_KEY=your_key      # For Google/Gemini models
export OPENROUTER_API_KEY=your_key  # For OpenRouter models
```

You can also create a `.env` file in the project root with these variables.

### Dependencies

The analysis script requires additional Python packages:

```bash
pip install matplotlib seaborn pandas scipy
```

## Typical Workflow

1. **Configure agents** - Edit `AGENT_CONFIGS` in `run_reliability_eval.py` (uncomment desired models)

2. **Set environment variables** - Ensure API keys are set for your models and benchmarks

3. **Run all phases at once** (recommended):
   ```bash
   python reliability_eval/run_reliability_eval.py \
       --phases baseline fault prompt structural safety abstention \
       --n 3 --max_concurrent 10 --benchmark gaia
   ```

   Or run phases separately:

   ```bash
   # Baseline evaluation (consistency + predictability)
   python reliability_eval/run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline

   # Robustness evaluations
   python reliability_eval/run_reliability_eval.py --n 3 --max_tasks 50 --phases fault prompt structural

   # Safety analysis on collected traces
   python reliability_eval/run_reliability_eval.py --phases safety

   # Abstention detection on collected traces
   python reliability_eval/run_reliability_eval.py --phases abstention
   ```

4. **Analyze results**:
   ```bash
   python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark gaia --use_llm_safety
   ```

5. **Review outputs** in `reliability_eval/analysis/`

## Troubleshooting

### Common Issues

**API Rate Limits:**
- Use `--max_concurrent 1` to avoid rate limiting if needed
- The script includes automatic retry logic with backoff

**Missing API Keys:**
- The script will warn about missing keys and prompt for confirmation
- Ensure the correct provider key is set for each configured agent

**Failed Runs:**
- Check `reliability_eval/reliability_eval_log.json` for failed run details
- Use `--retry_failed` to retry failed runs
- Use `--continue_run_id` to continue a specific run

**No Results Found:**
- Verify the benchmark directory exists in `results/`
- Check that result files have the `*_UPLOAD.json` naming pattern

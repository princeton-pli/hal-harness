# Quick Start Guide - Reliability Evaluation

## TL;DR - Run Everything

```bash
# 1. Install dependencies
pip install pandas matplotlib seaborn numpy

# 2. Run evaluations (skip SWE-bench if you don't have Docker)
cd /Users/sr4049/princeton/projects/hal-harness
python reliability_eval/run_consistency_eval.py --k 5 --max_tasks 20 --skip_swebench

# 3. Analyze results
python reliability_eval/analyze_consistency.py

# 4. View results
open reliability_eval/analysis/consistency_report.md
open reliability_eval/analysis/consistency_heatmap.png
```

## What Gets Evaluated

**Agents:**
- `hal_agent_gpt4o_mini` - GPT-4o-mini (cheap)
- `hal_agent_gemini_flash` - Gemini 1.5 Flash (cheap)

**Benchmarks:**
- **GAIA** - General QA tasks (~5-10 min per agent)
- **TauBench airline** - Tool-calling tasks (~10-15 min per agent)
- **SWE-bench mini** - Code generation (~20-30 min per agent, requires Docker)

**Total runs:** 2 agents × 2-3 benchmarks × 5 repetitions × 20 tasks = 400-600 API calls

**Estimated cost:** $2-5 USD (using cheap models)

**Estimated time:**
- Without SWE-bench: ~1 hour
- With SWE-bench: ~2-3 hours

## Expected Output Structure

```
hal-harness/
├── results/                          # HAL evaluation results
│   ├── gaia/
│   │   └── run_*/
│   │       └── results.json
│   ├── taubench_airline/
│   │   └── run_*/
│   │       └── results.json
│   └── swebench_verified_mini/
│       └── run_*/
│           └── results.json
│
└── reliability_eval/
    ├── run_log.json                  # Execution progress log
    └── analysis/                     # Analysis outputs
        ├── consistency_detailed.csv
        ├── consistency_aggregated.csv
        ├── consistency_heatmap.png
        ├── consistency_vs_accuracy.png
        └── consistency_report.md
```

## Customizing the Evaluation

### Run More/Fewer Repetitions

```bash
# K=3 for faster testing
python reliability_eval/run_consistency_eval.py --k 3 --max_tasks 10

# K=10 for more robust statistics
python reliability_eval/run_consistency_eval.py --k 10 --max_tasks 50
```

### Run Only Specific Benchmarks

Edit `run_consistency_eval.py` and comment out benchmarks you don't want:

```python
BENCHMARK_CONFIGS = [
    {
        "name": "gaia",
        "benchmark_name": "gaia",
        # ...
    },
    # Comment out benchmarks you want to skip
    # {
    #     "name": "taubench_airline",
    #     ...
    # },
]
```

### Add Your Own Agent

Add to `AGENT_CONFIGS` in `run_consistency_eval.py`:

```python
AGENT_CONFIGS.append({
    "name": "my_agent_name",
    "agent_dir": "agents/my_agent_dir",
    "agent_function": "main.run",
    "model_name": "gpt-4o-2024-11-20",
    "provider_args": []
})
```

## Interpreting Results

### Consistency Score (C_out)

The key metric from `consistency_aggregated.csv`:

| Value | Interpretation |
|-------|----------------|
| 0.9 - 1.0 | **Highly deterministic** - Agent consistently succeeds or fails on each task |
| 0.7 - 0.9 | **Moderately consistent** - Some variance but generally predictable |
| 0.4 - 0.7 | **Inconsistent** - Significant randomness in outcomes |
| 0.0 - 0.4 | **Highly stochastic** - Maximum variance given accuracy level |

### Example Analysis

```csv
agent,benchmark,C_out_mean,C_out_std,p_hat_mean,count
hal_agent_gpt4o_mini,gaia,0.85,0.12,0.65,20
hal_agent_gemini_flash,gaia,0.72,0.18,0.68,20
```

**Interpretation:**
- Both agents have similar accuracy (~65-68%)
- GPT-4o-mini is more consistent (C_out=0.85 vs 0.72)
- Gemini has higher variance in consistency (std=0.18)

This shows that **consistency is independent of capability** - both agents solve similar tasks, but GPT-4o-mini does so more predictably.

## Troubleshooting

### "No module named 'hal'"

Make sure you've installed HAL:
```bash
pip install -e .
```

### "OPENAI_API_KEY not found"

Set your API keys in `.env`:
```bash
cp .env.template .env
# Edit .env and add your keys
```

### Evaluation stuck/hanging

- Check logs in `results/*/run_*/hal_eval.log`
- Some benchmarks (especially SWE-bench) can be slow
- Use `--max_tasks 5` for quick testing

### Analysis shows "No results found"

- Verify evaluations completed: `ls -la results/*/
- Check for `results.json` files: `find results/ -name "results.json"`
- Run evaluation script first before analysis

## Next Steps

After computing C_out, you can:

1. **Extend to more metrics** - Add trajectory consistency (C_traj) and resource consistency (C_res)
2. **Add more agents** - Compare different architectures and prompting strategies
3. **Scale up** - Increase K to 10-20 for publication-quality statistics
4. **Analyze patterns** - Correlate consistency with model size, cost, latency

See the main `README.md` for full documentation.

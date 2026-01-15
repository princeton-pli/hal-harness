# R_struct: Structural Robustness Metric

## Overview

**R_struct (Structural Robustness)** measures how well an agent maintains performance when the environment structure changes while preserving semantic meaning. Higher R_struct indicates greater robustness to environmental variations.

**Formula**: `R_struct = min(acc_perturbed / acc_baseline, 1.0)`

**Range**: [0, 1] where:
- **1.0** = Perfect robustness (no performance degradation)
- **0.0** = Complete failure under structural perturbations

---

## Why R_struct Matters

Real-world production environments are not static:

1. **API Evolution**: Endpoints change versions, parameters get renamed (snake_case ↔ camelCase)
2. **Database Migrations**: Column names change, schemas evolve (flat ↔ nested)
3. **File System Changes**: Paths reorganize, naming conventions vary
4. **Data Format Variations**: Dates, numbers, booleans represented differently across systems

**Example**: An agent that works with `/api/v1/users` but fails with `/api/v2/users` (same functionality) has low structural robustness.

---

## Implementation Status

### ✅ Fully Implemented

**Core Framework**:
- [hal/utils/structural_perturbations.py](../hal/utils/structural_perturbations.py) - Complete perturbation framework
- `StructuralPerturbator` class with 4 perturbation types
- `PerturbedEnvironmentWrapper` for benchmark integration
- Configuration system with 3 strength presets (mild, medium, severe)

**Evaluation Pipeline**:
- [run_structural_robustness_eval.py](run_structural_robustness_eval.py) - Run evaluations with perturbations
- [analyze_structural_robustness.py](analyze_structural_robustness.py) - Analyze results and compute R_struct

**Metrics Computed**:
- **R_struct_overall**: Aggregate robustness score
- **R_struct by type**: Per-perturbation-type scores (API, database, file, data format)
- **Degradation**: Performance loss (1 - R_struct)
- **Task sensitivity**: Which tasks are most affected

**Visualizations**:
- Agent comparison plots (R_struct, degradation, baseline vs perturbed)
- Task sensitivity distributions
- Heatmaps (agent × perturbation type)

---

## Perturbation Types

### 1. API Perturbations

**Endpoint Changes**:
```python
# Original
GET /api/v1/users/123

# Versioned (mild)
GET /api/v2/users/123

# Shortened (severe)
GET /users/123
```

**Parameter Naming**:
```python
# Original (snake_case)
{"user_id": 123, "first_name": "John"}

# camelCase (mild)
{"userId": 123, "firstName": "John"}

# kebab-case (medium)
{"user-id": 123, "first-name": "John"}

# Abbreviated (severe)
{"uid": 123, "fname": "John"}
```

**Response Structure**:
```python
# Original (unwrapped)
{"id": 123, "name": "John"}

# Wrapped (medium)
{"status": "success", "data": {"id": 123, "name": "John"}}
```

### 2. Database Perturbations

**Column Naming**:
```sql
-- Original
SELECT user_id, first_name FROM users

-- camelCase
SELECT userId, firstName FROM users

-- Abbreviated
SELECT uid, fname FROM users
```

**Table Naming**:
```sql
-- Original
FROM orders

-- Prefixed
FROM tbl_orders

-- Suffixed
FROM orders_records
```

**Schema Structure**:
```python
# Original (flat)
{"user_id": 123, "user_name": "John"}

# Nested (severe)
{"user": {"id": 123, "name": "John"}}
```

### 3. File System Perturbations

**Path Changes**:
```
Original: /data/users.json
+1 depth:  /data/data1/users.json
-1 depth:  /users.json
```

**Naming Convention**:
```
Original: user_data.json (snake_case)
camelCase: userData.json
PascalCase: UserData.json
```

**Format Changes**:
```
Original: data.json
CSV:      data.csv  (same content, different format)
XML:      data.xml
YAML:     data.yaml
```

### 4. Data Format Perturbations

**Date Formats**:
```python
ISO:  "2024-01-15"
US:   "01/15/2024"
EU:   "15/01/2024"
Unix: 1705276800
```

**Number Formats**:
```python
Numeric:           1234.56
String:            "1234.56"
String w/ commas:  "1,234.56"
```

**Boolean Formats**:
```python
Boolean: true
String:  "true"
Numeric: 1
Yes/No:  "yes"
```

---

## How It Works

### 1. Perturbation Configuration

Choose strength preset or custom configuration:

```python
from hal.utils.structural_perturbations import (
    create_perturbator,
    PerturbationStrength,
    PerturbationConfig
)

# Use preset
perturbator = create_perturbator(
    perturbation_type="all",
    strength="medium"
)

# Custom configuration
config = PerturbationConfig(
    api_parameter_case="camelCase",
    db_column_naming="abbreviated",
    file_naming_case="PascalCase",
    date_format="us"
)
perturbator = StructuralPerturbator("all", config)
```

**Preset Strengths**:
- **Mild**: Only naming conventions (snake_case → camelCase)
- **Medium**: Naming + structure (versioning, wrapping, path depth +1)
- **Severe**: Complete restructuring (abbreviations, nesting, format changes)

### 2. Baseline Evaluation

Run normal evaluation without perturbations:

```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/my_agent \
    --agent_function main.run \
    --agent_name "My Agent (baseline)" \
    --max_tasks 50
```

### 3. Perturbed Evaluation

Run with structural perturbations enabled:

```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/my_agent \
    --agent_function main.run \
    --agent_name "My Agent (perturbed_medium)" \
    -A enable_structural_perturbations=true \
    -A perturbation_strength=medium \
    -A perturbation_type=all \
    --max_tasks 50
```

### 4. Analysis

Compute R_struct from baseline and perturbed runs:

```bash
python reliability_eval/analyze_structural_robustness.py \
    --results_dir results/ \
    --benchmark taubench_airline \
    --output_dir reliability_eval/analysis
```

**Metrics Calculated**:
```python
R_struct = min(acc_perturbed / acc_baseline, 1.0)
degradation = 1.0 - R_struct
```

---

## Usage Guide

### Step 1: Run Baseline and Perturbed Evaluations

```bash
# Option A: Run both with script
python reliability_eval/run_structural_robustness_eval.py \
    --benchmark taubench_airline \
    --perturbation_strength medium \
    --max_tasks 50

# Option B: Run manually
# 1. Baseline
hal-eval --benchmark taubench_airline \
    --agent_dir agents/hal_generalist_agent \
    --agent_function main.run \
    --agent_name "HAL Agent (baseline)" \
    --max_tasks 50

# 2. Perturbed
hal-eval --benchmark taubench_airline \
    --agent_dir agents/hal_generalist_agent \
    --agent_function main.run \
    --agent_name "HAL Agent (perturbed_medium)" \
    -A enable_structural_perturbations=true \
    -A perturbation_strength=medium \
    --max_tasks 50
```

### Step 2: Analyze Results

```bash
python reliability_eval/analyze_structural_robustness.py \
    --results_dir results/ \
    --benchmark taubench_airline \
    --output_dir reliability_eval/analysis
```

**Outputs**:
- `agent_r_struct.csv`: Agent-level R_struct, degradation, accuracies
- `task_sensitivity_*.csv`: Per-task sensitivity to perturbations
- `r_struct_comparison.png`: 4-panel agent comparison
- `task_sensitivity_*.png`: Task-level distributions per agent
- `r_struct_report.md`: Comprehensive analysis report

### Step 3: Interpret Results

**High R_struct (≥ 0.9)**:
- Highly robust to environmental changes
- Minimal performance degradation
- Good: Safe for production deployment

**Medium R_struct (0.7 - 0.9)**:
- Moderately robust
- Some performance loss under perturbations
- Caution: May need environment-specific tuning

**Low R_struct (< 0.7)**:
- Sensitive to environmental structure
- Significant performance degradation
- Risk: Brittle, may fail in production

---

## Example Results

### Agent Comparison

```csv
agent,perturbation_strength,R_struct,degradation,acc_baseline,acc_perturbed
gpt-4o,mild,0.980,0.020,0.850,0.833
gpt-4o,medium,0.923,0.077,0.850,0.785
gpt-4o,severe,0.847,0.153,0.850,0.720
claude-sonnet-4,mild,0.965,0.035,0.820,0.791
claude-sonnet-4,medium,0.885,0.115,0.820,0.726
claude-sonnet-4,severe,0.793,0.207,0.820,0.650
```

**Interpretation**:
- GPT-4o is most robust (R_struct_medium = 0.923)
- Both agents handle mild perturbations well (> 0.96)
- Severe perturbations cause 15-21% degradation
- Claude Sonnet 4 shows higher sensitivity to severe perturbations

### Task-Level Sensitivity

**Most Sensitive Tasks** (failed under perturbation):
1. Task 12: API parameter naming caused failure
2. Task 27: Date format change broke parsing
3. Task 41: Nested schema structure not handled

**Most Robust Tasks** (no change):
1. Task 5: Simple text processing (no structure dependency)
2. Task 18: Handled all naming conventions
3. Task 34: Robust date parsing logic

---

## Integration with Other Metrics

R_struct complements other robustness metrics:

**vs R_fault (Fault Robustness)**:
- R_fault: Performance under API/tool failures (availability issues)
- R_struct: Performance under structure changes (compatibility issues)
- Both needed for production robustness

**vs S_prompt (Prompt Robustness)**:
- S_prompt: Robustness to prompt phrasing
- R_struct: Robustness to environment structure
- Different dimensions of robustness

**Relationship**:
```
High R_fault + High R_struct = Robust to failures AND changes
High R_fault + Low R_struct  = Handles failures BUT brittle to changes
Low R_fault + High R_struct  = Handles changes BUT fails on errors
Low R_fault + Low R_struct   = Fragile (worst case)
```

---

## Technical Details

### Perturbation Application

Perturbations are applied at multiple levels:

1. **API Level**: Endpoint and parameter transformation
2. **Data Level**: Format and structure changes
3. **File Level**: Path and format modifications

```python
# Example: API perturbation
original_params = {"user_id": 123, "first_name": "John"}
perturbed_params = perturbator.perturb_api_params(original_params)
# Result: {"userId": 123, "firstName": "John"}

# Example: Data format perturbation
original_date = "2024-01-15"
perturbed_date = perturbator.perturb_date(original_date)
# Result: "01/15/2024" (US format)
```

### Benchmark Integration

For benchmarks to support R_struct, they need to:

1. Check for `enable_structural_perturbations` agent arg
2. Initialize `StructuralPerturbator` with config
3. Wrap environment interactions through perturbator

**Example Integration**:
```python
# In benchmark __init__
if agent_args.get('enable_structural_perturbations') == 'true':
    from hal.utils.structural_perturbations import create_perturbator

    strength = agent_args.get('perturbation_strength', 'medium')
    ptype = agent_args.get('perturbation_type', 'all')

    self.perturbator = create_perturbator(
        perturbation_type=ptype,
        strength=strength
    )
```

### Supported Benchmarks

**Priority Order**:
1. ✅ TauBench (airline, retail) - Rich API interactions
2. ⏳ AssistantBench - Web scraping, data formats
3. ⏳ SciCode - File I/O, data handling

**Note**: Not all benchmarks may support all perturbation types. API-heavy benchmarks work best.

---

## Limitations

1. **Benchmark Dependency**: Requires benchmark-level integration
   - Mitigation: Provide integration examples and templates

2. **Semantic Preservation**: Perturbations must preserve meaning
   - Mitigation: Conservative perturbation rules, validation

3. **Evaluation Cost**: Requires baseline + perturbed runs
   - Mitigation: Can test specific perturbation types separately

4. **Agent Awareness**: Agents unaware of perturbations may fail
   - Expected: This is what we're measuring!

---

## Best Practices

### For Evaluation

1. **Test Multiple Strengths**: Run mild, medium, severe to understand robustness curve
2. **Compare to Baseline**: Always run baseline first for accurate R_struct calculation
3. **Isolate Perturbation Types**: Test API, database, file separately to identify weaknesses
4. **Use Sufficient Tasks**: 50+ tasks for reliable statistics

### For Agent Development

1. **Target R_struct ≥ 0.85**: Good robustness for production
2. **Use Robust Parsing**: Libraries that handle multiple formats (e.g., dateutil for dates)
3. **Avoid Hardcoded Assumptions**: Don't assume snake_case, specific paths, etc.
4. **Test Locally**: Apply perturbations during development

**Robust Code Examples**:

```python
# BAD: Hardcoded assumptions
user_id = data["user_id"]  # Fails if key is "userId"

# GOOD: Flexible key lookup
user_id = data.get("user_id") or data.get("userId") or data.get("uid")

# BETTER: Normalize keys
def normalize_keys(data):
    # Convert all keys to snake_case for internal use
    return {to_snake_case(k): v for k, v in data.items()}

# BAD: Hardcoded date parsing
date = datetime.strptime(date_str, "%Y-%m-%d")

# GOOD: Flexible date parsing
from dateutil import parser
date = parser.parse(date_str)  # Handles multiple formats
```

### For Deployment

1. **Monitor R_struct in Staging**: Test against varied environment configs
2. **Set Minimum Thresholds**: Require R_struct ≥ 0.8 for production deployment
3. **Document Environment Assumptions**: If R_struct is low, document required environment
4. **Provide Migration Guides**: Help users adapt to structural changes

---

## Troubleshooting

### Low R_struct scores

**Problem**: Agent shows R_struct < 0.7

**Solutions**:
1. Identify which perturbation type causes most failures (check task sensitivity)
2. Review agent code for hardcoded assumptions
3. Use more flexible parsing libraries
4. Add fallback logic for multiple naming conventions

### Benchmark integration errors

**Problem**: Perturbations not applied during evaluation

**Solutions**:
1. Verify benchmark supports structural perturbations
2. Check agent args are passed correctly (-A flags)
3. Ensure perturbator is initialized in benchmark __init__
4. Review perturbator logs for applied perturbations

### Inconsistent results

**Problem**: R_struct varies significantly across runs

**Solutions**:
1. Use more tasks (50+ recommended)
2. Check if baseline and perturbed runs use same task set
3. Verify perturbations are applied consistently
4. Review for non-deterministic agent behavior

---

## Future Enhancements

**Planned**:
1. ✅ Support for TauBench
2. ⏳ Support for AssistantBench, SciCode
3. ⏳ Combination perturbations (test multiple types simultaneously)
4. ⏳ Adaptive perturbations (harder perturbations for robust agents)
5. ⏳ Correlation analysis with other metrics

**Research Directions**:
- Optimal perturbation strength for discriminating agent robustness
- Transfer of structural robustness across benchmarks
- Relationship between R_struct and generalization ability
- Adversarial perturbations that break semantic preservation

---

## References

- [DESIGN_R_STRUCT_S_COST.md](DESIGN_R_STRUCT_S_COST.md) - Detailed design document
- [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) - Overall integration status
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - All implemented metrics
- [Main README](../README.md) - HAL documentation

---

**Status**: Fully Implemented ✅

**Last Updated**: 2026-01-09

**Contact**: See main README for support

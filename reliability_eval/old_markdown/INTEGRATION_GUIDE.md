# Integration Guide for New Reliability Metrics

This guide provides step-by-step instructions for integrating the new reliability evaluation frameworks (C_traj, R_fault, V_heal, V_ttr, S_comp) into HAL.

## Table of Contents

1. [C_traj Integration (Already Complete)](#c_traj-integration)
2. [Fault Injection Integration](#fault-injection-integration)
3. [Compliance Monitoring Integration](#compliance-monitoring-integration)
4. [Testing the Integrations](#testing-the-integrations)

---

## C_traj Integration (Already Complete ✅)

The C_traj (Trajectory Consistency) metric is already fully integrated and operational!

### What's Implemented

- ✅ Trajectory extraction from `raw_logging_results`
- ✅ Jensen-Shannon divergence computation
- ✅ Integration into `analyze_consistency.py`
- ✅ Visualization in `plot_trajectory_consistency()`
- ✅ Automatic inclusion in reports

### How to Use

```bash
# 1. Run normal consistency evaluation
python reliability_eval/run_consistency_eval.py --k 5 --max_tasks 50

# 2. Analyze (C_traj computed automatically)
python reliability_eval/analyze_consistency.py \
    --results_dir results/ \
    --benchmark taubench_airline

# 3. View results
cat reliability_eval/analysis/consistency_report.md | grep "C_traj"
open reliability_eval/analysis/trajectory_consistency.png
```

**No changes needed** - C_traj works with existing evaluation data!

---

## Fault Injection Integration

To enable fault robustness (R_fault) and recoverability (V_heal, V_ttr) metrics, you need to integrate fault injection into the agent execution pipeline.

### Step 1: Import Fault Injector in Agent Runner

**File**: `hal/agent_runner.py`

**Location**: At the top of the file, add:

```python
from hal.utils.fault_injection import FaultInjector, FaultEvent
```

### Step 2: Initialize Fault Injector

**Location**: In the `AgentRunner.__init__()` or evaluation setup:

```python
def run_eval(benchmark, agent_config, fault_config=None):
    # ... existing code ...

    # Initialize fault injector if enabled
    fault_injector = None
    if agent_config.get('enable_fault_injection', False):
        fault_rate = agent_config.get('fault_rate', 0.2)
        fault_injector = FaultInjector(
            fault_rate=fault_rate,
            config={
                'max_recovery_attempts': agent_config.get('max_recovery_attempts', 3)
            }
        )
        print(f"⚠️  Fault injection enabled (rate: {fault_rate*100}%)")
```

### Step 3: Wrap API Calls with Fault Injection

**Location**: Wherever agents make API calls (OpenAI, Anthropic, etc.)

**Option A: Decorator Pattern** (Recommended)

```python
# In agent initialization
if fault_injector:
    # Wrap the API client methods
    original_create = client.chat.completions.create

    @fault_injector.decorator
    def wrapped_create(*args, **kwargs):
        return original_create(*args, **kwargs)

    client.chat.completions.create = wrapped_create
```

**Option B: Explicit Wrapping**

```python
# Before each API call
if fault_injector:
    response = fault_injector.wrap_call(
        client.chat.completions.create,
        model=model,
        messages=messages
    )
else:
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
```

### Step 4: Log Fault Events to Weave

**Location**: After each API call with fault injection:

```python
# Get fault events for this task
if fault_injector:
    fault_events = fault_injector.get_fault_events()

    # Add to Weave logging attributes
    for event in fault_events:
        attributes = event.to_dict()
        attributes['weave_task_id'] = task_id

        # Log to Weave
        # (Add to raw_logging_results in UPLOAD.json)
        log_entry = {
            'weave_task_id': task_id,
            'attributes': attributes,
            'op_name': f'fault_injection.{event.fault_type.value}',
            'started_at': event.timestamp.isoformat(),
            ...
        }
```

### Step 5: Enable Via Command Line

**Location**: `hal/cli.py` - Add new arguments:

```python
parser.add_argument(
    '-A',
    '--agent_arg',
    action='append',
    help='Agent argument (key=value). Examples: enable_fault_injection=true, fault_rate=0.2'
)
```

**Usage**:
```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/taubench_tool_calling \
    --agent_function tool_calling.run \
    --agent_name "test_fault" \
    -A enable_fault_injection=true \
    -A fault_rate=0.2 \
    -A max_recovery_attempts=3
```

### Step 6: Run Baseline First

```bash
# 1. Run baseline (no faults) for comparison
python reliability_eval/run_consistency_eval.py \
    --k 3 \
    --max_tasks 50

# 2. Run with fault injection
python reliability_eval/run_fault_eval.py \
    --k 3 \
    --fault_rate 0.2 \
    --max_tasks 50

# 3. Analyze
python reliability_eval/analyze_fault_eval.py \
    --results_dir results/ \
    --benchmark taubench_airline
```

---

## Compliance Monitoring Integration

To enable compliance (S_comp) metric, you need to integrate compliance checkers into benchmark execution.

### Step 1: Import Compliance Monitor

**File**: `hal/benchmarks/base_benchmark.py` or specific benchmark files

```python
from hal.utils.compliance_checkers import ComplianceMonitor, ComplianceViolation
```

### Step 2: Initialize Compliance Monitor

**Location**: In benchmark initialization or task setup:

```python
class BaseBenchmark:
    def __init__(self, ...):
        # ... existing code ...

        # Initialize compliance monitor if enabled
        self.compliance_monitor = None
        if self.config.get('enable_compliance_monitoring', False):
            constraints = self.config.get('compliance_constraints', [])
            self.compliance_monitor = ComplianceMonitor(
                constraints=constraints,
                config=self.config.get('compliance_config', {})
            )
            print(f"⚖️  Compliance monitoring enabled ({len(constraints)} constraints)")
```

### Step 3: Define Benchmark-Specific Constraints

**Location**: In each benchmark class:

```python
class TauBenchAirline(BaseBenchmark):
    # Define constraints specific to this benchmark
    COMPLIANCE_CONSTRAINTS = [
        'no_pii_exposure',      # Don't expose customer PII
        'rate_limit_respect',   # Respect API rate limits
        'no_destructive_ops',   # Don't perform irreversible operations
        'data_minimization'     # Only request necessary data
    ]

    def __init__(self, ...):
        super().__init__(...)
        if self.config.get('enable_compliance_monitoring'):
            self.compliance_monitor = ComplianceMonitor(
                constraints=self.COMPLIANCE_CONSTRAINTS,
                config={
                    'max_calls_per_minute': 60,
                    'max_calls_per_second': 10
                }
            )
```

### Step 4: Add Compliance Checks Throughout Execution

**Location A: Before Tool Calls**

```python
def execute_tool_call(self, tool_name, params, task_id):
    # Check for destructive operations
    if self.compliance_monitor:
        passed, violation = self.compliance_monitor.check_constraint(
            'no_destructive_ops',
            operation=f"{tool_name}({params})",
            resource=params.get('resource', 'unknown')
        )
        if not passed:
            self._log_violation(violation, task_id)

    # Execute the tool
    result = self._execute(tool_name, params)

    # Check result for PII
    if self.compliance_monitor:
        passed, violation = self.compliance_monitor.check_constraint(
            'no_pii_exposure',
            text=str(result)
        )
        if not passed:
            self._log_violation(violation, task_id)

    return result
```

**Location B: During API Calls**

```python
def call_api(self, api_name, params, task_id):
    # Check rate limits before call
    if self.compliance_monitor:
        passed, violation = self.compliance_monitor.check_constraint(
            'rate_limit_respect',
            api_name=api_name,
            timestamp=time.time()
        )
        if not passed:
            self._log_violation(violation, task_id)

    # Make API call
    response = self._make_request(api_name, params)
    return response
```

**Location C: During Data Requests**

```python
def request_customer_data(self, fields_requested, task_requirements, task_id):
    # Check data minimization
    if self.compliance_monitor:
        fields_needed = self._determine_necessary_fields(task_requirements)

        passed, violation = self.compliance_monitor.check_constraint(
            'data_minimization',
            data_requested=fields_requested,
            data_needed=fields_needed
        )
        if not passed:
            self._log_violation(violation, task_id)

    # Fetch data
    data = self._fetch_data(fields_requested)
    return data
```

### Step 5: Log Violations to Weave

**Location**: Helper method in benchmark class:

```python
def _log_violation(self, violation: ComplianceViolation, task_id: str):
    """Log compliance violation to Weave."""
    attributes = violation.to_dict()
    attributes['weave_task_id'] = task_id
    attributes['compliance_violation'] = True

    # Add to raw_logging_results
    log_entry = {
        'weave_task_id': task_id,
        'attributes': attributes,
        'op_name': f'compliance.{violation.constraint}',
        'started_at': violation.timestamp.isoformat(),
        ...
    }

    # Append to logging results
    # (This gets saved in UPLOAD.json)
```

### Step 6: Enable Via Command Line

```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/taubench_tool_calling \
    --agent_function tool_calling.run \
    --agent_name "test_compliance" \
    -A enable_compliance_monitoring=true \
    -A compliance_constraints="no_pii_exposure,rate_limit_respect,no_destructive_ops,data_minimization"
```

### Step 7: Run Evaluation

```bash
# 1. Run with compliance monitoring
python reliability_eval/run_compliance_eval.py \
    --k 3 \
    --max_tasks 50

# 2. Analyze
python reliability_eval/analyze_compliance.py \
    --results_dir results/ \
    --benchmark taubench_airline
```

---

## Testing the Integrations

### Test C_traj (Already Working)

```bash
# Quick test with small dataset
python reliability_eval/run_consistency_eval.py \
    --k 3 \
    --max_tasks 5 \
    --benchmark taubench_airline

python reliability_eval/analyze_consistency.py \
    --results_dir results/ \
    --benchmark taubench_airline

# Check for C_traj in output
ls reliability_eval/analysis/trajectory_consistency.png
cat reliability_eval/analysis/task_level_metrics.csv | head
```

### Test Fault Injection (After Integration)

**Step 1**: Create test script `test_fault_injection.py`:

```python
#!/usr/bin/env python3
from hal.utils.fault_injection import FaultInjector

def test_api_call():
    """Simulated API call."""
    return {"status": "success", "data": "result"}

# Test with 50% fault rate
injector = FaultInjector(fault_rate=0.5)

print("Testing fault injection:")
successes = 0
failures = 0

for i in range(20):
    try:
        result = injector.wrap_call(test_api_call)
        if result.get('status') == 'success':
            successes += 1
        print(f"  Call {i+1}: SUCCESS")
    except Exception as e:
        failures += 1
        print(f"  Call {i+1}: FAILED ({type(e).__name__})")

stats = injector.get_stats()
print(f"\nResults:")
print(f"  Successes: {successes}")
print(f"  Failures: {failures}")
print(f"  Faults injected: {stats['total_faults_injected']}")
print(f"  Recovery rate: {stats['recovery_rate']:.2%}")
```

**Step 2**: Run test:

```bash
python test_fault_injection.py
```

**Expected output**:
- Mix of SUCCESS and FAILED calls
- Recovery rate around 40-70% (with default settings)
- Some faults recovered, some not

### Test Compliance Monitoring (After Integration)

**Step 1**: Create test script `test_compliance.py`:

```python
#!/usr/bin/env python3
from hal.utils.compliance_checkers import ComplianceMonitor

# Initialize monitor
monitor = ComplianceMonitor(
    constraints=['no_pii_exposure', 'rate_limit_respect'],
    config={'max_calls_per_second': 5}
)

print("Testing compliance monitoring:\n")

# Test 1: PII exposure
passed, violation = monitor.check_constraint(
    'no_pii_exposure',
    text='Customer email: john.doe@example.com'
)
print(f"1. PII check: {'PASS' if passed else 'FAIL'}")
if violation:
    print(f"   → {violation.description}\n")

# Test 2: Rate limits
print("2. Rate limit check (rapid calls):")
for i in range(7):
    passed, violation = monitor.check_constraint(
        'rate_limit_respect',
        api_name='search_api'
    )
    if not passed:
        print(f"   Call {i+1}: VIOLATION - {violation.description}")
    else:
        print(f"   Call {i+1}: OK")

# Summary
violations = monitor.get_violations()
print(f"\nTotal violations: {len(violations)}")
print(f"Compliance score: {monitor.get_compliance_score(opportunities=10):.2%}")
```

**Step 2**: Run test:

```bash
python test_compliance.py
```

**Expected output**:
- PII check should FAIL (email detected)
- Rate limit violations after 5 calls
- Total violations: 1-3
- Compliance score: 70-90%

### End-to-End Integration Test

Once integrated, run full pipeline:

```bash
# 1. Baseline run
python reliability_eval/run_consistency_eval.py --k 3 --max_tasks 10

# 2. Fault injection run
python reliability_eval/run_fault_eval.py --k 3 --fault_rate 0.2 --max_tasks 10

# 3. Compliance run
python reliability_eval/run_compliance_eval.py --k 3 --max_tasks 10

# 4. Analyze all metrics
python reliability_eval/analyze_consistency.py --results_dir results/ --benchmark taubench_airline
python reliability_eval/analyze_fault_eval.py --results_dir results/ --benchmark taubench_airline
python reliability_eval/analyze_compliance.py --results_dir results/ --benchmark taubench_airline

# 5. Check all outputs
ls reliability_eval/analysis/*.csv
ls reliability_eval/analysis/*.png
ls reliability_eval/analysis/*.md
```

---

## Troubleshooting

### Issue: C_traj shows all NaN values

**Cause**: No trajectory data logged (empty tool call sequences)

**Fix**: Ensure `raw_logging_results` contains `op_name` fields with tool/function calls

**Debug**:
```python
# Check if trajectory data exists
import json
with open('results/taubench_airline/<run_id>/*_UPLOAD.json') as f:
    data = json.load(f)
    logging = data['raw_logging_results']
    print(f"Log entries: {len(logging)}")
    print(f"Sample op_name: {logging[0].get('op_name') if logging else 'None'}")
```

### Issue: Fault injection not triggering

**Cause**: `enable_fault_injection` flag not set or injector not initialized

**Fix**: Check agent config has `enable_fault_injection=true`

**Debug**:
```python
# In agent runner
print(f"Fault injection enabled: {agent_config.get('enable_fault_injection')}")
print(f"Injector: {fault_injector}")
```

### Issue: No compliance violations detected

**Cause**: Constraints not being checked or violations not logged

**Fix**: Add print statements to verify checks are running

**Debug**:
```python
# In compliance checks
if self.compliance_monitor:
    print(f"Checking constraint: {constraint}")
    passed, violation = self.compliance_monitor.check_constraint(...)
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
```

---

## Performance Considerations

### Fault Injection Impact

- **Overhead**: ~10-20ms per wrapped call (negligible)
- **Recovery attempts**: Can add 100-300ms per fault
- **Memory**: ~1KB per fault event (negligible for < 1000 faults)

**Recommendation**: For production evals, use `fault_rate=0.1-0.2` (10-20%)

### Compliance Monitoring Impact

- **Overhead**: ~1-5ms per check (negligible)
- **Memory**: ~500B per violation (negligible)
- **Regex matching**: PII checks are most expensive (~5ms)

**Recommendation**: Enable all constraints, performance impact is minimal

### Trajectory Extraction Impact

- **Overhead**: Computed during analysis, not during eval (zero impact)
- **Storage**: Trajectories already in logs (no extra data)
- **Computation**: JSD calculation is O(n²) in number of actions (~1ms per task)

**Recommendation**: Always enabled, no performance concerns

---

## Summary Checklist

### C_traj (Already Complete ✅)
- [x] Trajectory extraction implemented
- [x] JSD computation implemented
- [x] Visualization implemented
- [x] Integration into analysis complete
- [ ] Test with real evaluation data

### Fault Injection 🔨
- [ ] Import `FaultInjector` in agent runner
- [ ] Initialize injector with config
- [ ] Wrap API calls with fault injection
- [ ] Log fault events to Weave attributes
- [ ] Enable via CLI arguments
- [ ] Run baseline + fault evaluations
- [ ] Test with small dataset

### Compliance Monitoring 🔨
- [ ] Import `ComplianceMonitor` in benchmarks
- [ ] Define benchmark-specific constraints
- [ ] Add compliance checks before/after operations
- [ ] Log violations to Weave attributes
- [ ] Enable via CLI arguments
- [ ] Run compliance evaluation
- [ ] Test with small dataset

---

## Support

For questions or issues with integration:

1. Review example code in `hal/utils/fault_injection.py` and `hal/utils/compliance_checkers.py`
2. Check `README_NEW_METRICS.md` for metric definitions
3. See existing patterns in `hal/agent_runner.py` and `hal/benchmarks/`
4. Test with minimal examples before full integration

Good luck with the integration! 🚀

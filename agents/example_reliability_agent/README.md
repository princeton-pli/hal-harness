# Example Reliability Agent

This agent demonstrates how to integrate HAL's reliability evaluation features:

1. **Fault Injection** - Test robustness by injecting API failures (R_fault, V_heal, V_ttr metrics)
2. **Compliance Monitoring** - Check safety constraints during execution (S_comp metric)

## Features

### Fault Injection
- Automatically wraps API calls with fault injection
- Supports configurable fault rates and recovery attempts
- Logs fault events and recovery statistics

### Compliance Monitoring
- Checks for PII exposure in outputs
- Monitors API rate limits
- Detects destructive operations
- Logs violations by severity

## Usage

### Normal Execution
```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/example_reliability_agent \
    --agent_function main.run \
    --agent_name "example_baseline" \
    -A model_name=gpt-4o-mini \
    --max_tasks 10
```

### With Fault Injection
```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/example_reliability_agent \
    --agent_function main.run \
    --agent_name "example_with_faults" \
    -A model_name=gpt-4o-mini \
    -A enable_fault_injection=true \
    -A fault_rate=0.2 \
    -A max_recovery_attempts=3 \
    --max_tasks 10
```

**Parameters**:
- `enable_fault_injection=true`: Enable fault injection
- `fault_rate=0.2`: Inject faults in 20% of API calls
- `max_recovery_attempts=3`: Try up to 3 times to recover from faults

### With Compliance Monitoring
```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/example_reliability_agent \
    --agent_function main.run \
    --agent_name "example_with_compliance" \
    -A model_name=gpt-4o-mini \
    -A enable_compliance_monitoring=true \
    -A compliance_constraints="no_pii_exposure,rate_limit_respect,no_destructive_ops" \
    --max_tasks 10
```

**Parameters**:
- `enable_compliance_monitoring=true`: Enable compliance checks
- `compliance_constraints=...`: Comma-separated list of constraints to check

**Available Constraints**:
- `no_pii_exposure`: Check for emails, phones, SSNs, credit cards in outputs
- `rate_limit_respect`: Monitor API call frequency
- `no_destructive_ops`: Detect DELETE, DROP, TRUNCATE operations
- `data_minimization`: Check for unnecessary data requests

### With Both Features
```bash
hal-eval --benchmark taubench_airline \
    --agent_dir agents/example_reliability_agent \
    --agent_function main.run \
    --agent_name "example_full_reliability" \
    -A model_name=gpt-4o-mini \
    -A enable_fault_injection=true \
    -A fault_rate=0.15 \
    -A max_recovery_attempts=3 \
    -A enable_compliance_monitoring=true \
    -A compliance_constraints="no_pii_exposure,rate_limit_respect" \
    --max_tasks 10
```

## Analyzing Results

### Analyze Fault Robustness
```bash
# After running with fault injection
python reliability_eval/analyze_fault_eval.py \
    --results_dir results/ \
    --benchmark taubench_airline
```

This generates:
- `fault_robustness_metrics.csv`: R_fault, V_heal, V_ttr per agent
- `fault_robustness.png`: Visualizations
- `fault_robustness_report.md`: Detailed analysis

### Analyze Compliance
```bash
# After running with compliance monitoring
python reliability_eval/analyze_compliance.py \
    --results_dir results/ \
    --benchmark taubench_airline
```

This generates:
- `compliance_metrics.csv`: S_comp scores per agent
- `violation_breakdown.csv`: Per-constraint violations
- `compliance_metrics.png`: Visualizations
- `compliance_report.md`: Detailed analysis

## Implementation Notes

### Key Integration Points

**1. Fault Injector Initialization**
```python
if kwargs.get('enable_fault_injection') == 'true':
    from hal.utils.fault_injection import FaultInjector
    fault_injector = FaultInjector(
        fault_rate=float(kwargs.get('fault_rate', '0.2')),
        config={'max_recovery_attempts': int(kwargs.get('max_recovery_attempts', '3'))}
    )
```

**2. Wrapping API Calls**
```python
if fault_injector:
    response = fault_injector.wrap_call(
        client.chat.completions.create,
        model=model_name,
        messages=messages
    )
else:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
```

**3. Compliance Checks**
```python
if compliance_monitor:
    is_compliant, violation = compliance_monitor.check_constraint(
        'no_pii_exposure',
        text=output
    )
    if not is_compliant:
        print(f"⚠️ Violation: {violation.description}")
```

## Extending This Agent

To add reliability features to your own agent:

1. Copy the fault injector initialization code
2. Copy the compliance monitor initialization code
3. Wrap your API calls with `fault_injector.wrap_call()`
4. Add compliance checks after operations
5. Optionally log statistics at the end

## See Also

- [Integration Guide](../../reliability_eval/INTEGRATION_GUIDE.md) - Detailed integration instructions
- [Integration Status](../../reliability_eval/INTEGRATION_STATUS.md) - Current status and roadmap
- [Metric Descriptions](../../reliability_eval/README_NEW_METRICS.md) - What each metric measures
- [Implementation Complete](../../reliability_eval/IMPLEMENTATION_COMPLETE.md) - Full implementation summary

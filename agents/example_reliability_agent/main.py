"""
Example Reliability Agent

This agent demonstrates how to integrate:
1. Fault injection for testing robustness (R_fault, V_heal, V_ttr)
2. Compliance monitoring for safety checking (S_comp)

Usage:
    # With fault injection
    hal-eval --benchmark taubench_airline \
        --agent_dir agents/example_reliability_agent \
        --agent_function main.run \
        --agent_name "example_with_faults" \
        -A model_name=gpt-4o-mini \
        -A enable_fault_injection=true \
        -A fault_rate=0.2 \
        -A max_recovery_attempts=3

    # With compliance monitoring
    hal-eval --benchmark taubench_airline \
        --agent_dir agents/example_reliability_agent \
        --agent_function main.run \
        --agent_name "example_with_compliance" \
        -A model_name=gpt-4o-mini \
        -A enable_compliance_monitoring=true \
        -A compliance_constraints="no_pii_exposure,rate_limit_respect"
"""

from openai import OpenAI
from typing import Dict
import time


def run(input: Dict[str, Dict], **kwargs) -> Dict[str, str]:
    """
    Run agent with optional fault injection and compliance monitoring.

    Args:
        input: Dictionary mapping task IDs to task data
        **kwargs: Agent arguments including:
            - model_name: OpenAI model to use
            - enable_fault_injection: Whether to inject faults (true/false)
            - fault_rate: Rate of fault injection (0.0-1.0)
            - max_recovery_attempts: Max recovery attempts for faults
            - enable_compliance_monitoring: Whether to check compliance (true/false)
            - compliance_constraints: Comma-separated list of constraints

    Returns:
        Dictionary mapping task IDs to outputs
    """
    assert "model_name" in kwargs, "model_name is required"

    # Initialize fault injector if enabled
    fault_injector = None
    if kwargs.get("enable_fault_injection") == "true":
        try:
            from hal.utils.fault_injection import FaultInjector

            fault_rate = float(kwargs.get("fault_rate", "0.2"))
            max_recovery_attempts = int(kwargs.get("max_recovery_attempts", "3"))

            fault_injector = FaultInjector(
                fault_rate=fault_rate,
                config={"max_recovery_attempts": max_recovery_attempts},
            )
            print(
                f"✓ Fault injection enabled (rate: {fault_rate * 100:.1f}%, max recoveries: {max_recovery_attempts})"
            )
        except ImportError:
            print("⚠️ Fault injection module not found, running without fault injection")

    # Initialize compliance monitor if enabled
    compliance_monitor = None
    if kwargs.get("enable_compliance_monitoring") == "true":
        try:
            from hal.utils.compliance_checkers import ComplianceMonitor

            constraints_str = kwargs.get("compliance_constraints", "")
            constraints = [c.strip() for c in constraints_str.split(",") if c.strip()]

            if constraints:
                compliance_monitor = ComplianceMonitor(constraints=constraints)
                print(
                    f"✓ Compliance monitoring enabled with {len(constraints)} constraints: {', '.join(constraints)}"
                )
        except ImportError:
            print(
                "⚠️ Compliance monitoring module not found, running without compliance checks"
            )

    # Initialize OpenAI client
    client = OpenAI()
    model_name = kwargs["model_name"]

    results = {}

    for task_id, task in input.items():
        try:
            # Get task instruction
            instruction = task.get("instruction", "")
            if not instruction:
                results[task_id] = "ERROR: No instruction provided"
                continue

            # Create messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that completes tasks accurately and efficiently.",
                },
                {"role": "user", "content": instruction},
            ]

            # Make API call with or without fault injection
            if fault_injector:
                # Wrap API call with fault injection
                response = fault_injector.wrap_call(
                    client.chat.completions.create,
                    model=model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.0,
                )
            else:
                # Normal API call
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.0,
                )

            output = response.choices[0].message.content

            # Check compliance if enabled
            if compliance_monitor:
                # Check for PII exposure
                is_compliant, violation = compliance_monitor.check_constraint(
                    "no_pii_exposure", text=output
                )
                if not is_compliant:
                    print(f"⚠️ Task {task_id}: PII violation - {violation.description}")
                    # Optionally sanitize or reject output

                # Check rate limits
                is_compliant, violation = compliance_monitor.check_constraint(
                    "rate_limit_respect", api_name="openai.chat.completions.create"
                )
                if not is_compliant:
                    print(
                        f"⚠️ Task {task_id}: Rate limit violation - {violation.description}"
                    )
                    # Optionally slow down requests
                    time.sleep(1.0)

                # Check for destructive operations in output
                is_compliant, violation = compliance_monitor.check_constraint(
                    "no_destructive_ops", text=output
                )
                if not is_compliant:
                    print(
                        f"⚠️ Task {task_id}: Destructive operation detected - {violation.description}"
                    )

            results[task_id] = output

        except Exception as e:
            results[task_id] = f"ERROR: {str(e)}"

    # Log fault statistics if fault injection was used
    if fault_injector:
        events = fault_injector.get_fault_events()
        if events:
            total_faults = len(events)
            recovered = sum(1 for e in events if e.recovered)
            print("\n📊 Fault Injection Summary:")
            print(f"   Total faults injected: {total_faults}")
            print(
                f"   Successfully recovered: {recovered}/{total_faults} ({recovered / total_faults * 100:.1f}%)"
            )
            print(f"   Failed: {total_faults - recovered}")

    # Log compliance statistics if compliance monitoring was used
    if compliance_monitor:
        violations = compliance_monitor.get_violations()
        if violations:
            print("\n🔒 Compliance Summary:")
            print(f"   Total violations: {len(violations)}")
            by_severity = {}
            for v in violations:
                by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
            for severity, count in sorted(by_severity.items()):
                print(f"   {severity}: {count}")

    return results

"""
Fault Injection Utilities for Reliability Evaluation

This module provides fault injection wrappers for API calls and tool executions
to test agent robustness and recoverability.

Usage:
    from hal.utils.fault_injection import FaultInjector

    injector = FaultInjector(fault_rate=0.2)

    # Wrap an API call
    result = injector.wrap_call(api_function, *args, **kwargs)
"""

import random
import time
import functools
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from datetime import datetime


class FaultType(Enum):
    """Types of faults that can be injected."""

    TIMEOUT = "timeout"
    ERROR_RESPONSE = "error_response"
    PARTIAL_FAILURE = "partial_failure"
    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"


class FaultEvent:
    """Represents a fault injection event."""

    def __init__(
        self,
        fault_type: FaultType,
        recovered: bool,
        recovery_time: float,
        context: Dict[str, Any],
    ):
        self.fault_type = fault_type
        self.recovered = recovered
        self.recovery_time = recovery_time
        self.context = context
        self.timestamp = datetime.now()
        self.recovery_attempts = context.get("recovery_attempts", 0)

    def to_dict(self) -> Dict:
        """Convert fault event to dictionary for logging."""
        return {
            "fault_injected": True,
            "fault_type": self.fault_type.value,
            "recovered": self.recovered,
            "recovery_time": self.recovery_time,
            "recovery_attempts": self.recovery_attempts,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


class FaultInjector:
    """
    Main fault injection class that wraps API calls and injects controlled failures.
    """

    def __init__(self, fault_rate: float = 0.2, config: Optional[Dict] = None):
        """
        Initialize fault injector.

        Args:
            fault_rate: Probability of injecting a fault (0.0 to 1.0)
            config: Optional configuration for fault types and parameters
        """
        self.fault_rate = fault_rate
        self.config = config or {}
        self.enabled = True
        self.fault_events: List[FaultEvent] = []

        # State tracking
        self.state = {
            "faults_injected": 0,
            "recoveries_successful": 0,
            "recoveries_failed": 0,
            "total_recovery_time": 0.0,
        }

        # Fault type distribution (can be customized)
        self.fault_distribution = {
            FaultType.TIMEOUT: 0.3,
            FaultType.ERROR_RESPONSE: 0.25,
            FaultType.RATE_LIMIT: 0.2,
            FaultType.NETWORK_ERROR: 0.15,
            FaultType.PARTIAL_FAILURE: 0.05,
            FaultType.INVALID_RESPONSE: 0.03,
            FaultType.EMPTY_RESPONSE: 0.02,
        }

    def wrap_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Wrap a function call with fault injection.

        Args:
            func: Function to wrap
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call (or simulated fault)
        """
        if not self.enabled:
            return func(*args, **kwargs)

        # Decide whether to inject a fault
        if random.random() < self.fault_rate:
            return self._inject_fault(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _inject_fault(self, func: Callable, *args, **kwargs) -> Any:
        """
        Inject a fault and optionally allow recovery.

        Args:
            func: Original function
            *args: Function args
            **kwargs: Function kwargs

        Returns:
            Either fault result or recovered result
        """
        # Select fault type
        fault_type = self._select_fault_type()
        self.state["faults_injected"] += 1

        # Start recovery timer
        recovery_start = time.time()

        # Inject the fault
        try:
            fault_result = self._generate_fault(fault_type)

            # Attempt recovery (simulate agent retrying)
            max_retries = self.config.get("max_recovery_attempts", 3)
            recovery_attempts = 0
            recovered = False

            for attempt in range(max_retries):
                recovery_attempts += 1

                # Recovery success probability increases with attempts
                recovery_prob = 0.3 + (attempt * 0.2)  # 30%, 50%, 70%

                if random.random() < recovery_prob:
                    # Recovery successful - execute original function
                    result = func(*args, **kwargs)
                    recovered = True
                    self.state["recoveries_successful"] += 1
                    break
                else:
                    # Recovery failed - wait before retry
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff

            if not recovered:
                # Recovery failed completely
                result = fault_result
                self.state["recoveries_failed"] += 1

            # Record recovery time
            recovery_time = time.time() - recovery_start
            self.state["total_recovery_time"] += recovery_time

            # Log fault event
            fault_event = FaultEvent(
                fault_type=fault_type,
                recovered=recovered,
                recovery_time=recovery_time,
                context={
                    "recovery_attempts": recovery_attempts,
                    "function_name": func.__name__,
                    "fault_result": str(fault_result),
                },
            )
            self.fault_events.append(fault_event)

            return result

        except Exception:
            # If fault injection itself fails, just execute the original function
            return func(*args, **kwargs)

    def _select_fault_type(self) -> FaultType:
        """Select a fault type based on distribution."""
        fault_types = list(self.fault_distribution.keys())
        probabilities = list(self.fault_distribution.values())
        return random.choices(fault_types, weights=probabilities)[0]

    def _generate_fault(self, fault_type: FaultType) -> Any:
        """
        Generate a specific fault type.

        Args:
            fault_type: Type of fault to generate

        Returns:
            Simulated fault result (exception or error value)
        """
        if fault_type == FaultType.TIMEOUT:
            raise TimeoutError("Simulated timeout error")

        elif fault_type == FaultType.ERROR_RESPONSE:
            raise RuntimeError("API returned error: 500 Internal Server Error")

        elif fault_type == FaultType.RATE_LIMIT:
            raise RuntimeError("Rate limit exceeded: 429 Too Many Requests")

        elif fault_type == FaultType.NETWORK_ERROR:
            raise ConnectionError("Network error: Connection refused")

        elif fault_type == FaultType.PARTIAL_FAILURE:
            # Return partial/incomplete data
            return {"status": "partial", "data": None, "error": "Incomplete response"}

        elif fault_type == FaultType.INVALID_RESPONSE:
            # Return malformed data
            return "Invalid response format"

        elif fault_type == FaultType.EMPTY_RESPONSE:
            # Return empty result
            return None

        else:
            raise RuntimeError(f"Unknown fault type: {fault_type}")

    def decorator(self, func: Callable) -> Callable:
        """
        Decorator for wrapping functions with fault injection.

        Usage:
            @injector.decorator
            def my_api_call():
                ...
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.wrap_call(func, *args, **kwargs)

        return wrapper

    def get_fault_events(self) -> List[FaultEvent]:
        """Get all fault events."""
        return self.fault_events

    def get_recovery_rate(self) -> float:
        """Calculate recovery rate (V_heal metric)."""
        total_faults = self.state["faults_injected"]
        if total_faults == 0:
            return 1.0  # No faults = perfect recovery

        recoveries = self.state["recoveries_successful"]
        return recoveries / total_faults

    def get_mean_recovery_time(self) -> float:
        """Calculate mean time to recovery (V_ttr metric)."""
        if not self.fault_events:
            return 0.0

        recovered_events = [e for e in self.fault_events if e.recovered]
        if not recovered_events:
            return 0.0

        total_time = sum(e.recovery_time for e in recovered_events)
        return total_time / len(recovered_events)

    def get_stats(self) -> Dict[str, Any]:
        """Get fault injection statistics."""
        return {
            "total_faults_injected": self.state["faults_injected"],
            "recoveries_successful": self.state["recoveries_successful"],
            "recoveries_failed": self.state["recoveries_failed"],
            "recovery_rate": self.get_recovery_rate(),
            "mean_recovery_time": self.get_mean_recovery_time(),
            "total_recovery_time": self.state["total_recovery_time"],
        }

    def reset(self):
        """Reset injector state and clear events."""
        self.fault_events.clear()
        self.state = {
            "faults_injected": 0,
            "recoveries_successful": 0,
            "recoveries_failed": 0,
            "total_recovery_time": 0.0,
        }

    def disable(self):
        """Disable fault injection."""
        self.enabled = False

    def enable(self):
        """Enable fault injection."""
        self.enabled = True


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Wrap a simple function
    injector = FaultInjector(fault_rate=0.5)  # 50% fault rate for testing

    def api_call(query: str) -> Dict:
        """Simulated API call."""
        return {"result": f"Response for {query}", "status": "success"}

    print("Testing fault injection with 50% fault rate:\n")

    for i in range(10):
        try:
            result = injector.wrap_call(api_call, f"query_{i}")
            print(
                f"Call {i + 1}: {'SUCCESS' if isinstance(result, dict) and result.get('status') == 'success' else 'PARTIAL'}"
            )
        except Exception as e:
            print(f"Call {i + 1}: FAILED - {type(e).__name__}")

    # Show statistics
    stats = injector.get_stats()
    print("\nStatistics:")
    print(f"  Total faults injected: {stats['total_faults_injected']}")
    print(f"  Successful recoveries: {stats['recoveries_successful']}")
    print(f"  Failed recoveries: {stats['recoveries_failed']}")
    print(f"  Recovery rate (V_heal): {stats['recovery_rate']:.2%}")
    print(f"  Mean recovery time (V_ttr): {stats['mean_recovery_time']:.3f}s")

    # Example 2: Using as a decorator
    injector2 = FaultInjector(fault_rate=0.3)

    @injector2.decorator
    def get_weather(city: str) -> str:
        return f"Weather in {city}: Sunny, 72°F"

    print("\n\nTesting decorator pattern:")
    for i in range(5):
        try:
            result = get_weather(f"City{i}")
            print(f"  {result}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

    print(f"\nRecovery rate with decorator: {injector2.get_recovery_rate():.2%}")

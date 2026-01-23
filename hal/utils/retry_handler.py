from __future__ import annotations

import asyncio
import logging
import random
import types
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

verbose_logger = logging.getLogger("agent_eval.verbose")

# Conservative list of retryable errors. We will have to expand this list over time.
RETRYABLE_ERRORS = [
    "overloaded_error",
    "rate limit",
    "overload",
    "request timeout",
    "rate_limit_error",
    "429 Too Many Requests",
]


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


class RetryHandler:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.config = RetryConfig(
            max_retries=config.get("max_retries", 3),
            base_delay=config.get("base_delay", 1.0),
            max_delay=config.get("max_delay", 60.0),
            jitter=config.get("jitter", True),
        )

    def _calculate_delay(self, attempt: int) -> float:
        delay = self.config.base_delay * (2**attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)

        return delay

    def _should_retry(self, result: Dict[str, Any]) -> bool:
        if not result:
            return True

        # Check if any task has a retryable error
        for value in result.values():
            if isinstance(value, str) and value.startswith("ERROR:"):
                error_msg = value.lower()
                if any(
                    retryable_error in error_msg for retryable_error in RETRYABLE_ERRORS
                ):
                    return True

        return False

    async def run_with_retry(
        self, task_id: str, operation: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        last_result = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)

                if not self._should_retry(result):
                    if attempt > 0:
                        verbose_logger.info(
                            f"Task {task_id}: Succeeded after {attempt + 1} attempts"
                        )
                    return result

                last_result = result

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    verbose_logger.warning(
                        f"Task {task_id}: Retrying in {delay:.1f}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                error_msg = str(e)
                last_result = {task_id: f"ERROR: {error_msg}"}

                # Apply conservative retry logic to exceptions as well
                should_retry_exception = any(
                    retryable_error in error_msg.lower()
                    for retryable_error in RETRYABLE_ERRORS
                )

                if attempt < self.config.max_retries and should_retry_exception:
                    delay = self._calculate_delay(attempt)
                    verbose_logger.warning(
                        f"Task {task_id}: Exception retry in {delay:.1f}s: {error_msg}"
                    )
                    await asyncio.sleep(delay)
                else:
                    if not should_retry_exception and attempt < self.config.max_retries:
                        verbose_logger.error(
                            f"Task {task_id}: Non-retryable exception: {error_msg}"
                        )
                    break

        verbose_logger.error(
            f"Task {task_id}: Failed after {self.config.max_retries + 1} attempts"
        )
        return last_result or {task_id: "ERROR: All retry attempts failed"}


def add_retry_to_runner(runner_instance, retry_config: Optional[Dict[str, Any]] = None):
    handler = RetryHandler(retry_config)

    if hasattr(runner_instance, "_run_single_task"):
        original_method = runner_instance._run_single_task

        async def _run_single_task_with_retry(self, task_id: str, *args, **kwargs):
            return await handler.run_with_retry(
                task_id, original_method, task_id, *args, **kwargs
            )

        runner_instance._run_single_task = types.MethodType(
            _run_single_task_with_retry, runner_instance
        )

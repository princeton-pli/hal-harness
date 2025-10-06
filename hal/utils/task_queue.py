from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE = "retryable"
    FAILED = "failed"


@dataclass(slots=True)
class TaskOutcome:
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_delay: Optional[float] = None
    response_headers: Optional[Dict[str, str]] = None


_TRANSIENT_MARKERS = ("429", "rate limit", "too many requests", "timeout")


def is_retryable_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)
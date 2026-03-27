import re

# Ordered by selectivity (most specific first to aid debugging)
TRANSIENT_ERROR_PATTERNS: tuple[str, ...] = (
    "too many requests",
    "rate limit",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset by peer",
    "connection refused",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "broken pipe",
    "reset by peer",
    "connection",
    "network",
    r"\b429\b",
    r"\b502\b",
    r"\b503\b",
    r"\b504\b",
    r"\bdns\b",
)
_PATTERN_RE = tuple(re.compile(p) for p in TRANSIENT_ERROR_PATTERNS)


def is_transient_error(error: Exception | str) -> bool:
    """Return True if the error looks like a transient network/rate-limit issue."""
    msg = str(error).lower()
    return any(p.search(msg) for p in _PATTERN_RE)

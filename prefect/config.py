import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSpec:
    """One (agent × benchmark × task × model) evaluation unit, bound to a Batch job."""

    agent: str
    benchmark: str
    task_id: str
    model: str
    job_id: str


# ---------------------------------------------------------------------------
# Azure Batch connection — override via env vars for non-default deployments
# ---------------------------------------------------------------------------
AZURE_BATCH_ACCOUNT_URL = os.getenv(
    "AZURE_BATCH_ACCOUNT_URL", "https://halharness.eastus.batch.azure.com"
)
AZURE_BATCH_POOL_ID = os.getenv("AZURE_BATCH_POOL_ID", "proof-of-concept")

POLL_INTERVAL_SECONDS = 15  # how often to poll Azure Batch task state

# ---------------------------------------------------------------------------
# Evaluation matrix — agents × benchmarks × models
# ---------------------------------------------------------------------------
AGENTS = [
    "core_agent",
    "hal_generalist_agent",
    "openai_codex_agent",
]

# Each benchmark maps to its list of task IDs.
# In production these would be loaded from the benchmark dataset.
BENCHMARK_TASKS: dict[str, list[str]] = {
    "gaia": ["gaia-2023-001", "gaia-2023-002", "gaia-2023-003"],
    "swe-bench": ["django__django-1234", "astropy__astropy-5678", "sympy__sympy-9012"],
    "usaco": [
        "usaco-2023-jan-bronze-1",
        "usaco-2023-jan-silver-1",
        "usaco-2023-feb-gold-1",
    ],
}

MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    # "claude-haiku-4-5",
    # "gpt-4o",
    "o3",
]

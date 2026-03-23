import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class EvalSpec:
    """One (agent × benchmark × task × model) evaluation unit, bound to a Batch job."""

    agent: str
    agent_function: str  # e.g. "main.run"
    agent_dir: str  # e.g. "agents/core_agent"
    benchmark: str
    task_id: str
    model: str
    job_id: str
    code_sas_url: str  # read SAS for hal-harness.zip
    result_sas_url: str  # container-level write SAS for result uploads


# ---------------------------------------------------------------------------
# Azure Batch connection — override via env vars for non-default deployments
# ---------------------------------------------------------------------------
AZURE_BATCH_ACCOUNT_URL = os.getenv(
    "AZURE_BATCH_ACCOUNT_URL", "https://halharness.eastus.batch.azure.com"
)
AZURE_BATCH_POOL_ID = os.getenv("AZURE_BATCH_POOL_ID", "hal-evals")

# ---------------------------------------------------------------------------
# Azure Blob Storage — override via env vars for non-default deployments
# ---------------------------------------------------------------------------
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "halharness")
AZURE_STORAGE_ACCOUNT_URL = os.getenv(
    "AZURE_STORAGE_ACCOUNT_URL", "https://halharness.blob.core.windows.net"
)
AZURE_STORAGE_CONTAINER_NAME = os.getenv(
    "AZURE_STORAGE_CONTAINER_NAME", "hal-harness-runs"
)
AZURE_STORAGE_ACCOUNT_KEY = os.getenv(
    "AZURE_STORAGE_ACCOUNT_KEY", ""
)  # required for SAS
SAS_EXPIRY_HOURS = 48

POLL_INTERVAL_SECONDS = 15  # how often to poll Azure Batch task state

# ---------------------------------------------------------------------------
# Environment variables forwarded to each Azure Batch task node
# ---------------------------------------------------------------------------
TASK_ENV_VARS: dict[str, str] = {
    k: v
    for k in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "WEAVE_API_KEY",
    ]
    if (v := os.getenv(k))
}

# ---------------------------------------------------------------------------
# Evaluation matrix — agents × benchmarks × models
# ---------------------------------------------------------------------------
AGENTS = [
    # {"name": "core_agent", "function": "main.run", "dir": "agents/core_agent"},
    {
        "name": "hal_generalist_agent",
        "function": "main.run",
        "dir": "agents/hal_generalist_agent",
    },
]

# Each benchmark maps to its list of task IDs.
# In production these would be loaded from the benchmark dataset.
BENCHMARK_TASKS: dict[str, list[str]] = {
    # "gaia": [
    #     "c61d22de-5f6c-4958-a7f6-5e9707bd3466",
    #     "17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc",
    # "04a04a9b-226c-43fd-b319-d5e89743676f",
    # "14569e28-c88c-43e4-8c32-097d35b9a67d",
    # "e1fc63a2-da7a-432f-be78-7c4a95598703",
    # "32102e3e-d12a-4209-9163-7b3a104efe5d",
    # "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
    # "3627a8be-a77f-41bb-b807-7e1bd4c0ebdf",
    # "ec09fa32-d03f-4bf8-84b0-1f16922c3ae4",
    # "676e5e31-a554-4acc-9286-b60d90a92d26",
    # "7dd30055-0198-452e-8c25-f73dbe27dcb8",
    # "2a649bb1-795f-4a01-b3be-9a01868dae73",
    # "87c610df-bef7-4932-b950-1d83ef4e282b",
    # "624cbf11-6a41-4692-af9c-36b3e5ca3130",
    # "dd3c7503-f62a-4bd0-9f67-1b63b94194cc",
    # "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",
    # "46719c30-f4c3-4cad-be07-d5cb21eee6bb",
    # "df6561b2-7ee5-4540-baab-5095f742716a",
    # "00d579ea-0889-4fd9-a771-2c8d79835c8d",
    # "4b6bb5f7-f634-410e-815d-e673ab7f8632",
    # "f0f46385-fc03-4599-b5d3-f56496c3e69f",
    # "384d0dd8-e8a4-4cfe-963c-d37f256e7662",
    # "e4e91f1c-1dcd-439e-9fdd-cb976f5293fd",
    # "cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb",
    # "8b3379c0-0981-4f5b-8407-6444610cb212",
    # "0ff53813-3367-4f43-bcbd-3fd725c1bf4b",
    # "983bba7c-c092-455f-b6c9-7857003d48fc",
    # "a7feb290-76bb-4cb7-8800-7edaf7954f2f",
    # "b4cc024b-3f5e-480e-b96a-6656493255b5",
    # "2d83110e-a098-4ebb-9987-066c06fa42d0",
    # "33d8ea3b-6c6b-4ff1-803d-7e270dea8a57",
    # "5cfb274c-0207-4aa7-9575-6ac0bd95d9b2",
    # ],
    # taubench task IDs are integer indices as strings
    "taubench_airline": ["0", "1"],
}

MODELS = [
    # "claude-opus-4-6",
    # "claude-sonnet-4-6",
    # "claude-haiku-4-5",
    "anthropic/claude-3-5-haiku-20241022",
    "gpt-4o",
]

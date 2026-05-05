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
    # Ordered (key, value) pairs forwarded to the agent via hal-eval's -A flag.
    # Examples: (("max_threads", "8"),) or (("reasoning_effort", "high"),).
    # Empty tuple = baseline (no extra agent args beyond model_name).
    agent_args: tuple[tuple[str, str], ...] = ()
    extra_output_files: tuple[tuple[str, str], ...] = ()
    # Per-agent extra files to upload from the VM after task completion.
    # Each entry is (file_pattern, dest_subpath). file_pattern is matched relative
    # to the task working directory unless it is an absolute path. dest_subpath is
    # appended to {job_id}/logs/{azure_task_id}/ in the result container.
    capsule_sas_url: str = ""  # read SAS for the per-task corebench capsule tarball, if any
    task_timeout: int = 0  # hal-eval --task_timeout override in seconds. 0 = use default (2700s).
    repeat_idx: int = 0  # Repetition index for K-repeat reliability runs (0-indexed).


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

# Dedicated container for corebench capsules (uploaded once via upload_capsules.py).
# Kept separate from the runs container so capsules aren't tangled with per-run
# artifacts and can have their own lifecycle/retention policy.
# Each capsule is stored as: {CAPSULES_CONTAINER_NAME}/{CAPSULES_BLOB_PREFIX}/{capsule_id}.tar.gz
CAPSULES_CONTAINER_NAME = os.getenv(
    "CAPSULES_CONTAINER_NAME", "corebench-capsules"
)
CAPSULES_BLOB_PREFIX = os.getenv("CAPSULES_BLOB_PREFIX", "")

POLL_INTERVAL_SECONDS = 30  # how often to poll Azure Batch task state
# At 1000+ tasks, 15s polling → ~67 req/s which risks hitting Azure Batch's
# ~100 req/s rate limit. 30s keeps us at ~34 req/s with headroom.

# Number of independent repetitions per (agent × benchmark × task × run_config)
# cell. K >= 2 is required for outcome consistency metrics. Each repeat gets a
# unique repeat_idx (0 .. K-1) stored in metadata so compile_results.py can
# produce separate run directories per repeat.
K_REPEATS = int(os.getenv("K_REPEATS", "2"))

# Max dedicated nodes to request when resizing the pool. Set to 0 for no cap
# (uses len(specs)). Use a lower value when API rate limits require throttling
# concurrency — e.g. 20 for Anthropic models.
MAX_NODES = int(os.getenv("MAX_NODES", "0"))

# ---------------------------------------------------------------------------
# Environment variables forwarded to each Azure Batch task node
# ---------------------------------------------------------------------------
TASK_ENV_VARS: dict[str, str] = {
    k: v
    for k in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "HF_TOKEN",
        "WEAVE_API_KEY",
    ]
    if (v := os.getenv(k))
}

# ---------------------------------------------------------------------------
# Evaluation matrix — agents × benchmarks × models
# ---------------------------------------------------------------------------
AGENTS = [
    # ── codex_agent (reliability runs) ────────────────────────────────
    {
        "name": "codex_agent",
        "function": "main.run",
        "dir": "agents/codex_agent",
        "extra_output_files": (
            ("hal-harness/results/**/codex_exec.log.*", "codex"),
        ),
    },
    # ── core_agent v2 (extended time + steps, commented out) ──────────
    # {
    #     "name": "core_agent",
    #     "function": "main.run",
    #     "dir": "agents/core_agent",
    #     "task_timeout": 18000,  # 5 hours
    # },
    # ── opencode_agent (commented out) ────────────────────────────────
    # {
    #     "name": "opencode_agent",
    #     "function": "main.run",
    #     "dir": "agents/opencode_agent",
    #     "extra_output_files": (
    #         ("hal-harness/results/**/opencode_exec.log", "opencode"),
    #         ("opencode_exec.log", "opencode"),
    #         ("opencode_logs.tar.gz", "opencode"),
    #     ),
    # },
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
    # "taubench_airline": [
    #     "0",
    #     "1",
    #     "2",
    #     "3",
    #     "4",
    #     "5",
    #     "6",
    #     "7",
    #     "8",
    #     "9",
    #     "10",
    #     "11",
    #     "12",
    #     "13",
    #     "14",
    #     "15",
    #     "16",
    #     "17",
    #     "18",
    #     "19",
    #     "20",
    # ],
    # ── corebench v2 mainline (39 capsules, from core_test.json.bak.main42) ──
    # 3 capsules excluded from the 42-capsule original by downstream:
    # capsule-8536428, capsule-4180912, capsule-6800638.
    "corebench_hard": [
        "capsule-5507257", "capsule-3449234",
        "capsule-8807709", "capsule-6049678", "capsule-2804717",
        "capsule-3418007", "capsule-1624349", "capsule-9832712",
        "capsule-2816027", "capsule-3821950", "capsule-9054015",
        "capsule-3301293", "capsule-1724988", "capsule-4299879",
        "capsule-0851068", "capsule-1394704", "capsule-0504157",
        "capsule-3593259", "capsule-3639589", "capsule-2345790",
        "capsule-7716865", "capsule-4933686", "capsule-9240688",
        "capsule-1175539", "capsule-2708693",
        "capsule-4252248", "capsule-4671827", "capsule-7186268",
        "capsule-5136217", "capsule-7800694",
        "capsule-8412128", "capsule-7655932", "capsule-9294029",
        "capsule-6295990", "capsule-9477017", "capsule-0201673",
        "capsule-0152700", "capsule-2242462", "capsule-3762736",
    ]
}

# ---------------------------------------------------------------------------
# Run matrix — explicit list of (model, agent_args) pairs. Each entry is one
# cell in the evaluation matrix and gets crossed with the agents × benchmarks
# × tasks dimensions in flow.py. This is NOT a cartesian product of models and
# ablation axes — it's a flat list so you can express arbitrary combinations
# (e.g. "gpt-5.4 at four reasoning levels, but gpt-5.3-codex only at medium").
#
# Each entry:
#   (model_name, ((key1, val1), (key2, val2), ...))
# The agent_args tuple is forwarded to the agent via hal-eval's -A flag
# (transported as HAL_AGENT_ARG_<key>=<value> env vars on the Batch node).
#
# Recognized agent_args keys (by codex_agent):
#   - max_threads: int (stringified). Enables Codex CLI's subagent workflow
#     via `agents.max_threads` config + a prompt directive telling the model
#     to spawn parallel subagents for independent subtasks.
#   - reasoning_effort: "low" | "medium" | "high" | "xhigh"
#     (Codex CLI's `model_reasoning_effort` config).
# ---------------------------------------------------------------------------
RUN_CONFIGS: list[tuple[str, tuple[tuple[str, str], ...]]] = [
    # ── codex_agent reliability runs (12 ablations × K_REPEATS) ───────
    # Reasoning effort sweep on gpt-5.4
    ("gpt-5.4", (("reasoning_effort", "low"),)),
    ("gpt-5.4", (("reasoning_effort", "medium"),)),
    ("gpt-5.4", (("reasoning_effort", "high"),)),
    ("gpt-5.4", (("reasoning_effort", "xhigh"),)),
    # Subagent count sweep on gpt-5.4 at medium reasoning
    ("gpt-5.4", (("reasoning_effort", "medium"), ("max_threads", "1"))),
    ("gpt-5.4", (("reasoning_effort", "medium"), ("max_threads", "3"))),
    ("gpt-5.4", (("reasoning_effort", "medium"), ("max_threads", "6"))),
    ("gpt-5.4", (("reasoning_effort", "medium"), ("max_threads", "9"))),
    # Model sweep at medium reasoning
    ("gpt-5", (("reasoning_effort", "medium"),)),
    ("gpt-5.1", (("reasoning_effort", "medium"),)),
    ("gpt-5.2", (("reasoning_effort", "medium"),)),
    ("gpt-5.3-codex", (("reasoning_effort", "medium"),)),
]

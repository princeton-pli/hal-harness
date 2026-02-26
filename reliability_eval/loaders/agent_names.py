"""Agent name parsing and model metadata lookups."""

import re
from typing import Dict

import pandas as pd

from reliability_eval.constants import (
    MODEL_CATEGORY,
    MODEL_METADATA,
    PROVIDER_ORDER,
)


def get_model_category(agent_name: str) -> str:
    """Get model category (small/large/reasoning) from agent name."""
    # Match the longest key first to avoid e.g. 'gpt_5_2' matching before 'gpt_5_2_medium'
    best_match = "unknown"
    best_len = 0
    for model_key, category in MODEL_CATEGORY.items():
        if model_key in agent_name and len(model_key) > best_len:
            best_match = category
            best_len = len(model_key)
    return best_match


def get_model_metadata(agent_name: str) -> Dict:
    """Get metadata for a model, with fallback for unknown models."""
    return MODEL_METADATA.get(agent_name, {"date": "2024-01-01", "provider": "Unknown"})


def get_provider(agent_name: str) -> str:
    """Get provider for an agent name."""
    return get_model_metadata(agent_name).get("provider", "Unknown")


def strip_agent_prefix(name: str) -> str:
    """Strip scaffold prefixes from agent name and convert to natural readable format."""
    # Remove common scaffold prefixes
    name = re.sub(r"^taubench_codex[-_]", "", name)
    name = re.sub(r"^taubench_toolcalling[-_]", "", name)
    name = re.sub(r"^taubench_fewshot[-_]", "", name)
    name = re.sub(r"^gaia_generalist[-_]", "", name)

    # Map to natural readable names
    display_names = {
        "gpt_4_turbo": "GPT-4 Turbo",
        "gpt_4o_mini": "GPT-4o mini",
        "gpt_o1": "o1",
        "gpt_5_2": "GPT 5.2",
        "gpt_5_2_medium": "GPT 5.2 (medium)",
        "gpt_5_2_xhigh": "GPT 5.2 (xhigh)",
        "gpt_5_2_codex_medium": "GPT 5.2 Codex (medium)",
        "gemini_2_flash": "Gemini 2.0 Flash",
        "gemini_2_5_flash": "Gemini 2.5 Flash",
        "gemini_2_5_pro": "Gemini 2.5 Pro",
        "gemini_3_pro": "Gemini 3.0 Pro",
        "claude_haiku_3_5": "Claude 3.5 Haiku",
        "claude_sonnet_3_7": "Claude 3.7 Sonnet",
        "claude_sonnet_4_5": "Claude 4.5 Sonnet",
        "claude_opus_4_5": "Claude 4.5 Opus",
    }

    return display_names.get(name, name)


def sort_agents_by_provider_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort dataframe by provider first, then by release date within each provider."""
    df = df.copy()
    df["release_date"] = df["agent"].map(
        lambda x: get_model_metadata(x).get("date", "2024-01-01")
    )
    df["provider"] = df["agent"].map(
        lambda x: get_model_metadata(x).get("provider", "Unknown")
    )
    df["release_timestamp"] = pd.to_datetime(df["release_date"])
    df["provider_order"] = df["provider"].map(PROVIDER_ORDER)
    df = df.sort_values(["provider_order", "release_timestamp"])
    df = df.drop(["provider_order"], axis=1)
    return df


def extract_agent_name(run_dir_name: str, benchmark: str) -> str:
    """Extract clean agent name from run directory name."""
    parts = run_dir_name.split("_")

    if run_dir_name.startswith(f"{benchmark}_"):
        prefix_len = len(benchmark.split("_"))
        agent_parts = parts[prefix_len:]
    else:
        agent_parts = parts[1:]

    # Remove trailing timestamp (numeric suffix)
    if agent_parts and agent_parts[-1].isdigit():
        agent_parts = agent_parts[:-1]

    # Remove repetition markers (rep1, rep2, etc.)
    if agent_parts and re.match(r"^rep\d+$", agent_parts[-1]):
        agent_parts = agent_parts[:-1]

    filtered_parts = []
    skip_keywords = [
        "fault",
        "compliance",
        "perturbed",
        "baseline",
        "struct",
        "prompt",
        "sensitivity",
    ]
    for part in agent_parts:
        if part in skip_keywords or "pct" in part:
            break
        # Also skip repN patterns in the middle (just in case)
        if re.match(r"^rep\d+$", part):
            break
        filtered_parts.append(part)

    return "_".join(filtered_parts)

"""OpenCode agent setup – constants and utilities for CLI installation and configuration."""

import json
import os
import shutil
import subprocess
from typing import Optional


# ── Constants ────────────────────────────────────────────────

OPENCODE_INSTALL_CMD = "curl -fsSL https://opencode.ai/install | bash"

KNOWN_BINARY_LOCATIONS = [
    "~/.local/bin/opencode",
    "~/.opencode/bin/opencode",
    "~/.local/share/opencode/bin/opencode",
    "/usr/local/bin/opencode",
    "/usr/bin/opencode",
    "/opt/homebrew/bin/opencode",
]

OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "logLevel": "DEBUG",
    "permission": {
        "*": "allow",
        "external_directory": "allow",
        "doom_loop": "allow",
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "websearch": "allow",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "task": "allow",
        "skill": "allow",
    },
}


# ── Utility Functions ────────────────────────────────────────


def find_opencode_binary() -> Optional[str]:
    """Search for the opencode binary in PATH and known locations."""
    found = shutil.which("opencode")
    if found:
        return found

    for loc in KNOWN_BINARY_LOCATIONS:
        expanded = os.path.expanduser(loc)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    search_roots = [os.path.expanduser("~"), "/usr/local", "/usr", "/opt"]
    for root in search_roots:
        for sub in ("bin", "local/bin", "local/share/opencode/bin"):
            candidate = os.path.join(root, sub, "opencode")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    return None


def install_system_deps() -> None:
    """Install system dependencies (git, curl, procps) via apt-get."""
    subprocess.run(
        "sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true",
        shell=True,
    )
    subprocess.run(["sudo", "apt-get", "update"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "git", "curl", "procps"], check=True)


def install_opencode_cli() -> None:
    """Run the official OpenCode CLI installer."""
    subprocess.run(OPENCODE_INSTALL_CMD, shell=True, check=True)


def setup_opencode_config(cwd: str = "/workspace") -> str:
    """Write project-level OpenCode config (.opencode/config.json).

    Returns the path to the written config file.
    """
    config_dir = os.path.join(cwd, ".opencode")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.json")
    with open(config_file, "w") as f:
        json.dump(OPENCODE_CONFIG, f, indent=2)
    return config_file


def _build_provider_config(model_name: str, kwargs: dict) -> dict:
    """Build per-model provider options based on agent kwargs.

    Supports:
      - thinking_budget (int): Anthropic extended thinking with budgetTokens
      - reasoning_effort ("low"|"medium"|"high"|"xhigh"): OpenAI/Google reasoning
    """
    provider_config: dict = {
        "anthropic": {},
        "openai": {},
        # Map GEMINI_API_KEY to Google provider's apiKey so OpenCode can
        # authenticate. OpenCode's google provider doesn't auto-detect
        # GEMINI_API_KEY.
        "google": {
            "options": {"apiKey": os.environ.get("GEMINI_API_KEY", "")},
        } if os.environ.get("GEMINI_API_KEY") else {},
    }

    thinking_budget = kwargs.get("thinking_budget")
    reasoning_effort = kwargs.get("reasoning_effort")

    # Determine provider from model name prefix
    if model_name.startswith("anthropic/"):
        bare_model = model_name.split("/", 1)[1]
        model_opts: dict = {}
        if thinking_budget is not None:
            model_opts["thinking"] = {
                "type": "enabled",
                "budgetTokens": int(thinking_budget),
            }
        if model_opts:
            provider_config["anthropic"] = {
                "models": {bare_model: {"options": model_opts}}
            }
    elif model_name.startswith("openai/") or model_name.startswith("gpt-"):
        bare_model = model_name.split("/", 1)[-1]
        model_opts = {}
        if reasoning_effort:
            model_opts["reasoningEffort"] = reasoning_effort
        if model_opts:
            provider_config["openai"] = {
                "models": {bare_model: {"options": model_opts}}
            }
    elif model_name.startswith("gemini/") or model_name.startswith("google/"):
        bare_model = model_name.split("/", 1)[-1]
        model_opts = {}
        if reasoning_effort:
            model_opts["reasoningEffort"] = reasoning_effort
        if model_opts:
            # Merge model options into existing google config (preserves apiKey)
            provider_config["google"]["models"] = {
                bare_model: {"options": model_opts}
            }

    return provider_config


def setup_global_config(model_name: str = "", kwargs: Optional[dict] = None) -> str:
    """Write global OpenCode config (~/.config/opencode/opencode.json).

    When model_name and kwargs are provided, injects per-model provider options
    (thinking budget, reasoning effort) into the config.

    Returns the path to the written config file.
    """
    global_config_dir = os.path.expanduser("~/.config/opencode")
    os.makedirs(global_config_dir, exist_ok=True)

    provider_config = _build_provider_config(model_name, kwargs or {})

    global_config = {
        "$schema": "https://opencode.ai/config.json",
        "logLevel": "DEBUG",
        "provider": provider_config,
    }
    global_config_path = os.path.join(global_config_dir, "opencode.json")
    with open(global_config_path, "w") as f:
        json.dump(global_config, f, indent=2)
    return global_config_path

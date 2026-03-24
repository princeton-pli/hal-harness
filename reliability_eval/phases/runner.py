"""Command building, execution, and environment setup for reliability_eval phases."""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from reliability_eval.config import AGENT_CONFIGS, BENCHMARK_CONFIGS


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def load_environment():
    """Load environment variables from .env file if available."""
    env_file = Path(".env")
    if env_file.exists():
        print(f"📄 Loading environment variables from {env_file}")
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            print("✅ Loaded .env file using python-dotenv")
        except ImportError:
            print("⚠️  python-dotenv not found, manually parsing .env")
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and not os.getenv(key):
                            os.environ[key] = value
            print("✅ Loaded .env file manually")
    else:
        print(f"⚠️  No .env file found at {env_file.absolute()}")


def check_api_keys(
    agent_configs: list[dict] | None = None, require_wandb: bool = True
):
    """Check that required API keys are available for configured models."""
    required_vars = ["WANDB_API_KEY"] if require_wandb else []

    configs = agent_configs if agent_configs is not None else AGENT_CONFIGS

    # Check which providers are in use
    providers_in_use = {cfg.get("provider", "openai") for cfg in configs}
    models_in_use = {cfg["model_name"] for cfg in configs}

    # Add provider-specific keys
    if "openai" in providers_in_use or any(
        "gpt" in m or "o1" in m for m in models_in_use
    ):
        required_vars.append("OPENAI_API_KEY")

    if "anthropic" in providers_in_use or any("claude" in m for m in models_in_use):
        required_vars.append("ANTHROPIC_API_KEY")

    if "google" in providers_in_use or any("gemini" in m for m in models_in_use):
        required_vars.append("GEMINI_API_KEY")

    # Check for OpenRouter
    if any("openrouter/" in m for m in models_in_use):
        required_vars.append("OPENROUTER_API_KEY")

    # Validate
    missing = [var for var in required_vars if not os.getenv(var)]
    found = [var for var in required_vars if os.getenv(var)]

    if found:
        print(f"✅ API keys found: {', '.join(found)}")

    if missing:
        print(f"\n⚠️  Warning: Missing API keys: {', '.join(missing)}")
        print("   Some evaluations may fail.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != "y":
            exit(1)


# =============================================================================
# COMMAND BUILDING
# =============================================================================

_AGENT_FUNCTION_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*$")


def _resolve_hal_eval_command() -> list[str]:
    """Use the installed CLI when available, otherwise the current interpreter."""
    hal_eval_bin = shutil.which("hal-eval")
    if hal_eval_bin:
        return [hal_eval_bin]
    return [sys.executable, "-m", "hal.cli"]


def _validate_agent_config(agent_config: dict) -> None:
    """Validate agent_function and agent_dir before use in subprocess commands."""
    agent_function = agent_config.get("agent_function", "")
    if not _AGENT_FUNCTION_RE.match(agent_function):
        raise ValueError(
            f"Invalid agent_function {agent_function!r}: must match "
            r"'^[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*$'"
        )

    agent_dir = agent_config.get("agent_dir", "")
    if not isinstance(agent_dir, str):
        raise ValueError(f"Invalid agent_dir {agent_dir!r}: must be a string")


def build_base_command(
    agent_config: dict,
    benchmark_config: dict,
    agent_name_suffix: str,
    max_tasks: int | None,
    conda_env: str | None = None,
    max_concurrent: int | None = None,
    run_id: str | None = None,
    continue_run: bool = False,
    results_dir: str | None = None,
) -> list[str]:
    """Build the base hal-eval command."""
    requires_agent_entrypoint = not benchmark_config.get("external_runner")
    if requires_agent_entrypoint:
        _validate_agent_config(agent_config)
    benchmark_name = benchmark_config["benchmark_name"]
    agent_name = f"{agent_config['name']}{agent_name_suffix}"

    cmd = [
        *_resolve_hal_eval_command(),
        "--benchmark",
        benchmark_name,
        "--agent_name",
        agent_name,
        "-A",
        f"model_name={agent_config['model_name']}",
        "-A",
        f"provider={agent_config.get('provider', 'openai')}",
        "-A",
        f"benchmark_name={benchmark_name}",
        "-A",
        "temperature=0.0",
        "--max_concurrent",
        str(max_concurrent or benchmark_config.get("max_concurrent", 1)),
    ]

    if requires_agent_entrypoint:
        cmd[3:3] = [
            "--agent_dir",
            agent_config["agent_dir"],
            "--agent_function",
            agent_config["agent_function"],
        ]

    # Only add --max_tasks if explicitly set (None means run all tasks)
    if max_tasks is not None:
        cmd.extend(["--max_tasks", str(max_tasks)])

    for key, value in (agent_config.get("extra_agent_args") or {}).items():
        cmd.extend(["-A", f"{key}={value}"])

    # Pass reasoning_effort if specified (for models like GPT-5.2, Gemini 2.5, Claude with thinking)
    if agent_config.get("reasoning_effort"):
        cmd.extend(["-A", f"reasoning_effort={agent_config['reasoning_effort']}"])

    # Use custom task timeout if specified (default: 600s = 10 min)
    # Agentic scaffolds with high reasoning effort need longer timeouts
    task_timeout = agent_config.get("task_timeout")
    if task_timeout:
        cmd.extend(["--task_timeout", str(task_timeout)])

    if run_id:
        cmd.extend(["--run_id", run_id])

    if continue_run:
        cmd.append("--continue_run")

    if conda_env:
        cmd.extend(["--conda_env_name", conda_env])

    if benchmark_config.get("requires_docker", False):
        cmd.append("--docker")

    if benchmark_config.get("requires_vm", False):
        cmd.append("--vm")

    if results_dir and results_dir != "results":
        cmd.extend(["--results_dir", results_dir])

    # Pass specific task IDs if configured for this benchmark
    task_ids = benchmark_config.get("task_ids")
    if task_ids:
        cmd.extend(["--task_ids", ",".join(sorted(task_ids, key=int))])

    return cmd


def add_baseline_args(cmd: list[str], benchmark_config: dict) -> list[str]:
    """Add arguments for baseline phase (consistency_outcome + predictability_rate_confidence_correlation/predictability_calibration + safety_compliance)."""
    # Predictability: confidence scoring
    cmd.extend(["-A", "compute_confidence=true"])
    cmd.extend(["-A", "store_confidence_details=true"])
    cmd.extend(["-A", "store_conversation_history=true"])

    # Compliance: constraint monitoring
    constraints = benchmark_config.get("compliance_constraints", [])
    if constraints:
        cmd.extend(["-A", "enable_compliance_monitoring=true"])
        cmd.extend(["-A", f"compliance_constraints={','.join(constraints)}"])

    return cmd


def add_fault_args(cmd: list[str], fault_rate: float) -> list[str]:
    """Add arguments for fault injection phase (robustness_fault_injection)."""
    cmd.extend(["-A", "enable_fault_injection=true"])
    cmd.extend(["-A", f"fault_rate={fault_rate}"])
    cmd.extend(["-A", "track_recovery=true"])
    return cmd


def add_prompt_sensitivity_args(
    cmd: list[str],
    num_variations: int,
    variation_strength: str = "mild",
    variation_index: int | None = None,
) -> list[str]:
    """Add arguments for prompt sensitivity phase (S_prompt).

    Args:
        cmd: Command list to extend
        num_variations: Number of variations to generate
        variation_strength: Strength of variations (mild, medium, strong, naturalistic)
        variation_index: If set, run only this specific variation (0=original, 1..N=variations)
    """
    cmd.extend(["--prompt_sensitivity"])
    cmd.extend(["--num_variations", str(num_variations)])
    cmd.extend(["--variation_strength", variation_strength])
    if variation_index is not None:
        cmd.extend(["--variation_index", str(variation_index)])
    return cmd


def add_structural_args(cmd: list[str], strength: str, ptype: str) -> list[str]:
    """Add arguments for structural perturbation phase (robustness_structural)."""
    cmd.extend(["-A", "enable_structural_perturbations=true"])
    cmd.extend(["-A", f"perturbation_strength={strength}"])
    cmd.extend(["-A", f"perturbation_type={ptype}"])
    return cmd


# =============================================================================
# EXECUTION
# =============================================================================


def run_command(cmd: list[str], max_retries: int = 3) -> tuple[bool, float, str | None]:
    """Run a command with real-time output and retry logic."""
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            # Run with real-time output (no capture)
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            return True, elapsed, None

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            error_msg = f"Return code: {e.returncode}"

            # Retry on non-final attempts
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"\n⚠️  Command failed on attempt {attempt + 1}/{max_retries}")
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            return False, elapsed, error_msg

    return False, 0, "Max retries exceeded"


def get_valid_combinations(benchmark_filter: str | None = None) -> list[tuple]:
    """Get valid agent-benchmark combinations."""
    combinations = []
    for agent_config in AGENT_CONFIGS:
        for bench_name in agent_config.get("benchmarks", []):
            if bench_name in BENCHMARK_CONFIGS:
                if benchmark_filter and bench_name != benchmark_filter:
                    continue
                combinations.append(
                    (agent_config, BENCHMARK_CONFIGS[bench_name], bench_name)
                )
    return combinations

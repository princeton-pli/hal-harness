#!/usr/bin/env python3
"""
Reliability Evaluation Script for Outcome Consistency (C_out)

This script runs K repetitions of agent evaluations across multiple benchmarks
to compute outcome consistency as defined in the reliability framework.

Usage:
    python run_consistency_eval.py --k 5 --max_tasks 20

Configuration:
    - Runs experiments sequentially to avoid rate limits
    - Uses cheaper models (gpt-4o-mini, gemini-1.5-flash)
    - Evaluates on: GAIA, TauBench (airline), SWE-bench (mini)
"""

import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime
import os
import time

# Configuration
# Note: Different benchmarks may need different agents
# TauBench uses taubench_few_shot, GAIA uses hal_generalist_agent, etc.

AGENT_CONFIGS = [
    # TauBench agents with tool calling
    # GPT-4o-mini for reliability testing
    {
        "name": "taubench_toolcalling_gpt_4o_mini",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "gpt-4o-mini-2024-07-18",
        "benchmarks": ["taubench_airline"],
        "extra_agent_args": {
            "provider": "openai",
            "temperature": 0.0,
            "compute_confidence": "true",
            "store_conversation_history": "true"
        }
    },
    # Anthropic models
    # {
    #     "name": "taubench_toolcalling_claude_haiku_3_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-3-5-haiku-20241022",
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "anthropic",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_claude_sonnet_3_7",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "openrouter/anthropic/claude-3-7-sonnet-20250219",
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "openai",  # OpenRouter uses OpenAI-compatible API
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_claude_sonnet_4_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-sonnet-4-5",
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "anthropic",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_claude_opus_4_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-opus-4-5",  # Use dashes, not dots
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "anthropic",
    #         "temperature": 0.0
    #     }
    # },
    # Google Gemini models
    # {
    #     "name": "taubench_toolcalling_gemini_2_flash",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.0-flash", # gemini/gemini-2.0-flash, gemini/gemini-2.5-flash, gemini/gemini-2.5-pro, gemini/gemini-3-pro-preview
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_2_5_flash",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.5-flash", # gemini/gemini-2.0-flash, gemini/gemini-2.5-flash, gemini/gemini-2.5-pro, gemini/gemini-3-pro-preview
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_2_5_pro",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.5-pro", # gemini/gemini-2.0-flash, gemini/gemini-2.5-flash, gemini/gemini-2.5-pro, gemini/gemini-3-pro-preview
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_3_pro",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-3-pro-preview", # gemini/gemini-2.0-flash, gemini/gemini-2.5-flash, gemini/gemini-2.5-pro, gemini/gemini-3-pro-preview
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # GAIA agents
    # {
    #     "name": "gaia_gemini_3_flash",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gemini/gemini-3-flash-preview",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "gaia_gemini_flash",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gemini/gemini-2.0-flash",
    #     "benchmarks": ["gaia"],
    #     "extra_agent_args": {}
    # },
    # SWE-bench agents
    # {
    #     "name": "swebench_gpt4o_mini",
    #     "agent_dir": "agents/hal_generalist_agent",
    #     "agent_function": "main.run",
    #     "model_name": "gpt-4o-mini-2024-07-18",
    #     "benchmarks": ["swebench_verified_mini"],
    #     "extra_agent_args": {}
    # },
]

BENCHMARK_CONFIGS = [
    # {
    #     "name": "gaia",
    #     "benchmark_name": "gaia",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "extra_args": []
    # },
    # {
    #     "name": "taubench_retail",
    #     "benchmark_name": "taubench_retail",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "extra_args": []
    # },
    {
        "name": "taubench_airline",
        "benchmark_name": "taubench_airline",
        "requires_docker": False,
        "requires_vm": False,
        "extra_args": []
    },
    # {
    #     "name": "swebench_verified_mini",
    #     "benchmark_name": "swebench_verified_mini",
    #     "requires_docker": True,
    #     "requires_vm": False,
    #     "extra_args": []
    # },
]


def check_environment():
    """Check that required environment variables are set, loading from .env if available"""

    # Try to load .env file
    env_file = Path(".env")
    if env_file.exists():
        print(f"📄 Loading environment variables from {env_file}")
        try:
            # Try using python-dotenv if available
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ Loaded .env file using python-dotenv")
        except ImportError:
            # Fallback: manually parse .env file
            print("⚠️  python-dotenv not found, manually parsing .env")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Only set if not already in environment
                        if key and not os.getenv(key):
                            os.environ[key] = value
            print("✅ Loaded .env file manually")
    else:
        print(f"⚠️  No .env file found at {env_file.absolute()}")
        print("   Relying on existing environment variables")

    # Check required variables
    required_vars = ["OPENAI_API_KEY", "WANDB_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"\n⚠️  Warning: Missing environment variables: {', '.join(missing)}")
        print("   Some evaluations may fail without proper API keys.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    else:
        print(f"✅ Required API keys found: {', '.join(required_vars)}")

    # Check for other API keys based on models being used
    models_in_use = {cfg['model_name'] for cfg in AGENT_CONFIGS}

    # Check for Gemini API key if using Gemini models
    if any('gemini' in model for model in models_in_use):
        if not os.getenv("GEMINI_API_KEY"):
            print("\n⚠️  Warning: GEMINI_API_KEY not set. Gemini evaluations will fail.")
        else:
            print("✅ GEMINI_API_KEY found")

    # Check for Anthropic API key if using Claude models
    if any('claude' in model for model in models_in_use):
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\n⚠️  Warning: ANTHROPIC_API_KEY not set. Claude evaluations will fail.")
        else:
            print("✅ ANTHROPIC_API_KEY found")


def run_evaluation(agent_config, benchmark_config, k_runs, max_tasks, conda_env, run_number, total_runs, max_retries=3):
    """Run a single evaluation with the specified configuration, with retry logic for network errors"""

    # Use agent name directly - results are already in benchmark-specific directory
    agent_name = agent_config['name']

    print("\n" + "="*80)
    print(f"🔄 Run {run_number}/{total_runs}")
    print(f"📊 Agent: {agent_config['name']}")
    print(f"📋 Benchmark: {benchmark_config['name']}")
    print(f"🔢 Model: {agent_config['model_name']}")
    print("="*80)

    # Build the hal-eval command
    cmd = [
        "hal-eval",
        "--benchmark", benchmark_config["benchmark_name"],
        "--agent_dir", agent_config["agent_dir"],
        "--agent_function", agent_config["agent_function"],
        "--agent_name", agent_name,
        "-A", f"model_name={agent_config['model_name']}",
        "-A", f"benchmark_name={benchmark_config['benchmark_name']}",  # Required by hal_generalist_agent
        "--max_concurrent", "1",  # Sequential to avoid rate limits
        "--max_tasks", str(max_tasks),
    ]

    # Add extra agent-specific arguments (e.g., provider, temperature)
    for key, value in agent_config.get("extra_agent_args", {}).items():
        cmd.extend(["-A", f"{key}={value}"])

    # Add conda environment if specified
    if conda_env:
        cmd.extend(["--conda_env_name", conda_env])

    # Add Docker flag if required
    if benchmark_config.get("requires_docker", False):
        cmd.append("--docker")

    # Add VM flag if required
    if benchmark_config.get("requires_vm", False):
        cmd.append("--vm")

    # Add any extra benchmark-specific arguments
    for arg in benchmark_config.get("extra_args", []):
        cmd.append(arg)

    print("\n🚀 Running command:")
    print(f"   {' '.join(cmd)}\n")

    # Run the evaluation with retry logic for network errors
    for attempt in range(max_retries):
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = time.time() - start_time
            print(f"\n✅ Evaluation completed in {elapsed:.1f}s")
            if result.stdout:
                print(result.stdout)
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time

            # Check if it's a network error (DNS, connection issues)
            is_network_error = False
            error_output = (e.stdout or "") + (e.stderr or "")
            network_error_indicators = [
                "nodename nor servname provided",
                "Errno 8",
                "Connection refused",
                "Connection reset",
                "Failed to resolve",
                "Name or service not known",
                "Temporary failure in name resolution",
                "Network is unreachable"
            ]

            for indicator in network_error_indicators:
                if indicator in error_output:
                    is_network_error = True
                    break

            # If it's a network error and we have retries left, retry with backoff
            if is_network_error and attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"\n⚠️  Network error detected on attempt {attempt + 1}/{max_retries}")
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            # Otherwise, it's a final failure
            print(f"\n❌ Evaluation failed after {elapsed:.1f}s (attempt {attempt + 1}/{max_retries})")
            print(f"Return code: {e.returncode}")

            if e.stdout:
                print(f"\nStdout:\n{e.stdout}")

            if e.stderr:
                print(f"\nStderr:\n{e.stderr}")

            if not e.stdout and not e.stderr:
                print("\n⚠️  No error output captured. The command may have failed silently.")
                print("   Try running the command manually to see the full error:")
                print(f"   {' '.join(cmd)}")

            return False

    # Should not reach here, but just in case
    return False


def run_k_repetitions(k_runs, max_tasks, conda_env, skip_swebench=False):
    """Run K repetitions of each agent-benchmark combination"""

    # Filter out SWE-bench if requested
    benchmarks = BENCHMARK_CONFIGS
    if skip_swebench:
        benchmarks = [b for b in benchmarks if 'swebench' not in b['name']]
        print("⏭️  Skipping SWE-bench evaluations (requires Docker)")

    # Build list of valid agent-benchmark combinations
    combinations = []
    for agent_config in AGENT_CONFIGS:
        for benchmark_config in benchmarks:
            # Check if agent supports this benchmark
            if benchmark_config['name'] in agent_config.get('benchmarks', []):
                combinations.append((agent_config, benchmark_config))

    total_runs = len(combinations) * k_runs
    run_number = 0

    results_log = {
        "start_time": datetime.now().isoformat(),
        "k_runs": k_runs,
        "max_tasks": max_tasks,
        "conda_env": conda_env,
        "results": []
    }

    print(f"\n🎯 Starting {total_runs} evaluation runs")
    print(f"   K={k_runs} repetitions")
    print(f"   {len(combinations)} valid agent-benchmark combinations")
    print(f"   {max_tasks} tasks per benchmark\n")

    for agent_config, benchmark_config in combinations:
        for k in range(k_runs):
            run_number += 1

            print(f"\n📍 Repetition {k+1}/{k_runs} for {agent_config['name']} on {benchmark_config['name']}")

            success = run_evaluation(
                agent_config,
                benchmark_config,
                k_runs,
                max_tasks,
                conda_env,
                run_number,
                total_runs
            )

            results_log["results"].append({
                "agent": agent_config["name"],
                "benchmark": benchmark_config["name"],
                "repetition": k + 1,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })

            # Save progress after each run
            log_path = Path("reliability_eval/run_log.json")
            log_path.parent.mkdir(exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(results_log, f, indent=2)

            # Add a small delay between runs to avoid rate limits
            if run_number < total_runs:
                print("\n⏸️  Waiting 5 seconds before next run...")
                time.sleep(5)

    results_log["end_time"] = datetime.now().isoformat()

    # Final save
    with open(log_path, 'w') as f:
        json.dump(results_log, f, indent=2)

    print("\n" + "="*80)
    print("✨ All evaluations completed!")
    print(f"📊 Results logged to: {log_path}")
    print("="*80)

    # Summary
    successful = sum(1 for r in results_log["results"] if r["success"])
    print(f"\n📈 Summary: {successful}/{total_runs} runs completed successfully")

    return results_log


def main():
    parser = argparse.ArgumentParser(
        description="Run consistency evaluation across agents and benchmarks"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of repetitions per task (default: 5)"
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=20,
        help="Maximum number of tasks per benchmark (default: 20)"
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default=None,
        help="Conda environment name (optional, uses current environment if not specified)"
    )
    parser.add_argument(
        "--skip_swebench",
        action="store_true",
        help="Skip SWE-bench evaluations (requires Docker)"
    )

    args = parser.parse_args()

    print("🔬 Reliability Evaluation - Outcome Consistency")
    print(f"   K repetitions: {args.k}")
    print(f"   Max tasks: {args.max_tasks}")
    print(f"   Environment: {args.conda_env if args.conda_env else 'current (uv/conda/system)'}")

    # Check environment
    check_environment()

    # Run evaluations
    run_k_repetitions(
        k_runs=args.k,
        max_tasks=args.max_tasks,
        conda_env=args.conda_env,
        skip_swebench=args.skip_swebench
    )


if __name__ == "__main__":
    main()

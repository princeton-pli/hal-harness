#!/usr/bin/env python3
"""
Reliability Evaluation Script for Prompt Sensitivity (S_prompt)

This script runs evaluations with prompt variations to measure:
- Prompt sensitivity: How performance varies across different phrasings of the same task
- Task-level variance: Which tasks are most sensitive to prompt changes
- Performance stability: How robust agents are to prompt perturbations

Usage:
    python run_prompt_sensitivity_eval.py --num_variations 3 --max_tasks 20

Configuration:
    - Runs with --prompt_sensitivity flag
    - Uses num_variations prompt paraphrases per task
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
AGENT_CONFIGS = [
    # TauBench agents with tool calling (NOW SUPPORTED!)
    # OpenAI models
    {
        "name": "taubench_toolcalling_gpt_5_2",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "gpt-5.2-2025-12-11",
        "benchmarks": ["taubench_airline", "taubench_retail"],
        "extra_agent_args": {
            "provider": "openai",
            "temperature": 0.0
        }
    },
    # Anthropic models
    # {
    #     "name": "taubench_toolcalling_claude_haiku_3_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-3-5-haiku-20241022",
    #     "benchmarks": ["taubench_airline"],
    #     "extra_agent_args": {
    #         "provider": "anthropic",
    #         "temperature": 0.0
    #     }
    # },
    # {
    #     "name": "taubench_toolcalling_claude_sonnet_4_5",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "claude-sonnet-4-5",
    #     "benchmarks": ["taubench_airline"],
    #     "extra_agent_args": {
    #         "provider": "anthropic",
    #         "temperature": 0.0
    #     }
    # },
    # Google Gemini models
    # {
    #     "name": "taubench_toolcalling_gemini_2_5_flash",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.5-flash",
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "google",
    #         "temperature": 0.0
    #     }
    # },
]

BENCHMARK_CONFIGS = [
    # Example: GAIA (SUPPORTED)
    # {
    #     "name": "gaia",
    #     "benchmark_name": "gaia",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "extra_args": []
    # },

    # TauBench (NOT SUPPORTED - prompts come from environment objects)
    {
        "name": "taubench_airline",
        "benchmark_name": "taubench_airline",
        "requires_docker": False,
        "requires_vm": False,
        "extra_args": []
    },
    # {
    #     "name": "taubench_retail",
    #     "benchmark_name": "taubench_retail",
    #     "requires_docker": False,
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
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print("✅ Loaded .env file using python-dotenv")
        except ImportError:
            print("⚠️  python-dotenv not found, manually parsing .env")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
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

    # Check for provider-specific API keys
    models_in_use = {cfg['model_name'] for cfg in AGENT_CONFIGS}

    if any('gemini' in model for model in models_in_use):
        if not os.getenv("GEMINI_API_KEY"):
            print("\n⚠️  Warning: GEMINI_API_KEY not set. Gemini evaluations will fail.")
        else:
            print("✅ GEMINI_API_KEY found")

    if any('claude' in model for model in models_in_use):
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\n⚠️  Warning: ANTHROPIC_API_KEY not set. Claude evaluations will fail.")
        else:
            print("✅ ANTHROPIC_API_KEY found")


def run_evaluation(agent_config, benchmark_config, num_variations, max_tasks, conda_env, run_number, total_runs, max_retries=3):
    """Run a single evaluation with prompt sensitivity enabled"""

    agent_name = agent_config['name']

    print("\n" + "="*80)
    print(f"🔄 Run {run_number}/{total_runs}")
    print(f"📊 Agent: {agent_config['name']}")
    print(f"📋 Benchmark: {benchmark_config['name']}")
    print(f"🔢 Model: {agent_config['model_name']}")
    print(f"🔀 Variations: {num_variations} (+ 1 original)")
    print("="*80)

    # Build the hal-eval command
    cmd = [
        "hal-eval",
        "--benchmark", benchmark_config["benchmark_name"],
        "--agent_dir", agent_config["agent_dir"],
        "--agent_function", agent_config["agent_function"],
        "--agent_name", agent_name,
        "-A", f"model_name={agent_config['model_name']}",
        "-A", f"benchmark_name={benchmark_config['benchmark_name']}",
        "--max_concurrent", "1",
        "--max_tasks", str(max_tasks),
        "--prompt_sensitivity",  # Enable prompt sensitivity
        "--num_variations", str(num_variations),
    ]

    # Add extra agent-specific arguments
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

            # Check if it's a network error
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
                wait_time = 10 * (attempt + 1)
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

    return False


def run_evaluations(num_variations, max_tasks, conda_env, skip_swebench=False):
    """Run evaluations for each agent-benchmark combination"""

    # Check if configurations are empty
    if not AGENT_CONFIGS:
        print("\n❌ Error: AGENT_CONFIGS is empty!")
        print("   Please uncomment and configure at least one agent in run_prompt_sensitivity_eval.py")
        print("   NOTE: TauBench is not supported. Use GAIA, USACO, or SWE-bench.")
        return None

    if not BENCHMARK_CONFIGS:
        print("\n❌ Error: BENCHMARK_CONFIGS is empty!")
        print("   Please uncomment and configure at least one benchmark in run_prompt_sensitivity_eval.py")
        print("   NOTE: TauBench is not supported. Use GAIA, USACO, or SWE-bench.")
        return None

    # Filter out SWE-bench if requested
    benchmarks = BENCHMARK_CONFIGS
    if skip_swebench:
        benchmarks = [b for b in benchmarks if 'swebench' not in b['name']]
        print("⏭️  Skipping SWE-bench evaluations (requires Docker)")

    # Build list of valid agent-benchmark combinations
    combinations = []
    for agent_config in AGENT_CONFIGS:
        for benchmark_config in benchmarks:
            if benchmark_config['name'] in agent_config.get('benchmarks', []):
                combinations.append((agent_config, benchmark_config))

    if not combinations:
        print("\n❌ Error: No valid agent-benchmark combinations found!")
        print("   Check that:")
        print("   1. Agent configs have a 'benchmarks' field")
        print("   2. Benchmark names in 'benchmarks' match the BENCHMARK_CONFIGS names")
        print("   3. At least one agent supports at least one configured benchmark")
        return None

    total_runs = len(combinations)
    run_number = 0

    results_log = {
        "start_time": datetime.now().isoformat(),
        "num_variations": num_variations,
        "max_tasks": max_tasks,
        "conda_env": conda_env,
        "results": []
    }

    print(f"\n🎯 Starting {total_runs} evaluation runs")
    print(f"   {num_variations} variations (+ 1 original) per task")
    print(f"   {len(combinations)} valid agent-benchmark combinations")
    print(f"   {max_tasks} tasks per benchmark\n")

    for agent_config, benchmark_config in combinations:
        run_number += 1

        print(f"\n📍 Running {agent_config['name']} on {benchmark_config['name']}")

        success = run_evaluation(
            agent_config,
            benchmark_config,
            num_variations,
            max_tasks,
            conda_env,
            run_number,
            total_runs
        )

        results_log["results"].append({
            "agent": agent_config["name"],
            "benchmark": benchmark_config["name"],
            "success": success,
            "timestamp": datetime.now().isoformat()
        })

        # Save progress after each run
        log_path = Path("reliability_eval/prompt_sensitivity_run_log.json")
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
        description="Run prompt sensitivity evaluation across agents and benchmarks"
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=3,
        help="Number of prompt variations to generate per task (default: 3)"
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

    print("🔬 Reliability Evaluation - Prompt Sensitivity")
    print(f"   Prompt variations: {args.num_variations}")
    print(f"   Max tasks: {args.max_tasks}")
    print(f"   Environment: {args.conda_env if args.conda_env else 'current (uv/conda/system)'}")

    # Check environment
    check_environment()

    # Run evaluations
    run_evaluations(
        num_variations=args.num_variations,
        max_tasks=args.max_tasks,
        conda_env=args.conda_env,
        skip_swebench=args.skip_swebench
    )


if __name__ == "__main__":
    main()

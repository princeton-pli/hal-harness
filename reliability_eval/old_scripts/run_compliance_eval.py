#!/usr/bin/env python3
"""
Reliability Evaluation Script for Compliance (S_comp)

This script runs evaluations with compliance monitoring to measure:
- S_comp: Adherence to behavioral constraints and safety guardrails

Usage:
    python run_compliance_eval.py --k 3 --max_tasks 50

Configuration:
    - Runs with compliance monitoring enabled
    - Tracks constraint violations during execution
    - Evaluates on multiple benchmarks
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
    # TauBench agents with compliance monitoring
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
    #     "name": "taubench_toolcalling_gpt_4o_mini",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-4o-mini-2024-07-18",
    #     "benchmarks": ["taubench_airline", "taubench_retail"],
    #     "extra_agent_args": {
    #         "provider": "openai",
    #         "temperature": 0.0
    #     }
    # },
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
]

BENCHMARK_CONFIGS = [
    {
        "name": "taubench_airline",
        "benchmark_name": "taubench_airline",
        "requires_docker": False,
        "requires_vm": False,
        "extra_args": [],
        "compliance_constraints": [
            "no_pii_exposure",        # Don't expose customer PII in logs
            "rate_limit_respect",     # Respect API rate limits
            "no_destructive_ops",     # Don't perform irreversible operations
            "data_minimization"       # Only request necessary data
        ]
    },
    # {
    #     "name": "taubench_retail",
    #     "benchmark_name": "taubench_retail",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "extra_args": [],
    #     "compliance_constraints": [
    #         "no_pii_exposure",
    #         "rate_limit_respect",
    #         "no_destructive_ops",
    #         "data_minimization"
    #     ]
    # },
]


def check_environment():
    """Check that required environment variables are set"""
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

    # Check for model-specific API keys
    models_in_use = {cfg['model_name'] for cfg in AGENT_CONFIGS}

    if any('claude' in model for model in models_in_use):
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("\n⚠️  Warning: ANTHROPIC_API_KEY not set. Claude evaluations will fail.")
        else:
            print("✅ ANTHROPIC_API_KEY found")


def run_evaluation(agent_config, benchmark_config, max_tasks, conda_env, run_number, total_runs, max_retries=3):
    """Run a single evaluation with compliance monitoring enabled"""

    agent_name = f"{agent_config['name']}_compliance"
    constraints = ','.join(benchmark_config.get('compliance_constraints', []))

    print("\n" + "="*80)
    print(f"🔄 Run {run_number}/{total_runs}")
    print(f"📊 Agent: {agent_config['name']}")
    print(f"📋 Benchmark: {benchmark_config['name']}")
    print(f"🔢 Model: {agent_config['model_name']}")
    print(f"⚖️  Compliance Constraints: {len(benchmark_config.get('compliance_constraints', []))}")
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
        "-A", "enable_compliance_monitoring=true",
        "-A", f"compliance_constraints={constraints}",
        "--max_concurrent", "1",
        "--max_tasks", str(max_tasks),
    ]

    # Add extra agent-specific arguments
    for key, value in agent_config.get("extra_agent_args", {}).items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        cmd.extend(["-A", f"{key}={value}"])

    if conda_env:
        cmd.extend(["--conda_env_name", conda_env])

    if benchmark_config.get("requires_docker", False):
        cmd.append("--docker")

    if benchmark_config.get("requires_vm", False):
        cmd.append("--vm")

    for arg in benchmark_config.get("extra_args", []):
        cmd.append(arg)

    print("\n🚀 Running command:")
    print(f"   {' '.join(cmd)}\n")

    # Run with retry logic
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

            # Check for network errors
            is_network_error = False
            error_output = (e.stdout or "") + (e.stderr or "")
            network_error_indicators = [
                "nodename nor servname provided",
                "Errno 8",
                "Connection refused",
                "Connection reset",
                "Failed to resolve",
            ]

            for indicator in network_error_indicators:
                if indicator in error_output:
                    is_network_error = True
                    break

            if is_network_error and attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"\n⚠️  Network error detected on attempt {attempt + 1}/{max_retries}")
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue

            print(f"\n❌ Evaluation failed after {elapsed:.1f}s (attempt {attempt + 1}/{max_retries})")
            print(f"Return code: {e.returncode}")

            if e.stdout:
                print(f"\nStdout:\n{e.stdout}")
            if e.stderr:
                print(f"\nStderr:\n{e.stderr}")

            return False

    return False


def run_k_repetitions(k_runs, max_tasks, conda_env):
    """Run K repetitions of each agent-benchmark combination with compliance monitoring"""

    combinations = []
    for agent_config in AGENT_CONFIGS:
        for benchmark_config in BENCHMARK_CONFIGS:
            if benchmark_config['name'] in agent_config.get('benchmarks', []):
                combinations.append((agent_config, benchmark_config))

    total_runs = len(combinations) * k_runs
    run_number = 0

    results_log = {
        "start_time": datetime.now().isoformat(),
        "k_runs": k_runs,
        "max_tasks": max_tasks,
        "conda_env": conda_env,
        "evaluation_type": "compliance",
        "results": []
    }

    print(f"\n🎯 Starting {total_runs} compliance evaluation runs")
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

            # Save progress
            log_path = Path("reliability_eval/compliance_run_log.json")
            log_path.parent.mkdir(exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(results_log, f, indent=2)

            # Delay between runs
            if run_number < total_runs:
                print("\n⏸️  Waiting 5 seconds before next run...")
                time.sleep(5)

    results_log["end_time"] = datetime.now().isoformat()

    with open(log_path, 'w') as f:
        json.dump(results_log, f, indent=2)

    print("\n" + "="*80)
    print("✨ All compliance evaluations completed!")
    print(f"📊 Results logged to: {log_path}")
    print("="*80)

    successful = sum(1 for r in results_log["results"] if r["success"])
    print(f"\n📈 Summary: {successful}/{total_runs} runs completed successfully")

    return results_log


def main():
    parser = argparse.ArgumentParser(
        description="Run compliance evaluation with constraint monitoring"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of repetitions per task (default: 3)"
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=50,
        help="Maximum number of tasks per benchmark (default: 50)"
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default=None,
        help="Conda environment name (optional)"
    )

    args = parser.parse_args()

    print("🔬 Reliability Evaluation - Compliance")
    print(f"   K repetitions: {args.k}")
    print(f"   Max tasks: {args.max_tasks}")
    print(f"   Environment: {args.conda_env if args.conda_env else 'current (uv/conda/system)'}")

    check_environment()

    run_k_repetitions(
        k_runs=args.k,
        max_tasks=args.max_tasks,
        conda_env=args.conda_env
    )


if __name__ == "__main__":
    main()

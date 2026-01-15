#!/usr/bin/env python3
"""
Unified Reliability Evaluation Script

This script runs a comprehensive panel of reliability metrics efficiently:

PHASE 1 - Baseline (K repetitions) → Multiple metrics from same runs:
  - C_out: Outcome Consistency (from K repetitions)
  - P_rc/P_cal: Predictability (from confidence scores)
  - S_comp: Compliance (from constraint monitoring)

PHASE 2 - Fault Injection → R_fault (Fault Robustness)

PHASE 3 - Prompt Sensitivity → S_prompt (requires prompt variations)

PHASE 4 - Structural Perturbations → R_struct (Structural Robustness)

Usage:
    # Run all phases
    python run_reliability_eval.py --k 5 --max_tasks 50

    # Run specific phases only
    python run_reliability_eval.py --k 5 --max_tasks 50 --phases baseline fault

    # Quick test run
    python run_reliability_eval.py --k 2 --max_tasks 5 --phases baseline

Configuration:
    Edit AGENT_CONFIGS below to specify which models to evaluate.
    Edit BENCHMARK_CONFIGS to specify which benchmarks to run on.
"""

import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# CONFIGURATION - Edit these to customize your evaluation
# =============================================================================

AGENT_CONFIGS = [
    # -------------------------------------------------------------------------
    # OpenAI Models
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_toolcalling_gpt_4o_mini",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-4o-mini-2024-07-18",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_4_turbo",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-4-turbo-2024-04-09",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_o1",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "o1-2024-12-17",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gpt_5_2",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gpt-5.2-2025-12-11",
    #     "provider": "openai",
    #     "benchmarks": ["taubench_airline"],
    # },
    # -------------------------------------------------------------------------
    # Anthropic Models
    # -------------------------------------------------------------------------

    {
        "name": "taubench_toolcalling_claude_haiku_3_5",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "benchmarks": ["taubench_airline"],
    },
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
    {
        "name": "taubench_toolcalling_claude_sonnet_4_5",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "claude-sonnet-4-5",
        "provider": "anthropic",
        "benchmarks": ["taubench_airline"],
    },
    {
        "name": "taubench_toolcalling_claude_opus_4_5",
        "agent_dir": "agents/taubench_tool_calling",
        "agent_function": "tool_calling.run",
        "model_name": "claude-opus-4-5",  # Use dashes, not dots
        "provider": "anthropic",
        "benchmarks": ["taubench_airline"],
    },

    # -------------------------------------------------------------------------
    # Google Gemini Models
    # -------------------------------------------------------------------------
    # {
    #     "name": "taubench_toolcalling_gemini_2_flash",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.0-flash", 
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_2_5_flash",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.5-flash",
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_2_5_pro",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-2.5-pro",
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
    # {
    #     "name": "taubench_toolcalling_gemini_3_pro",
    #     "agent_dir": "agents/taubench_tool_calling",
    #     "agent_function": "tool_calling.run",
    #     "model_name": "gemini/gemini-3-pro-preview",
    #     "provider": "google",
    #     "benchmarks": ["taubench_airline"],
    # },
]

BENCHMARK_CONFIGS = {
    "taubench_airline": {
        "benchmark_name": "taubench_airline",
        "requires_docker": False,
        "requires_vm": False,
        "max_concurrent": 1,  # Sequential to avoid rate limits
        "compliance_constraints": [
            "no_pii_exposure",
            "no_destructive_ops",
            "data_minimization",
            "rate_limit_respect",
        ],
    },
    # "taubench_retail": {
    #     "benchmark_name": "taubench_retail",
    #     "requires_docker": False,
    #     "requires_vm": False,
    #     "max_concurrent": 1,
    #     "compliance_constraints": [
    #         "no_pii_exposure",
    #         "no_destructive_ops",
    #         "data_minimization",
    #         "rate_limit_respect",
    #     ],
    # },
}

# Phase-specific settings
PHASE_SETTINGS = {
    "baseline": {
        "description": "Baseline runs (C_out, C_traj, C_conf, C_res, P_rc/P_cal)",
        "k_runs": 3,  # Will be overridden by --k argument
    },
    "fault": {
        "description": "Fault injection (R_fault)",
        "k_runs": 3,
        "fault_rate": 0.2,
    },
    "prompt": {
        "description": "Prompt sensitivity (S_prompt)",
        "num_variations": 3,
    },
    "structural": {
        "description": "Structural perturbations (R_struct)",
        "perturbation_strength": "medium",
        "perturbation_type": "all",
    },
    "safety": {
        "description": "LLM-based safety analysis (S_harm, S_comp)",
        "model": "gpt-4o-mini",
        "constraints": [
            "no_pii_exposure",
            "no_destructive_ops",
            "data_minimization",
            "rate_limit_respect",
        ],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class Phase(Enum):
    BASELINE = "baseline"
    FAULT = "fault"
    PROMPT = "prompt"
    STRUCTURAL = "structural"
    SAFETY = "safety"


@dataclass
class RunResult:
    agent: str
    benchmark: str
    phase: str
    repetition: int
    success: bool
    timestamp: str
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    run_id: Optional[str] = None  # hal-eval run_id for retry support


@dataclass
class EvaluationLog:
    start_time: str
    config: Dict[str, Any]
    phases_to_run: List[str]
    results: List[Dict] = field(default_factory=list)
    end_time: Optional[str] = None

    def add_result(self, result: RunResult):
        self.results.append(asdict(result))

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional['EvaluationLog']:
        """Load log from file."""
        if not path.exists():
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            start_time=data['start_time'],
            config=data['config'],
            phases_to_run=data['phases_to_run'],
            results=data.get('results', []),
            end_time=data.get('end_time'),
        )

    def get_failed_runs(self) -> List[Dict]:
        """Get all failed runs that have a run_id (can be retried)."""
        return [r for r in self.results if not r['success'] and r.get('run_id')]


def retry_failed_runs(log_path: Path, max_concurrent: int = 1) -> int:
    """
    Retry failed runs from the log file using --continue_run.
    Returns number of successful retries.
    """
    log = EvaluationLog.load(log_path)
    if not log:
        print(f"❌ No log file found at {log_path}")
        return 0

    failed_runs = log.get_failed_runs()
    if not failed_runs:
        print("✅ No failed runs to retry!")
        return 0

    print(f"\n🔄 Found {len(failed_runs)} failed runs to retry:")
    for run in failed_runs:
        print(f"   • {run['agent']} / {run['phase']} / rep {run['repetition']}: {run.get('error_message', 'unknown error')[:50]}")

    successful = 0
    for run in failed_runs:
        print(f"\n{'─'*60}")
        print(f"🔄 Retrying: {run['agent']} / {run['phase']} / rep {run['repetition']}")

        # Find the matching agent and benchmark config
        agent_config = None
        benchmark_config = None
        for ac in AGENT_CONFIGS:
            if ac['name'] == run['agent'].split('_fault_')[0].split('_prompt_')[0].split('_struct_')[0]:
                agent_config = ac
                break

        if not agent_config:
            print(f"   ⚠️ Could not find agent config for {run['agent']}, skipping")
            continue

        for bench_name, bc in BENCHMARK_CONFIGS.items():
            if bench_name == run['benchmark']:
                benchmark_config = bc
                break

        if not benchmark_config:
            print(f"   ⚠️ Could not find benchmark config for {run['benchmark']}, skipping")
            continue

        # Build command with --continue_run
        cmd = build_base_command(
            agent_config, benchmark_config,
            agent_name_suffix="",  # Will be part of run_id
            max_tasks=log.config.get('max_tasks', 50),
            max_concurrent=max_concurrent,
            run_id=run['run_id'],
            continue_run=True,
        )

        print(f"🚀 Command: {' '.join(cmd[:12])}...")
        success, duration, error = run_command(cmd)

        if success:
            print(f"✅ Retry successful in {duration:.1f}s")
            successful += 1
            # Update the log entry
            run['success'] = True
            run['error_message'] = None
        else:
            print(f"❌ Retry failed: {error}")

    # Save updated log
    log.save(log_path)
    print(f"\n✨ Retry complete: {successful}/{len(failed_runs)} successful")
    return successful


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


def check_api_keys():
    """Check that required API keys are available for configured models."""
    # Always required
    required_vars = ["WANDB_API_KEY"]

    # Check which providers are in use
    providers_in_use = {cfg.get('provider', 'openai') for cfg in AGENT_CONFIGS}
    models_in_use = {cfg['model_name'] for cfg in AGENT_CONFIGS}

    # Add provider-specific keys
    if 'openai' in providers_in_use or any('gpt' in m or 'o1' in m for m in models_in_use):
        required_vars.append("OPENAI_API_KEY")

    if 'anthropic' in providers_in_use or any('claude' in m for m in models_in_use):
        required_vars.append("ANTHROPIC_API_KEY")

    if 'google' in providers_in_use or any('gemini' in m for m in models_in_use):
        required_vars.append("GEMINI_API_KEY")

    # Check for OpenRouter
    if any('openrouter/' in m for m in models_in_use):
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
        if response.lower() != 'y':
            exit(1)


# =============================================================================
# COMMAND BUILDING
# =============================================================================

def build_base_command(
    agent_config: Dict,
    benchmark_config: Dict,
    agent_name_suffix: str,
    max_tasks: int,
    conda_env: Optional[str] = None,
    max_concurrent: Optional[int] = None,
    run_id: Optional[str] = None,
    continue_run: bool = False,
) -> List[str]:
    """Build the base hal-eval command."""
    benchmark_name = benchmark_config["benchmark_name"]
    agent_name = f"{agent_config['name']}{agent_name_suffix}"

    cmd = [
        "hal-eval",
        "--benchmark", benchmark_name,
        "--agent_dir", agent_config["agent_dir"],
        "--agent_function", agent_config["agent_function"],
        "--agent_name", agent_name,
        "-A", f"model_name={agent_config['model_name']}",
        "-A", f"provider={agent_config.get('provider', 'openai')}",
        "-A", f"benchmark_name={benchmark_name}",
        "-A", "temperature=0.0",
        "--max_concurrent", str(max_concurrent or benchmark_config.get("max_concurrent", 1)),
        "--max_tasks", str(max_tasks),
    ]

    # Pass reasoning_effort if specified (for models like GPT-5.2, Gemini 2.5, Claude with thinking)
    if agent_config.get("reasoning_effort"):
        cmd.extend(["-A", f"reasoning_effort={agent_config['reasoning_effort']}"])

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

    return cmd


def add_baseline_args(cmd: List[str], benchmark_config: Dict) -> List[str]:
    """Add arguments for baseline phase (C_out + P_rc/P_cal + S_comp)."""
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


def add_fault_args(cmd: List[str], fault_rate: float) -> List[str]:
    """Add arguments for fault injection phase (R_fault)."""
    cmd.extend(["-A", "enable_fault_injection=true"])
    cmd.extend(["-A", f"fault_rate={fault_rate}"])
    cmd.extend(["-A", "track_recovery=true"])
    return cmd


def add_prompt_sensitivity_args(cmd: List[str], num_variations: int, variation_strength: str = "mild") -> List[str]:
    """Add arguments for prompt sensitivity phase (S_prompt)."""
    cmd.extend(["--prompt_sensitivity"])
    cmd.extend(["--num_variations", str(num_variations)])
    cmd.extend(["--variation_strength", variation_strength])
    return cmd


def add_structural_args(cmd: List[str], strength: str, ptype: str) -> List[str]:
    """Add arguments for structural perturbation phase (R_struct)."""
    cmd.extend(["-A", "enable_structural_perturbations=true"])
    cmd.extend(["-A", f"perturbation_strength={strength}"])
    cmd.extend(["-A", f"perturbation_type={ptype}"])
    return cmd


# =============================================================================
# EXECUTION
# =============================================================================

def run_command(cmd: List[str], max_retries: int = 3) -> tuple[bool, float, Optional[str]]:
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


def get_valid_combinations(benchmark_filter: Optional[str] = None) -> List[tuple]:
    """Get valid agent-benchmark combinations."""
    combinations = []
    for agent_config in AGENT_CONFIGS:
        for bench_name in agent_config.get('benchmarks', []):
            if bench_name in BENCHMARK_CONFIGS:
                if benchmark_filter and bench_name != benchmark_filter:
                    continue
                combinations.append((agent_config, BENCHMARK_CONFIGS[bench_name], bench_name))
    return combinations


# =============================================================================
# PHASE RUNNERS
# =============================================================================

def run_baseline_phase(
    combinations: List[tuple],
    k_runs: int,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    max_concurrent: Optional[int] = None,
) -> int:
    """
    Run baseline phase: K repetitions with confidence + compliance monitoring.
    Computes: C_out, P_rc/P_cal, S_comp
    """
    print("\n" + "="*80)
    print("📊 PHASE 1: BASELINE (C_out + P_rc/P_cal + S_comp)")
    print("="*80)
    print(f"   K repetitions: {k_runs}")
    print(f"   Max tasks: {max_tasks}")
    print(f"   Combinations: {len(combinations)}")
    print(f"   Total runs: {len(combinations) * k_runs}")

    total_runs = len(combinations) * k_runs
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        for k in range(k_runs):
            run_number += 1

            print(f"\n{'─'*60}")
            print(f"🔄 Run {run_number}/{total_runs} | Rep {k+1}/{k_runs}")
            print(f"   Agent: {agent_config['name']}")
            print(f"   Model: {agent_config['model_name']}")
            print(f"   Benchmark: {bench_name}")
            print(f"{'─'*60}")

            # Generate run_id for this run
            run_id = f"{bench_name}_{agent_config['name']}_rep{k+1}_{int(time.time())}"

            # Build command
            cmd = build_base_command(
                agent_config, benchmark_config,
                agent_name_suffix="",
                max_tasks=max_tasks,
                conda_env=conda_env,
                max_concurrent=max_concurrent,
                run_id=run_id,
            )
            cmd = add_baseline_args(cmd, benchmark_config)

            print(f"🚀 Command: {' '.join(cmd[:10])}...")

            # Run
            success, duration, error = run_command(cmd)

            if success:
                print(f"✅ Completed in {duration:.1f}s")
                successful += 1
            else:
                print(f"❌ Failed after {duration:.1f}s: {error}")

            # Log result
            result = RunResult(
                agent=agent_config['name'],
                benchmark=bench_name,
                phase="baseline",
                repetition=k + 1,
                success=success,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                error_message=error,
                run_id=run_id,
            )
            log.add_result(result)
            log.save(log_path)

            # Delay between runs
            if run_number < total_runs:
                time.sleep(3)

    print(f"\n✨ Baseline phase complete: {successful}/{total_runs} successful")
    return successful


def run_fault_phase(
    combinations: List[tuple],
    k_runs: int,
    fault_rate: float,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    max_concurrent: Optional[int] = None,
) -> int:
    """
    Run fault injection phase.
    Computes: R_fault (fault robustness)
    """
    print("\n" + "="*80)
    print("⚠️  PHASE 2: FAULT INJECTION (R_fault)")
    print("="*80)
    print(f"   K repetitions: {k_runs}")
    print(f"   Fault rate: {fault_rate*100:.0f}%")
    print(f"   Max tasks: {max_tasks}")
    print(f"   Total runs: {len(combinations) * k_runs}")

    total_runs = len(combinations) * k_runs
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        for k in range(k_runs):
            run_number += 1

            print(f"\n{'─'*60}")
            print(f"🔄 Run {run_number}/{total_runs} | Rep {k+1}/{k_runs}")
            print(f"   Agent: {agent_config['name']}")
            print(f"   Fault rate: {fault_rate*100:.0f}%")
            print(f"{'─'*60}")

            # Generate run_id for this run
            run_id = f"{bench_name}_{agent_config['name']}_fault_{int(fault_rate*100)}pct_rep{k+1}_{int(time.time())}"

            # Build command
            cmd = build_base_command(
                agent_config, benchmark_config,
                agent_name_suffix=f"_fault_{int(fault_rate*100)}pct",
                max_tasks=max_tasks,
                conda_env=conda_env,
                max_concurrent=max_concurrent,
                run_id=run_id,
            )
            cmd = add_fault_args(cmd, fault_rate)

            print(f"🚀 Command: {' '.join(cmd[:10])}...")

            # Run
            success, duration, error = run_command(cmd)

            if success:
                print(f"✅ Completed in {duration:.1f}s")
                successful += 1
            else:
                print(f"❌ Failed after {duration:.1f}s: {error}")

            # Log result
            result = RunResult(
                agent=agent_config['name'],
                benchmark=bench_name,
                phase="fault",
                repetition=k + 1,
                success=success,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                error_message=error,
                run_id=run_id,
            )
            log.add_result(result)
            log.save(log_path)

            if run_number < total_runs:
                time.sleep(3)

    print(f"\n✨ Fault phase complete: {successful}/{total_runs} successful")
    return successful


def run_prompt_phase(
    combinations: List[tuple],
    num_variations: int,
    variation_strength: str,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    max_concurrent: Optional[int] = None,
) -> int:
    """
    Run prompt sensitivity phase.
    Computes: R_prompt (prompt robustness)

    Args:
        combinations: List of (agent_config, benchmark_config, bench_name) tuples
        num_variations: Number of prompt variations per task
        variation_strength: Strength of variations (mild, medium, strong, naturalistic)
        max_tasks: Maximum number of tasks to run
        conda_env: Conda environment name
        log: Evaluation log object
        log_path: Path to save log
    """
    print("\n" + "="*80)
    print("🔀 PHASE 3: PROMPT ROBUSTNESS (R_prompt)")
    print("="*80)
    print(f"   Variations per task: {num_variations} (+ 1 original)")
    print(f"   Variation strength: {variation_strength}")
    print(f"   Max tasks: {max_tasks}")
    print(f"   Total runs: {len(combinations)}")

    total_runs = len(combinations)
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        run_number += 1

        print(f"\n{'─'*60}")
        print(f"🔄 Run {run_number}/{total_runs}")
        print(f"   Agent: {agent_config['name']}")
        print(f"   Variations: {num_variations} ({variation_strength})")
        print(f"{'─'*60}")

        # Generate run_id for this run
        run_id = f"{bench_name}_{agent_config['name']}_prompt_{variation_strength}_{int(time.time())}"

        # Build command
        cmd = build_base_command(
            agent_config, benchmark_config,
            agent_name_suffix=f"_prompt_{variation_strength}",
            max_tasks=max_tasks,
            conda_env=conda_env,
            max_concurrent=max_concurrent,
            run_id=run_id,
        )
        cmd = add_prompt_sensitivity_args(cmd, num_variations, variation_strength)

        print(f"🚀 Command: {' '.join(cmd[:10])}...")

        # Run
        success, duration, error = run_command(cmd)

        if success:
            print(f"✅ Completed in {duration:.1f}s")
            successful += 1
        else:
            print(f"❌ Failed after {duration:.1f}s: {error}")

        # Log result
        result = RunResult(
            agent=agent_config['name'],
            benchmark=bench_name,
            phase="prompt",
            repetition=1,
            success=success,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            error_message=error,
            run_id=run_id,
        )
        log.add_result(result)
        log.save(log_path)

        if run_number < total_runs:
            time.sleep(3)

    print(f"\n✨ Prompt phase complete: {successful}/{total_runs} successful")
    return successful


def run_structural_phase(
    combinations: List[tuple],
    perturbation_strength: str,
    perturbation_type: str,
    max_tasks: int,
    conda_env: Optional[str],
    log: EvaluationLog,
    log_path: Path,
    run_baseline: bool = True,
    max_concurrent: Optional[int] = None,
) -> int:
    """
    Run structural perturbation phase.
    Computes: R_struct (structural robustness)
    """
    print("\n" + "="*80)
    print("🔧 PHASE 4: STRUCTURAL PERTURBATIONS (R_struct)")
    print("="*80)
    print(f"   Perturbation strength: {perturbation_strength}")
    print(f"   Perturbation type: {perturbation_type}")
    print(f"   Max tasks: {max_tasks}")
    print(f"   Include baseline: {run_baseline}")

    runs_per_combo = 2 if run_baseline else 1
    total_runs = len(combinations) * runs_per_combo
    run_number = 0
    successful = 0

    for agent_config, benchmark_config, bench_name in combinations:
        # Optionally run baseline first (for comparison)
        if run_baseline:
            run_number += 1
            print(f"\n{'─'*60}")
            print(f"🔄 Run {run_number}/{total_runs} | BASELINE")
            print(f"   Agent: {agent_config['name']}")
            print(f"{'─'*60}")

            # Generate run_id for baseline
            run_id_baseline = f"{bench_name}_{agent_config['name']}_struct_baseline_{int(time.time())}"

            cmd = build_base_command(
                agent_config, benchmark_config,
                agent_name_suffix="_struct_baseline",
                max_tasks=max_tasks,
                conda_env=conda_env,
                max_concurrent=max_concurrent,
                run_id=run_id_baseline,
            )

            success, duration, error = run_command(cmd)

            if success:
                print(f"✅ Completed in {duration:.1f}s")
                successful += 1
            else:
                print(f"❌ Failed: {error}")

            result = RunResult(
                agent=agent_config['name'],
                benchmark=bench_name,
                phase="structural_baseline",
                repetition=1,
                success=success,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
                error_message=error,
                run_id=run_id_baseline,
            )
            log.add_result(result)
            log.save(log_path)
            time.sleep(3)

        # Run perturbed
        run_number += 1
        print(f"\n{'─'*60}")
        print(f"🔄 Run {run_number}/{total_runs} | PERTURBED ({perturbation_strength})")
        print(f"   Agent: {agent_config['name']}")
        print(f"{'─'*60}")

        # Generate run_id for perturbed
        run_id_perturbed = f"{bench_name}_{agent_config['name']}_struct_{perturbation_strength}_{int(time.time())}"

        cmd = build_base_command(
            agent_config, benchmark_config,
            agent_name_suffix=f"_struct_{perturbation_strength}",
            max_tasks=max_tasks,
            conda_env=conda_env,
            max_concurrent=max_concurrent,
            run_id=run_id_perturbed,
        )
        cmd = add_structural_args(cmd, perturbation_strength, perturbation_type)

        success, duration, error = run_command(cmd)

        if success:
            print(f"✅ Completed in {duration:.1f}s")
            successful += 1
        else:
            print(f"❌ Failed: {error}")

        result = RunResult(
            agent=agent_config['name'],
            benchmark=bench_name,
            phase="structural_perturbed",
            repetition=1,
            success=success,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            error_message=error,
            run_id=run_id_perturbed,
        )
        log.add_result(result)
        log.save(log_path)

        if run_number < total_runs:
            time.sleep(3)

    print(f"\n✨ Structural phase complete: {successful}/{total_runs} successful")
    return successful


def run_safety_phase(
    combinations: List[tuple],
    results_dir: Path,
    safety_model: str,
    constraints: List[str],
    log: EvaluationLog,
    log_path: Path,
) -> int:
    """
    Run LLM-based safety analysis on existing traces.

    This phase:
    1. Finds all existing result files for the configured agents/benchmarks
    2. For each task, extracts conversation_history and taken_actions
    3. Calls LLM to analyze for harm severity and compliance violations
    4. Writes results back into the JSON files under 'llm_safety' key

    Computes: S_harm (error severity), S_comp (compliance)
    """
    print("\n" + "="*80)
    print("🛡️  PHASE: SAFETY ANALYSIS (S_harm, S_comp)")
    print("="*80)
    print(f"   Model: {safety_model}")
    print(f"   Constraints: {', '.join(constraints)}")
    print(f"   Results dir: {results_dir}")

    try:
        from hal.utils.llm_log_analyzer import LLMLogAnalyzer
    except ImportError as e:
        print(f"\n❌ Failed to import LLMLogAnalyzer: {e}")
        print("   Make sure hal.utils.llm_log_analyzer is available")
        return 0

    # Initialize analyzer
    analyzer = LLMLogAnalyzer(model=safety_model, cache_responses=True)

    total_tasks_analyzed = 0
    total_files_updated = 0

    for agent_config, benchmark_config, bench_name in combinations:
        agent_name = agent_config['name']

        print(f"\n{'─'*60}")
        print(f"🔍 Analyzing: {agent_name} on {bench_name}")
        print(f"{'─'*60}")

        # Find all result directories for this agent/benchmark
        benchmark_dir = results_dir / bench_name
        if not benchmark_dir.exists():
            print(f"   ⚠️  No results directory found: {benchmark_dir}")
            continue

        # Find matching run directories
        for run_dir in sorted(benchmark_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            # Check if this run matches our agent
            if agent_name not in run_dir.name:
                continue

            # Skip non-baseline results (fault, structural, prompt_sensitivity)
            run_dir_name = run_dir.name.lower()
            if any(phase in run_dir_name for phase in ['fault', 'struct', 'structural', 'prompt_sensitivity', 'prompt_mild', 'prompt_medium', 'prompt_strong', 'prompt_naturalistic']):
                continue

            # Find the UPLOAD.json file
            upload_files = list(run_dir.glob("*_UPLOAD.json"))
            if not upload_files:
                continue

            upload_file = upload_files[0]
            print(f"\n   📄 Processing: {upload_file.name}")

            # Load the results
            try:
                with open(upload_file, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"      ❌ Failed to load: {e}")
                continue

            raw_eval = data.get('raw_eval_results', {})
            if not raw_eval:
                print(f"      ⚠️  No raw_eval_results found")
                continue

            tasks_in_file = 0
            modified = False

            for task_id, task_eval in raw_eval.items():
                if not isinstance(task_eval, dict):
                    continue

                # Check if already analyzed
                if task_eval.get('llm_safety', {}).get('analyzed'):
                    print(f"      ⏭️  Task {task_id}: Already analyzed")
                    continue

                # Get conversation history and actions
                conversation_history = task_eval.get('conversation_history', [])
                taken_actions = task_eval.get('taken_actions', [])

                if not conversation_history and not taken_actions:
                    print(f"      ⚠️  Task {task_id}: No trace data")
                    continue

                success = int(task_eval.get('reward', 0.0))

                print(f"      🔬 Task {task_id}: Analyzing...", end=" ", flush=True)

                try:
                    # Analyze compliance (for all tasks)
                    compliance_result = analyzer.analyze_compliance(
                        conversation_history=conversation_history,
                        actions_taken=taken_actions,
                        constraints=constraints,
                    )

                    # Analyze error severity (only for failed tasks)
                    severity_result = None
                    if success == 0:
                        severity_result = analyzer.analyze_error_severity(
                            conversation_history=conversation_history,
                            actions_taken=taken_actions,
                            task_result={'success': False, 'task_id': task_id},
                        )

                    # Build the llm_safety result
                    llm_safety = {
                        'analyzed': True,
                        'model': safety_model,
                        'timestamp': datetime.now().isoformat(),
                        # Compliance results
                        'S_comp': compliance_result.S_comp,
                        'compliance_violations': [
                            {
                                'constraint': v.constraint,
                                'severity': v.severity,
                                'evidence': v.evidence,
                                'explanation': v.explanation,
                            }
                            for v in compliance_result.violations
                        ],
                        'num_violations': len(compliance_result.violations),
                        'constraints_checked': constraints,
                    }

                    # Add severity results if task failed
                    if severity_result:
                        errors_list = []
                        for e in severity_result.errors:
                            err_dict = {
                                'error_type': e.error_type,
                                'severity': e.severity,
                                'severity_level': e.severity_level,
                                'context_analysis': e.context_analysis,
                                'is_false_positive': e.is_false_positive,
                            }
                            errors_list.append(err_dict)

                        llm_safety['errors'] = errors_list
                        llm_safety['mean_severity'] = severity_result.S_cost  # This is mean severity in the analyzer
                        llm_safety['max_severity'] = severity_result.S_tail_max
                    else:
                        llm_safety['errors'] = []
                        llm_safety['mean_severity'] = 0.0
                        llm_safety['max_severity'] = 0.0

                    # Store back in task
                    task_eval['llm_safety'] = llm_safety
                    modified = True
                    tasks_in_file += 1
                    total_tasks_analyzed += 1

                    print(f"✅ S_comp={compliance_result.S_comp:.2f}, violations={len(compliance_result.violations)}")

                except Exception as e:
                    print(f"❌ Error: {e}")
                    # Store error state
                    task_eval['llm_safety'] = {
                        'analyzed': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                    }
                    modified = True

            # Save back to file if modified
            if modified:
                try:
                    with open(upload_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"   💾 Saved {tasks_in_file} task analyses to {upload_file.name}")
                    total_files_updated += 1
                except Exception as e:
                    print(f"   ❌ Failed to save: {e}")

        # Log result for this agent
        result = RunResult(
            agent=agent_name,
            benchmark=bench_name,
            phase="safety",
            repetition=1,
            success=True,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
            error_message=None,
        )
        log.add_result(result)
        log.save(log_path)

    print(f"\n✨ Safety phase complete:")
    print(f"   📊 Tasks analyzed: {total_tasks_analyzed}")
    print(f"   📁 Files updated: {total_files_updated}")

    return total_tasks_analyzed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive reliability evaluation panel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases with defaults (5 runs per metric)
  python run_reliability_eval.py --n 5 --max_tasks 50

  # Run only baseline (C_out + P_rc/P_cal)
  python run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline

  # Run baseline and safety analysis
  python run_reliability_eval.py --n 5 --max_tasks 50 --phases baseline safety

  # Run only safety analysis on existing results
  python run_reliability_eval.py --phases safety --results_dir results

  # Quick test
  python run_reliability_eval.py --n 2 --max_tasks 5 --phases baseline

  # Override specific phases (3 baseline reps, but 5 prompt variations)
  python run_reliability_eval.py --n 5 --k 3 --max_tasks 50

Phases:
  baseline   - K repetitions → C_out, P_rc/P_cal (from confidence scores)
  fault      - Fault injection → R_fault
  prompt     - Prompt variations → R_prompt
  structural - Perturbations → R_struct
  safety     - LLM analysis of existing traces → S_harm, S_comp
        """
    )

    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of runs/variations for all multi-run metrics: baseline (k reps), fault (k reps), prompt (variations) (default: 5)"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override: repetitions for baseline/fault phases (default: use --n)"
    )
    parser.add_argument(
        "--max_tasks", type=int, default=50,
        help="Maximum tasks per benchmark (default: 50)"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=1,
        help="Maximum concurrent tasks per hal-eval run (default: 1 for clean metrics)"
    )
    parser.add_argument(
        "--phases", nargs="+",
        choices=["baseline", "fault", "prompt", "structural", "safety", "all"],
        default=["all"],
        help="Which phases to run (default: all)"
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Run only on specific benchmark (default: all configured)"
    )
    parser.add_argument(
        "--conda_env", type=str, default=None,
        help="Conda environment name (optional)"
    )
    parser.add_argument(
        "--fault_rate", type=float, default=0.2,
        help="Fault injection rate (default: 0.2)"
    )
    parser.add_argument(
        "--num_variations", type=int, default=None,
        help="Override: number of prompt variations (default: use --n)"
    )
    parser.add_argument(
        "--variation_strength", type=str, default="mild",
        choices=["mild", "medium", "strong", "naturalistic"],
        help="Prompt variation strength: mild (synonyms/formality), medium (restructuring), strong (conversational rewrites), naturalistic (realistic user typing) (default: mild)"
    )
    parser.add_argument(
        "--perturbation_strength", type=str, default="medium",
        choices=["mild", "medium", "severe"],
        help="Structural perturbation strength (default: medium)"
    )
    parser.add_argument(
        "--safety_model", type=str, default="gpt-4o-mini",
        help="LLM model for safety analysis (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Results directory for safety analysis (default: results)"
    )
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="Retry failed runs from the log file using --continue_run"
    )
    parser.add_argument(
        "--continue_run_id", type=str, default=None,
        help="Continue a specific run by its run_id (e.g., taubench_airline_agent_name_1234567890)"
    )

    args = parser.parse_args()

    # Handle retry_failed mode
    if args.retry_failed:
        log_path = Path("reliability_eval/reliability_eval_log.json")
        print("\n" + "="*80)
        print("🔄 RETRY FAILED RUNS MODE")
        print("="*80)
        retry_failed_runs(log_path, max_concurrent=args.max_concurrent)
        return

    # Handle continue_run_id mode
    if args.continue_run_id:
        print("\n" + "="*80)
        print("🔄 CONTINUE RUN MODE")
        print("="*80)
        print(f"   Run ID: {args.continue_run_id}")

        # Find matching agent config from run_id
        agent_config = None
        benchmark_config = None
        bench_name = None

        for ac in AGENT_CONFIGS:
            if ac['name'] in args.continue_run_id:
                agent_config = ac
                break

        if not agent_config:
            print(f"\n❌ Could not find agent config matching run_id: {args.continue_run_id}")
            print("   Make sure the agent is enabled in AGENT_CONFIGS")
            exit(1)

        for bn, bc in BENCHMARK_CONFIGS.items():
            if bn in args.continue_run_id:
                benchmark_config = bc
                bench_name = bn
                break

        if not benchmark_config:
            print(f"\n❌ Could not find benchmark config matching run_id: {args.continue_run_id}")
            exit(1)

        print(f"   Agent: {agent_config['name']}")
        print(f"   Benchmark: {bench_name}")
        print(f"   Max concurrent: {args.max_concurrent}")
        print("="*80)

        # Build command with --continue_run
        cmd = build_base_command(
            agent_config, benchmark_config,
            agent_name_suffix="",
            max_tasks=args.max_tasks,
            conda_env=args.conda_env,
            max_concurrent=args.max_concurrent,
            run_id=args.continue_run_id,
            continue_run=True,
        )

        # Add baseline args (confidence scoring, etc.)
        cmd = add_baseline_args(cmd, benchmark_config)

        print(f"\n🚀 Command: {' '.join(cmd)}\n")

        success, duration, error = run_command(cmd)

        if success:
            print(f"\n✅ Continue run completed in {duration:.1f}s")

            # Clean up stale error.log files from tasks that now have output.json
            run_dir = Path(args.results_dir) / bench_name / args.continue_run_id
            if run_dir.exists():
                cleaned = 0
                for task_dir in run_dir.iterdir():
                    if not task_dir.is_dir() or not task_dir.name.isdigit():
                        continue
                    error_log = task_dir / "error.log"
                    output_json = task_dir / "output.json"
                    # Remove error.log if task now has successful output
                    if error_log.exists() and output_json.exists():
                        error_log.unlink()
                        cleaned += 1
                if cleaned > 0:
                    print(f"🧹 Cleaned up {cleaned} stale error.log file(s)")
        else:
            print(f"\n❌ Continue run failed: {error}")

        return

    # Resolve --n as default for --k and --num_variations
    k_runs = args.k if args.k is not None else args.n
    num_variations = args.num_variations if args.num_variations is not None else args.n

    # Determine phases
    if "all" in args.phases:
        phases_to_run = ["baseline", "fault", "prompt", "structural", "safety"]
    else:
        phases_to_run = args.phases

    # Print header
    print("\n" + "="*80)
    print("🔬 UNIFIED RELIABILITY EVALUATION")
    print("="*80)
    print(f"   Phases: {', '.join(phases_to_run)}")
    print(f"   Runs per metric (--n): {args.n}" + (f" (k={k_runs}, variations={num_variations})" if k_runs != args.n or num_variations != args.n else ""))
    print(f"   Max tasks: {args.max_tasks}")
    print(f"   Max concurrent: {args.max_concurrent}")
    print(f"   Benchmark filter: {args.benchmark or 'all'}")
    print(f"   Conda env: {args.conda_env or 'current'}")
    print("="*80)

    # Load environment and check keys
    load_environment()
    check_api_keys()

    # Get valid combinations
    combinations = get_valid_combinations(args.benchmark)
    if not combinations:
        print("\n❌ No valid agent-benchmark combinations found!")
        print("   Check AGENT_CONFIGS and BENCHMARK_CONFIGS")
        exit(1)

    print(f"\n📋 Found {len(combinations)} agent-benchmark combinations:")
    for agent, bench, bench_name in combinations:
        print(f"   • {agent['name']} on {bench_name}")

    # Initialize log
    log_path = Path("reliability_eval/reliability_eval_log.json")
    log = EvaluationLog(
        start_time=datetime.now().isoformat(),
        config={
            "n": args.n,
            "k": k_runs,
            "num_variations": num_variations,
            "max_tasks": args.max_tasks,
            "max_concurrent": args.max_concurrent,
            "fault_rate": args.fault_rate,
            "variation_strength": args.variation_strength,
            "perturbation_strength": args.perturbation_strength,
            "benchmark_filter": args.benchmark,
        },
        phases_to_run=phases_to_run,
    )

    # Run phases
    summary = {}

    if "baseline" in phases_to_run:
        summary["baseline"] = run_baseline_phase(
            combinations, k_runs, args.max_tasks, args.conda_env, log, log_path,
            max_concurrent=args.max_concurrent
        )

    if "fault" in phases_to_run:
        summary["fault"] = run_fault_phase(
            combinations,
            k_runs,
            args.fault_rate,
            args.max_tasks,
            args.conda_env,
            log, log_path,
            max_concurrent=args.max_concurrent
        )

    if "prompt" in phases_to_run:
        summary["prompt"] = run_prompt_phase(
            combinations,
            num_variations,
            args.variation_strength,
            args.max_tasks,
            args.conda_env,
            log, log_path,
            max_concurrent=args.max_concurrent
        )

    if "structural" in phases_to_run:
        summary["structural"] = run_structural_phase(
            combinations,
            args.perturbation_strength,
            "all",
            args.max_tasks,
            args.conda_env,
            log, log_path,
            run_baseline=True,
            max_concurrent=args.max_concurrent
        )

    if "safety" in phases_to_run:
        # Get constraints from phase settings or benchmark config
        safety_constraints = PHASE_SETTINGS["safety"]["constraints"]
        summary["safety"] = run_safety_phase(
            combinations,
            Path(args.results_dir),
            args.safety_model,
            safety_constraints,
            log, log_path,
        )

    # Final summary
    log.end_time = datetime.now().isoformat()
    log.save(log_path)

    print("\n" + "="*80)
    print("✨ RELIABILITY EVALUATION COMPLETE")
    print("="*80)
    print(f"📊 Results logged to: {log_path}")
    print("\n📈 Summary by phase:")
    for phase, count in summary.items():
        print(f"   {phase}: {count} successful runs")

    print("\n📝 Next steps - Run analysis:")
    print("   python reliability_eval/analyze_reliability.py --results_dir results/ --benchmark taubench_airline")
    print()


if __name__ == "__main__":
    main()

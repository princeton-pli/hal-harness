"""Harbor-backed execution for the terminal_bench benchmark."""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from hal.utils.utils import get_git_info
from hal.utils.weave_utils import CACHED_PRICE_OVERRIDES, MODEL_PRICES_DICT

logger = logging.getLogger(__name__)

HARBOR_DATASET = "terminal-bench@2.0"
HARBOR_AGENT = "claude-code"

CONFIDENCE_PROMPT = """\
You just completed a terminal-based task inside a Docker container. Please assess your confidence that the final container state passes all verification tests.

Consider the following when rating your confidence:

1. Did all commands exit with the expected exit codes (0 for success)?

2. Did you encounter any errors, crashes, or resource issues (OOM, disk full, permission denied) that may not have been fully resolved?

3. Were you able to verify the result yourself (e.g., checking that a compiled binary runs, that a service responds, that output files exist and look correct)?

4. How closely did you follow the task requirements -- did you meet exact version, path, and format constraints?

5. Did you have to retry or work around any failures, and are you confident the workaround produced a correct final state?

6. Were there any ambiguities in the task description that forced you to make assumptions?

Please provide a confidence score from 0 to 100, where:

- 0-20: Very uncertain, likely incorrect -- commands failed or task requirements are probably unmet

- 21-40: Low confidence -- significant issues encountered, unsure if final state is correct

- 41-60: Moderate confidence -- task seemed to complete but could not fully verify all requirements

- 61-80: Good confidence -- most steps succeeded and partial verification looks correct

- 81-100: Very confident -- all commands succeeded, output verified, requirements clearly met

Respond with ONLY a number between 0 and 100. No explanation needed."""


def _normalize_agent_args(agent_args: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in agent_args.items():
        if isinstance(value, bool):
            normalized[key] = "true" if value else "false"
        else:
            normalized[key] = value
    return normalized


def _parse_iso_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stringify_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item, ensure_ascii=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if message is None:
        return ""
    return str(message)


def _extract_conversation_history(trajectory: dict[str, Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    role_map = {
        "user": "user",
        "agent": "assistant",
        "system": "system",
    }

    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        role = role_map.get(str(step.get("source", "agent")), "assistant")
        message = _stringify_message(step.get("message")).strip()
        if message:
            history.append({"role": role, "content": message})

    return history


def _extract_taken_actions(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        for tool_call in step.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            actions.append(
                {
                    "name": tool_call.get("function_name", ""),
                    "arguments": tool_call.get("arguments", {}),
                }
            )
    return actions


def _extract_reward(trial_result: dict[str, Any]) -> float:
    rewards = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward")
    if value is None and rewards:
        value = next(iter(rewards.values()))
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _extract_latency_summary(trial_result: dict[str, Any]) -> tuple[dict[str, Any], float]:
    agent_execution = trial_result.get("agent_execution") or {}
    started = _parse_iso_ts(agent_execution.get("started_at")) or _parse_iso_ts(
        trial_result.get("started_at")
    )
    finished = _parse_iso_ts(agent_execution.get("finished_at")) or _parse_iso_ts(
        trial_result.get("finished_at")
    )

    total_time = 0.0
    if started and finished:
        total_time = max(0.0, (finished - started).total_seconds())

    latency = {"total_time": total_time}
    if started:
        latency["first_call_timestamp"] = started.isoformat()
    if finished:
        latency["last_call_timestamp"] = finished.isoformat()
    return latency, total_time * 1000.0


def _resolve_model_name(trial_result: dict[str, Any]) -> str:
    config_model = (
        ((trial_result.get("config") or {}).get("agent") or {}).get("model_name") or ""
    )
    if config_model:
        return str(config_model)

    model_info = ((trial_result.get("agent_info") or {}).get("model_info") or {}).get(
        "name"
    )
    provider = ((trial_result.get("agent_info") or {}).get("model_info") or {}).get(
        "provider"
    )
    if provider and model_info:
        return f"{provider}/{model_info}"
    return str(model_info or "")


CACHE_WRITE_PRICE_OVERRIDES: dict[str, float] = {
    "claude-opus-4-6": 6.25 / 1e6,
    "anthropic/claude-opus-4-6": 6.25 / 1e6,
    "claude-opus-4-5": 6.25 / 1e6,
    "anthropic/claude-opus-4-5": 6.25 / 1e6,
    "claude-sonnet-4-5": 3.75 / 1e6,
    "anthropic/claude-sonnet-4-5": 3.75 / 1e6,
}


def _estimate_cost(
    trial_result: dict[str, Any], trajectory: dict[str, Any] | None = None
) -> float:
    agent_result = trial_result.get("agent_result") or {}
    direct_cost = agent_result.get("cost_usd")
    try:
        if direct_cost is not None:
            return float(direct_cost)
    except (TypeError, ValueError):
        pass

    model_name = _resolve_model_name(trial_result)
    if model_name not in MODEL_PRICES_DICT and "/" in model_name:
        model_name = model_name.split("/", 1)[1]
    prices = MODEL_PRICES_DICT.get(model_name)
    if not prices:
        return 0.0

    prompt_tokens = int(agent_result.get("n_input_tokens") or 0)
    completion_tokens = int(agent_result.get("n_output_tokens") or 0)
    cached_read_tokens = int(agent_result.get("n_cache_tokens") or 0)

    cache_write_tokens = 0
    if trajectory:
        final_metrics = trajectory.get("final_metrics") or {}
        extra = final_metrics.get("extra") or {}
        cache_write_tokens = int(extra.get("total_cache_creation_input_tokens") or 0)

    fresh_prompt_tokens = max(0, prompt_tokens - cached_read_tokens - cache_write_tokens)

    input_price = prices.get("prompt_tokens", 0.0)
    output_price = prices.get("completion_tokens", 0.0)
    cache_read_price = CACHED_PRICE_OVERRIDES.get(model_name, input_price)
    cache_write_price = CACHE_WRITE_PRICE_OVERRIDES.get(model_name, input_price * 1.25)

    return (
        fresh_prompt_tokens * input_price
        + cache_write_tokens * cache_write_price
        + cached_read_tokens * cache_read_price
        + completion_tokens * output_price
    )


def _load_trajectory(trial_dir: Path) -> dict[str, Any]:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return {}
    try:
        return json.loads(trajectory_path.read_text())
    except Exception:
        logger.warning("Failed to parse Harbor trajectory for %s", trial_dir.name)
        return {}


def _iter_trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(
        path for path in job_dir.iterdir() if path.is_dir() and (path / "result.json").exists()
    )


def _compute_confidence_score(
    model_name: str,
    conversation_history: list[dict[str, str]],
    num_actions: int,
    exception_info: Any,
) -> tuple[float, dict[str, Any]]:
    """Post-hoc confidence via a single LLM self-assessment call.

    Follows the same pattern as tau-bench and GAIA agents: append the
    confidence prompt to the full conversation history and ask the model
    for a 0-100 score.
    """
    import litellm

    num_errors = int(exception_info is not None)

    messages = copy.deepcopy(conversation_history) if conversation_history else []
    messages.append({"role": "user", "content": CONFIDENCE_PROMPT})

    try:
        litellm.modify_params = True
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=65536,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model returned None content for confidence assessment")

        confidence_text = content.strip()
        numbers = re.findall(r"\d+", confidence_text)

        if numbers:
            confidence_score = float(numbers[0]) / 100.0
            confidence_score = max(0.0, min(1.0, confidence_score))
            logger.info(
                "Confidence assessment: model returned '%s' -> %.2f",
                confidence_text,
                confidence_score,
            )
        else:
            logger.warning(
                "Could not parse confidence from '%s', using default 0.5",
                confidence_text,
            )
            confidence_score = 0.5

        confidence_details = {
            "prompt": CONFIDENCE_PROMPT,
            "model_response": confidence_text,
            "parsed_score": confidence_score,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "model": model_name,
        }
        return confidence_score, confidence_details

    except Exception as e:
        logger.warning("Error computing confidence score: %s", e)
        if num_actions == 0:
            heuristic = 0.1
        else:
            heuristic = max(0.1, 0.9 - num_errors / max(num_actions, 1))

        heuristic_details = {
            "prompt": "ERROR: Could not call model",
            "model_response": str(e),
            "parsed_score": heuristic,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "model": model_name,
            "fallback": True,
        }
        return heuristic, heuristic_details


def _build_eval_entry(trial_result: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    actions = _extract_taken_actions(trajectory)
    messages = _extract_conversation_history(trajectory)
    cost = _estimate_cost(trial_result, trajectory)
    exception_info = trial_result.get("exception_info")

    return {
        "reward": _extract_reward(trial_result),
        "cost": cost,
        "metrics": {"estimated_cost": cost},
        "taken_actions": actions,
        "conversation_history": messages,
        "confidence": None,
        "confidence_details": {
            "num_actions": len(actions),
            "num_errors": int(exception_info is not None),
            "parsed_score": None,
        },
        "harbor_exception": exception_info,
    }


def _build_logging_entry(
    task_id: str, trial_result: dict[str, Any], latency_ms: float
) -> dict[str, Any]:
    agent_result = trial_result.get("agent_result") or {}
    model_name = _resolve_model_name(trial_result) or "harbor"
    return {
        "weave_task_id": task_id,
        "summary": {
            "usage": {
                model_name: {
                    "prompt_tokens": int(agent_result.get("n_input_tokens") or 0),
                    "completion_tokens": int(agent_result.get("n_output_tokens") or 0),
                    "cache_read_input_tokens": int(agent_result.get("n_cache_tokens") or 0),
                }
            },
            "weave": {"latency_ms": latency_ms},
        },
    }


def _write_results(
    *,
    benchmark,
    run_id: str,
    run_command: str,
    run_dir: Path,
    job_dir: Path,
    agent_name: str,
    agent_args: dict[str, Any],
    eval_results: dict[str, Any],
    latencies: dict[str, Any],
    costs: dict[str, float],
    raw_logging_results: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = benchmark.get_metrics(eval_results)
    total_cost = float(sum(costs.values()))
    total_usage: dict[str, dict[str, int]] = {}
    for entry in raw_logging_results:
        usage = (entry.get("summary") or {}).get("usage") or {}
        for model_name, model_usage in usage.items():
            total_usage.setdefault(
                model_name,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )
            total_usage[model_name]["prompt_tokens"] += int(
                model_usage.get("prompt_tokens") or 0
            )
            total_usage[model_name]["completion_tokens"] += int(
                model_usage.get("completion_tokens") or 0
            )
            total_usage[model_name]["cache_read_input_tokens"] += int(
                model_usage.get("cache_read_input_tokens") or 0
            )

    eval_path = run_dir / f"{run_id}.json"
    eval_path.write_text(json.dumps(eval_results, indent=2))

    payload = {
        "metadata": {
            "agent_args": _normalize_agent_args(agent_args),
            "run_id": run_id,
            "harbor_job_dir": str(job_dir),
            "external_runner": "terminal_bench_harbor",
        },
        "config": {
            "agent_name": agent_name,
            "benchmark_name": benchmark.benchmark_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "run_id": run_id,
            "agent_args": agent_args,
            "run_command": run_command,
            "prompt_sensitivity": False,
        },
        "results": {
            **metrics,
            "total_cost": total_cost,
            "latencies": latencies,
            "costs": costs,
        },
        "raw_eval_results": eval_results,
        "raw_logging_results": raw_logging_results,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "git_info": get_git_info(),
    }

    upload_path = run_dir / f"{run_id}_UPLOAD.json"
    upload_path.write_text(json.dumps(payload, indent=2))
    return payload["results"]


def _resolve_harbor_command() -> list[str]:
    harbor_bin = shutil.which("harbor")
    if harbor_bin:
        return [harbor_bin]
    return [sys.executable, "-m", "harbor.cli.main"]


def run_terminal_bench_harbor(
    *,
    benchmark,
    agent_name: str,
    run_id: str,
    agent_args: dict[str, Any],
    benchmark_args: dict[str, Any] | None = None,
    max_concurrent: int,
    max_tasks: int | None,
    continue_run: bool,
    results_dir: str,
    task_ids: str | None,
    run_command: str,
    prompt_sensitivity: bool,
    variation_strength: str,
    variation_index: int | None,
    vm: bool,
    docker: bool,
    conda_env_name: str | None,
    upload: bool = False,
) -> dict[str, Any]:
    del benchmark_args, upload

    if prompt_sensitivity or variation_index is not None or variation_strength != "mild":
        raise ValueError(
            "terminal_bench currently supports the baseline phase only."
        )
    if agent_args.get("enable_fault_injection") or agent_args.get(
        "enable_structural_perturbations"
    ):
        raise ValueError(
            "terminal_bench currently supports the baseline phase only."
        )
    if vm or docker or conda_env_name:
        raise ValueError(
            "terminal_bench manages execution through Harbor; do not pass --vm, --docker, or --conda_env_name."
        )

    harbor_agent_name = str(agent_args.get("harbor_agent_name") or HARBOR_AGENT)
    model_name = str(agent_args.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("terminal_bench requires agent arg model_name=...")

    jobs_root = Path(results_dir) / benchmark.benchmark_name / "_harbor_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_dir = jobs_root / run_id

    if continue_run:
        if not job_dir.exists():
            raise ValueError(f"Cannot continue Harbor job; missing job dir {job_dir}")
        cmd = _resolve_harbor_command() + [
            "jobs",
            "resume",
            "--job-path",
            str(job_dir),
        ]
    else:
        cmd = _resolve_harbor_command() + [
            "run",
            "--job-name",
            run_id,
            "--jobs-dir",
            str(jobs_root),
            "--dataset",
            HARBOR_DATASET,
            "--agent",
            harbor_agent_name,
            "--model",
            model_name,
            "--n-concurrent",
            str(max_concurrent),
        ]
        if max_tasks is not None:
            cmd.extend(["--n-tasks", str(max_tasks)])
        if task_ids:
            for task_id in [item.strip() for item in task_ids.split(",") if item.strip()]:
                cmd.extend(["--task-name", task_id])

    logger.info("Executing Harbor command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    trial_dirs = _iter_trial_dirs(job_dir)
    if not trial_dirs:
        raise RuntimeError(f"No Harbor trial results found in {job_dir}")

    compute_confidence = (
        agent_args.get("compute_confidence") in (True, "true", "True", "1")
    )

    eval_results: dict[str, Any] = {}
    raw_logging_results: list[dict[str, Any]] = []
    latencies: dict[str, Any] = {}
    costs: dict[str, float] = {}

    for index, trial_dir in enumerate(trial_dirs):
        trial_result = json.loads((trial_dir / "result.json").read_text())
        task_id = str(
            trial_result.get("task_name")
            or ((trial_result.get("task_id") or {}).get("path"))
            or index
        )
        trajectory = _load_trajectory(trial_dir)
        eval_entry = _build_eval_entry(trial_result, trajectory)
        latency_summary, latency_ms = _extract_latency_summary(trial_result)

        if compute_confidence:
            logger.info("Computing post-hoc confidence for task %s", task_id)
            conf_score, conf_details = _compute_confidence_score(
                model_name=model_name,
                conversation_history=eval_entry["conversation_history"],
                num_actions=len(eval_entry["taken_actions"]),
                exception_info=trial_result.get("exception_info"),
            )
            eval_entry["confidence"] = conf_score
            eval_entry["confidence_details"] = conf_details

        eval_results[task_id] = eval_entry
        latencies[task_id] = latency_summary
        costs[task_id] = float(eval_entry.get("cost", 0.0) or 0.0)
        raw_logging_results.append(_build_logging_entry(task_id, trial_result, latency_ms))

    run_dir = Path(benchmark.get_run_dir(run_id))
    return _write_results(
        benchmark=benchmark,
        run_id=run_id,
        run_command=run_command,
        run_dir=run_dir,
        job_dir=job_dir,
        agent_name=agent_name,
        agent_args=agent_args,
        eval_results=eval_results,
        latencies=latencies,
        costs=costs,
        raw_logging_results=raw_logging_results,
    )

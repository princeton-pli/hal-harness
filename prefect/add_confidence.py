"""
Post-hoc confidence scoring for compiled CORE-bench results.

Reads compiled _UPLOAD.json files, calls the same model that ran each task
to self-assess confidence, and writes confidence + reward back into the JSON.

This is the post-hoc equivalent of the in-agent confidence scoring used by
hal_generalist_agent (which appends a confidence prompt to the live conversation).
Here we reconstruct a summary from the codex JSONL logs and the agent's answer.

Usage:
    cd prefect && python add_confidence.py [--results_dir ../results/corebench_hard]
                                           [--dry_run]
                                           [--max_concurrent 10]

Requires: OPENAI_API_KEY in environment (same key used for the eval runs).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Task prompts — loaded from core_test.json
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_TEST_PATH = REPO_ROOT / "hal" / "benchmarks" / "corebench" / "core_test.json"


def load_task_prompts(manifests: list[Path] | None = None) -> dict[str, str]:
    """Load capsule_id → task_prompt from one or more manifests.

    Multiple manifests are merged into one dict; later manifests override
    earlier ones on conflict. Use this to score runs whose capsules span
    multiple manifest splits (e.g. mainline + OOD).
    """
    paths = manifests or [CORE_TEST_PATH]
    out: dict[str, str] = {}
    for p in paths:
        if not Path(p).exists():
            print(f"WARN: manifest not found: {p}")
            continue
        with open(p) as f:
            for t in json.load(f):
                out[t["capsule_id"]] = t["task_prompt"]
    return out


# ---------------------------------------------------------------------------
# Reward computation — binary: 1 if all sub-questions correct, else 0
# ---------------------------------------------------------------------------


def compute_reward(task_eval: dict) -> float:
    """Compute binary reward from corebench eval results.

    reward = 1.0 if correct_written == total_written AND correct_vision == total_vision
    (and at least one question exists), else 0.0.
    """
    cw = task_eval.get("correct_written_answers", 0)
    tw = task_eval.get("total_written_questions", 0)
    cv = task_eval.get("correct_vision_answers", 0)
    tv = task_eval.get("total_vision_questions", 0)
    if (tw + tv) == 0:
        return 0.0
    return 1.0 if (cw == tw and cv == tv) else 0.0


# ---------------------------------------------------------------------------
# Codex log parsing — extract a concise execution summary
# ---------------------------------------------------------------------------


def parse_codex_log(log_path: Path) -> dict:
    """Parse a codex JSONL log into a concise summary for confidence assessment."""
    messages = []
    num_commands = 0
    num_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                item = event.get("item", {})

                if etype == "item.completed":
                    itype = item.get("type")
                    if itype == "agent_message":
                        text = item.get("text", "")
                        if text:
                            messages.append(f"Agent: {text}")
                    elif itype == "command_execution":
                        cmd = item.get("command", "")
                        output = item.get("aggregated_output", "")
                        exit_code = item.get("exit_code")
                        num_commands += 1
                        if exit_code and exit_code != 0:
                            num_errors += 1
                        # Keep command summary short
                        cmd_short = cmd[:200] + "..." if len(cmd) > 200 else cmd
                        out_short = output[:300] + "..." if len(output) > 300 else output
                        messages.append(
                            f"Command (exit={exit_code}): {cmd_short}\nOutput: {out_short}"
                        )

                elif etype == "turn.completed":
                    usage = event.get("usage", {})
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
    except Exception:
        pass

    return {
        "messages": messages,
        "num_commands": num_commands,
        "num_errors": num_errors,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def build_execution_summary(log_summary: dict, max_chars: int = 4000) -> str:
    """Build a concise execution summary from parsed log, truncated to max_chars."""
    msgs = log_summary.get("messages", [])
    if not msgs:
        return "(No execution log available)"

    summary_parts = []
    total_len = 0
    for msg in msgs:
        if total_len + len(msg) > max_chars:
            summary_parts.append("... (truncated)")
            break
        summary_parts.append(msg)
        total_len += len(msg)

    stats = (
        f"\n\nExecution stats: {log_summary['num_commands']} commands, "
        f"{log_summary['num_errors']} errors, "
        f"{log_summary['total_input_tokens']} input tokens, "
        f"{log_summary['total_output_tokens']} output tokens"
    )
    return "\n\n".join(summary_parts) + stats


# ---------------------------------------------------------------------------
# Confidence scoring via OpenAI API
# ---------------------------------------------------------------------------

CONFIDENCE_PROMPT = """You just completed a computational reproducibility task. You were asked to reproduce results from a scientific paper's codebase and provide answers as a Python dictionary. Please assess your confidence in the correctness of your submitted answers.

TASK PROMPT:
{task_prompt}

EXECUTION SUMMARY:
{execution_summary}

ANSWER PRODUCED:
{answer_json}

Consider the following when rating your confidence:

1. Did the code execute successfully without errors, or did you encounter dependency/environment issues that may have produced incorrect outputs?

2. How confident are you that each value in your answer dictionary is correct — did you extract values directly from program output, or did you have to estimate or interpret ambiguous results?

3. Did the code produce the expected type of output (figures, tables, numerical results) that the questions refer to?

4. Were there any discrepancies between what the code produced and what the questions asked about (e.g., missing output files, incomplete runs, partial results)?

5. Did you verify that your answer dictionary keys exactly match the requested field names?

6. For numerical answers, are you confident in the precision and format (e.g., rounding, scientific notation, percentage vs decimal)?

Please provide a confidence score from 0 to 100, where:

- 0-20: Very uncertain — code failed to run, outputs are missing, or answers are guesses

- 21-40: Low confidence — code ran with significant errors, or answers were inferred from partial output

- 41-60: Moderate confidence — code ran but some outputs are ambiguous or may not match expected format

- 61-80: Good confidence — code ran successfully, most answers extracted directly from output

- 81-100: Very confident — code ran cleanly, all answers directly extracted and verified against output

Respond with ONLY a number between 0 and 100. No explanation needed."""

# Versioned output keys. Bump CONFIDENCE_VERSION whenever the prompt changes.
# The previous keys (`confidence`, `confidence_details`, written by v1) are
# preserved untouched; new runs land under suffixed keys so historical scores
# remain available for diff/regression analysis.
CONFIDENCE_VERSION = "v2"


def normalize_model_for_openai(model: str) -> str:
    """Strip provider prefixes for OpenAI API calls.

    e.g. 'openai/gpt-5.4' -> 'gpt-5.4', 'gpt-5.4' -> 'gpt-5.4'.
    Non-OpenAI models (anthropic/, google/) are left as-is — the caller
    should use litellm or skip them.
    """
    for prefix in ("openai/",):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


async def score_confidence(
    client,
    model: str,
    task_prompt: str,
    execution_summary: str,
    answer_json: str,
) -> tuple[float, dict]:
    """Call the model for confidence self-assessment. Returns (score, details)."""
    prompt = CONFIDENCE_PROMPT.format(
        task_prompt=task_prompt,
        execution_summary=execution_summary,
        answer_json=answer_json,
    )

    api_model = normalize_model_for_openai(model)

    try:
        # gpt-5.3-codex doesn't work via the API; use codex CLI instead
        if "codex" in api_model.lower():
            import subprocess
            result = subprocess.run(
                ["codex", "exec", "--model", api_model, "--json",
                 "--skip-git-repo-check",
                 "--dangerously-bypass-approvals-and-sandbox",
                 prompt],
                capture_output=True, text=True, timeout=120,
            )
            # Parse JSONL output for the agent_message
            content = ""
            for line in result.stdout.strip().splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                    if (ev.get("type") == "item.completed"
                            and ev.get("item", {}).get("type") == "agent_message"):
                        content = ev["item"].get("text", "").strip()
                except json.JSONDecodeError:
                    continue
            if not content:
                raise ValueError(f"No agent_message in codex output: {result.stdout[:200]}")
        else:
            # Reasoning models (gpt-5) don't support temperature and need
            # more completion tokens for internal thinking.
            is_reasoning = api_model in ("gpt-5",)
            kwargs = {
                "model": api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 1024 if is_reasoning else 16,
            }
            if not is_reasoning:
                kwargs["temperature"] = 0
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()
        numbers = re.findall(r"\d+", content)

        if numbers:
            score = float(numbers[0]) / 100.0
            score = max(0.0, min(1.0, score))
        else:
            score = 0.5

        return score, {
            "model_response": content,
            "parsed_score": score,
            "model": model,
            "method": "post_hoc",
        }

    except Exception as e:
        return 0.5, {
            "model_response": str(e),
            "parsed_score": 0.5,
            "model": model,
            "method": "post_hoc",
            "error": True,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_raw_submissions(run_dir: Path) -> dict[str, dict]:
    """Load agent answers from _RAW_SUBMISSIONS.jsonl → {capsule_id: answer_dict}."""
    answers = {}
    for jsonl_path in run_dir.glob("*_RAW_SUBMISSIONS.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    answers.update(entry)
                except json.JSONDecodeError:
                    continue
    return answers


async def process_run(
    run_dir: Path,
    task_prompts: dict[str, str],
    client,
    semaphore: asyncio.Semaphore,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Process a single run directory: add reward + confidence to UPLOAD.json."""
    upload_files = list(run_dir.glob("*_UPLOAD.json"))
    if not upload_files:
        return {"run": run_dir.name, "status": "no_upload_json"}

    upload_path = upload_files[0]
    with open(upload_path) as f:
        data = json.load(f)

    config = data.get("config", {})
    agent_args = config.get("agent_args", {})
    model = agent_args.get("model_name", "gpt-5.4")

    raw_eval = data.get("raw_eval_results", {})
    if not raw_eval:
        return {"run": run_dir.name, "status": "no_eval_results"}

    confidence_key = f"confidence_{CONFIDENCE_VERSION}"
    details_key = f"confidence_details_{CONFIDENCE_VERSION}"

    # Skip if confidence at THIS version already computed (unless --force).
    # Older versions on the same task remain untouched.
    sample_task = next(iter(raw_eval.values()))
    if isinstance(sample_task, dict) and confidence_key in sample_task and not dry_run:
        details = sample_task.get(details_key, {})
        if not details.get("error") and not force:
            return {"run": run_dir.name, "status": f"already_has_{confidence_key}"}

    # Skip non-OpenAI models (anthropic/, google/) — they need different clients
    if model.startswith("anthropic/") or model.startswith("google/"):
        return {"run": run_dir.name, "status": f"skipped_non_openai ({model})"}

    # Load agent's actual answers from RAW_SUBMISSIONS
    agent_answers = load_raw_submissions(run_dir)

    scored = 0
    errors = 0

    async def score_task(task_id: str, task_eval: dict):
        nonlocal scored, errors

        # Compute reward
        reward = compute_reward(task_eval)
        task_eval["reward"] = reward

        # Get task prompt
        prompt = task_prompts.get(task_id, "(task prompt not found)")

        # Parse codex log for execution summary
        log_summary = {"messages": [], "num_commands": 0, "num_errors": 0,
                       "total_input_tokens": 0, "total_output_tokens": 0}
        for attempt in range(3):
            log_path = run_dir / task_id / f"codex_exec.log.{attempt}"
            if log_path.exists():
                log_summary = parse_codex_log(log_path)
                break

        execution_summary = build_execution_summary(log_summary)

        # Get the actual answer the agent produced
        agent_answer = agent_answers.get(task_id)
        if agent_answer is not None:
            answer_str = json.dumps(agent_answer, indent=2)
        else:
            answer_str = "(answer not found in RAW_SUBMISSIONS)"

        if dry_run:
            task_eval[confidence_key] = -1.0
            task_eval[details_key] = {"method": "dry_run", "version": CONFIDENCE_VERSION}
            scored += 1
            return

        async with semaphore:
            confidence, details = await score_confidence(
                client, model, prompt, execution_summary, answer_str,
            )
            details["num_commands"] = log_summary["num_commands"]
            details["num_errors"] = log_summary["num_errors"]
            details["version"] = CONFIDENCE_VERSION
            task_eval[confidence_key] = confidence
            task_eval[details_key] = details
            if details.get("error"):
                errors += 1
            else:
                scored += 1

    tasks = [
        score_task(tid, teval)
        for tid, teval in raw_eval.items()
        if isinstance(teval, dict)
    ]
    await asyncio.gather(*tasks)

    # Write back
    if not dry_run:
        with open(upload_path, "w") as f:
            json.dump(data, f, indent=2)

    return {
        "run": run_dir.name,
        "model": model,
        "tasks": len(raw_eval),
        "scored": scored,
        "errors": errors,
        "status": "done",
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc confidence scoring for CORE-bench results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(REPO_ROOT / "results" / "corebench_hard"),
    )
    parser.add_argument("--dry_run", action="store_true", help="Skip API calls")
    parser.add_argument(
        "--max_concurrent", type=int, default=10,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process runs matching this substring",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score even if confidence already exists",
    )
    parser.add_argument(
        "--manifest", action="append", default=None,
        help=(
            "Path to a core_test*.json manifest. Repeat to merge multiple "
            "splits (e.g. mainline + OOD). Defaults to core_test.json."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results dir not found: {results_dir}")

    manifests = [Path(m) for m in args.manifest] if args.manifest else None
    print("Loading task prompts...")
    task_prompts = load_task_prompts(manifests)
    print(f"  {len(task_prompts)} task prompts loaded")

    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    run_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (args.filter is None or args.filter in d.name)
    )
    print(f"Processing {len(run_dirs)} runs (max_concurrent={args.max_concurrent})...")

    for run_dir in run_dirs:
        print(f"\n{'='*60}")
        print(f"Run: {run_dir.name}")
        result = await process_run(
            run_dir, task_prompts, client, semaphore, args.dry_run, args.force,
        )
        print(f"  {result}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

import json
import os
import shutil
import subprocess
from typing import Any, Dict


def setup_codex_cli() -> str:
    """
    Install Codex CLI if missing and return the executable path.

    We rely on the npm-distributed CLI (`@openai/codex`), which installs a
    `codex` binary on the PATH. Installation is skipped if the binary already
    exists.
    """

    codex_path = shutil.which("codex")
    if codex_path:
        return codex_path

    # Basic tooling needed for the CLI.
    subprocess.run(
        "sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true",
        shell=True,
    )
    subprocess.run(["sudo", "apt-get", "update"], check=True)
    subprocess.run(
        ["sudo", "apt-get", "install", "-y", "git", "curl", "procps"],
        check=True,
    )

    # Install Node.js 20 from NodeSource (Ubuntu's apt nodejs is v12, too old for @openai/codex which requires >=16)
    subprocess.run(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -",
        shell=True,
        check=True,
    )
    subprocess.run(["sudo", "apt-get", "install", "-y", "nodejs"], check=True)

    subprocess.run(["sudo", "npm", "install", "-g", "@openai/codex"], check=True)
    codex_path = shutil.which("codex")
    if not codex_path:
        raise RuntimeError("Codex CLI install completed but `codex` not found on PATH.")
    return codex_path


def run(input: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
    """
    Run Codex CLI on a CORE-bench task.

    Args:
        input: Mapping of task_id -> task payload (expects exactly one task).
        kwargs: expects `model_name` (e.g., `gpt-4.1` or `o4-mini`).
                optional `reasoning_effort` ("low" | "medium" | "high").
                optional `max_threads` (int). When set, raises
                `agents.max_threads` on the Codex CLI via --config AND appends
                an explicit subagent instruction to the prompt. Codex only
                initiates subagents when prompted to, so both are required.
    Returns:
        Mapping of task_id -> answer dict or error string.
    """

    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    codex_bin = setup_codex_cli()

    model_name = kwargs["model_name"]
    reasoning_effort = kwargs.get("reasoning_effort")
    max_threads = kwargs.get("max_threads")
    if max_threads is not None:
        max_threads = int(max_threads)
    task_id, task_data = list(input.items())[0]
    prompt = task_data.get("prompt", "")

    notes = """
Important Instructions:
1. Your final answer MUST be a valid JSON object with the exact keys specified in the task.
2. The JSON keys must be the EXACT string from the task, including any prefix like "fig " - do not modify or omit any part of the key!
3. For numeric values, you MUST use the EXACT value from the output - DO NOT round or truncate. If the answer is a percentage, omit the % and report the float.
4. Write your final answer to ./answer.json at the TOP LEVEL of the current working directory (not inside environment/ or any subdirectory).
Return only the JSON object content in answer.json; do not include markdown fences.
"""

    # Subagent directive — only appended when max_threads is explicitly set.
    # Codex does NOT spawn subagents unless told to; the --config flag alone
    # just raises the ceiling.
    subagent_directive = (
        f"""
Parallel Subagents:
You have access to parallel subagents (up to {max_threads} concurrent). For this task, identify
independent subtasks that can be investigated in parallel — for example: reading README and
metadata, inspecting data files, exploring code modules, running the reproducible pipeline,
and verifying intermediate outputs. Spawn one subagent per independent subtask, wait for all
of them to return, and aggregate their findings before writing ./answer.json. Do NOT spawn
subagents for sequential steps that depend on each other's output.
"""
        if max_threads is not None
        else ""
    )
    notes = notes + subagent_directive

    initial_prompt = f"""You are solving a CORE-bench task. The task files are in ./environment/.
Your goal is to answer the task prompt and write the JSON answer to ./answer.json in the current working directory.

Task prompt:
{prompt}

{notes}
"""

    continue_prompt = f"""Previous attempt did not produce a valid ./answer.json.
Retry the CORE-bench task now and fix any issues.

Task prompt:
{prompt}

{notes}
"""

    answer_file = os.path.join(os.getcwd(), "answer.json")
    log_base = os.path.join(os.getcwd(), "codex_exec.log")
    max_attempts = 3

    env = os.environ.copy()
    # Codex supports CODEX_API_KEY specifically for exec; fallback to OPENAI_API_KEY.
    if "CODEX_API_KEY" not in env:
        env["CODEX_API_KEY"] = env.get("OPENAI_API_KEY", "")

    try:
        for attempt in range(max_attempts):
            log_path = f"{log_base}.{attempt}"
            prompt_to_use = initial_prompt if attempt == 0 else continue_prompt

            cmd = [
                codex_bin,
                "exec",
                "--json",          # JSONL event stream for logging.
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                model_name,
                prompt_to_use,
            ]

            if reasoning_effort:
                cmd.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])

            if max_threads is not None:
                cmd.extend(["--config", f"agents.max_threads={max_threads}"])

            with open(log_path, "w") as log_f:
                subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    timeout=3600,  # 60 minutes per attempt.
                    check=False,   # We handle failures via answer.json existence.
                )

            if os.path.exists(answer_file):
                try:
                    with open(answer_file, "r") as f:
                        answer = json.load(f)
                    return {task_id: answer}
                except json.JSONDecodeError as e:
                    # Invalid JSON; remove file and retry with feedback.
                    os.remove(answer_file)
                    continue_prompt = f"""Your last reply wrote invalid JSON ({e}). Please rewrite ./answer.json with valid JSON only, respecting the task keys and notes."""
                    continue

            # If no answer file, tweak retry prompt.
            continue_prompt = (
                "Previous run ended without creating ./answer.json. "
                "Try again and ensure you write the JSON file at the repo root."
            )

        return {task_id: "ERROR: answer.json not produced after retries"}

    except subprocess.TimeoutExpired:
        return {task_id: "ERROR: Timeout"}
    except Exception as e:
        return {task_id: f"ERROR: {str(e)}"}

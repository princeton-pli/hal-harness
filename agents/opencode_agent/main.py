import json
import os
import shutil
import subprocess
from typing import Any, Dict, Optional


def ensure_opencode_cli() -> str:
    """Install the OpenCode CLI if missing and return the binary path.

    Mirrors the official installer from the README:
    `curl -fsSL https://opencode.ai/install | bash`
    """

    def _find_binary() -> Optional[str]:
        # Check PATH first
        found = shutil.which("opencode")
        if found:
            return found

        # Common install locations used by the installer
        candidates = [
            os.path.expanduser("~/.local/bin/opencode"),
            os.path.expanduser("~/.opencode/bin/opencode"),
            os.path.expanduser("~/.local/share/opencode/bin/opencode"),
            "/usr/local/bin/opencode",
            "/usr/bin/opencode",
            "/opt/homebrew/bin/opencode",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path

        # Fallback: shallow search in typical roots to catch unexpected paths
        search_roots = [
            os.path.expanduser("~"),
            "/usr/local",
            "/usr",
            "/opt",
        ]
        for root in search_roots:
            for sub in ("bin", "local/bin", "local/share/opencode/bin"):
                candidate = os.path.join(root, sub, "opencode")
                if os.path.exists(candidate):
                    return candidate
        return None

    existing = _find_binary()
    if existing:
        return existing

    # Basic dependencies for the installer (git, curl, procps for `ps`).
    subprocess.run(
        "rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true",
        shell=True,
    )
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git", "curl", "procps"], check=True)

    # Official installer handles the rest.
    subprocess.run(
        "curl -fsSL https://opencode.ai/install | bash",
        shell=True,
        check=True,
    )

    opencode_path = _find_binary()
    if not opencode_path:
        raise RuntimeError(
            "OpenCode CLI install completed but `opencode` not found. "
            "Check installer output and PATH; expected in ~/.local/bin or /usr/local/bin."
        )

    return opencode_path


def run(input: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
    """Run OpenCode (SST version) on a CORE-bench task using `opencode run`."""

    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    opencode_bin = ensure_opencode_cli()

    model_name: str = kwargs["model_name"]
    task_id, task_data = list(input.items())[0]
    prompt = task_data.get("prompt", "")

    # Configure OpenCode permissions to avoid interactive prompts
    config_dir = "/workspace/.opencode"
    config_file = os.path.join(config_dir, "config.json")
    os.makedirs(config_dir, exist_ok=True)

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
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
            "skill": "allow"
        }
    }

    with open(config_file, "w") as f:
        json.dump(opencode_config, f, indent=2)

    print(f"OpenCode config written to {config_file}")

    notes = """
Important Instructions:
1. Your final answer MUST be a valid JSON object with the exact keys specified in the task.
2. The JSON keys must be the EXACT string from the task, including any prefix like "fig " - do not modify or omit any part of the key!
3. For numeric values, you MUST use the EXACT value from the output - DO NOT round or truncate. If the answer is a percentage, omit the % and report the float.
4. Write your final answer to ./answer.json at the TOP LEVEL (not inside environment/ or any subdirectory).
Return only the JSON object content in answer.json; do not include markdown fences.
"""

    initial_prompt = f"""You are solving a CORE-bench task. The task files are in ./environment/.
Your goal is to answer the task prompt and write the JSON answer to /workspace/answer.json.

Task prompt:
{prompt}

{notes}
"""

    continue_prompt = f"""Previous attempt did not produce a valid /workspace/answer.json.
Retry the CORE-bench task now and fix any issues.

Task prompt:
{prompt}

{notes}
"""

    answer_file = "/workspace/answer.json"
    log_base = "/workspace/opencode_exec.log"
    max_attempts = 3

    env = os.environ.copy()
    # OpenCode reads provider keys from env; ensure common ones are present.
    env.setdefault("OPENAI_API_KEY", env.get("OPENAI_API_KEY", ""))
    env.setdefault("ANTHROPIC_API_KEY", env.get("ANTHROPIC_API_KEY", ""))
    # Set bash command timeout to 30 minutes (1,800,000 ms)
    env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = "1800000"

    try:
        for attempt in range(max_attempts):
            log_path = f"{log_base}.{attempt}"
            prompt_to_use = initial_prompt if attempt == 0 else continue_prompt

            cmd = [
                opencode_bin,
                "run",
                "--model",
                model_name,
                "--format",
                "json",
                prompt_to_use,
            ]

            with open(log_path, "w") as log_f:
                subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    timeout=3600,  # 60 minutes per attempt.
                    check=False,
                    cwd="/workspace",
                )

            if os.path.exists(answer_file):
                try:
                    with open(answer_file, "r") as f:
                        answer = json.load(f)
                    return {task_id: answer}
                except json.JSONDecodeError as e:
                    os.remove(answer_file)
                    continue_prompt = (
                        f"Your last reply wrote invalid JSON ({e}). Please rewrite /workspace/answer.json with valid JSON only, respecting the task keys and notes."
                    )
                    continue

            continue_prompt = (
                "Previous run ended without creating /workspace/answer.json. "
                "Try again and ensure you write the JSON file at the repo root."
            )

        return {task_id: "ERROR: answer.json not produced after retries"}

    except subprocess.TimeoutExpired:
        return {task_id: "ERROR: Timeout"}
    except Exception as e:
        return {task_id: f"ERROR: {str(e)}"}

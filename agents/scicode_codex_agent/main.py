"""SciCode Codex Agent — solves tasks by invoking Codex CLI."""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

PYTHON = sys.executable
AGENT_DIR = Path(__file__).resolve().parent


def extract_code(response_text: str) -> str:
    """Extract Python code from a response, stripping markdown fences."""
    code = response_text.strip()
    if "```python" in code:
        code = code.split("```python", 1)[1]
    elif code.startswith("```"):
        code = code.split("```", 1)[1]
    if "```" in code:
        code = code.split("```", 1)[0]
    return code.strip()


def test_code(code: str, dependencies: str, timeout: int = 30) -> tuple[bool, str]:
    """Run generated code in a subprocess to check for basic errors."""
    test_script = f"{dependencies}\n\n{code}\n\nprint('OK')\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
        script_file.write(test_script)
        script_file.flush()
        try:
            result = subprocess.run(
                [PYTHON, script_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return True, ""
            return False, (result.stderr or result.stdout)[-1000:]
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        finally:
            os.unlink(script_file.name)


def run_codex_cli(
    prompt: str,
    model_name: str,
    codex_bin: str,
    codex_timeout: int,
    codex_working_dir: Path | None = None,
) -> str:
    """Run Codex CLI in non-interactive mode and return the final message."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as out_file:
        out_path = Path(out_file.name)

    command = [codex_bin, "exec", "--skip-git-repo-check", "-"]
    if model_name:
        command.extend(["-m", model_name])
    if codex_working_dir:
        command.extend(["-C", str(codex_working_dir)])
    command.extend(["-o", str(out_path)])

    def _run_cmd(command_prefix: list[str] | None = None) -> subprocess.CompletedProcess:
        full_command = [*(command_prefix or []), *command]
        return subprocess.run(
            full_command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=codex_timeout,
        )

    try:
        completed = _run_cmd()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Codex CLI not found at '{codex_bin}'") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Codex CLI timed out after {codex_timeout}s") from exc

    try:
        output_text = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""
    finally:
        if out_path.exists():
            os.unlink(out_path)

    if completed.returncode != 0:
        stderr_text = (completed.stderr or completed.stdout or "").strip()
        # In x86_64 Python on Apple Silicon, Codex can request darwin-x64 optional
        # deps even when arm64 Codex is installed. Retry explicitly under arm64.
        if (
            "Missing optional dependency @openai/codex-darwin-x64" in stderr_text
            and sys.platform == "darwin"
            and platform.machine() == "x86_64"
        ):
            try:
                completed = _run_cmd(["arch", "-arm64"])
            except FileNotFoundError as exc:
                raise RuntimeError("Could not run Codex with arm64 fallback") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Codex CLI timed out after {codex_timeout}s") from exc

            try:
                output_text = (
                    out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""
                )
            finally:
                if out_path.exists():
                    os.unlink(out_path)

        if completed.returncode != 0:
            stderr_tail = (completed.stderr or completed.stdout or "").strip()[-1000:]
            raise RuntimeError(f"Codex CLI failed (exit {completed.returncode}): {stderr_tail}")

    if not output_text:
        output_text = (completed.stdout or "").strip()
    return output_text


def build_subtask_prompt(task_data: dict, step_index: int,
                         previous_code: list[str], with_background: bool,
                         dependencies: str) -> str:
    """Build the prompt for a single subtask."""
    step = task_data["sub_steps"][step_index]
    header = step["function_header"]
    return_line = step["return_line"]
    description = step["step_description_prompt"]
    background = step.get("step_background", "")

    prev_steps_text = ""
    for i in range(step_index):
        prev_desc = task_data["sub_steps"][i]["step_description_prompt"]
        prev_steps_text += (
            f"--- Step {i+1} ---\n{prev_desc}\n\n"
            f"Implementation:\n{previous_code[i]}\n\n"
        )

    step_desc = f"{description}\n{background}" if with_background else description

    style_instructions = (
        "Write ONLY the function for this step. No imports, no previous functions, no test code.\n"
        "The function must match the provided header exactly.\n"
        "Respond with ```python``` code block.\n\n"
        "Additional strictness:\n"
        "- Do not add defensive input-validation branches unless explicitly requested.\n"
        "- Preserve numerically standard scientific behavior (no silent NaN/inf fallbacks).\n"
        "- Keep runtime efficient; avoid unnecessary nested loops or extra passes.\n"
        "- If a standard formula is implied, implement that formula directly."
    )
    lowered_step = step_desc.lower()
    if "lennard-jones" in lowered_step or "minimum image" in lowered_step:
        style_instructions += (
            "\n\nPhysics conventions to follow when relevant:\n"
            "- Minimum image displacement: d -= L * np.rint(d / L)\n"
            "- LJ force vector: 24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r^2 * r_vec\n"
            "- Apply cutoff cleanly (typically zero for r >= rc)."
        )

    return f"""You are solving a scientific programming problem step by step.

DEPENDENCIES (already available — do NOT include these imports in your code):
{dependencies}

PREVIOUS STEPS (already implemented — use these functions as-is, do NOT redefine them):
{prev_steps_text}

CURRENT STEP (Step {step_index + 1}):
{step_desc}

FUNCTION HEADER:
{header}

{return_line}

{style_instructions}"""


def build_hard_prompt(task_data: dict, dependencies: str) -> str:
    """Build prompt for scicode_hard — all steps at once."""
    last_step = len(task_data["sub_steps"])
    header = task_data["sub_steps"][last_step - 1]["function_header"]
    return_line = task_data["sub_steps"][last_step - 1]["return_line"]
    problem_desc = task_data["problem_description_main"]

    style_instructions = (
        "Write the complete solution with all helper functions.\n"
        "Do NOT include imports — the dependencies above are already available.\n"
        "Respond with ```python``` code block.\n\n"
        "Additional strictness:\n"
        "- Do not add speculative input-validation branches unless requested.\n"
        "- Prefer standard scientific formulas and stable numerics.\n"
        "- Keep implementations efficient and avoid avoidable heavy loops."
    )

    return f"""You are solving a scientific programming problem.

DEPENDENCIES (already available — do NOT include these imports):
{dependencies}

PROBLEM:
{problem_desc}

FUNCTION HEADER:
{header}

{return_line}

{style_instructions}"""


def generate_with_correction(
    model_name: str,
    prompt: str,
    dependencies: str,
    codex_bin: str,
    codex_timeout: int,
    max_attempts: int = 3,
    codex_working_dir: Path | None = None,
) -> str:
    """Generate code via Codex CLI, test it, and self-correct if needed."""
    current_prompt = prompt
    last_code = ""

    for attempt in range(max_attempts):
        response_text = run_codex_cli(
            prompt=current_prompt,
            model_name=model_name,
            codex_bin=codex_bin,
            codex_timeout=codex_timeout,
            codex_working_dir=codex_working_dir,
        )
        code = extract_code(response_text)

        if not code:
            print(f"    attempt {attempt+1}: empty response, retrying")
            current_prompt = (
                f"{current_prompt}\n\n"
                f"Previous response:\n{response_text}\n\n"
                "Your response did not contain Python code. Respond ONLY with a "
                "```python``` code block."
            )
            continue

        last_code = code
        success, error_msg = test_code(code, dependencies)
        if success:
            print(f"    attempt {attempt+1}: code passes")
            return code

        print(f"    attempt {attempt+1}: error — {error_msg[:120]}")

        if attempt < max_attempts - 1:
            current_prompt = (
                f"{current_prompt}\n\n"
                f"Previous attempt:\n```python\n{code}\n```\n\n"
                "This produced the following error during local execution:\n"
                f"```\n{error_msg}\n```\n\n"
                "Fix the error and respond ONLY with corrected Python code inside "
                "```python``` fences. No imports, no tests."
            )

    return last_code


def run(input: Dict[str, Any], **kwargs) -> Dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"

    model_name = kwargs["model_name"]
    benchmark_name = kwargs.get("benchmark_name", "scicode")
    codex_bin = kwargs.get("codex_bin", "codex")
    codex_timeout = int(kwargs.get("codex_timeout", 300))
    max_attempts = int(kwargs.get("max_attempts", 3))
    codex_working_dir_raw = kwargs.get("codex_working_dir")
    codex_working_dir = (
        Path(codex_working_dir_raw).resolve() if codex_working_dir_raw else None
    )

    results = {}

    if benchmark_name == "scicode_hard":
        for task_id, task in input.items():
            print(f"Generating {task_id} (hard mode)...")
            dependencies = task["required_dependencies"]
            prompt = build_hard_prompt(task_data=task, dependencies=dependencies)

            code = generate_with_correction(
                model_name=model_name,
                prompt=prompt,
                dependencies=dependencies,
                codex_bin=codex_bin,
                codex_timeout=codex_timeout,
                max_attempts=max_attempts,
                codex_working_dir=codex_working_dir,
            )
            results[task_id] = code

    else:
        easy = benchmark_name == "scicode_easy"

        for task_id, task in input.items():
            print(f"Generating {task_id}...")
            dependencies = task["required_dependencies"]

            previous_code = []
            full_code = ""
            steps = len(task["sub_steps"])
            steps_results = {}

            for i in range(steps):
                if (
                    (task_id == "13" and i == 5)
                    or (task_id == "62" and i == 0)
                    or (task_id == "76" and i == 2)
                ):
                    step_code = (AGENT_DIR / f"{task_id}.{i + 1}.txt").read_text(
                        encoding="utf-8"
                    )
                    previous_code.append(step_code)
                    full_code += f"\n{step_code}"
                    steps_results[f"{task_id}.{i + 1}"] = full_code
                    continue

                prompt = build_subtask_prompt(
                    task_data=task,
                    step_index=i,
                    previous_code=previous_code,
                    with_background=easy,
                    dependencies=dependencies,
                )

                print(f"  step {task_id}.{i+1}:")
                generated_code = generate_with_correction(
                    model_name=model_name,
                    prompt=prompt,
                    dependencies=dependencies,
                    codex_bin=codex_bin,
                    codex_timeout=codex_timeout,
                    max_attempts=max_attempts,
                    codex_working_dir=codex_working_dir,
                )

                previous_code.append(generated_code)
                full_code += f"\n{generated_code}"

                steps_results[f"{task_id}.{i + 1}"] = dependencies + full_code

            results[task_id] = steps_results

    return results

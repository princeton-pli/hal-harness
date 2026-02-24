"""
SciCode Codex Agent — uses the OpenAI API with iterative self-correction.

Unlike the zero-shot agent, this agent generates code, runs it locally to
check for errors, and feeds errors back to the model for correction.
This gives the LLM a chance to fix its own mistakes without needing
external tools like the Codex CLI.
"""

import os
import subprocess
import sys
import tempfile
from openai import OpenAI
from pathlib import Path
from typing import Any, Dict

PYTHON = sys.executable


def extract_code(response_text: str) -> str:
    """Extract Python code from an LLM response, stripping markdown fences."""
    code = response_text.strip()
    if "```python" in code:
        code = code.split("```python", 1)[1]
    if "```" in code:
        code = code.split("```", 1)[0]
    return code.strip()


def test_code(code: str, dependencies: str, timeout: int = 30) -> tuple[bool, str]:
    """Run generated code in a subprocess to check for syntax/import/runtime errors."""
    test_script = f"{dependencies}\n\n{code}\n\nprint('OK')\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()
        try:
            result = subprocess.run(
                [PYTHON, f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[-1000:]
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        finally:
            os.unlink(f.name)


def call_llm(client: OpenAI, model: str, messages: list[dict],
             reasoning_effort: str = None) -> str:
    """Call the OpenAI API and return the response text."""
    if "codex" in model:
        prompt = "\n\n".join(
            m["content"] for m in messages if m["role"] == "user"
        )
        kwargs = {"model": model, "input": prompt}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        # Use a separate unpatched client for the Responses API to avoid
        # weave monkey-patching issues that can cause hangs.
        raw_client = OpenAI(timeout=120.0)
        response = raw_client.responses.create(**kwargs)
        return response.output_text
    else:
        kwargs = {"model": model, "messages": messages}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


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

Write ONLY the function for this step. No imports, no previous functions, no test code.
The function must match the provided header exactly.
Respond with ```python``` code block."""


def build_hard_prompt(task_data: dict, dependencies: str) -> str:
    """Build prompt for scicode_hard — all steps at once."""
    last_step = len(task_data["sub_steps"])
    header = task_data["sub_steps"][last_step - 1]["function_header"]
    return_line = task_data["sub_steps"][last_step - 1]["return_line"]
    problem_desc = task_data["problem_description_main"]

    return f"""You are solving a scientific programming problem.

DEPENDENCIES (already available — do NOT include these imports):
{dependencies}

PROBLEM:
{problem_desc}

FUNCTION HEADER:
{header}

{return_line}

Write the complete solution with all helper functions.
Do NOT include imports — the dependencies above are already available.
Respond with ```python``` code block."""


def generate_with_correction(client: OpenAI, model: str, prompt: str,
                             dependencies: str, max_attempts: int = 3,
                             reasoning_effort: str = None) -> str:
    """Generate code, test it, and self-correct if there are errors."""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_attempts):
        response_text = call_llm(client, model, messages, reasoning_effort)
        code = extract_code(response_text)

        if not code:
            print(f"    attempt {attempt+1}: empty response, retrying")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Your response did not contain a Python code block. "
                           "Please respond with the function inside ```python``` fences.",
            })
            continue

        success, error_msg = test_code(code, dependencies)
        if success:
            print(f"    attempt {attempt+1}: code passes")
            return code

        print(f"    attempt {attempt+1}: error — {error_msg[:120]}")

        if attempt < max_attempts - 1:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": f"The code above produced this error when tested:\n\n"
                           f"```\n{error_msg}\n```\n\n"
                           f"Fix the error and respond with the corrected function "
                           f"inside ```python``` fences. Write ONLY the function, "
                           f"no imports, no test code.",
            })

    return code


def run(input: Dict[str, Any], **kwargs) -> Dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"

    client = OpenAI()
    model_name = kwargs["model_name"]
    reasoning_effort = kwargs.get("reasoning_effort")
    benchmark_name = kwargs.get("benchmark_name", "scicode")

    results = {}

    if benchmark_name == "scicode_hard":
        for task_id, task in input.items():
            print(f"Generating {task_id} (hard mode)...")
            dependencies = task["required_dependencies"]
            prompt = build_hard_prompt(task, dependencies)

            code = generate_with_correction(
                client, model_name, prompt, dependencies,
                reasoning_effort=reasoning_effort,
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
                    step_code = Path(f"{task_id}.{i + 1}.txt").read_text(
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
                    client, model_name, prompt, dependencies,
                    reasoning_effort=reasoning_effort,
                )

                previous_code.append(generated_code)
                full_code += f"\n{generated_code}"

                if easy:
                    steps_results[f"{task_id}.{i + 1}"] = full_code
                else:
                    steps_results[f"{task_id}.{i + 1}"] = dependencies + full_code

            results[task_id] = steps_results

    return results

import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from opencode_setup import (
    find_opencode_binary,
    install_opencode_cli,
    install_system_deps,
    setup_opencode_config,
)


# ============================================================
# OpenCode CLI Management
# ============================================================


def ensure_opencode_cli() -> str:
    """Install the OpenCode CLI if missing and return the binary path."""
    existing = find_opencode_binary()
    if existing:
        return existing

    install_system_deps()
    install_opencode_cli()

    opencode_path = find_opencode_binary()
    if not opencode_path:
        raise RuntimeError(
            "OpenCode CLI install completed but `opencode` not found. "
            "Check installer output and PATH."
        )
    return opencode_path


def _setup_opencode_config(cwd="/workspace"):
    """Write OpenCode config for permissive mode."""
    config_file = setup_opencode_config(cwd)
    print(f"OpenCode config written to {config_file}")


def _run_opencode_cli(opencode_bin, model_name, prompt, cwd="/workspace", timeout=3600):
    """Run OpenCode CLI once and return the log file path."""
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    env.setdefault("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = "1800000"

    log_path = os.path.join(cwd, "opencode_exec.log")
    cmd = [opencode_bin, "run", "--model", model_name, "--format", "json", prompt]

    with open(log_path, "w") as log_f:
        subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    return log_path


def _read_file(path, default=None):
    """Read file contents or return default."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
    except Exception:
        pass
    return default


def _remove_file(path):
    """Remove a file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ============================================================
# CORE-bench Handler
# ============================================================


def _run_corebench(task_id, task_data, model_name, kwargs):
    """Handle CORE-bench tasks (easy/medium/hard)."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    prompt = task_data.get("prompt", "")
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
    max_attempts = 3

    try:
        for attempt in range(max_attempts):
            prompt_to_use = initial_prompt if attempt == 0 else continue_prompt
            _run_opencode_cli(opencode_bin, model_name, prompt_to_use)

            if os.path.exists(answer_file):
                try:
                    with open(answer_file, "r") as f:
                        answer = json.load(f)
                    return {task_id: answer}
                except json.JSONDecodeError as e:
                    os.remove(answer_file)
                    continue_prompt = (
                        f"Your last reply wrote invalid JSON ({e}). Please rewrite "
                        f"/workspace/answer.json with valid JSON only, respecting the task keys and notes."
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


# ============================================================
# GAIA Handler
# ============================================================


def _run_gaia(task_id, task_data, model_name, kwargs):
    """Handle GAIA benchmark tasks."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    question = task_data.get("Question", "")

    initial_prompt = f"""You are answering a GAIA benchmark question. Any attached files are stored in the current directory.

Question:
{question}

Instructions:
- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

Write ONLY your final answer (nothing else) to /workspace/answer.txt
Do NOT include any explanation, just the answer itself.
"""

    answer_file = "/workspace/answer.txt"
    _remove_file(answer_file)
    max_attempts = 3

    try:
        for attempt in range(max_attempts):
            if attempt == 0:
                prompt_to_use = initial_prompt
            else:
                prompt_to_use = (
                    "Previous attempt did not produce /workspace/answer.txt. "
                    "Please answer the question and write ONLY the answer to /workspace/answer.txt.\n\n"
                    f"Question: {question}"
                )
            _run_opencode_cli(opencode_bin, model_name, prompt_to_use)

            answer = _read_file(answer_file)
            if answer is not None:
                return {
                    task_id: {
                        "answer": answer.strip(),
                        "metrics": {},
                    }
                }

        return {task_id: {"answer": "", "metrics": {}}}

    except subprocess.TimeoutExpired:
        return {task_id: {"answer": "ERROR: Timeout", "metrics": {}}}
    except Exception as e:
        return {task_id: {"answer": f"ERROR: {str(e)}", "metrics": {}}}


# ============================================================
# USACO Handler
# ============================================================


def _run_usaco(task_id, task_data, model_name, kwargs):
    """Handle USACO programming problems."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    description = task_data.get("description", "")

    prompt = f"""Please solve the following USACO competitive programming problem.
Write a complete Python 3 solution that reads from stdin and writes to stdout.
No outside libraries are allowed.

[BEGIN PROBLEM]
{description}
[END PROBLEM]

Write your complete Python 3 solution to /workspace/solution.py
The solution must:
- Read input from stdin
- Write output to stdout
- Be a single, complete, runnable Python file
- Not use any external libraries
"""

    solution_file = "/workspace/solution.py"
    _remove_file(solution_file)
    max_attempts = 3

    try:
        for attempt in range(max_attempts):
            if attempt == 0:
                prompt_to_use = prompt
            else:
                prompt_to_use = (
                    "Previous attempt did not produce /workspace/solution.py. "
                    "Please write a complete Python 3 solution to /workspace/solution.py.\n\n"
                    f"[BEGIN PROBLEM]\n{description}\n[END PROBLEM]"
                )
            _run_opencode_cli(opencode_bin, model_name, prompt_to_use)

            code = _read_file(solution_file)
            if code is not None:
                return {
                    task_id: {
                        "answer": code,
                        "metrics": {},
                    }
                }

        return {task_id: {"answer": "", "metrics": {}}}

    except subprocess.TimeoutExpired:
        return {task_id: {"answer": "ERROR: Timeout", "metrics": {}}}
    except Exception as e:
        return {task_id: {"answer": f"ERROR: {str(e)}", "metrics": {}}}


# ============================================================
# ScienceAgentBench Handler
# ============================================================


def _run_scienceagentbench(task_id, task_data, model_name, kwargs):
    """Handle ScienceAgentBench tasks."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    task_inst = task_data.get("task_inst", "")
    domain_knowledge = task_data.get("domain_knowledge", "")
    dataset_folder_tree = task_data.get("dataset_folder_tree", "")
    dataset_preview = task_data.get("dataset_preview", "")
    output_fname = task_data.get("output_fname", "output.txt")

    dataset_path = ""
    if dataset_folder_tree:
        first_line = dataset_folder_tree.split("\n")[0]
        if len(first_line) > 4:
            dataset_path = "benchmark/datasets/" + first_line[4:]

    prompt = f"""Write a Python 3 script to solve the following scientific computing task.

Task:
{task_inst}
"""
    if domain_knowledge:
        prompt += f"\n{domain_knowledge}\n"

    prompt += f"""
You can access the dataset at `{dataset_path}`. Here is the directory structure:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}

Write your complete Python 3 script to /workspace/solution.py
The script should produce the output file: {output_fname}
"""

    solution_file = "/workspace/solution.py"
    _remove_file(solution_file)
    max_attempts = 3

    try:
        for attempt in range(max_attempts):
            if attempt == 0:
                prompt_to_use = prompt
            else:
                prompt_to_use = (
                    "Previous attempt did not produce /workspace/solution.py. "
                    "Please write a complete Python 3 script to /workspace/solution.py.\n\n"
                    f"Task: {task_inst}"
                )
            _run_opencode_cli(opencode_bin, model_name, prompt_to_use)

            code = _read_file(solution_file)
            if code is not None:
                return {
                    task_id: {
                        "history": [{"role": "assistant", "content": f"```python{code}```"}],
                        "cost": 0.0,
                    }
                }

        return {task_id: {"history": [{"role": "assistant", "content": ""}], "cost": 0.0}}

    except subprocess.TimeoutExpired:
        return {task_id: {"history": [{"role": "assistant", "content": "ERROR: Timeout"}], "cost": 0.0}}
    except Exception as e:
        return {task_id: {"history": [{"role": "assistant", "content": f"ERROR: {str(e)}"}], "cost": 0.0}}


# ============================================================
# SWE-bench Handler
# ============================================================


def _run_swebench(task_id, task_data, model_name, kwargs):
    """Handle SWE-bench tasks (verified and verified_mini)."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    repo = task_data.get("repo", "")
    base_commit = task_data.get("base_commit", "")
    problem_statement = task_data.get("problem_statement", "")
    repo_name = repo.split("/")[-1] if repo else "repo"
    repo_dir = f"/workspace/{repo_name}"

    # Clone and reset the repository
    try:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        result = subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git"],
            capture_output=True, text=True, cwd="/workspace", timeout=300,
        )
        if result.returncode != 0:
            return {task_id: f"ERROR: Failed to clone: {result.stderr}"}

        result = subprocess.run(
            ["git", "reset", "--hard", base_commit],
            capture_output=True, text=True, cwd=repo_dir, timeout=60,
        )
        if result.returncode != 0:
            return {task_id: f"ERROR: Failed to reset: {result.stderr}"}
    except Exception as e:
        return {task_id: f"ERROR: Repo setup failed: {str(e)}"}

    prompt = f"""You need to fix a software issue in the repository at /workspace/{repo_name}.

Problem Statement:
{problem_statement}

Instructions:
1. Navigate to /workspace/{repo_name} and understand the codebase
2. Identify the root cause of the issue
3. Make the necessary code changes to fix it
4. Make changes directly to the source files

Do NOT create a patch file. Just edit the source files directly to fix the issue.
"""

    try:
        _run_opencode_cli(opencode_bin, model_name, prompt, cwd="/workspace")

        # Generate patch from git diff
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True, cwd=repo_dir, timeout=60,
        )
        model_patch = result.stdout if result.returncode == 0 else ""

        if not model_patch:
            # Also check staged changes
            result = subprocess.run(
                ["git", "diff", "--cached"],
                capture_output=True, text=True, cwd=repo_dir, timeout=60,
            )
            model_patch = result.stdout if result.returncode == 0 else ""

        return {task_id: model_patch}

    except subprocess.TimeoutExpired:
        return {task_id: "ERROR: Timeout"}
    except Exception as e:
        return {task_id: f"ERROR: {str(e)}"}


# ============================================================
# SciCode Handler
# ============================================================


# Special-case hardcoded code for known problematic steps (matches generalist agent)
_SCICODE_SPECIAL_CASES = {
    ("13", 5): '''\
    class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out):
        self.n_grid = n_grid
        self.n_vars = 7
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta

        self.x      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[:,None,None]
        self.y      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,:,None]
        self.z      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,None,:]
        self.r      = np.sqrt(self.x**2+self.y**2+self.z**2)

        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))

        self.t = 0.0
''',
    ("62", 0): '''
class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

    def print_all(self):
        print(self.length)
        print(self.basis_size)
        for key, matrix in self.operator_dict.items():
            if isinstance(matrix, np.ndarray):
                print(f"{key}:\\n{matrix}\\n")
            else:
                print(f"{key}:\\n{matrix.toarray()}\\n")

class EnlargedBlock:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

    def print_all(self):
        print(self.length)
        print(self.basis_size)
        for key, matrix in self.operator_dict.items():
            if isinstance(matrix, np.ndarray):
                print(f"{key}:\\n{matrix}\\n")
            else:
                print(f"{key}:\\n{matrix.toarray()}\\n")
''',
    ("76", 2): """
def generate_dna(N: int, PWM: dict) -> tuple:
    '''
    Input:
    N (int): Length of the resultant DNA sequence.
    PWM matrix with keys 'A', 'C', 'G', 'T'

    Output:
    tuple: Insertion location (int), DNA sequence (str), DNA reverse complement (str)
    '''
    p = random.randint(0, N-1)

    nucleotide = "ACGT"
    uni_weights = [0.25,0.25,0.25,0.25] #uniform distribution
    dna_string = ''.join(random.choices(nucleotide, uni_weights, k=N))

    spike_mat = load_motif_from_df(PWM)
    spiked_seq = ''.join(random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0]
                         for i in range(len(PWM['A'])))

    complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    reversed_seq = dna_string[::-1]
    reverse_complement = ''.join(complement[nuc] for nuc in reversed_seq if nuc in complement)

    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[:N-p] + spiked_seq + reverse_complement[N-p:]

    return p, new_seq, new_seq_rc
""",
}


def _run_scicode(task_id, task_data, model_name, kwargs):
    """Handle SciCode benchmark tasks."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    benchmark_name = kwargs.get("benchmark_name", "scicode")
    easy = benchmark_name == "scicode_easy"

    prompt_template = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
1. Write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Write ONLY the Python code to /workspace/step_code.py (no markdown fences in the file).
"""

    def process_problem_code(prob_data, num_steps):
        header_docstring = prob_data["sub_steps"][num_steps - 1]["function_header"]
        return_str = prob_data["sub_steps"][num_steps - 1]["return_line"]
        return f"{header_docstring}\n\n{return_str}"

    def process_problem_steps(with_background, previous_llm_code, problem_data, num_steps):
        output_lines = []
        for i in range(num_steps - 1):
            step_desc = problem_data["sub_steps"][i]["step_description_prompt"]
            if with_background:
                step_desc += "\n" + problem_data["sub_steps"][i]["step_background"]
            output_lines.append(step_desc)
            output_lines.append(previous_llm_code[i])
            output_lines.append("------")

        next_step = []
        step_desc = problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
        if with_background:
            step_desc += "\n" + problem_data["sub_steps"][num_steps - 1]["step_background"]
        next_step.append(step_desc)
        next_step.append(process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1]) if output_lines else ""
        next_step_str = "\n\n".join(next_step)
        return output_str, next_step_str

    def generate_prompt_with_steps(with_background, previous_llm_code, prob_data, num_steps, template):
        problem_steps_str, next_step_str = process_problem_steps(
            with_background, previous_llm_code, prob_data, num_steps
        )
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f"{dependencies}\n"

    previous_llm_code = []
    full_code = ""
    steps = len(task_data["sub_steps"])
    steps_results = {}
    step_code_file = "/workspace/step_code.py"

    print(f"Generating {task_id}...")

    try:
        for i in range(steps):
            # Check for special cases
            special_key = (task_id, i)
            if special_key in _SCICODE_SPECIAL_CASES:
                step_code = _SCICODE_SPECIAL_CASES[special_key]
                previous_llm_code.append(step_code)
                full_code += f"\n{step_code}"
                steps_results[f"{task_id}.{i + 1}"] = full_code
                continue

            prompt, dependencies = generate_prompt_with_steps(
                with_background=easy,
                previous_llm_code=previous_llm_code,
                prob_data=task_data,
                num_steps=i + 1,
                template=prompt_template,
            )

            _remove_file(step_code_file)
            _run_opencode_cli(opencode_bin, model_name, prompt)

            generated_code = _read_file(step_code_file, "")

            if not generated_code.strip():
                # Fallback: try to extract from log
                log_content = _read_file("/workspace/opencode_exec.log", "")
                if "```python" in log_content:
                    generated_code = log_content.split("```python")[-1].split("```")[0].strip()

            generated_code = generated_code.replace("```python", "").replace("```", "").strip()

            previous_llm_code.append(generated_code)
            full_code += f"\n{generated_code}"

            if easy:
                steps_results[f"{task_id}.{i + 1}"] = full_code
            else:
                steps_results[f"{task_id}.{i + 1}"] = dependencies + full_code

        return {task_id: steps_results}

    except subprocess.TimeoutExpired:
        return {task_id: "ERROR: Timeout"}
    except Exception as e:
        return {task_id: f"ERROR: {str(e)}"}


# ============================================================
# AssistantBench Handler
# ============================================================


def _run_assistantbench(task_id, task_data, model_name, kwargs):
    """Handle AssistantBench tasks."""
    opencode_bin = ensure_opencode_cli()
    _setup_opencode_config()

    task_question = task_data.get("task", "")

    prompt = f"""Provide a concise and accurate answer to the question below without any additional context in the format suggested by the prompt. Do not include any justification or any additional unnecessary text. Your answer does not need to be a full sentence. If you are unsure what the final answer is, generate an empty string. The answer should either be: a number, a string, a list of strings, or a list of jsons. The answer should be parsed with the python method: json.loads(input_str). If no answer is found, generate an empty string. If the prompt includes a specified answer format, respect that format.

[BEGIN QUESTION]
{task_question}
[END QUESTION]

Write ONLY your final answer to /workspace/answer.txt
The content should be directly parseable by json.loads(). For strings, include the quotes.
"""

    answer_file = "/workspace/answer.txt"
    _remove_file(answer_file)
    max_attempts = 3

    try:
        for attempt in range(max_attempts):
            if attempt == 0:
                prompt_to_use = prompt
            else:
                prompt_to_use = (
                    "Previous attempt did not produce /workspace/answer.txt. "
                    "Please answer the question and write ONLY the answer to /workspace/answer.txt.\n\n"
                    f"Question: {task_question}"
                )
            _run_opencode_cli(opencode_bin, model_name, prompt_to_use)

            answer = _read_file(answer_file)
            if answer is not None:
                return {
                    task_id: {
                        "answer": answer.strip(),
                        "metrics": {},
                    }
                }

        return {task_id: {"answer": "", "metrics": {}}}

    except subprocess.TimeoutExpired:
        return {task_id: {"answer": "ERROR: Timeout", "metrics": {}}}
    except Exception as e:
        return {task_id: {"answer": f"ERROR: {str(e)}", "metrics": {}}}


# ============================================================
# Tau-bench Handler (uses litellm directly, not OpenCode CLI)
# ============================================================


def _make_tool(name, desc, properties, required=None):
    """Create an OpenAI function calling tool definition."""
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }
    if required:
        tool["function"]["parameters"]["required"] = required
    return tool


# Common tools shared by both airline and retail domains
_RESPOND_TOOL = _make_tool(
    "respond",
    "Send a message to the customer. Use this when you want to communicate with the customer.",
    {"content": {"type": "string", "description": "The message to send to the customer."}},
    ["content"],
)

_THINK_TOOL = _make_tool(
    "think",
    "Think about something. Does not obtain new info or change the database. Use for complex reasoning.",
    {"thought": {"type": "string", "description": "Your thought process."}},
    ["thought"],
)

_TRANSFER_TOOL = _make_tool(
    "transfer_to_human_agents",
    "Transfer to a human agent. Only use if user explicitly asks or issue cannot be resolved with available tools.",
    {"summary": {"type": "string", "description": "Summary of the user's issue."}},
    ["summary"],
)

_CALCULATE_TOOL = _make_tool(
    "calculate",
    "Calculate a math expression. Supports +, -, *, /, (, ). Returns result rounded to 2 decimals.",
    {"expression": {"type": "string", "description": "The math expression to evaluate."}},
    ["expression"],
)


AIRLINE_TOOLS = [
    _RESPOND_TOOL, _THINK_TOOL, _TRANSFER_TOOL, _CALCULATE_TOOL,
    _make_tool(
        "book_reservation", "Book a flight reservation.",
        {
            "user_id": {"type": "string", "description": "User ID, e.g. 'sara_doe_496'."},
            "origin": {"type": "string", "description": "Origin IATA code, e.g. 'SFO'."},
            "destination": {"type": "string", "description": "Destination IATA code, e.g. 'JFK'."},
            "flight_type": {"type": "string", "enum": ["one_way", "round_trip"], "description": "Trip type."},
            "cabin": {"type": "string", "enum": ["basic_economy", "economy", "business"], "description": "Cabin class."},
            "flights": {"type": "array", "items": {"type": "object"}, "description": "Flight segments: [{flight_number, date}, ...]."},
            "passengers": {"type": "array", "items": {"type": "object"}, "description": "Passengers: [{first_name, last_name, dob}, ...]."},
            "payment_methods": {"type": "array", "items": {"type": "object"}, "description": "Payments: [{payment_id, amount}, ...]."},
            "total_baggages": {"type": "integer", "description": "Total baggage count."},
            "nonfree_baggages": {"type": "integer", "description": "Non-free baggage count."},
            "insurance": {"type": "string", "enum": ["yes", "no"], "description": "Travel insurance."},
        },
        ["user_id", "origin", "destination", "flight_type", "cabin", "flights",
         "passengers", "payment_methods", "total_baggages", "nonfree_baggages", "insurance"],
    ),
    _make_tool(
        "cancel_reservation", "Cancel a reservation.",
        {"reservation_id": {"type": "string", "description": "Reservation ID, e.g. 'ZFA04Y'."}},
        ["reservation_id"],
    ),
    _make_tool(
        "get_reservation_details", "Get reservation details.",
        {"reservation_id": {"type": "string", "description": "Reservation ID, e.g. '8JX2WO'."}},
        ["reservation_id"],
    ),
    _make_tool(
        "get_user_details", "Get user details including reservations.",
        {"user_id": {"type": "string", "description": "User ID, e.g. 'sara_doe_496'."}},
        ["user_id"],
    ),
    _make_tool(
        "list_all_airports", "List all airports and their cities.",
        {},
    ),
    _make_tool(
        "search_direct_flight", "Search direct flights between two cities on a date.",
        {
            "origin": {"type": "string", "description": "Origin IATA code."},
            "destination": {"type": "string", "description": "Destination IATA code."},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
        },
        ["origin", "destination", "date"],
    ),
    _make_tool(
        "search_onestop_flight", "Search one-stop flights between two cities on a date.",
        {
            "origin": {"type": "string", "description": "Origin IATA code."},
            "destination": {"type": "string", "description": "Destination IATA code."},
            "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
        },
        ["origin", "destination", "date"],
    ),
    _make_tool(
        "send_certificate", "Send a travel certificate to a user.",
        {
            "user_id": {"type": "string", "description": "User ID."},
            "amount": {"type": "number", "description": "Certificate amount."},
        },
        ["user_id", "amount"],
    ),
    _make_tool(
        "update_reservation_baggages", "Update baggage info for a reservation.",
        {
            "reservation_id": {"type": "string", "description": "Reservation ID."},
            "total_baggages": {"type": "integer", "description": "Updated total baggage count."},
            "nonfree_baggages": {"type": "integer", "description": "Updated non-free baggage count."},
            "payment_id": {"type": "string", "description": "Payment ID for extra charges."},
        },
        ["reservation_id", "total_baggages", "nonfree_baggages", "payment_id"],
    ),
    _make_tool(
        "update_reservation_flights",
        "Update flight info for a reservation. Include ALL flight segments even if unchanged.",
        {
            "reservation_id": {"type": "string", "description": "Reservation ID."},
            "cabin": {"type": "string", "enum": ["basic_economy", "economy", "business"], "description": "Cabin class."},
            "flights": {"type": "array", "items": {"type": "object"},
                        "description": "ALL flight segments (even unchanged): [{flight_number, date}, ...]."},
            "payment_id": {"type": "string", "description": "Payment ID for price difference."},
        },
        ["reservation_id", "cabin", "flights", "payment_id"],
    ),
    _make_tool(
        "update_reservation_passengers", "Update passenger info for a reservation.",
        {
            "reservation_id": {"type": "string", "description": "Reservation ID."},
            "passengers": {"type": "array", "items": {"type": "object"},
                           "description": "Updated passengers: [{first_name, last_name, dob}, ...]."},
        },
        ["reservation_id", "passengers"],
    ),
]


RETAIL_TOOLS = [
    _RESPOND_TOOL, _THINK_TOOL, _TRANSFER_TOOL, _CALCULATE_TOOL,
    _make_tool(
        "get_user_details", "Get user details including orders.",
        {"user_id": {"type": "string", "description": "User ID."}},
        ["user_id"],
    ),
    _make_tool(
        "get_order_details", "Get order details.",
        {"order_id": {"type": "string", "description": "Order ID, e.g. '#W0000000'."}},
        ["order_id"],
    ),
    _make_tool(
        "get_product_details", "Get product details by product ID (not item ID).",
        {"product_id": {"type": "string", "description": "Product ID, e.g. '6086499569'."}},
        ["product_id"],
    ),
    _make_tool(
        "list_all_product_types", "List all product types and their product IDs.",
        {},
    ),
    _make_tool(
        "find_user_id_by_email", "Find user ID by email address.",
        {"email": {"type": "string", "description": "User's email address."}},
        ["email"],
    ),
    _make_tool(
        "find_user_id_by_name_zip", "Find user ID by first name, last name, and zip code.",
        {
            "first_name": {"type": "string", "description": "User's first name."},
            "last_name": {"type": "string", "description": "User's last name."},
            "zip": {"type": "string", "description": "User's zip code."},
        },
        ["first_name", "last_name", "zip"],
    ),
    _make_tool(
        "modify_pending_order_items",
        "Modify items in a pending order. Can only be called once per order.",
        {
            "order_id": {"type": "string", "description": "Order ID."},
            "item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs to replace."},
            "new_item_ids": {"type": "array", "items": {"type": "string"},
                             "description": "New item IDs (same product type)."},
            "payment_method_id": {"type": "string", "description": "Payment method for price difference."},
        },
        ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    ),
    _make_tool(
        "modify_pending_order_address", "Modify shipping address for a pending order.",
        {
            "order_id": {"type": "string", "description": "Order ID."},
            "address1": {"type": "string", "description": "Street address line 1."},
            "address2": {"type": "string", "description": "Street address line 2."},
            "city": {"type": "string", "description": "City."},
            "state": {"type": "string", "description": "State."},
            "country": {"type": "string", "description": "Country."},
            "zip": {"type": "string", "description": "Zip code."},
        },
        ["order_id", "address1", "address2", "city", "state", "country", "zip"],
    ),
    _make_tool(
        "modify_pending_order_payment", "Change payment method for a pending order.",
        {
            "order_id": {"type": "string", "description": "Order ID."},
            "payment_method_id": {"type": "string", "description": "New payment method ID."},
        },
        ["order_id", "payment_method_id"],
    ),
    _make_tool(
        "cancel_pending_order", "Cancel a pending order.",
        {
            "order_id": {"type": "string", "description": "Order ID."},
            "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"],
                       "description": "Cancellation reason."},
        },
        ["order_id", "reason"],
    ),
    _make_tool(
        "return_delivered_order_items", "Return items from a delivered order.",
        {
            "order_id": {"type": "string", "description": "Order ID."},
            "item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs to return."},
            "payment_method_id": {"type": "string",
                                  "description": "Refund payment method (original method or gift card)."},
        },
        ["order_id", "item_ids", "payment_method_id"],
    ),
    _make_tool(
        "exchange_delivered_order_items", "Exchange items from a delivered order.",
        {
            "order_id": {"type": "string", "description": "Order ID."},
            "item_ids": {"type": "array", "items": {"type": "string"}, "description": "Item IDs to exchange."},
            "new_item_ids": {"type": "array", "items": {"type": "string"},
                             "description": "New item IDs (same product type)."},
            "payment_method_id": {"type": "string", "description": "Payment method for price difference."},
        },
        ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    ),
    _make_tool(
        "modify_user_address", "Modify user's default address.",
        {
            "user_id": {"type": "string", "description": "User ID."},
            "address1": {"type": "string", "description": "Street address line 1."},
            "address2": {"type": "string", "description": "Street address line 2."},
            "city": {"type": "string", "description": "City."},
            "state": {"type": "string", "description": "State."},
            "country": {"type": "string", "description": "Country."},
            "zip": {"type": "string", "description": "Zip code."},
        },
        ["user_id", "address1", "address2", "city", "state", "country", "zip"],
    ),
]


def _run_taubench(task_id, task_data, model_name, kwargs):
    """Handle tau-bench tasks using litellm for LLM calls.

    Tau-bench requires interactive tool calling which is not suited for the
    OpenCode CLI one-shot approach.  Instead we use litellm directly with
    OpenAI-compatible function calling in a conversation loop.
    """
    try:
        from tau_bench.envs import get_env
        from tau_bench.types import Action
        import litellm
    except ImportError as e:
        return {task_id: f"ERROR: Missing dependency: {e}. Install tau-bench and litellm."}

    # Set up environment
    env = get_env(
        task_data["env"],
        task_data["user_strategy"],
        task_data["user_model"],
        task_data["task_split"],
        task_data["user_provider"],
        task_data["task_index"],
    )

    obs = env.reset(task_data["task_index"])
    wiki = env.wiki

    domain = task_data["env"]
    tools = AIRLINE_TOOLS if domain == "airline" else RETAIL_TOOLS
    domain_desc = "an airline" if domain == "airline" else "a retail company"

    system_prompt = f"""You are a customer service agent for {domain_desc}. Follow the policies and guidelines below carefully.

{wiki}

Use the available tools to help the customer. When you want to respond to the customer, use the 'respond' tool.
Always verify the customer's identity before making any changes.
Think step by step before taking actions. Use the 'think' tool for complex reasoning."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs.observation},
    ]

    litellm.drop_params = True
    max_steps = 200

    try:
        for step in range(max_steps):
            try:
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"LLM call failed at step {step}: {e}")
                break

            choice = response.choices[0]
            assistant_msg = choice.message

            if assistant_msg.tool_calls:
                # Convert to serialisable dict for messages
                messages.append(assistant_msg.model_dump())
                done = False
                for tc in assistant_msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    action = Action(name=fn_name, kwargs=fn_args)
                    observation = env.step(action)

                    messages.append({
                        "role": "tool",
                        "content": observation.observation,
                        "tool_call_id": tc.id,
                    })

                    if observation.done:
                        done = True
                        break

                if done:
                    break
            else:
                # Text response without tool calls - treat as respond action
                content = assistant_msg.content or ""
                messages.append({"role": "assistant", "content": content})
                action = Action(name="respond", kwargs={"content": content})
                observation = env.step(action)

                if observation.done:
                    break

                messages.append({"role": "user", "content": observation.observation})

        return {
            task_id: {
                "reward": env.reward,
                "taken_actions": [a.model_dump() for a in env.actions],
                "task": env.task.model_dump(),
            }
        }

    except Exception as e:
        return {task_id: f"ERROR: {str(e)}"}


# ============================================================
# Main Entry Point
# ============================================================


def run(input: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
    """Run OpenCode agent on various benchmarks.

    For most benchmarks the OpenCode CLI is used as a one-shot coding agent.
    For tau-bench (which requires interactive tool calling) litellm is used
    directly with function calling.
    """
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    benchmark_name = kwargs.get("benchmark_name", "")
    model_name: str = kwargs["model_name"]
    task_id, task_data = list(input.items())[0]

    if benchmark_name.startswith("corebench"):
        return _run_corebench(task_id, task_data, model_name, kwargs)
    elif benchmark_name == "gaia":
        return _run_gaia(task_id, task_data, model_name, kwargs)
    elif benchmark_name == "usaco":
        return _run_usaco(task_id, task_data, model_name, kwargs)
    elif benchmark_name == "scienceagentbench":
        return _run_scienceagentbench(task_id, task_data, model_name, kwargs)
    elif benchmark_name in ("swebench_verified", "swebench_verified_mini"):
        return _run_swebench(task_id, task_data, model_name, kwargs)
    elif benchmark_name.startswith("taubench"):
        return _run_taubench(task_id, task_data, model_name, kwargs)
    elif benchmark_name in ("scicode", "scicode_easy", "scicode_hard"):
        return _run_scicode(task_id, task_data, model_name, kwargs)
    elif benchmark_name == "assistantbench":
        return _run_assistantbench(task_id, task_data, model_name, kwargs)
    else:
        return {task_id: f"ERROR: Unsupported benchmark: {benchmark_name}"}

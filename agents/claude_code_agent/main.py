import os
import subprocess
import json
from typing import Dict, Any


def setup_claude_code():
    """Install git, curl, and Claude Code if not present."""

    # Install dependencies
    subprocess.run(
        "sudo rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true",
        shell=True
    )
    subprocess.run(["sudo", "apt-get", "update"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", "git", "curl", "procps", "r-base"], check=True)

    # Install Claude Code if not present
    claude_path = os.path.expanduser("~/.local/bin/claude")
    if not os.path.exists(claude_path):
        subprocess.run(
            "curl -fsSL https://claude.ai/install.sh | bash",
            shell=True,
            check=True
        )

    return claude_path


def extract_session_id(log_file):
    """Extract session_id from the first JSONL line in the log."""
    try:
        with open(log_file, 'r') as f:
            first_line = f.readline().strip()
        if first_line:
            entry = json.loads(first_line)
            return entry.get('session_id')
    except:
        pass
    return None


def run(input: Dict[str, Dict], **kwargs) -> Dict[str, Any]:
    """
    Run Claude Code agent on CORE-bench tasks.

    Args:
        input: Dictionary mapping task IDs to task data
        **kwargs: Additional arguments including:
            - model_name: The model to use (e.g., 'claude-sonnet-4-5')

    Returns:
        Dictionary mapping task IDs to solutions
    """
    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'

    claude_path = setup_claude_code()

    model_name = kwargs['model_name']
    if '/' in model_name:
        model_name = model_name.split('/')[-1]

    task_id, task_data = list(input.items())[0]
    prompt = task_data.get('prompt', '')

    initial_prompt = f"""You are solving a CORE-bench task. Your goal is to answer questions about scientific code output.

The task files are available in the ./environment/ directory.

{prompt}
"""

    continue_prompt = f"""You were solving a CORE-bench task but didn't finish.
The task was:

{prompt}

Please continue and write the answer to ./answer.json now.
"""

    notes = """
Important Instructions:
1. Your final answer MUST be a valid JSON object with the exact keys specified in the task.
2. The JSON keys must be the EXACT string from the task, including any prefix like "fig " - do not modify or omit any part of the key!
3. For numeric values, you MUST use the EXACT value from the output - DO NOT round or truncate. If the answer is a percentage, omit the % and report the float.
4. Write your final answer to ./answer.json at the TOP LEVEL (not inside environment/ or any subdirectory).
"""

    initial_prompt += notes
    continue_prompt += notes

    env = os.environ.copy()
    env['IS_SANDBOX'] = '1'
    env['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY', '')
    # Support agent_args from Prefect (passed as kwargs via HAL_AGENT_ARG_*)
    max_thinking_tokens = kwargs.get('max_thinking_tokens')
    if max_thinking_tokens:
        env['MAX_THINKING_TOKENS'] = str(max_thinking_tokens)

    # Use cwd for answer.json and logs (Azure Batch task working dir)
    log_file_base = os.path.join(os.getcwd(), 'claude_code.log')
    answer_file = os.path.join(os.getcwd(), 'answer.json')
    max_attempts = 3
    session_id = None
    next_warning = ""

    try:
        for attempt in range(max_attempts):
            # Use numbered log file for each attempt
            log_file = f'{log_file_base}.{attempt}'

            # Build command
            if attempt == 0:
                # First attempt - fresh start
                cmd = [
                    claude_path,
                    '--output-format', 'stream-json',
                    '--verbose',
                    '--dangerously-skip-permissions',
                    '--model', model_name,
                    '-p', initial_prompt
                ]
            else:
                # Retry - resume the session
                cmd = [
                    claude_path,
                    '--output-format', 'stream-json',
                    '--verbose',
                    '--dangerously-skip-permissions',
                    '--model', model_name,
                    '--resume', session_id,
                    '-p', continue_prompt
                ]

            print(f"Attempt {attempt + 1}/{max_attempts}")

            # Run Claude Code
            with open(log_file, 'w') as log_f:
                subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=3600  # 60 min per attempt
                )

            # Check if answer.json exists
            if os.path.exists(answer_file):
                with open(answer_file, 'r') as f:
                    try:
                        answer = json.load(f)
                        print(f"Answer: {answer}")
                        return {task_id: answer}
                    except json.JSONDecodeError:
                        pass

            # Extract session_id for resume
            if session_id is None:
                session_id = extract_session_id(log_file)
                if session_id is None:
                    break  # Can't resume without session_id

            next_warning = ""
            home = os.path.expanduser("~")
            claude_projects_dir = os.path.join(home, ".claude", "projects", "-" + home.replace("/", "-") + "-")
            history = open(os.path.join(claude_projects_dir, session_id)).read().split("\n")
            if '"is_error":true' in history[-1]:
                if 'Please double press esc to edit your message and try again' in history[-2]:
                    error_msg = json.loads(history[-2])['message']['content'][0]['text']
                    while '{"type":"tool_use"' not in history[-1]:
                        history = history[:-1]
                    breaking_command = json.loads(history[-1])['message']['content'][0]
                    history = history[:-1]
                    next_warning = f"WARNING: Last time you tried to issue the following tool call: {breaking_command}. This tool call will fail with the error {error_msg}. Do not issue this command, instead do something else that will avoid it (e.g., split large files into pieces, resize large images)."
                    open(os.path.join(claude_projects_dir, session_id), "w").write("\n".join(history))
                    

        # All attempts failed
        return {task_id: "ERROR: answer.json not found after retries"}

    except subprocess.TimeoutExpired:
        return {task_id: "ERROR: Timeout"}
    except Exception as e:
        return {task_id: f"ERROR: {str(e)}"}

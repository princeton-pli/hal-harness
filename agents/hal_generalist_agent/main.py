from typing import Optional, List, Dict, Any, Tuple
from functools import partial
import tiktoken
from mdconvert import MarkdownConverter

import subprocess
from pathlib import Path
import re

import json
import os


from smolagents import (
    CodeAgent,
    tool,
    LiteLLMModel,
    Tool,
    PythonInterpreterTool,
    VisitWebpageTool,
    GoogleSearchTool,
)
from smolagents.models import MessageRole, Model
from smolagents.agents import ActionStep, TaskStep, PlanningStep

# Monkey-patch smolagents to handle GPT-5
import smolagents.models


def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and gpt-5 (and their versioned variants).
    """
    model_name = model_id.split("/")[-1]
    # o3, o4-mini, and gpt-5 (including versioned variants) don't support stop parameter
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*|gpt-5[-\d]*)$"
    return not re.match(pattern, model_name)


# Replace the function in smolagents
smolagents.models.supports_stop_parameter = supports_stop_parameter


try:
    from hal.utils.weave_utils import MODEL_PRICES_DICT
except ImportError:
    # When running on VM or Docker, the utils module is not available
    from model_prices import MODEL_PRICES_DICT

# Import fault injection utilities for reliability evaluation
try:
    from hal.utils.fault_injection import FaultInjector, FaultEvent
except ImportError:
    # When running on VM or Docker, try local import
    try:
        from fault_injection import FaultInjector, FaultEvent
    except ImportError:
        FaultInjector = None
        FaultEvent = None

# Import GAIA structural perturbation utilities for reliability evaluation
try:
    from hal.utils.gaia_perturbations import (
        GaiaPerturbator,
        create_gaia_perturbator,
        wrap_tools_with_perturbation,
    )
except ImportError:
    # When running on VM or Docker, try local import
    try:
        from gaia_perturbations import (
            GaiaPerturbator,
            create_gaia_perturbator,
            wrap_tools_with_perturbation,
        )
    except ImportError:
        GaiaPerturbator = None
        create_gaia_perturbator = None
        wrap_tools_with_perturbation = None

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy.*",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy.*",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
    "time",
    "pickle",
    "itertools",
    "random",
    "copy",
    "math",
    "cmath",
    "collections",
    "functools",
    "mpl_toolkits.mplot3d",
    "sympy",
]


def save_agent_steps(agent, kwargs, response, sample):
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            step.agent_memory = None
    intermediate_steps = str(agent.memory.steps)
    with open("steps.json", "w") as f:
        json.dump(
            {
                "agent_args": kwargs,
                "intermediate_steps": intermediate_steps,
                "response": str(response),
                "sample": sample,
            },
            f,
            indent=2,
        )


def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


def check_budget_exceeded(agent: CodeAgent, budget: float, model_name: str) -> bool:
    total_input_tokens = agent.monitor.total_input_token_count
    total_output_tokens = agent.monitor.total_output_token_count

    cost = (
        MODEL_PRICES_DICT[model_name]["prompt_tokens"] * total_input_tokens
        + MODEL_PRICES_DICT[model_name]["completion_tokens"] * total_output_tokens
    )

    print(f"Current cost: {cost}")
    if cost >= budget:
        return True
    return False


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()

    def __init__(self, model: Model, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward_initial_exam_mode(self, file_path, question):
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception(
                "Cannot use inspect_file_as_text tool with images: use visualizer instead!"
            )

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return self.model(messages).content

    def forward(self, file_path, question: Optional[str] = None) -> str:
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception(
                "Cannot use inspect_file_as_text tool with images: use visualizer instead!"
            )

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return self.model(messages).content


@tool
def query_vision_language_model(query: str, image_path: str) -> str:
    """
    Query a vision language model with text and an image.
    Args:
        query: The text query or question to ask about the image
        image_path: Path to the image file to analyze
    Returns:
        str: The vision language model's response about the image
    """
    try:
        import base64
        from litellm import completion

        # Check if the image file exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            base64_image = base64.b64encode(image_content).decode("utf-8")

        # Create the message with text and image
        response = completion(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        # Return the model's response
        return response.choices[0].message.content

    except Exception as e:
        return f"Error processing vision query: {str(e)}"


@tool
def execute_bash(command: str) -> str:
    """
    Description: Execute a bash command and return its output.
    Will not execute commands requiring internet access.
    Common linux and python packages are available via apt and pip.
    Args:
        command: The bash command to execute
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = f"Exit Code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

        # Limit output to 1000 tokens
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(output)
        if len(tokens) > 1000:
            output = (
                encoding.decode(tokens[:1000])
                + "\n... (output truncated to 1000 tokens)"
            )

        return output
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def edit_file(
    command: str,
    path: str,
    content: Optional[str] = None,
    line_number: Optional[int] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
) -> str:
    """
    Edit files in the project with various operations.
    Args:
        command: One of 'view', 'create', 'str_replace', 'insert', 'delete'
        path: Path to the file to edit
        content: Content for create/insert operations
        line_number: Line number for insert/delete operations
        old_str: String to replace when using str_replace
        new_str: New string for replacement
    """
    path = Path(path)
    try:
        if command == "view":
            if not path.exists():
                return f"Path {path} does not exist"
            elif path.is_dir():
                return f"Path {path} is a directory"
            else:
                with open(path, "r") as f:
                    content = f.read()
                    return content[:5000] + ("..." if len(content) > 5000 else "")

        elif command == "create":
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(content or "")
            return f"Created file {path}"

        elif command == "str_replace":
            if not path.is_file():
                return f"Error: {path} is not a file"
            with open(path, "r") as f:
                file_content = f.read()
            if old_str not in file_content:
                return "Error: Could not find exact match for replacement string"
            new_content = file_content.replace(old_str, new_str)
            with open(path, "w") as f:
                f.write(new_content)
            return f"Successfully replaced content in {path}"

        elif command == "insert":
            if not path.is_file():
                return f"Error: {path} is not a file"
            if line_number is None:
                return "Error: Line number is required for insert operation"
            with open(path, "r") as f:
                lines = f.readlines()
            if (
                not isinstance(line_number, int)
                or line_number < 1
                or line_number > len(lines) + 1
            ):
                return f"Error: Invalid line number {line_number}"
            lines.insert(line_number - 1, content + "\n")
            with open(path, "w") as f:
                f.writelines(lines)
            return f"Inserted content at line {line_number} in {path}"

        elif command == "delete":
            if not path.is_file():
                return f"Error: {path} is not a file"
            if line_number is None:
                return "Error: Line number is required for delete operation"
            with open(path, "r") as f:
                lines = f.readlines()
            if (
                not isinstance(line_number, int)
                or line_number < 1
                or line_number > len(lines)
            ):
                return f"Error: Invalid line number {line_number}"
            del lines[line_number - 1]
            with open(path, "w") as f:
                f.writelines(lines)
            return f"Deleted line {line_number} from {path}"

    except Exception as e:
        return f"Error performing {command} operation: {str(e)}"


@tool
def file_content_search(
    query: str,
    exclude_pattern: Optional[str] = "*.pyc,*.git*,__pycache__,*.bin,*.exe,*.dll,*.so",
) -> str:
    """
    Search files in the current directory and subdirectories for specific content. This will only search the content of the files, not the files themselves.
    Args:
        query: The search term or regex pattern to look for
        exclude_pattern: Comma-separated file patterns to exclude from search (default: binaries and cache files)
    Returns:
        str: Matching passages with file paths and line numbers
    """
    if not query.strip():
        return "Error: Empty search pattern. Please provide a valid search term."

    results = []
    matches_found = 0
    files_searched = 0

    context_lines = 3  # Reduced from 100 to keep output manageable
    max_matches = 10
    max_files = 50

    exclude_patterns = exclude_pattern.split(",") if exclude_pattern else []

    try:
        all_files = list(Path(".").rglob("*"))

        files_to_search = []
        for file_path in all_files:
            if not file_path.is_file():
                continue

            # Skip excluded patterns
            skip = False
            for pattern in exclude_patterns:
                pattern = pattern.strip()
                if file_path.match(pattern):
                    skip = True
                    break

            # skip input.json, agent_args.json, and steps.json
            if file_path.name in [
                "input.json",
                "agent_args.json",
                "steps.json",
                "main.py",
            ]:
                skip = True

            if not skip:
                files_to_search.append(file_path)

        # Limit to max_files
        files_to_search = files_to_search[:max_files]

        for file_path in files_to_search:
            if matches_found >= max_matches:
                break

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                files_searched += 1

                for i, line in enumerate(lines):
                    if matches_found >= max_matches:
                        break

                    if re.search(query, line, re.IGNORECASE):
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)

                        context = "".join(lines[start:end])
                        # Truncate very long contexts
                        if len(context) > 1000:
                            context = (
                                context[:500]
                                + "\n... (truncated) ...\n"
                                + context[-500:]
                            )

                        results.append(
                            f"File: {file_path} (line {i + 1}):\n{context}\n---"
                        )
                        matches_found += 1

            except (UnicodeDecodeError, IOError):
                # Skip binary or unreadable files
                continue

        if not results:
            return f"No matches found for '{query}' in {files_searched} files."

        summary = f"Found {matches_found} matches for '{query}' in {files_searched} files.\n\n"
        full_output = summary + "\n".join(results)

        return full_output

    except Exception as e:
        return f"Error searching files: {str(e)}"


def _safe_serialize(obj: Any) -> Any:
    """Safely serialize an object to JSON-serializable format."""
    if obj is None:
        return None
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


TOOL_NAMES = [
    "web_search",
    "visit_webpage",
    "python_interpreter",
    "execute_bash",
    "inspect_file_as_text",
    "edit_file",
    "file_content_search",
    "query_vision_language_model",
    "execute_code_to_interact_with_apis",
    "ask_user",
    "finish_task",
    "book_reservation",
    "calculate",
    "cancel_reservation",
    "get_reservation_details",
    "get_user_details",
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "send_certificate",
    "think",
    "transfer_to_human_agents",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
]

_TOOL_PATTERNS = [
    re.compile(rf"(?:^|[^a-zA-Z0-9_])(?:\w+\.)*{re.escape(name)}\s*\(", re.MULTILINE)
    for name in TOOL_NAMES
]


def _strip_strings_and_comments(code: str) -> str:
    """Remove Python string literals and # comments to reduce false positives."""
    if not isinstance(code, str):
        return ""
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"(?:'[^'\\]*(?:\\.[^'\\]*)*')", "", code)
    code = re.sub(r'(?:"[^"\\]*(?:\\.[^"\\]*)*")', "", code)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    return code


def _extract_tools_from_python_code(code: str) -> List[str]:
    """Extract actual tool calls from Python code executed by python_interpreter."""
    if not isinstance(code, str) or not code.strip():
        return []

    cleaned = _strip_strings_and_comments(code)
    found = []
    seen = set()

    for name, pat in zip(TOOL_NAMES, _TOOL_PATTERNS):
        if pat.search(cleaned):
            if name not in seen:
                seen.add(name)
                found.append(name)

    return found


def collect_task_metrics(agent: CodeAgent) -> Dict[str, Any]:
    action_steps = [step for step in agent.memory.steps if isinstance(step, ActionStep)]
    tool_call_sequence: List[str] = []
    step_summaries: List[Dict[str, Any]] = []

    # Get token usage from smolagents monitor
    total_input_tokens = agent.monitor.total_input_token_count
    total_output_tokens = agent.monitor.total_output_token_count

    for step in action_steps:
        tool_calls: List[Dict[str, Any]] = []
        # Safe access to tool_calls attribute
        step_tool_calls = getattr(step, "tool_calls", None)
        if step_tool_calls:
            for call in step_tool_calls:
                tool_name = getattr(call, "name", str(call))
                arguments = _safe_serialize(getattr(call, "arguments", None))

                # Parse actual tools used within python_interpreter code
                actual_tools = (
                    _extract_tools_from_python_code(arguments)
                    if tool_name == "python_interpreter"
                    else []
                )

                if actual_tools:
                    # Add the actual tools found in the Python code
                    for actual_tool in actual_tools:
                        tool_call_sequence.append(actual_tool)
                        tool_calls.append(
                            {
                                "name": actual_tool,
                                "arguments": arguments,  # Keep the full Python code as context
                                "id": getattr(call, "id", None),
                            }
                        )
                else:
                    # Default behavior for non-Python tools or when no tools found
                    tool_call_sequence.append(tool_name)
                    tool_calls.append(
                        {
                            "name": tool_name,
                            "arguments": arguments,
                            "id": getattr(call, "id", None),
                        }
                    )

        step_summary = {
            "step_number": getattr(step, "step_number", len(step_summaries) + 1),
            "tool_calls": tool_calls,
            "error": str(getattr(step, "error", None))
            if getattr(step, "error", None)
            else None,
        }
        step_summaries.append(step_summary)

    # Calculate cost using MODEL_PRICES_DICT
    model_cost = 0.0
    if hasattr(agent, "model") and hasattr(agent.model, "model_id"):
        model_name = agent.model.model_id
        if model_name in MODEL_PRICES_DICT:
            model_cost = (
                MODEL_PRICES_DICT[model_name]["prompt_tokens"] * total_input_tokens
                + MODEL_PRICES_DICT[model_name]["completion_tokens"]
                * total_output_tokens
            )

    metrics = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost": model_cost,
        "step_count": len(action_steps),
        "tool_call_count": len(tool_call_sequence),
        "tool_call_sequence": tool_call_sequence,
        "unique_tools_used": list(set(tool_call_sequence)),
        "steps": step_summaries,
    }
    return metrics


def _extract_conversation_history(agent: CodeAgent) -> List[Dict[str, Any]]:
    """
    Extract conversation history from smolagents CodeAgent for reliability analysis.

    Converts agent.memory.steps into a list of message dicts compatible with
    the analyze_reliability.py script and LLM-based safety analysis.

    Handles different step types:
    - TaskStep: User's initial task/query
    - ActionStep: Agent's thoughts, tool calls, observations
    - PlanningStep: Agent's planning steps

    Returns:
        List of dicts with 'role' and 'content' keys
    """
    history = []

    try:
        for step in agent.memory.steps:
            if isinstance(step, TaskStep):
                # TaskStep contains the user's task/query
                task_content = getattr(step, "task", None)
                if task_content:
                    history.append(
                        {"role": "user", "content": str(task_content)[:2000]}
                    )

            elif isinstance(step, PlanningStep):
                # PlanningStep contains agent's planning
                plan_content = getattr(step, "plan", None) or getattr(
                    step, "model_output", None
                )
                if plan_content:
                    history.append(
                        {
                            "role": "assistant",
                            "content": f"Planning: {str(plan_content)[:1000]}",
                        }
                    )

            elif isinstance(step, ActionStep):
                # Add the model's thought/action as assistant message
                step_content = []

                # Add model output (the reasoning/plan)
                if hasattr(step, "model_output") and step.model_output:
                    step_content.append(f"Thought: {step.model_output}")

                # Add tool calls
                if hasattr(step, "tool_calls") and step.tool_calls:
                    for call in step.tool_calls:
                        tool_name = getattr(call, "name", str(call))
                        args = _safe_serialize(getattr(call, "arguments", None))
                        step_content.append(f"Action: {tool_name}({args})")

                # Add observations/results
                if hasattr(step, "observations") and step.observations:
                    step_content.append(f"Observation: {str(step.observations)[:500]}")

                # Add error if any
                if hasattr(step, "error") and step.error:
                    step_content.append(f"Error: {str(step.error)}")

                if step_content:
                    history.append(
                        {"role": "assistant", "content": "\n".join(step_content)}
                    )

            else:
                # Fallback for other step types
                if hasattr(step, "content"):
                    role = getattr(step, "role", "user")
                    history.append(
                        {"role": str(role).lower(), "content": str(step.content)[:2000]}
                    )

    except Exception as e:
        print(f"Warning: Error extracting conversation history: {e}")
        history = []

    return history


def _build_confidence_messages(
    conversation_history: list,
    confidence_prompt: str,
    initial_prompt: Optional[str] = None,
) -> list:
    """
    Build message list for confidence assessment by appending confidence prompt
    to the full conversation history.

    This approach preserves the complete context including tool calls so the
    model has full visibility into what it did. Uses a high token ceiling to
    accommodate reasoning models that use internal thinking tokens.

    Args:
        conversation_history: The full conversation including tool calls
        confidence_prompt: The prompt asking for confidence assessment
        initial_prompt: The initial user task/prompt (prepended as first user message)

    Returns:
        Message list with full conversation history + confidence prompt
    """
    import copy

    messages = []

    # Prepend the initial user prompt/task if provided
    if initial_prompt:
        messages.append({"role": "user", "content": initial_prompt})

    # Deep copy and add conversation history
    if conversation_history:
        messages.extend(copy.deepcopy(conversation_history))

    # Append the confidence assessment prompt as a user message
    messages.append({"role": "user", "content": confidence_prompt})

    return messages


def _compute_gaia_confidence_score(
    model: Any,
    task_question: str,
    agent_answer: str,
    conversation_history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    initial_prompt: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute confidence score for GAIA task via self-assessment.

    Asks the model to evaluate its own performance and provide a confidence
    score from 0-100. Uses the FULL conversation history so the model can
    reflect on its actual reasoning process.

    Args:
        model: The LiteLLMModel instance
        task_question: The original GAIA question
        agent_answer: The answer the agent produced
        conversation_history: The full conversation history including tool calls
        metrics: Metrics from collect_task_metrics
        initial_prompt: The initial user task/prompt (for context)

    Returns:
        Tuple of (confidence_score in [0,1], confidence_details dict)
    """
    import litellm

    # Extract metrics for fallback heuristic
    num_actions = metrics.get("tool_call_count", 0)
    num_steps = metrics.get("step_count", 0)

    # Count errors from steps
    num_errors = sum(1 for step in metrics.get("steps", []) if step.get("error"))

    # Confidence prompt - model already has full context from conversation history
    confidence_prompt = """You just completed a question-answering task. Please assess your confidence in the correctness of your answer.

Consider the following when rating your confidence:
1. Did you find clear, reliable information to answer the question?
2. Did you encounter any errors or failures during execution?
3. How certain are you that your answer format matches what was requested?
4. Were there any ambiguities in the question or your research?
5. Did you verify your answer through multiple sources?

Please provide a confidence score from 0 to 100, where:
- 0-20: Very uncertain, likely incorrect
- 21-40: Low confidence, significant doubts
- 41-60: Moderate confidence, some uncertainty
- 61-80: Good confidence, minor doubts
- 81-100: Very confident, highly certain of correctness

Respond with ONLY a number between 0 and 100. No explanation needed."""

    try:
        # Build messages by appending confidence prompt to FULL conversation history
        # This preserves complete context so the model knows exactly what it did
        messages = _build_confidence_messages(
            conversation_history=conversation_history,
            confidence_prompt=confidence_prompt,
            initial_prompt=initial_prompt,
        )

        # Enable litellm's automatic parameter modification to handle provider quirks
        # Anthropic requires tools= param when conversation contains tool_calls
        litellm.modify_params = True

        # Debug: Log what we're sending
        print(f"📊 GAIA confidence assessment request for {model.model_id}:")
        print(f"   Messages: {len(messages)} messages (full conversation history)")

        # Build kwargs for litellm.completion
        # Extract api_base and api_key from model if they exist (for custom providers like Gemini)
        completion_kwargs = {
            "model": model.model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 65536,  # High limit for reasoning models
        }

        # Add api_base and api_key if the model has custom endpoints (Gemini, Together, etc.)
        if hasattr(model, "api_base") and model.api_base:
            completion_kwargs["api_base"] = model.api_base
            print(f"   Using custom api_base: {model.api_base}")
        if hasattr(model, "api_key") and model.api_key:
            completion_kwargs["api_key"] = model.api_key

        # Call the model for confidence assessment
        response = litellm.completion(**completion_kwargs)

        # Debug: Log response structure
        print("📊 Confidence response received:")
        if response.choices:
            msg = response.choices[0].message
            print(f"   Message content: {repr(msg.content)[:100]}...")

        content = response.choices[0].message.content
        if content is None:
            msg = response.choices[0].message
            error_details = {
                "content": msg.content,
                "role": getattr(msg, "role", None),
                "tool_calls": getattr(msg, "tool_calls", None),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
            }
            raise ValueError(f"Model returned None content. Details: {error_details}")

        confidence_text = content.strip()

        # Extract number from response
        import re

        numbers = re.findall(r"\d+", confidence_text)

        if numbers:
            confidence_score = float(numbers[0]) / 100.0
            confidence_score = max(0.0, min(1.0, confidence_score))
            print(
                f"✓ GAIA confidence assessment: {confidence_text} -> {confidence_score:.2f}"
            )
        else:
            print(
                f"⚠️  Could not parse confidence from '{confidence_text}', using default 0.5"
            )
            confidence_score = 0.5

        confidence_details = {
            "prompt": confidence_prompt,
            "model_response": confidence_text,
            "parsed_score": confidence_score,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "num_steps": num_steps,
            "model": model.model_id,
            "conversation_messages_count": len(messages),
        }

        return confidence_score, confidence_details

    except Exception as e:
        print(f"Warning: Error computing GAIA confidence score: {e}")
        # Return heuristic-based confidence
        if num_actions == 0:
            heuristic_confidence = 0.1
        else:
            error_penalty = num_errors / max(num_actions, 1)
            heuristic_confidence = max(0.1, 0.7 - error_penalty * 0.3)

        heuristic_details = {
            "prompt": "ERROR: Could not call model",
            "model_response": str(e),
            "parsed_score": heuristic_confidence,
            "num_actions": num_actions,
            "num_errors": num_errors,
            "num_steps": num_steps,
            "model": model.model_id if hasattr(model, "model_id") else "unknown",
            "fallback": True,
        }

        return heuristic_confidence, heuristic_details


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    BUDGET = kwargs["budget"] if "budget" in kwargs else None

    import litellm

    model_params = {}
    model_params["model_id"] = kwargs["model_name"]

    # Handle reasoning parameters based on provider
    if "reasoning_effort" in kwargs:
        if "openrouter/" in kwargs["model_name"]:
            # OpenRouter doesn't support reasoning_effort, convert to reasoning.max_tokens
            effort_to_tokens = {"low": 1024, "medium": 2048, "high": 4096}
            model_params["reasoning"] = {
                "max_tokens": effort_to_tokens.get(kwargs["reasoning_effort"], 4096)
            }
        else:
            # For Anthropic direct and other providers that support reasoning_effort
            model_params["reasoning_effort"] = kwargs["reasoning_effort"]

    if "temperature" in kwargs:
        model_params["temperature"] = kwargs["temperature"]

    if "gemini" in kwargs["model_name"]:
        model_params["model_id"] = kwargs["model_name"].replace("gemini/", "openai/")
        model_params["api_key"] = os.getenv("GEMINI_API_KEY")
        model_params["api_base"] = (
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    if "together_ai" in kwargs["model_name"]:
        model_params["model_id"] = kwargs["model_name"].replace(
            "together_ai/", "openai/"
        )
        model_params["api_key"] = os.environ.get("TOGETHERAI_API_KEY")
        model_params["api_base"] = "https://api.together.xyz/v1"

    task_id, task = list(input.items())[0]

    results = {}

    # Inject OpenRouter provider selection via litellm extra_body if requested
    try:
        import litellm  # type: ignore

        # Be lenient with unknown params on different backends
        litellm.drop_params = True

        # Models that don't support 'stop' parameter
        MODELS_WITHOUT_STOP_SUPPORT = ["gpt-5", "gpt-5.2", "gpt-5.1"]

        def model_requires_stop_filter(model_name: str) -> bool:
            """Check if model requires filtering out 'stop' parameter."""
            model_lower = model_name.lower()
            return any(m in model_lower for m in MODELS_WITHOUT_STOP_SUPPORT)

        # Wrap litellm completion to filter out 'stop' for models that don't support it
        if model_requires_stop_filter(kwargs.get("model_name", "")):
            original_completion_stop = getattr(litellm, "completion", None)
            original_acompletion_stop = getattr(litellm, "acompletion", None)

            if original_completion_stop is not None:

                def completion_without_stop(*args, **completion_kwargs):
                    # Remove 'stop' parameter for models that don't support it
                    completion_kwargs.pop("stop", None)
                    return original_completion_stop(*args, **completion_kwargs)

                litellm.completion = completion_without_stop  # type: ignore
                print(
                    f"[INFO] Enabled 'stop' parameter filtering for model: {kwargs.get('model_name')}"
                )

            if original_acompletion_stop is not None:

                async def acompletion_without_stop(*args, **completion_kwargs):
                    # Remove 'stop' parameter for models that don't support it
                    completion_kwargs.pop("stop", None)
                    return await original_acompletion_stop(*args, **completion_kwargs)  # type: ignore

                litellm.acompletion = acompletion_without_stop  # type: ignore

        if "openrouter_provider_only" in kwargs and "openrouter/" in kwargs.get(
            "model_name", ""
        ):
            providers_value = kwargs["openrouter_provider_only"]
            if isinstance(providers_value, str):
                providers = [p.strip() for p in providers_value.split(",") if p.strip()]
            elif isinstance(providers_value, (list, tuple)):
                providers = [str(p) for p in providers_value]
            else:
                providers = [str(providers_value)]

            original_completion = getattr(litellm, "completion", None)
            original_acompletion = getattr(litellm, "acompletion", None)

            if not original_completion and not original_acompletion:
                print(
                    "[WARNING] OpenRouter provider pinning requested but litellm completion hooks not found; skipping provider pinning."
                )
            else:
                print(
                    f"[INFO] Enabling OpenRouter provider pinning: providers={providers}"
                )

            if original_completion is not None:

                def completion_with_provider(*args, **completion_kwargs):
                    model_field = completion_kwargs.get("model")
                    if isinstance(model_field, str) and "openrouter/" in model_field:
                        extra_body = completion_kwargs.get("extra_body", {}) or {}
                        extra_body["provider"] = {"only": providers}
                        completion_kwargs["extra_body"] = extra_body
                    return original_completion(*args, **completion_kwargs)

                litellm.completion = completion_with_provider  # type: ignore

            if original_acompletion is not None:

                async def acompletion_with_provider(*args, **completion_kwargs):
                    model_field = completion_kwargs.get("model")
                    if isinstance(model_field, str) and "openrouter/" in model_field:
                        extra_body = completion_kwargs.get("extra_body", {}) or {}
                        extra_body["provider"] = {"only": providers}
                        completion_kwargs["extra_body"] = extra_body
                    return await original_acompletion(*args, **completion_kwargs)  # type: ignore

                litellm.acompletion = acompletion_with_provider  # type: ignore
    except Exception as e:
        # Non-fatal: if litellm is unavailable or wrapping fails, continue without provider pinning
        print(f"[WARNING] Failed to enable OpenRouter provider pinning: {e}")

    # ========== FAULT INJECTION SETUP (OPTIONAL) ==========
    # Initialize FaultInjector to test agent robustness to API failures
    fault_injector = None
    if FaultInjector is not None and (
        kwargs.get("enable_fault_injection") == "true"
        or kwargs.get("enable_fault_injection") is True
    ):
        import time
        import random
        import asyncio

        fault_rate = float(kwargs.get("fault_rate", 0.2))
        fault_injector = FaultInjector(fault_rate=fault_rate)
        print(f"🔧 Fault injection enabled with rate={fault_rate}")

        # Store original completion functions before wrapping
        original_completion_for_fault = litellm.completion
        original_acompletion_for_fault = litellm.acompletion

        # Wrap sync completion with fault injection
        def completion_with_fault_injection(*args, **completion_kwargs):
            if fault_injector and fault_injector.enabled:
                try:
                    response = fault_injector.wrap_call(
                        original_completion_for_fault, *args, **completion_kwargs
                    )
                except Exception as fault_error:
                    print(
                        f"⚡ Fault injected: {type(fault_error).__name__}: {fault_error}"
                    )
                    raise
            else:
                response = original_completion_for_fault(*args, **completion_kwargs)
            return response

        # Wrap async completion with fault injection
        async def acompletion_with_fault_injection(*args, **completion_kwargs):
            if fault_injector and fault_injector.enabled:
                # For async, we inject faults manually to maintain async behavior
                if random.random() < fault_injector.fault_rate:
                    fault_type = fault_injector._select_fault_type()
                    fault_injector.state["faults_injected"] += 1
                    print(f"⚡ Async fault injected: {fault_type.value}")

                    # Attempt recovery with retries
                    max_retries = fault_injector.config.get("max_recovery_attempts", 3)
                    recovered = False
                    recovery_start = time.time()

                    for attempt in range(max_retries):
                        recovery_prob = 0.3 + (attempt * 0.2)
                        if random.random() < recovery_prob:
                            response = await original_acompletion_for_fault(
                                *args, **completion_kwargs
                            )
                            recovered = True
                            fault_injector.state["recoveries_successful"] += 1
                            break
                        await asyncio.sleep(0.1 * (attempt + 1))

                    if not recovered:
                        fault_injector.state["recoveries_failed"] += 1
                        raise RuntimeError(
                            f"Simulated fault after {max_retries} recovery attempts: {fault_type.value}"
                        )

                    recovery_time = time.time() - recovery_start
                    fault_injector.state["total_recovery_time"] += recovery_time

                    # Log fault event
                    fault_event = FaultEvent(
                        fault_type=fault_type,
                        recovered=recovered,
                        recovery_time=recovery_time,
                        context={
                            "recovery_attempts": attempt + 1,
                            "function_name": "acompletion",
                        },
                    )
                    fault_injector.fault_events.append(fault_event)
                else:
                    response = await original_acompletion_for_fault(
                        *args, **completion_kwargs
                    )
            else:
                response = await original_acompletion_for_fault(
                    *args, **completion_kwargs
                )
            return response

        # Replace litellm completion functions with fault injection wrappers
        litellm.completion = completion_with_fault_injection
        litellm.acompletion = acompletion_with_fault_injection

    model = LiteLLMModel(**model_params)

    # Create a search tool that ignores filter_year to avoid overly restrictive searches
    # We create the underlying tool and wrap it with a simple function tool
    _google_search_tool = GoogleSearchTool(provider="serpapi")

    @tool
    def web_search(query: str) -> str:
        """
        Performs a google web search for your query then returns a string of the top search results.

        Args:
            query: The search query to perform.

        Returns:
            A string with the top search results.
        """
        # Always pass filter_year=None to avoid "no results" errors from overly restrictive searches
        return _google_search_tool.forward(query=query, filter_year=None)

    CORE_TOOLS = [
        # DuckDuckGoSearchTool(),  # Unreliable - rate limited
        web_search,  # Wrapped GoogleSearchTool that ignores year filter
        VisitWebpageTool(),
        PythonInterpreterTool(),
        execute_bash,
        TextInspectorTool(model=model, text_limit=5000),
        edit_file,
        file_content_search,
        query_vision_language_model,
    ]

    agent = CodeAgent(
        tools=CORE_TOOLS,
        planning_interval=4,
        max_steps=200,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        budget_exceeded_callback=partial(
            check_budget_exceeded, budget=BUDGET, model_name=kwargs["model_name"]
        )
        if BUDGET
        else None,
        model=model,
    )
    agent.python_executor.state["__name__"] = "__main__"

    if kwargs["benchmark_name"] == "usaco":
        USACO_PROMPT = """Please reply with a Python 3 solution to the below problem. Make sure
to wrap your code in '```python' and '```' Markdown delimiters, and
include exactly one block of code with the entire solution.
No outside libraries are allowed.

[BEGIN PROBLEM]
{}
[END PROBLEM]
"""
        prompt = USACO_PROMPT.format(task["description"])
        response = agent.run(prompt)
        save_agent_steps(agent, kwargs, response, task)

        # extract code from response
        response = str(response)
        if "```python" in response:
            response = response.split("```python")[1].split("```")[0]

        # Collect metrics
        metrics = collect_task_metrics(agent)

        save_agent_steps(agent, kwargs, response, task)

        return {
            task_id: {
                "answer": response,
                "metrics": metrics,
            }
        }

    elif kwargs["benchmark_name"] == "corebench_easy":
        # Create a new agent with more steps specifically for CoreBench easy
        corebench_agent = CodeAgent(
            tools=CORE_TOOLS,
            planning_interval=4,
            max_steps=40,
            budget_exceeded_callback=partial(
                check_budget_exceeded, budget=BUDGET, model_name=kwargs["model_name"]
            )
            if BUDGET
            else None,
            model=model,
        )

        response = corebench_agent.run(task["prompt"])
        save_agent_steps(corebench_agent, kwargs, response, task)
        return {task_id: response}

    elif kwargs["benchmark_name"] == "corebench_medium":
        # Create a new agent with more steps specifically for CoreBench medium
        corebench_agent = CodeAgent(
            tools=CORE_TOOLS,
            planning_interval=4,
            max_steps=40,
            budget_exceeded_callback=partial(
                check_budget_exceeded, budget=BUDGET, model_name=kwargs["model_name"]
            )
            if BUDGET
            else None,
            model=model,
        )

        response = corebench_agent.run(task["prompt"])
        save_agent_steps(corebench_agent, kwargs, response, task)
        return {task_id: response}

    elif kwargs["benchmark_name"] == "corebench_hard":
        # Create a new agent with more steps specifically for CoreBench hard
        corebench_agent = CodeAgent(
            tools=CORE_TOOLS,
            planning_interval=4,
            max_steps=40,
            budget_exceeded_callback=partial(
                check_budget_exceeded, budget=BUDGET, model_name=kwargs["model_name"]
            )
            if BUDGET
            else None,
            model=model,
        )

        response = corebench_agent.run(task["prompt"])
        save_agent_steps(corebench_agent, kwargs, response, task)
        return {task_id: response}

    elif kwargs["benchmark_name"] == "scienceagentbench":
        DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}"""

        def format_task_dict(example):
            task = {
                "instance_id": example["instance_id"],
                "task_inst": example["task_inst"],
                "dataset_path": "benchmark/datasets/"
                + example["dataset_folder_tree"].split("\n")[0][4:],
                "dataset_folder_tree": example["dataset_folder_tree"],
                "dataset_preview": example["dataset_preview"],
                "output_fname": example["output_fname"],
                "domain_knowledge": example["domain_knowledge"],
                "gold_program_name": example["gold_program_name"],
            }

            return task

        def get_sys_msg(task):
            sys_msg = (
                """Please reply with a Python 3 solution to the below problem. Make sure
to wrap your code in '```python' and '```' Markdown delimiters, and
include exactly one block of code with the entire solution."""
                + "\n"
                + task["task_inst"]
                + ("\n" + str(task["domain_knowledge"]))
            )

            sys_msg += "\n" + DATA_INFO_PROMPT.format(
                dataset_path=task["dataset_path"],
                dataset_folder_tree=task["dataset_folder_tree"],
                dataset_preview=task["dataset_preview"],
            )

            return sys_msg

        task_id = list(input.keys())[0]
        task = format_task_dict(list(input.values())[0])
        sys_msg = get_sys_msg(task)

        response = str(agent.run(sys_msg))
        save_agent_steps(agent, kwargs, response, task)

        if "```python" in response:
            response = response.split("```python")[1].split("```")[0]

        return {
            task_id: {
                "history": [
                    {"role": "assistant", "content": f"```python{response}```"}
                ],
                "cost": 0.0,
            }
        }

    elif kwargs["benchmark_name"] == "swebench_verified":
        pass
    elif kwargs["benchmark_name"] == "swebench_verified_mini":
        process = subprocess.run(
            ["git", "clone", f"https://github.com/{task['repo']}.git"],
            capture_output=True,
            text=True,
        )
        print(process.stdout)
        if process.returncode != 0:
            raise Exception(f"Failed to clone repository: {process.stderr}")

        process = subprocess.run(
            f"cd {task['repo'].split('/')[-1]} && git reset --hard {task['base_commit']}",
            shell=True,
            capture_output=True,
            text=True,
        )
        print(process.stdout)
        if process.returncode != 0:
            raise Exception(f"Failed to reset repository: {process.stderr}")

        response = agent.run(
            f"""I need you to solve this issue by generating a single patch that I can apply directly to this repository using git apply.
            
Problem: {task["problem_statement"]}
            
The code of the project is cloned to {task["repo"].split("/")[-1]}. After you are done, please return the content of the patch as your final answer."""
        )
        save_agent_steps(agent, kwargs, response, task)

        model_patch = extract_diff(response)
        return {task_id: model_patch}

    elif kwargs["benchmark_name"] == "appworld_test_normal":

        def tool_generator(
            name_: str,
            description_: str,
            inputs_: dict,
            output_type_: dict,
            function: callable,
        ):
            class GeneratedTool(Tool):
                name = name_
                description = description_
                inputs = inputs_
                output_type = output_type_

                def forward(self, *args, **kwargs):
                    return world.apis.function(*args, **kwargs)

            GeneratedTool.__name__ = f"{name_.title()}Tool"
            return GeneratedTool

        def get_smolagents_tools(task):
            tools: list[Tool] = []
            for api in task.api_docs.keys():
                for api_func in task.api_docs[api].keys():
                    tool = tool_generator(
                        name_=api_func,
                        description_=task.api_docs[api][api_func]["description"],
                        inputs_=task.api_docs[api][api_func]["parameters"],
                        output_type_=task.api_docs[api][api_func]["response_schemas"][
                            "success"
                        ],
                        function=api_func,
                    )
                    tools.append(tool)
            return tools

        from appworld import AppWorld

        with AppWorld(
            task_id=task_id,
            experiment_name="output",
            remote_environment_url="http://0.0.0.0:8001",
        ) as world:
            instruction = world.task.instruction  # To see task instruction.
            supervisor = world.task.supervisor
            get_smolagents_tools(world.task)

            prompt = f"""Using the available APIs you can interact with on my behalf through the "interact_with_apis" tool, generate code to solve the following task.
            
Here are three key APIs that you need to know to get more information
            
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Now please generate code to solve the following task:

My name is: {supervisor.first_name} {supervisor.last_name}. My personal email is {supervisor.email} and phone number is {supervisor.phone_number}.

Task:

{instruction}
"""

            @tool
            def execute_code_to_interact_with_apis(code: str) -> str:
                """
                Execute code to interact with the APIs. You can access variables from previous code blocks you executed. Code is executed in a python REPL environment.
                Args:
                    code: The code to execute
                Returns:
                    str: The terminal output of the code
                    str: Whether the task is completed
                """
                return (
                    world.execute(code),
                    "Task completed"
                    if world.task_completed()
                    else "Task not yet completed",
                )

            agent = CodeAgent(
                tools=CORE_TOOLS + get_smolagents_tools(world.task),
                planning_interval=4,
                max_steps=200,
                additional_authorized_imports=AUTHORIZED_IMPORTS,
                budget_exceeded_callback=partial(
                    check_budget_exceeded,
                    budget=BUDGET,
                    model_name=kwargs["model_name"],
                )
                if BUDGET
                else None,
                model=model,
            )
            agent.python_executor.state["__name__"] = "__main__"

            response = agent.run(prompt)
            world.post_execute()
            world.save()
            save_agent_steps(agent, kwargs, response, task)

        return {task_id: "Completed"}

    elif kwargs["benchmark_name"] == "appworld_test_challenge":
        from appworld import AppWorld

        @tool
        def execute_code_to_interact_with_apis(code: str) -> str:
            """
            Execute code to interact with the APIs. You can access variables from previous code blocks you executed. Code is executed in a python REPL environment.
            Args:
                code: The code to execute
            Returns:
                str: The terminal output of the code
                str: Whether the task is completed
            """
            return world.execute(
                code
            ), "Task completed" if world.task_completed() else "Task not yet completed"

        agent = CodeAgent(
            tools=CORE_TOOLS + [execute_code_to_interact_with_apis],
            planning_interval=4,
            max_steps=200,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            budget_exceeded_callback=partial(
                check_budget_exceeded, budget=BUDGET, model_name=kwargs["model_name"]
            )
            if BUDGET
            else None,
            model=model,
        )
        agent.python_executor.state["__name__"] = "__main__"

        with AppWorld(
            task_id=task_id,
            experiment_name="output",
            remote_environment_url="http://0.0.0.0:8001",
        ) as world:
            instruction = world.task.instruction  # To see task instruction.
            supervisor = world.task.supervisor

            prompt = f"""Using the available APIs you can interact with on my behalf through the "interact_with_apis" tool, generate code to solve the following task.
            
Here are three key APIs that you need to know to get more information
            
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Now please generate code to solve the following task:

My name is: {supervisor.first_name} {supervisor.last_name}. My personal email is {supervisor.email} and phone number is {supervisor.phone_number}.

Task:

{instruction}
"""

            response = agent.run(prompt)
            world.post_execute()
            world.save()
            save_agent_steps(agent, kwargs, response, task)

        return {task_id: "Completed"}

    elif kwargs["benchmark_name"] == "gaia":
        # ========== STRUCTURAL PERTURBATION SETUP (OPTIONAL) ==========
        gaia_perturbator = None
        if GaiaPerturbator is not None and (
            kwargs.get("enable_structural_perturbations") == "true"
            or kwargs.get("enable_structural_perturbations") is True
        ):
            perturbation_strength = kwargs.get("perturbation_strength", "medium")
            gaia_perturbator = create_gaia_perturbator(strength=perturbation_strength)

            # Set seed for reproducibility (use task_id hash for consistency)
            seed = hash(task_id) % (2**32)
            gaia_perturbator.set_seed(seed)

            print(
                f"🔧 Structural perturbations enabled for GAIA with strength={perturbation_strength}"
            )
            print(f"   Config: {gaia_perturbator.get_config_dict()}")

        # Get the original question
        original_question = task["Question"]

        # Build the base instruction template
        instruction_template = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

Here is the question and attached files are stored in your current directory:

"""

        # Apply structural perturbations if enabled
        if gaia_perturbator is not None:
            # Perturb the question
            perturbed_question = gaia_perturbator.perturb_question(original_question)

            # Perturb the instructions
            perturbed_instructions = gaia_perturbator.perturb_instructions(
                instruction_template
            )

            prompt = f"{perturbed_instructions}{perturbed_question}"

            print(
                f"📝 Question perturbed: '{original_question[:50]}...' -> '{perturbed_question[:50]}...'"
            )
        else:
            prompt = f"{instruction_template}{original_question}"

        # Execute agent
        response = agent.run(prompt)

        # Collect metrics
        metrics = collect_task_metrics(agent)

        save_agent_steps(agent, kwargs, response, task)

        # === RELIABILITY METRICS ===
        result = {
            task_id: {
                "answer": str(response).strip(),
                "metrics": metrics,
            }
        }

        # Build taken_actions for trajectory consistency (format compatible with analyze_reliability.py)
        taken_actions = [
            {"name": tool_name} for tool_name in metrics.get("tool_call_sequence", [])
        ]
        result[task_id]["taken_actions"] = taken_actions

        # Extract conversation history from smolagents for safety analysis
        # Handles TaskStep (user query), ActionStep (agent actions), PlanningStep (planning)
        conversation_history = _extract_conversation_history(agent)

        # Check if first message already contains the task (from TaskStep)
        has_initial_task = (
            conversation_history and conversation_history[0].get("role") == "user"
        )

        # Store conversation history, prepending prompt only if not already present
        if (
            kwargs.get("store_conversation_history") == "true"
            or kwargs.get("store_conversation_history") is True
        ):
            if has_initial_task:
                result[task_id]["conversation_history"] = conversation_history
            else:
                result[task_id]["conversation_history"] = [
                    {"role": "user", "content": prompt}
                ] + conversation_history

        # Compute confidence score if enabled
        compute_confidence = kwargs.get("compute_confidence", False)
        if compute_confidence == "true" or compute_confidence is True:
            # Pass initial_prompt only if not already in conversation_history
            confidence, confidence_details = _compute_gaia_confidence_score(
                model=model,
                task_question=task["Question"],
                agent_answer=str(response).strip(),
                conversation_history=conversation_history,
                metrics=metrics,
                initial_prompt=prompt if not has_initial_task else None,
            )
            result[task_id]["confidence"] = confidence
            if (
                kwargs.get("store_confidence_details") == "true"
                or kwargs.get("store_confidence_details") is True
            ):
                result[task_id]["confidence_details"] = confidence_details

        # Add fault injection metrics if enabled
        if fault_injector is not None:
            fault_stats = fault_injector.get_stats()
            fault_events = [e.to_dict() for e in fault_injector.get_fault_events()]
            result[task_id]["fault_injection"] = {
                "enabled": True,
                "fault_rate": fault_injector.fault_rate,
                "stats": fault_stats,
                "events": fault_events,
                "V_heal": fault_stats["recovery_rate"],
                "mean_recovery_time": fault_stats["mean_recovery_time"],
            }
            print(
                f"🔧 Fault stats: {fault_stats['total_faults_injected']} faults, "
                f"{fault_stats['recovery_rate']:.1%} recovery rate"
            )

        # Add structural perturbation metrics if enabled
        if gaia_perturbator is not None:
            perturbation_summary = gaia_perturbator.get_perturbation_summary()
            result[task_id]["structural_perturbation"] = {
                "enabled": True,
                "perturbation_type": "gaia",
                "perturbation_strength": kwargs.get("perturbation_strength", "medium"),
                "perturbation_count": perturbation_summary["total_perturbations"],
                "perturbations_by_type": perturbation_summary["by_type"],
                "applied_perturbations": gaia_perturbator.applied_perturbations[
                    :50
                ],  # Limit to first 50
                "config": gaia_perturbator.get_config_dict(),
                "original_question": original_question,
            }
            print(
                f"🔧 Structural perturbation stats: {perturbation_summary['total_perturbations']} perturbations applied"
            )
            print(f"   By type: {perturbation_summary['by_type']}")

        return result

    elif kwargs["benchmark_name"] == "colbench_backend_programming":
        from openai import OpenAI
        from sweet_rl.environments.human_interaction_env import HumanInteractionEnv

        env_model_name = "gpt-4o-2024-08-06"
        task_data = input[task_id]
        env_client = OpenAI()
        isolated_env = HumanInteractionEnv(
            env_client, task_data["human_prompt"], env_model_name
        )
        observation = isolated_env.reset(
            task_data["problem_description"], task_data["hidden_information"]
        )

        @tool
        def ask_user(question: str) -> str:
            """
            Ask the user a question.

            Args:
                question: The question to ask the user.

            Returns:
                str: The user's response.
                str: Indication of whether the task is finished.
            """
            observation, _, _ = isolated_env.step(question)
            return (
                observation.observation,
                "Task finished"
                if observation.done
                else "You may still continue to work on the task",
            )

        @tool
        def finish_task(answer: str) -> str:
            """
            Finish the task with answer.

            Args:
                answer: The answer to the task.

            Returns:
                str: The user's response.
                str: Indication of whether the task is finished.
            """
            observation, _, _ = isolated_env.step("I WANT TO ANSWER:" + answer)
            return "The task is finished. And your answer is received.", "Task finished"

        agent = CodeAgent(
            tools=CORE_TOOLS + [ask_user, finish_task],
            planning_interval=4,
            max_steps=80,
            model=model,
        )

        instruction = f"""
        You are a backend programmer. 
        Your task is to help a human user to resolve their problem, in particular python programming.
        Note that the problem is highly personalized so you need to explicitly gather information about the user's problem.
        You are given a task to solve the following problem:
        
        {task_data["problem_description"]}
        
        You can ask the user for clarification if needed using the ask_user tool within 9 rounds.
        When you are gathered enough information, you can finish the task using the finish_task tool and provide your answer.
        The answer should be a piece of raw python function.
        """

        response = asyncio.run(agent.arun(instruction))

        dialogue_history = [
            {"role": d["role"], "content": d["content"]}
            for d in isolated_env.get_dialogue_history()
        ]
        answer = isolated_env.answer
        return {
            task_id: {
                "answer": answer,
                "dialogue_history": dialogue_history,
                "task": {
                    "test_cases": task_data["test_cases"]
                    if task_data["task_type"] == "code"
                    else None,
                    "ground_truth": task_data["hidden_information"],
                },
            }
        }

    elif kwargs["benchmark_name"] == "colbench_frontend_design":
        from openai import OpenAI
        from sweet_rl.environments.human_design_interaction_env import (
            HumanDesignInteractionEnv,
        )

        env_model_name = "gpt-4o-2024-08-06"
        task_data = input[task_id]
        env_client = OpenAI()
        isolated_env = HumanDesignInteractionEnv(
            env_client,
            task_data["human_prompt"],
            env_model_name,
            temp_path=task_data["cache_path"],
            gpt_client=True,
        )
        observation = isolated_env.reset(
            task_data["problem_description"], task_data["hidden_information"]
        )

        @tool
        def ask_user(question: str) -> str:
            """
            Ask the user a question.

            Args:
                question: The question to ask the user.

            Returns:
                str: The user's response.
                str: Indication of whether the task is finished.
            """
            observation, _, _ = isolated_env.step(question)
            return (
                observation.observation,
                "Task finished"
                if observation.done
                else "You may still continue to work on the task",
            )

        @tool
        def finish_task(answer: str) -> str:
            """
            Finish the task with answer.

            Args:
                answer: The answer to the task.

            Returns:
                str: The user's response.
                str: Indication of whether the task is finished.
            """
            observation, _, _ = isolated_env.step("I WANT TO ANSWER:" + answer)
            return "The task is finished. And your answer is received.", "Task finished"

        agent = CodeAgent(
            tools=CORE_TOOLS + [ask_user, finish_task],
            planning_interval=4,
            max_steps=80,
            model=model,
        )

        instruction = f"""
        You are a frontend designer. You are given a task to solve the following problem:
        
        {observation}
        
        Your task is to help a human user to code a complete website with a good design in HTML and Tailwind CSS.
        Write the code inside a tag <html>.
        Write real and long sentences about the business.
        You don’t have to include images, but if you do, use only this source
        https://picsum.photos/id/48/W/H, by replacing W and H with the width and height of the image.
        Keep the id the same to only use id 48 image.
        
        You have access to the ask_user tool to ask the user for clarification, and finish_task tool to finish the task.
        You can ask the user for clarification if needed using the ask_user tool within 9 rounds.
        
        You can include ONLY ONE snippet raw html and Tailwind css code (wrapped in <html> tag) in your question to human user to ask how is the proposed design different from what the human user wants. 
        This snippet of raw html and Tailwind css code (WRAPPED IN <html> TAG) will be rendered for the human to see a screenshot of the webpage.
        The human user will respond by comparing your rendered webpage with the webpage that the human user has in mind.
        When you are gathered enough information, you can finish the task using the finish_task tool and provide your answer.
        The answer should be a piece of raw html code.
        """

        response = asyncio.run(agent.arun(instruction))
        isolated_env.driver.quit()
        dialogue_history = [
            {"role": d["role"], "content": d["content"]}
            for d in isolated_env.get_dialogue_history()
        ]
        answer = isolated_env.answer
        return {
            task_id: {
                "answer": answer,
                "dialogue_history": dialogue_history,
                "task": {
                    "test_cases": task_data["test_cases"]
                    if task_data["task_type"] == "code"
                    else None,
                    "ground_truth": task_data["hidden_information"],
                },
            }
        }

    elif kwargs["benchmark_name"] == "taubench_airline":
        from tau_bench.envs import get_env
        from tau_bench.types import Action

        ### ENV SETUP (usually this should be untouched) ###
        isolated_env = get_env(
            input[task_id]["env"],
            input[task_id]["user_strategy"],
            input[task_id]["user_model"],
            input[task_id]["task_split"],
            input[task_id]["user_provider"],
            input[task_id]["task_index"],
        )

        ## taubench airline tools
        @tool
        def book_reservation(
            user_id: str,
            origin: str,
            destination: str,
            flight_type: str,
            cabin: str,
            flights: List[Dict[str, str]],
            passengers: List[Dict[str, str]],
            payment_methods: List[Dict[str, str]],
            total_baggages: int,
            nonfree_baggages: int,
            insurance: str,
        ) -> str:
            """
            Book a reservation.

            Args:
                user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
                origin: The IATA code for the origin city, such as 'SFO'.
                destination: The IATA code for the destination city, such as 'JFK'.
                flight_type: Type of the trip ('one_way' or 'round_trip').
                cabin: Cabin class for the reservation ('basic_economy', 'economy', 'business').
                flights: An array of objects containing details about each piece of flight.
                        Each flight should have 'flight_number' (such as 'HAT001') and
                        'date' (in the format 'YYYY-MM-DD', such as '2024-05-01').
                passengers: An array of objects containing details about each passenger.
                        Each passenger should have 'first_name' (such as 'Noah'),
                        'last_name' (such as 'Brown'), and 'dob' (date of birth in the
                        format 'YYYY-MM-DD', such as '1990-01-01').
                payment_methods: An array of objects containing details about each payment method.
                                Each payment method should have 'payment_id' (such as 'credit_card_7815826',
                                'gift_card_7815826', 'certificate_7815826') and 'amount' (the amount to be paid).
                total_baggages: The total number of baggage items included in the reservation.
                nonfree_baggages: The number of non-free baggage items included in the reservation.
                insurance: Indicates whether travel insurance is added ('yes' or 'no').

            Returns:
                str: A JSON string of the reservation details if booking is successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """

            action = Action(
                name="book_reservation",
                kwargs={
                    "user_id": user_id,
                    "origin": origin,
                    "destination": destination,
                    "flight_type": flight_type,
                    "cabin": cabin,
                    "flights": flights,
                    "passengers": passengers,
                    "payment_methods": payment_methods,
                    "total_baggages": total_baggages,
                    "nonfree_baggages": nonfree_baggages,
                    "insurance": insurance,
                },
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def calculate(expression: str) -> str:
            """
            Calculate the result of a mathematical expression.

            Args:
                expression: The mathematical expression to calculate, such as '2 + 2'.
                        The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

            Returns:
                str: The result of the calculation or an error message if the calculation fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(name="calculate", kwargs={"expression": expression})
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def cancel_reservation(reservation_id: str) -> str:
            """
            Cancel the whole reservation.

            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.

            Returns:
                str: Confirmation message if cancellation is successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="cancel_reservation", kwargs={"reservation_id": reservation_id}
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def get_reservation_details(reservation_id: str) -> str:
            """
            Get the details of a reservation.

            Args:
                reservation_id: The reservation id, such as '8JX2WO'.

            Returns:
                str: A JSON string of the reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="get_reservation_details",
                kwargs={"reservation_id": reservation_id},
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def get_user_details(user_id: str) -> str:
            """
            Get the details of an user, including their reservations.

            Args:
                user_id: The user id, such as 'sara_doe_496'.

            Returns:
                str: A JSON string of the user details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(name="get_user_details", kwargs={"user_id": user_id})
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def list_all_airports() -> str:
            """
            List all airports and their cities.

            Returns:
                str: A JSON string containing all airports and their cities if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(name="list_all_airports", kwargs={})
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def search_direct_flight(origin: str, destination: str, date: str) -> str:
            """
            Search direct flights between two cities on a specific date.

            Args:
                origin: The origin city airport in three letters, such as 'JFK'.
                destination: The destination city airport in three letters, such as 'LAX'.
                date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.

            Returns:
                str: A JSON string of available direct flights if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="search_direct_flight",
                kwargs={"origin": origin, "destination": destination, "date": date},
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def search_onestop_flight(origin: str, destination: str, date: str) -> str:
            """
            Search direct flights between two cities on a specific date.

            Args:
                origin: The origin city airport in three letters, such as 'JFK'.
                destination: The destination city airport in three letters, such as 'LAX'.
                date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.

            Returns:
                str: A JSON string of available one-stop flights if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="search_onestop_flight",
                kwargs={"origin": origin, "destination": destination, "date": date},
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def send_certificate(user_id: str, amount: float) -> str:
            """
            Send a certificate to a user. Be careful!

            Args:
                user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
                amount: Certificate amount to send.

            Returns:
                str: Confirmation message if the certificate is sent successfully, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="send_certificate", kwargs={"user_id": user_id, "amount": amount}
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def think(thought: str) -> str:
            """
            Use the tool to think about something. It will not obtain new information or change the database,
            but just append the thought to the log. Use it when complex reasoning is needed.

            Args:
                thought: A thought to think about.

            Returns:
                str: Confirmation that the thought was logged.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(name="think", kwargs={"thought": thought})
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def transfer_to_human_agents(summary: str) -> str:
            """
            Transfer the user to a human agent, with a summary of the user's issue.
            Only transfer if the user explicitly asks for a human agent, or if the user's issue
            cannot be resolved by the agent with the available tools.

            Args:
                summary: A summary of the user's issue.

            Returns:
                str: Confirmation that the user was transferred to a human agent.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="transfer_to_human_agents", kwargs={"summary": summary}
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def update_reservation_baggages(
            reservation_id: str,
            total_baggages: int,
            nonfree_baggages: int,
            payment_id: str,
        ) -> str:
            """
            Update the baggage information of a reservation.

            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                total_baggages: The updated total number of baggage items included in the reservation.
                nonfree_baggages: The updated number of non-free baggage items included in the reservation.
                payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.

            Returns:
                str: Updated reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="update_reservation_baggages",
                kwargs={
                    "reservation_id": reservation_id,
                    "total_baggages": total_baggages,
                    "nonfree_baggages": nonfree_baggages,
                    "payment_id": payment_id,
                },
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def update_reservation_flights(
            reservation_id: str,
            cabin: str,
            flights: List[Dict[str, str]],
            payment_id: str,
        ) -> str:
            """
            Update the flight information of a reservation.

            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                cabin: Cabin class for the reservation ('basic_economy', 'economy', 'business').
                flights: An array of objects containing details about each piece of flight in the ENTIRE new reservation.
                        Even if the a flight segment is not changed, it should still be included in the array.
                payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.

            Returns:
                str: Updated reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="update_reservation_flights",
                kwargs={
                    "reservation_id": reservation_id,
                    "cabin": cabin,
                    "flights": flights,
                    "payment_id": payment_id,
                },
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def update_reservation_passengers(
            reservation_id: str, passengers: List[Dict[str, str]]
        ) -> str:
            """
            Update the passenger information of a reservation.

            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                passengers: An array of objects containing details about each passenger including 'first_name', 'last_name', and 'dob'.

            Returns:
                str: Updated reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name="update_reservation_passengers",
                kwargs={"reservation_id": reservation_id, "passengers": passengers},
            )
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        @tool
        def ask_user(question: str) -> str:
            """
            Ask the user a question.

            Args:
                question: The question to ask the user.

            Returns:
                str: The user's response.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(name="respond", kwargs={"content": question})
            observation = isolated_env.step(action)
            return (
                observation.observation,
                "User wants to end the conversation"
                if observation.done
                else "User does not want to end the conversation",
            )

        # get instruction from environment
        user_question = isolated_env.reset(input[task_id]["task_index"]).observation
        wiki = isolated_env.wiki
        with open("wiki.md", "w") as f:
            f.write(wiki)
        agent = CodeAgent(
            tools=CORE_TOOLS
            + [
                book_reservation,
                calculate,
                cancel_reservation,
                get_reservation_details,
                get_user_details,
                list_all_airports,
                search_direct_flight,
                search_onestop_flight,
                send_certificate,
                think,
                transfer_to_human_agents,
                update_reservation_baggages,
                update_reservation_flights,
                update_reservation_passengers,
                ask_user,
            ],
            planning_interval=4,
            budget_exceeded_callback=partial(
                check_budget_exceeded, budget=BUDGET, model_name=kwargs["model_name"]
            )
            if BUDGET
            else None,
            max_steps=200,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            model=model,
        )

        agent.python_executor.state["__name__"] = "__main__"
        ### YOUR AGENT CODE HERE ###
        instruction = f"""I added some useful information to the wiki in `wiki.md`. Please read it and then answer the user's question.

User's question: {user_question}
        """
        response = str(agent.run(instruction))
        action = Action(name="respond", kwargs={"content": response})
        observation = isolated_env.step(action)
        print("Final user's response: ", observation)

        save_agent_steps(agent, kwargs, response, task)

        # Collect metrics
        metrics = collect_task_metrics(agent)

        ### WHEN DONE WE RETURN THE ENV STATE ###
        return {
            task_id: {
                "reward": isolated_env.reward,
                "taken_actions": [
                    action.model_dump() for action in isolated_env.actions
                ],
                "task": isolated_env.task.model_dump(),
                "metrics": metrics,
            }
        }

    elif kwargs["benchmark_name"] == "scicode":
        from openai import OpenAI

        def process_problem_code(prob_data: dict, num_steps: int) -> str:
            """Process problem code and return the function header and return line"""
            header_docstring = prob_data["sub_steps"][num_steps - 1]["function_header"]
            return_str = prob_data["sub_steps"][num_steps - 1]["return_line"]
            string = f"{header_docstring}\n\n{return_str}"
            return string

        def process_problem_steps(
            with_background: bool,
            previous_llm_code: list[str],
            problem_data: dict,
            num_steps: int,
        ) -> tuple[str, str]:
            """Process problem data and return previous steps and next steps"""
            output_lines = []
            next_step = []
            for i in range(num_steps - 1):
                output_lines.append(
                    (
                        problem_data["sub_steps"][i]["step_description_prompt"]
                        + "\n"
                        + problem_data["sub_steps"][i]["step_background"]
                    )
                    if with_background
                    else problem_data["sub_steps"][i]["step_description_prompt"]
                )
                output_lines.append(previous_llm_code[i])
                output_lines.append("------")

            next_step.append(
                (
                    problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
                    + "\n"
                    + problem_data["sub_steps"][num_steps - 1]["step_background"]
                )
                if with_background
                else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"]
            )
            next_step.append(process_problem_code(problem_data, num_steps))
            output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
            next_step_str = "\n\n".join(next_step)
            return output_str, next_step_str

        def generate_prompt_with_steps(
            with_background: bool,
            previous_llm_code: list[str],
            prob_data: dict,
            num_steps: int,
            prompt_template: str,
        ) -> tuple[str, str]:
            """Generate prompt with steps for scicode and scicode easy benchmark"""
            # Parse the input file and extract the content
            problem_steps_str, next_step_str = process_problem_steps(
                with_background, previous_llm_code, prob_data, num_steps
            )
            dependencies = prob_data["required_dependencies"]
            assert next_step_str
            return prompt_template.format(
                problem_steps_str=problem_steps_str,
                next_step_str=next_step_str,
                dependencies=dependencies,
            ), f"{dependencies}\n"

        # Get the benchmark name from kwargs
        benchmark_name = kwargs["benchmark_name"]

        # Initialize results dictionary
        results = {}

        prompt_template = """
        PROBLEM DESCRIPTION:
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
5. Ensure your response is in the format of ```python```.

Example:
```python

[Insert the Python code here based on the provided function header and dependencies.]
```"""

        easy = True if benchmark_name == "scicode_easy" else False

        # Iterate through problems
        previous_llm_code = []
        full_code = ""
        steps = len(task["sub_steps"])
        print(f"Generating {task_id}...")
        steps_results = {}

        for i in range(steps):
            if task_id == "13" and i == 5:
                step_code = '''\
    class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out):
        """Constructor sets up coordinates, memory for variables.
        The variables:
            mesh points:
                x: the x coordinate for each mesh grid
                y: the y coordinate for each mesh grid
                z: the z coordinate for each mesh grid
                t: the time coordinate of the simulation
                r: the distance to the origin for each mesh grid
            evolving fields:
                E_x: the x component of the field E
                E_y: the y componnet of the field E
                E_z: the z component of the field E
                A_x: the x component of the field A
                A_y: the y component of the field A
                A_z: the z component of the field A
                phi: the scalar potential field phi values
            monitor variables:
                constraint: the current constraint violation value from the evolving fields.
                
        """

        self.n_grid = n_grid
        self.n_vars = 7
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta

        self.x      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[:,None,None]
        self.y      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,:,None]
        self.z      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,None,:]
        self.r      = np.sqrt(self.x**2+self.y**2+self.z**2)
        

        # set up all variables common to both approaches
        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))

        
        self.t = 0.0
'''
                previous_llm_code.append(step_code)
                full_code += f"\n{step_code}"
                steps_results[f"{task_id}.{i + 1}"] = full_code
                continue
            elif task_id == "62" and i == 0:
                step_code = """
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
                print(f"{key}:\n{matrix}\n")
            else:
                print(f"{key}:\n{matrix.toarray()}\n")

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
                print(f"{key}:\n{matrix}\n")
            else:
                print(f"{key}:\n{matrix.toarray()}\n")
"""
                previous_llm_code.append(step_code)
                full_code += f"\n{step_code}"
                steps_results[f"{task_id}.{i + 1}"] = full_code
                continue
            elif task_id == "76" and i == 2:
                step_code = """
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
"""
                previous_llm_code.append(step_code)
                full_code += f"\n{step_code}"
                steps_results[f"{task_id}.{i + 1}"] = full_code
                continue

            prompt, dependencies = generate_prompt_with_steps(
                with_background=easy,
                previous_llm_code=previous_llm_code,
                prob_data=task,
                num_steps=i + 1,
                prompt_template=prompt_template,
            )

            response = agent.run(prompt)
            response = str(response)
            generated_code = (
                response.replace("```python", "").replace("```", "").strip()
            )

            # Update previous_llm_code string with the generated code
            previous_llm_code.append(generated_code)
            full_code += f"\n{generated_code}"

            # Store the generated code for the current step
            if easy:
                steps_results[f"{task_id}.{i + 1}"] = full_code
            else:
                steps_results[f"{task_id}.{i + 1}"] = dependencies + full_code

        save_agent_steps(agent, kwargs, steps_results, task)

        return {task_id: steps_results}

    elif kwargs["benchmark_name"] == "assistantbench":
        asstbench_prompt = """Provide a concise and accurate answer to the question below without any additional context in the format suggested by the prompt. Do not include any justification or any additional unnecessary text. Your answer does not need to be a full sentence. If you are unsure what the final answer is, generate an empty string. The answer should either be: a number, a string, a list of strings, or a list of jsons. The answer should be parsed with the python method: json.loads(input_str). If no answer is found, generate an empty string. If the prompt includes a specified answer format, respect that format.

[BEGIN QUESTION]
{}
[END QUESTION]
"""
        prompt = asstbench_prompt.format(task["task"])
        response = agent.run(prompt)

        # Collect metrics
        metrics = collect_task_metrics(agent)

        save_agent_steps(agent, kwargs, response, task)

        return {
            task_id: {
                "answer": response,
                "metrics": metrics,
            }
        }

    else:
        raise ValueError(
            f"Unknown benchmark. HAL agent does not support this benchmark: {kwargs['benchmark_name']}"
        )

    results[task_id] = response

    return results

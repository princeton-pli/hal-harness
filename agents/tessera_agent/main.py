import json
import os

from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    PythonInterpreterTool,
    VisitWebpageTool,
)

load_dotenv()

AUTHORIZED_IMPORTS = [
    "requests",
    "json",
    "math",
    "re",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "pandas",
    "numpy",
    "sympy",
    "bs4",
    "io",
    "os",
    "pathlib",
    "csv",
    "statistics",
]

# Matches the GAIA answer-format instructions used by hal_generalist_agent
GAIA_INSTRUCTION = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

"""


def _build_prompt(task: dict, system_prompt: str, benchmark_name: str) -> str:
    if benchmark_name == "gaia":
        attachment_hint = ""
        if task.get("file_name"):
            attachment_hint = (
                f"\nAttached file in your current directory: {task['file_name']}\n"
            )
        context = f"Additional context: {system_prompt}\n\n" if system_prompt else ""
        return f"{GAIA_INSTRUCTION}{context}Here is the question:{attachment_hint}\n{task['Question']}"
    # Generic fallback for other benchmarks
    context = f"Additional context: {system_prompt}\n\n" if system_prompt else ""
    return context + json.dumps(task)


def run(input: dict, **kwargs) -> dict:
    assert "config_path" in kwargs, "config_path is required"
    assert len(input) == 1, "input must contain only one task"

    with open(kwargs["config_path"]) as f:
        config = json.load(f)

    assert "base_model" in config, "config must contain base_model"

    base_model = config["base_model"]
    system_prompt = config.get("system_prompt", "")
    benchmark_name = kwargs.get("benchmark_name", "gaia")

    task_id, task = list(input.items())[0]

    model = LiteLLMModel(model_id=base_model, temperature=0)

    tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        PythonInterpreterTool(),
    ]

    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=15,
        planning_interval=4,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
    )

    prompt = _build_prompt(task, system_prompt, benchmark_name)
    response = agent.run(prompt)
    return {task_id: str(response).strip()}

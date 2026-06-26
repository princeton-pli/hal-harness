import contextlib
import json

from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    MCPClient,
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

GAIA_INSTRUCTION = """Please answer the question below. You should:

- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.
- If the answer is a number, return only the number without any units unless specified otherwise.
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).
- If the answer is a comma separated list, apply the above rules to each element in the list.

"""


def _build_prompt(task: dict, system_prompt: str, benchmark_name: str, config_file_name: str = "") -> str:
    config_hint = (
        f'\nIMPORTANT: Before doing anything else, read your configuration file using:'
        f'\n  import pathlib; instructions = pathlib.Path("{config_file_name}").read_text()'
        f'\nFollow all instructions in that file for every task.\n'
    ) if config_file_name else ""
    if benchmark_name == "gaia":
        attachment_hint = ""
        if task.get("file_name"):
            attachment_hint = (
                f"\nAttached file in your current directory: {task['file_name']}\n"
            )
        context = f"Additional context: {system_prompt}\n\n" if system_prompt else ""
        return f"{GAIA_INSTRUCTION}{context}Here is the question:{attachment_hint}{config_hint}\n{task['Question']}"
    context = f"Additional context: {system_prompt}\n\n" if system_prompt else ""
    return context + config_hint + json.dumps(task)


def _connect_mcp_servers(tools_config: list, stack: contextlib.ExitStack) -> list:
    """Connects to each MCP server in tools_config, returns aggregated tool list."""
    mcp_tools = []
    for tool_cfg in tools_config:
        if tool_cfg.get("type") != "mcp":
            continue
        name = tool_cfg.get("name", "<unnamed>")
        transport = tool_cfg.get("transport", "sse")
        try:
            if transport == "sse":
                url = tool_cfg.get("url")
                if not url:
                    print(f"[tessera_agent] Skipping '{name}': no url for sse transport")
                    continue
                server_params = {"url": url}
            elif transport == "stdio":
                command = tool_cfg.get("command")
                if not command:
                    print(f"[tessera_agent] Skipping '{name}': no command for stdio transport")
                    continue
                from mcp import StdioServerParameters
                server_params = StdioServerParameters(
                    command=command,
                    args=tool_cfg.get("args", []),
                )
            else:
                print(f"[tessera_agent] Skipping '{name}': unknown transport '{transport}'")
                continue

            tools = stack.enter_context(MCPClient(server_params))
            mcp_tools.extend(tools)
            print(f"[tessera_agent] Connected to '{name}' via {transport} — {len(tools)} tool(s) available")
        except Exception as exc:
            print(f"[tessera_agent] Failed to connect to '{name}': {exc}")

    return mcp_tools


def run(input: dict, **kwargs) -> dict:
    assert "config_path" in kwargs, "config_path is required"
    assert len(input) == 1, "input must contain only one task"

    with open(kwargs["config_path"]) as f:
        config = json.load(f)

    assert "base_model" in config, "config must contain base_model"

    base_model = config["base_model"]
    system_prompt = config.get("system_prompt", "")
    tools_config = config.get("tools") or []
    benchmark_name = kwargs.get("benchmark_name", "gaia")
    config_file_name = config.get("config_file_name", "")
    config_file_content = config.get("config_file_content", "")

    if config_file_name and config_file_content:
        with open(config_file_name, "w", encoding="utf-8") as fh:
            fh.write(config_file_content)

    task_id, task = list(input.items())[0]
    prompt = _build_prompt(task, system_prompt, benchmark_name, config_file_name)

    model = LiteLLMModel(model_id=base_model, temperature=0)

    base_tools = [
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        PythonInterpreterTool(authorized_imports=AUTHORIZED_IMPORTS),
    ]

    with contextlib.ExitStack() as stack:
        mcp_tools = _connect_mcp_servers(tools_config, stack)
        agent = CodeAgent(
            tools=base_tools + mcp_tools,
            model=model,
            max_steps=15,
            planning_interval=4,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
        )
        response = agent.run(prompt)

    return {task_id: str(response).strip()}

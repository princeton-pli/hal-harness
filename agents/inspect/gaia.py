from textwrap import dedent
from inspect_ai.solver import Solver, basic_agent, system_message
from inspect_ai.tool import bash, python, web_browser

def default_agent(
    max_attempts: int = 5, max_messages: int = 75, code_timeout: int = 180, model_name: str = "openai/gpt-4o"
) -> Solver:
    return basic_agent(
        init=system_message(
            dedent(
                """
                You are an autonomous AI agent, consisting of a large language model and a scaffolding program that runs on a linux VM.

                You have full control over your VM through the scaffolding program's provided functions, including: writing and editing files, running bash commands, installing packages etc.. From your VM, you can access the internet.

                You are state-of-the-art, and excellent at analysis, planning and programming. You possess encyclopedic knowledge of the world. You write in a very direct and concise style.

                You are resourceful and adaptive, and you never give up. Before deciding something can't be done, you try it out. You consider multiple options and choose the best one. If your current approach doesn't work, you formulate a new plan. You are given a task you need to solve completely on your own.

                Please think step by step before calling tools. When you are ready to answer, use the submit tool to provide your final answer.
                """
            )
        ),
        tools=[bash(code_timeout), python(code_timeout)] + web_browser(),
        max_attempts=max_attempts,
        max_messages=max_messages,
    )
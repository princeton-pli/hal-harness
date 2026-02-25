from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from browser_use import Agent, Browser, BrowserConfig
import asyncio
import os

import warnings
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings(
    "ignore", category=PydanticDeprecatedSince20, message=r".*__fields__.*deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=PydanticDeprecatedSince20,
    message=r".*__fields_set__.*deprecated.*",
)


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"

    model_name = kwargs["model_name"].split("/")[-1]
    effort = (kwargs.get("reasoning_effort") or "").lower()
    EFFORT_TO_TOKENS = {"low": 1024, "medium": 2048, "high": 4096}

    if kwargs["model_name"].startswith("openrouter/"):
        # Strip the 'openrouter/' prefix; pass the rest directly to OpenRouter
        or_model = kwargs["model_name"].replace("openrouter/", "", 1)
        extra_body = None
        if effort in {"low", "medium", "high"}:
            EFFORT_TO_TOKENS = {"low": 1024, "medium": 2048, "high": 4096}
            extra_body = {
                "reasoning": {"max_tokens": EFFORT_TO_TOKENS[effort]},
                "include_reasoning": True,
            }

        llm = ChatOpenAI(
            model=or_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            extra_body=extra_body,
        )

    elif "openai" in kwargs["model_name"]:
        if "-2025" in kwargs["model_name"]:
            model_name = model_name.split("-2025")[0]
        api_key = os.getenv("OPENAI_API_KEY")
        llm_args = {"model": model_name, "api_key": api_key}
        if effort in {"low", "medium", "high"}:
            llm_args["reasoning_effort"] = effort
        if ("o3" in kwargs["model_name"]) or ("o4" in kwargs["model_name"]):
            llm = ChatOpenAI(**llm_args, disabled_params={"parallel_tool_calls": None})
        else:
            llm = ChatOpenAI(**llm_args)
    elif "anthropic" in kwargs["model_name"]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        llm = ChatAnthropic(model=model_name, api_key=api_key)
    elif "gemini" in kwargs["model_name"]:
        # Route Gemini models through OpenRouter with correct model name mapping
        if model_name == "gemini-2.0-flash":
            or_model = "google/gemini-2.0-flash-001"
        else:
            or_model = f"google/{model_name}"

        llm = ChatOpenAI(
            model=or_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    elif "deepseek" in kwargs["model_name"]:
        llm = ChatOpenAI(
            model=or_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            extra_body={"include_reasoning": True},
            disabled_params={"parallel_tool_calls": None},
        )
    else:
        raise ValueError(f"Unrecognized model_name: {kwargs['model_name']}")

    task_id = list(input.keys())[0]
    task_data = input[task_id]
    question = task_data["task"]

    async def _main():
        browser = Browser(config=BrowserConfig(headless=True))
        if "deepseek" in kwargs["model_name"]:
            agent = Agent(
                task=question,
                message_context="Provide a concise and accurate answer to the question below without any additional context in the format suggested by the prompt. Do not include any justification or any additional unnecessary text. Your answer does not need to be a full sentence. If you are unsure what the final answer is, generate an empty string. The answer should either be: a number, a string, a list of strings, or a list of jsons. The answer should be parsed with the python method: json.loads(input_str). If no answer is found, generate an empty string. If the prompt includes a specified answer format, respect that format.",
                llm=llm,
                use_vision=False,
                # planner_llm=planner_llm,
                # use_vision_for_planner=False,
                # planner_interval=2,
                browser=browser,
            )
        else:
            agent = Agent(
                task=question,
                message_context="Provide a concise and accurate answer to the question below without any additional context in the format suggested by the prompt. Do not include any justification or any additional unnecessary text. Your answer does not need to be a full sentence. If you are unsure what the final answer is, generate an empty string. The answer should either be: a number, a string, a list of strings, or a list of jsons. The answer should be parsed with the python method: json.loads(input_str). If no answer is found, generate an empty string. If the prompt includes a specified answer format, respect that format.",
                llm=llm,
                # planner_llm=planner_llm,
                # use_vision_for_planner=False,
                # planner_interval=2,
                browser=browser,
            )
        try:
            result = await agent.run(max_steps=20)
        finally:
            await browser.close()
        return result

    run_result = asyncio.run(_main())
    answer = run_result.final_result()

    # if answer is null or empty, return empty string
    if answer is None:
        answer = ""

    return {task_id: answer}

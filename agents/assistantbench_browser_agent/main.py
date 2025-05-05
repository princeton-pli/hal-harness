# This is an example agent that uses smolagents to generate answers for the AssistantBench benchmark.
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from browser_use import Agent, Browser, BrowserConfig
import asyncio
import os

import warnings
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings(
    "ignore",
    category=PydanticDeprecatedSince20,
    message=r".*__fields__.*deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=PydanticDeprecatedSince20,
    message=r".*__fields_set__.*deprecated.*"
)


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'

    model_name = kwargs['model_name'].split('/')[-1]

    if "openai" in kwargs['model_name']:
        if '-2025' in kwargs['model_name']:
            model_name = model_name.split('-2025')[0]
            api_key = os.getenv("OPENAI_API_KEY")
            llm_args = {"model": model_name, "api_key": api_key}
            if "medium" or "high" or "low" in kwargs['reasoning_effort']:
                llm_args["reasoning_effort"] = kwargs['reasoning_effort']
            if "o3" or "o4" in kwargs['model_name']:
                llm = ChatOpenAI(**llm_args, disabled_params={"parallel_tool_calls": None})
                planner_llm = ChatOpenAI(**llm_args, disabled_params={"parallel_tool_calls": None})
            else:
                llm = ChatOpenAI(**llm_args)
                planner_llm = ChatOpenAI(**llm_args)
    elif 'anthropic' in kwargs['model_name']:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        llm = ChatAnthropic(model=model_name, api_key=api_key)
        planner_llm = ChatAnthropic(model=model_name, api_key=api_key)
    elif 'gemini' in kwargs['model_name']:
        api_key = os.getenv("GEMINI_API_KEY")
        llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
        planner_llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif 'deepseek' in kwargs['model_name']:
        api_key = os.getenv("TOGETHERAI_API_KEY")
        llm = ChatTogether(model=kwargs['model_name'], api_key=api_key)
        planner_llm = ChatTogether(model=kwargs['model_name'], api_key=api_key)
    else:
        raise ValueError(f"Unrecognized model_name: {kwargs['model_name']}")

    
    task_id = list(input.keys())[0]
    task_data = input[task_id]
    question = task_data["task"]

    async def _main():
        browser = Browser(
        config=BrowserConfig(headless=True)
    )
        agent = Agent(
            task=question,
            message_context="Provide a concise and accurate answer to the question below without any additional context in the format suggested by the prompt. Do not include any justification or any additional unnecessary text. Your answer does not need to be a full sentence. If you are unsure what the final answer is, generate an empty string. The answer should either be: a number, a string, a list of strings, or a list of jsons. The answer should be parsed with the python method: json.loads(input_str). If no answer is found, generate an empty string. If the prompt includes a specified answer format, respect that format.",
            llm=llm,
            # planner_llm=planner_llm,
            # use_vision_for_planner=False,
            # planner_interval=2,
            browser=browser
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
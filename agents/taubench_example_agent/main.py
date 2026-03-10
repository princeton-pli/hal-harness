from tau_bench.envs import get_env
from tau_bench.types import Action

from openai import OpenAI


def _extract_response_text(response) -> str:
    """Best-effort text extraction for OpenAI Responses API outputs."""
    if getattr(response, "output_text", None):
        return response.output_text.strip()

    output_items = getattr(response, "output", []) or []
    text_chunks: list[str] = []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if (
                getattr(content, "type", None) == "output_text"
                and getattr(content, "text", None)
            ):
                text_chunks.append(content.text)

    return "".join(text_chunks).strip()


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    client = OpenAI()
    task_id = list(input.keys())[0]

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]["env"],
        input[task_id]["user_strategy"],
        input[task_id]["user_model"],
        input[task_id]["task_split"],
        input[task_id]["user_provider"],
        input[task_id]["task_index"],
    )
    # get instruction from environment
    instruction = isolated_env.reset(input[task_id]["task_index"]).observation

    ### YOUR AGENT CODE HERE ###
    request_kwargs = {
        "model": kwargs["model_name"],
        "input": [
            {"role": "user", "content": instruction},
        ],
        "max_output_tokens": 2000,
        "temperature": 1,
    }
    if "reasoning_effort" in kwargs:
        request_kwargs["reasoning"] = {"effort": kwargs["reasoning_effort"]}

    response = client.responses.create(**request_kwargs)

    ### ACTION ###
    action_text = _extract_response_text(response)
    action = Action(name=action_text, kwargs={})
    response = isolated_env.step(action)

    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {
        task_id: {
            "reward": isolated_env.reward,
            "taken_actions": [action.model_dump() for action in isolated_env.actions],
            "task": isolated_env.task.model_dump(),
        }
    }

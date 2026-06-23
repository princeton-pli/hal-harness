import json
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "config_path" in kwargs, "config_path is required"
    assert len(input) == 1, "input must contain only one task"

    with open(kwargs["config_path"]) as f:
        config = json.load(f)

    assert "base_model" in config, "config must contain base_model"
    assert "system_prompt" in config, "config must contain system_prompt"

    task_id, task = list(input.items())[0]

    response = litellm.completion(
        model=config["base_model"],
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": json.dumps(task)},
        ],
        temperature=0,
    )
    return {task_id: response.choices[0].message.content.strip()}

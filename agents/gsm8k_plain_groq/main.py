import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    response = litellm.completion(
        model=kwargs["model_name"],
        messages=[
            {
                "role": "user",
                "content": (
                    f"{task['question']}\n\n"
                    "Answer with a single number only. No units, no explanation."
                ),
            }
        ],
        max_tokens=64,
        temperature=0,
    )
    return {task_id: response.choices[0].message.content.strip()}

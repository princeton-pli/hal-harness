from openai import OpenAI


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    response = client.chat.completions.create(
        model=kwargs["model_name"],
        messages=[{"role": "user", "content": "Solve the following problem: " + task["problem_statement"]}],
        max_tokens=2000,
        temperature=0,
    )

    return {task_id: response.choices[0].message.content}

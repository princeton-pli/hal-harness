import re
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True

SYSTEM_PROMPT = """Think through the problem step by step before giving your final answer.

At the very end of your response, write your answer in this exact format:
FINAL ANSWER: <answer>

Rules for the final answer:
- Numbers: digits only, no commas (e.g. 17500 not 17,500)
- Keep it concise — just the value, no explanation"""


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    response = litellm.completion(
        model=kwargs["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["Question"]},
        ],
        max_tokens=1024,
        temperature=0,
    )
    text = response.choices[0].message.content.strip()

    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    else:
        answer = text.split("\n")[-1].strip()

    return {task_id: answer}

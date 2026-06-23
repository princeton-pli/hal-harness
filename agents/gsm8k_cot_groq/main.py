import re
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True

SYSTEM_PROMPT = """Solve the math problem step by step.

At the very end of your response, write your answer in this exact format:
FINAL ANSWER: <number>

Rules for the final answer:
- Digits only, no commas, no units (e.g. 17500 not 17,500)"""


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    response = litellm.completion(
        model=kwargs["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["question"]},
        ],
        max_tokens=512,
        temperature=0,
    )
    text = response.choices[0].message.content.strip()

    match = re.search(r"FINAL ANSWER:\s*([\d,.-]+)", text, re.IGNORECASE)
    answer = match.group(1).replace(",", "").strip() if match else text.split("\n")[-1].strip()

    return {task_id: answer}

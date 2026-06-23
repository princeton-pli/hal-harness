import os
import re
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True


SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer the question as accurately as possible.

At the end of your response, output your final answer on its own line in this exact format:
FINAL ANSWER: <your answer>

Rules for the final answer:
- Numbers: use digits, no commas (e.g. 17500 not 17,500)
- Dates: use the format requested, or YYYY-MM-DD if not specified
- Lists: comma-separated
- Keep it short — just the answer, no explanation"""


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]
    question = task["Question"]

    response = litellm.completion(
        model=kwargs["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=1024,
        temperature=0,
    )
    text = response.choices[0].message.content.strip()

    # Extract FINAL ANSWER tag
    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    else:
        # Fallback: ask the model to extract the answer
        extract = litellm.completion(
            model=kwargs["model_name"],
            messages=[
                {"role": "user", "content": f"Extract only the final answer from this response, as a short phrase or number:\n\n{text}"},
            ],
            max_tokens=64,
            temperature=0,
        )
        answer = extract.choices[0].message.content.strip()

    return {task_id: answer}

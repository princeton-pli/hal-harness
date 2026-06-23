import re
from openai import OpenAI


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Stage 1: reason through the problem
    reasoning_prompt = f"""You are a precise question-answering assistant. Think through the following question step by step, then give your final answer.

Rules for the final answer:
- It must be a number, a short phrase, or a comma-separated list
- Numbers: no units unless specified
- Strings: no articles, no abbreviations
- End your response with exactly: FINAL ANSWER: <your answer>

Question: {task["Question"]}"""

    reasoning_response = client.chat.completions.create(
        model=kwargs["model_name"],
        messages=[{"role": "user", "content": reasoning_prompt}],
        max_tokens=1000,
        temperature=0.2,
    )
    reasoning_text = reasoning_response.choices[0].message.content

    # Extract FINAL ANSWER if present, otherwise ask for just the answer
    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", reasoning_text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
    else:
        # Stage 2: extract answer from reasoning if model didn't follow format
        extraction_response = client.chat.completions.create(
            model=kwargs["model_name"],
            messages=[
                {"role": "user", "content": reasoning_prompt},
                {"role": "assistant", "content": reasoning_text},
                {"role": "user", "content": "Based on your reasoning above, give only the final answer with no explanation. Just the number, word, or short phrase."},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        answer = extraction_response.choices[0].message.content.strip()

    return {task_id: answer}

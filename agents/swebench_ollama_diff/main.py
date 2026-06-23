import re
from openai import OpenAI


SYSTEM_PROMPT = """You are an expert software engineer. When given a bug report or feature request, you produce a minimal, correct git patch in unified diff format.

Rules:
- Output ONLY the patch — no explanation, no markdown fences, no prose
- Use standard unified diff format: --- a/path/to/file, +++ b/path/to/file, @@ -L,S +L,S @@
- Make the smallest change that fixes the issue
- Preserve existing code style and indentation"""


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    # Stage 1: understand the problem and identify the file/function to change
    analysis = client.chat.completions.create(
        model=kwargs["model_name"],
        messages=[
            {"role": "system", "content": "You are an expert software engineer analyzing a bug report. Be concise."},
            {"role": "user", "content": f"Repository: {task['repo']}\n\nBug report:\n{task['problem_statement']}\n\nIn 2-3 sentences: what file(s) likely need changing and what is the fix?"},
        ],
        max_tokens=300,
        temperature=0,
    )
    analysis_text = analysis.choices[0].message.content

    # Stage 2: produce the patch
    patch_response = client.chat.completions.create(
        model=kwargs["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Repository: {task['repo']}\n\nBug report:\n{task['problem_statement']}\n\nAnalysis: {analysis_text}\n\nWrite the git patch:"},
        ],
        max_tokens=1500,
        temperature=0,
    )
    patch = patch_response.choices[0].message.content.strip()

    # Strip markdown fences if the model wrapped the diff anyway
    patch = re.sub(r"^```(?:diff|patch)?\n?", "", patch, flags=re.MULTILINE)
    patch = re.sub(r"\n?```$", "", patch, flags=re.MULTILINE)

    return {task_id: patch.strip()}

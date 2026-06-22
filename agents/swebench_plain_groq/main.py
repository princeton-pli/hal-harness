import re
import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True

SYSTEM_PROMPT = """You are an expert software engineer. Given a bug report, produce a minimal git patch in unified diff format.

Output ONLY the patch — no explanation, no markdown fences, no prose.
Use standard unified diff format: --- a/path, +++ b/path, @@ -L,S +L,S @@
Make the smallest change that fixes the issue."""


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert len(input) == 1, "input must contain only one task"

    task_id, task = list(input.items())[0]

    response = litellm.completion(
        model=kwargs["model_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Repository: {task['repo']}\n\nBug report:\n{task['problem_statement']}\n\nWrite the git patch:"},
        ],
        max_tokens=1500,
        temperature=0,
    )
    patch = response.choices[0].message.content.strip()

    patch = re.sub(r"^```(?:diff|patch)?\n?", "", patch, flags=re.MULTILINE)
    patch = re.sub(r"\n?```$", "", patch, flags=re.MULTILINE)

    return {task_id: patch.strip()}

# Disclaimer: this is not a functional agent and is only for demonstration purposes. This implementation is just a single model call.
from inspect_ai.tool import ToolError
from inspect_ai.util import sandbox
from typing import Any
    
    
from openai import AsyncOpenAI



async def list_files(dir: str):
    """List the files in a directory.

    Args:
        dir (str): Directory

    Returns:
        File listing of the directory
    """
    result = await sandbox().exec(["ls", dir])
    if result.success:
        return result.stdout
    else:
        raise ToolError(result.stderr)


async def run(sample: dict[str, Any], **kwargs) -> dict[str, Any]:
    
    # Example command executed in the sandbox
    result = await list_files(".")
    
    client = AsyncOpenAI()
    completion = await client.chat.completions.create(
        model="inspect",
        messages=sample["input"],
    )
    
    return {
        "output": str(result)
    }
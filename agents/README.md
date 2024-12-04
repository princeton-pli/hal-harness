# Adding New Agents to HAL

This guide explains how to add new agents to the HAL evaluation framework. An agent is a program that can solve tasks from one or more benchmarks.

## Agent Structure

Each agent should be in its own directory under `agents/` with the following structure:

```
agents/
  └── your_agent_name/
      ├── main.py          # Contains the main agent logic
      ├── requirements.txt # Python dependencies
      └── (other files)    # Additional files needed by your agent
```

## Agent Interface

Your agent must implement a `run` function with the following signature:

```python
def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Args:
        input: Dictionary mapping task IDs to task data
        **kwargs: Additional arguments passed via -A flags
    
    Returns:
        Dictionary mapping task IDs to solution strings
    """
```

## Example Agent

Here's a complete example of a minimal agent for the USACO benchmark:

```python
from openai import OpenAI

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # Recommended: Verify that required arguments are provided
    assert 'model_name' in kwargs, 'model_name is required'
    
    client = OpenAI()
    results = {}

    # Process each task
    for task_id, task in input.items(): # usually only one task is provided to each agent and tasks are run in parallel
        response = client.chat.completions.create(
            model=kwargs['model_name'],
            messages=[
                {"role": "user", "content": task['prompt']},
            ],
            max_tokens=2000,
            temperature=1,
        )
        results[task_id] = response.choices[0].message.content
        
    return results
```

## Key Features

1. **Input Format**: The `input` dictionary contains task IDs as keys and task data as values. The structure of task data varies by benchmark.

2. **Arguments**: Additional arguments can be passed to your agent using `-A` flags when running HAL:
   ```bash
   hal run --agent_name "my_agent" --agent_function "main.run" --agent_dir "agents/my_agent" -A model_name=gpt-4 -A temperature=0.7
   ```

3. **Requirements**: List all dependencies in `requirements.txt`. These will be installed:
   - In a conda environment if `--conda_env_name` is specified
   - On VMs if `--vm` flag is used
   - Must be manually installed if running locally without conda
   - If the agent requires more setup than just installing dependencies, feel free to run setup code as part of the `run` function.

4. **Output Format**: Your agent must return a dictionary mapping task IDs to solution strings. The format of solutions varies by benchmark:
   - USACO: Code solution as string
   - SWE-bench: Git patch as string
   - For benchmarks on which the entire environment is used for scoring, the agent should return a dictionary with task IDs as keys and e.g "complete" as values. This is in order to allow for continued runs. The entire environment will be saved and used for scoring.

## Advanced Features

1. **File Access**: For benchmarks that provide additional files (like SWE-bench), files are available in the working directory.

2. **Agent trace and cost logging**: The hal harness automatically logs the agent's trace and cost using W&B Weave.

## Benchmark-Specific Requirements

Different benchmarks have different input/output formats:

1. **USACO**:
   ```python
   input = {
       "task_id": {
           "prompt": "Problem description...",
           "test_cases": [{"input": "...", "output": "..."}]
       }
   }
   ```

2. **SWE-bench**:
   ```python
   input = {
       "instance_id": {
           "problem_statement": "Fix description...",
           "repo": "owner/repo",
           "base_commit": "commit_hash",
           "files": {"path/to/file": "local/path/to/file"}
       }
   }
   ```

3. **AppWorld**:
   ```python
   input = {
       "task_id": {
           "instruction": "Task description...",
           "environment": "Environment details..."
       }
   }
   ```

## Testing Your Agent

1. Create a test task:
   ```python
   test_input = {
       "test_task": {
           "prompt": "Write a hello world program"
       }
   }
   ```

2. Run your agent locally:
   ```python
   from your_agent import run
   result = run(test_input, model_name="gpt-4")
   print(result)
   ```

3. Test with HAL:
   ```bash
   hal run --agent_name "test_agent" --agent_function "main.run" --agent_dir "agents/your_agent" --benchmark "usaco" -A model_name=gpt-4o
   ```
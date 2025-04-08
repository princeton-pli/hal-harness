# Adding New Agents to HAL

This guide explains how to develop agents that you can evaluate with the HAL's evaluation harness. It should be really simple and does put (almost) no constraints on the agent's implementation.

## Basic Agent Structure

Each agent should be in its own directory under `agents/` with this structure:

```
agents/
  └── your_agent_name/
      ├── main.py          # Contains the main agent logic
      ├── requirements.txt # Python dependencies
      └── (other files)    # Additional files needed by your agent
```

## Core Agent Interface

Your agent must implement a `run` function with this signature:

```python
def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Args:
        input: Dictionary mapping task IDs to task data
        **kwargs: Additional arguments passed via -A flags
    
    Returns:
        Dictionary mapping task IDs to submissions
    """
```

## General Requirements

1. **Dependencies**: List all dependencies in `requirements.txt`. These will be installed:
   - On VMs if `--vm` flag is used
   - If you run evaluations locally, you must install the dependencies yourself. Then specify the conda environment name with `--conda_env_name` or run evaluations from the conda environment.

2. **Arguments**: Your agent can receive additional arguments via `-A` flags:
   ```bash
   hal-eval -A model_name=gpt-4 -A temperature=0.7 ...
   ```

3. **File Access**: For benchmarks that provide files (like SWE-bench), files are available in the working directory.

4. **Cost Logging**: The harness automatically logs agent traces and costs using Weave. However, you need to make sure to **not spawn new processes or threads that will not be logged by Weave**. Also make sure that your dependencies are compatible with `weave==0.51.41`.

## Benchmark-Specific Requirements

### USACO

**Input Format**:
```python
   input = {
       "task_id": {
           "name": "Good Bitstrings",
            "problem_link": "http://www.usaco.org/index.php?page=viewproblem2&cpid=1333",
            "test_data_link": "http://www.usaco.org/current/data/prob2_platinum_open23.zip",
            "solution_link": "http://www.usaco.org/current/data/sol_prob2_platinum_open23.html",
            "contest_link": "http://www.usaco.org/index.php?page=open23results",
            "inner_contest_link": "http://www.usaco.org/index.php?page=nov11problems",
            "problem_level": "platinum",
            "cp_id": "1333",
            "problem_id": "1333_platinum_good_bitstrings",
            "description": "[Description content truncated for brevity]",
            "num_tests": 21,
            "solution": "[Solution content truncated for brevity]",
            "runtime_limit_sentences": [],
            "memory_limit_sentences": [],
            "runtime_limit": 2,
            "memory_limit": 256,
            "samples": [{
              "input": "6\n1 1\n3 5\n4 7\n8 20\n4 10\n27 21",
              "output": "1\n5\n7\n10\n6\n13",
              "explanation": ""
            }],
            "description_no_samples": "[Description without samples content truncated]",
            "num_samples": 9999,
            "solution_python3": "[Python solution content truncated]",
            "solution_english": "[English solution explanation truncated]"  
       }
   }
   ```

**Output Format**: Return input dictionary with additional `response` key for each task ID that contains the python code solution.

**Example Agent**:
```python
def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'
    
    task_id, task = list(input.items())[0]
    
    client = OpenAI()

    results = {}

    response = client.chat.completions.create(
        model=kwargs['model_name'],
        messages=[
            {"role": "user", "content": "Solve the following problem: " + task['description']},
            ],
        max_tokens=2000,
        n=1,
        temperature=1,
    )
    
    results[task_id] = response.choices[0].message.content
    input[task_id]['response'] = results[task_id]
        
    return input
```

### SWE-bench

**Input Format**:
```python
{
    "instance_id": {
        "repo": "django/django",
        "instance_id": "django__django-11099",
        "base_commit": "[truncated]",
        "patch": "[truncated]",
        "test_patch": "[truncated]",
        "problem_statement": "[truncated]",
        "hints_text": "[truncated]",
        "created_at": "[truncated]",
        "version": "[truncated]",
        "FAIL_TO_PASS": "[truncated]",
        "PASS_TO_PASS": "[truncated]",
        "environment_setup_commit": "[truncated]"
    }
}
```

**Output Format**: Return a dictionary mapping instance IDs to git patch strings.

**Example Agent**:
```python
def run(input: dict, **kwargs):
    patches = {}
    for instance_id, task in input.items():
        # Generate patch to fix the bug
        patch = generate_patch(
            task["problem_statement"])
        patches[instance_id] = patch
    return patches
```

### AppWorld

**Input Format**:

```python
{
    "task_id": {
        "task_id": "[truncated]",
    }
}
```

**Note**: The task instruction can be accessed from the environment by the agent by calling `world.task.instruction`.

**Output Format**: Return a dictionary mapping task IDs to "complete" after modifying the environment.

**Requirements**:
- Must be run with `--vm` flag
- **Important:** set `remote_environment_url` to `http://0.0.0.0:8000` and `experiment_name` to `output`. An example is below and in `agents/appworld_example_agent/main.py`.

**Example Agent**:
```python
from appworld import AppWorld, load_task_ids

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    
    task_id = list(input.keys())[0]
    
    with AppWorld(task_id=task_id, experiment_name="output", remote_environment_url="http://0.0.0.0:8000") as world:
        world.task.instruction # To see task instruction.

        # indicate task completion:
        world.execute("apis.supervisor.complete_task()")
        
    return {task_id: "Completed"}
```

### Inspect AI Benchmarks

Inspect AI benchmarks support two types of agents:

1. **Custom Agents**: Implement custom agents as described in this document that can be run locally or on VMs.
2. **Inspect Solver Agents**: Use the `@solver` decorator from the Inspect AI framework. See the [Inspect AI documentation](https://inspect.ai-safety-institute.org.uk/agents.html) for details on how to implement solver agents.

#### GAIA

**Input Format**:
```python
{
    "sample_id": {
        "id": "[truncated]",
        "input": "[truncated]",
        "choices": null,
        "target": "[truncated]",
        "metadata": {
            "level": "999",
            "Annotator Metadata": {
                "Steps": "[truncated]",
                "Number of steps": "999",
                "How long did this take?": "[truncated]",
                "Tools": "[truncated]",
                "Number of tools": "999"
            }
        },
        "files": {},
        "setup": "[truncated]"
    }
}
```

**Output Format**: Return a dictionary mapping sample IDs to concise answers following the prompt's instructions (typically numbers, short phrases, or comma-separated lists).

#### CyBench

**Input Format**:
```python
{
    "sample_id": {
        "id": "[truncated]",
        "input": "[truncated]",
        "choices": null,
        "target": "[truncated]",
        "metadata": {
            "variant": "[truncated]",
            "challenge_metadata": {
                "first_solve_time": 999,
                "category": "[truncated]",
                "competition": "[truncated]"
            },
            "variant_metadata": {}
        },
        "files": {},
        "setup": null
    }
}
```

**Output Format**: Return a dictionary mapping sample IDs to solutions.

#### AgentHarm (Benign)

**Note**: Currently only supports Inspect solver agents. See [Inspect AI documentation](https://github.com/UKGovernmentBEIS/inspect_ai) for implementation details. Example inspect solver agent is in `agents/inspect/agentharm`.

### Example Custom Agent for Inspect Benchmarks (GAIA and Cybench)

```python
def run(input: dict, **kwargs):
    assert 'model_name' in kwargs, 'model_name is required'
    
    responses = {}
    for sample_id, sample in input.items():
        # Generate response using specified model
        response = generate_response(
            sample["input"],
            model=kwargs['model_name']
        )
        responses[sample_id] = response
    return responses
```

### Example Inspect Solver Agent

Can be found in `agents/inspect/gaia.py` and `agents/inspect/cybench.py`.

**Special Requirements**:
- For custom agents, use `-A` flags to pass keyword arguments
- For Inspect solvers, use `-I` flags to pass arguments to the inspect eval() function as detailed in the [Inspect AI documentation](https://inspect.ai-safety-institute.org.uk/models.html#generation-config)
- For SWE-Agent-v0.7 / Enigma-Agent, please add `keys.cfg` in the SWE-Agent / Enigma-Agent directory. For SWE-Agent-v1.0, please add `.env` in the SWE-Agent-v1.0 directory.

### ScienceAgentBench

**Input Format**:
```python
{
    "task_id": {
        "task_inst": "Task instruction text",
        "dataset_path": "Path to the dataset",
        "dataset_folder_tree": "Folder structure of the dataset",
        "dataset_preview": "Preview of the dataset contents",
        "output_fname": "Expected output filename",
        "domain_knowledge": "Additional domain knowledge",
        "gold_program_name": "Name of the gold standard program",
        "instance_id": "Unique identifier for the instance"
    }
}
```

**Output Format**: Return a dictionary mapping task IDs to solution trajectories that contain the agent's reasoning steps.

**Example Agent**:
```python
def run(input_dict: dict[str, dict], **kwargs) -> dict[str, str]:
    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input_dict) == 1, 'input must contain only one task'

    agent = ScienceAgent(
        kwargs['model_name'],
        context_cutoff=28000,
        use_self_debug=kwargs['use_self_debug'],
        use_knowledge=kwargs['use_knowledge']
    )

    task_id = list(input_dict.keys())[0]
    task = format_task_dict(list(input_dict.values())[0])
    out_fname = "pred_programs/pred_" + task["gold_program_name"]
    trajectory = agent.solve_task(task, out_fname=out_fname)

    return {task_id: trajectory}
```

**Special Requirements**:
- Include `use_self_debug` and `use_knowledge` flags to control agent behavior
- Output programs are stored in "pred_programs/" directory
- Docker is required for evaluation
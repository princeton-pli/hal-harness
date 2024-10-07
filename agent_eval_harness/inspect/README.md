# Using Agent Harness with Inspect Tasks

## Run Inspect Tasks

The agent harness can run any inspect task by providing a path to the task as the `benchmark`. The simplest way to run benchmarks is to use the `inspect_evals` package, which provides a collection of LLM evaluations for Inspect AI.

First install Inspect and Inspect Evals:

``` bash
pip install inspect_ai
pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
```

Then, run the agent-eval:

``` bash
agent-eval --agent_name gaia --benchmark inspect_evals/gaia --model openai/gpt-4o 
```

## Custom Inspect Agents

The agent harness supports using an external Solver as the agent for an Inspect task. You can do this by providing an `agent_function` which points to an Inspect Solver. This solver will be used as the Task's solver. For example:

``` bash
agent-eval --agent_name gaia --benchmark inspect_evals/gaia --model openai/gpt-4o --agent_dir agents/inspect --agent_function solver_agent.basic_agent
```

In the above example, the `basic_agent` solver from the file `agents/inspect/solver_agent.py` will be used as the solver when executing the Task.

Here is a trivial solver which add the `python` tool then runs the base Inspect generate tool loop:

``` python
@solver
def basic_agent(timeout: int = 60) -> Solver:

    async def solve(state: TaskState, generate: Generate):

        # Add tools
        state.tools.append(python(timeout=timeout))

        return await generate(state)

    return solve
```

## Custom External Agents

You may also use a completely external agent to complete the evaluation. In this scenario, the Inspect Task is primarily used to define the dataset and scoring function. When executed, this will read the Task's samples, execute the custom agent (passing the sample data to it), then score the results returned by the agent. To use a custom agent:

``` bash
agent-eval --agent_name gaia --benchmark inspect_evals/gaia --model openai/gpt-4o --agent_dir agents/inspect --agent_function custom_agent.run
```

In the above example, the `run` function from the file `agents/inspect/custom_agent.py` will be used as the solver when executing the Task.

Here is trivial example of a run function to illustrate the input/output semantics:

``` python
def run(tasks: dict[str, dict]) -> dict[str, str]:

    result = {}
    for id, sample in tasks.items():
        # TODO: Call external agent to compute completion
        completion = "Nam venenatis turpis mauris. Donec ut massa vel lacus maximus placerat."
        
        # Add this completion to the result
        result[id] = completion

    return result
```

### Agent Function Signature

The `--agent_function` argument must point to a Python function which expects a single argument, a `dict[str, dict[str, Any]]` which will contain sample data. The dictionary that is passed to the custom agent will include fields from the inspect Dataset, including:

```         
id: (int | str) Unique identifier for sample.
input: (str) The input to be submitted to the model.
choices: (list[str] | None) List of available answer choices (used only for multiple-choice evals).
target: (list[str] | str | None) Ideal target output. May be a literal value or narrative text to be used by a model grader.
metadata: (dict[str,Any] | None) Arbitrary metadata associated with the sample.
files (dict[str, str] | None) Files that go along with the sample
setup (str | None)  Setup script to run for sample
```

The agent function must return a dictionary with a key/value for each sample which is the sample's `id` and the sample's completion (as a string).

## Custom Parameters

You may pass parameters to an Inspect solver `agent_function` or to the task. To do this use `-A <name>=<value>` for agent args (passed to the Inspect Solver), or `-B <name>=<value>` for benchmark args (passed to the Inspect Task). You may repeat these args any number of times.

## Uploading Results

TODO: Test huggingface

TODO: Run gaia level 1

The Inspect runner will also upload results to HuggingFace, respecting the `--upload` key.

## Other Notes

### Code Structure

`inspect_runner.py` - The primary orchestrator for running Inspect evals `inspect/agent.py` - Functions related to running custom agents `inspect/inspect.py` - Functions related to loading and running Inspect Tasks and Inspect Solver agents `inspect/hf.py` - Simple HuggingFace helper function `inspect/log.py` - Simple logging helper functions `inspect/wave.py` - Simple weave helper functions

### Evaluation Flow

The flow is the same the `agent_runner` with the exception of some additional data that will produced in the results directory.

1)  An Inspect log file will be produced which includes the results and metrics, each sample and a complete transcript of the sample trajectory.
2)  The eval summary and result metrics are included in the benchmark `_UPLOAD` file.

##
import importlib
import inspect
from inspect_ai.solver import solver
from typing import Callable, cast

from inspect_ai.dataset import Dataset
from inspect_ai.solver import Generate, Solver, TaskState

def load_agent(agent_function: str) -> Callable:
    # parse the agent name
    module_name, function_name = agent_function.rsplit(".", 1)

    # attempt to load it from the module
    module = importlib.import_module(module_name)
    loaded_agent = getattr(module, function_name)
    return loaded_agent


def validate_agent(agent: Callable) -> None:

    # Get the signature of the function
    sig = inspect.signature(agent)

    # Get the parameters and return annotation
    parameters = sig.parameters
    return_annotation = sig.return_annotation

    # Check the number of parameters (should be exactly one)
    if len(parameters) != 1:
        raise RuntimeError("The agent function should accept only a single argument.")

    # Get the parameter name and annotation
    param_name, param_info = next(iter(parameters.items()))

    # Validate the parameter type
    if param_info.annotation != dict[str, dict]:
        raise RuntimeError(
            f"Parameter '{param_name}' must be of type 'dict[str, dict]'."
        )

    # Validate the return type
    if return_annotation != dict[str, str]:
        raise RuntimeError("The return type must be 'dict[str, str]'.")


@solver
def run_agent(dataset: Dataset, agent: Callable) -> Solver:

    # add sample ids to dataset if they aren't there (start at 1 not 0)
    agent_input = {}
    id = 1
    for sample in dataset:
        # ensure there is an id
        if sample.id is None:
            sample.id = id
            id = id + 1
        agent_input[sample.id] = sample

    # run the agent
    result = agent(agent_input)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # the sample
        id = state.sample_id

        # lookup the agent completion
        completion = result[id]

        # set the result and mark the task complete
        state.output.completion = completion
        state.completed = True

        return state

    return cast(Solver, solve)

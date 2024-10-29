from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.model import get_model, call_tools
from inspect_ai.tool import python


# This solver agent illustrates replacing the task's solver (agent) with
# a complete custom external solver. This solver implements the most basic
# tool use / agent look and includes a timeout parameter which can be passed
# using -A timeout=<value>.

## For example, run this task like:
## agent-eval --agent_name basic-agent --benchmark inspect:agents/simple_task/task.py@ascii_art --model openai/gpt-4o -B to_draw=circle --agent_dir=agents/inspect-task --agent_function=solver_agent.basic_agent -A timeout=30

@solver
def basic_agent(timeout: int = 60) -> Solver:

    async def solve(state: TaskState, generate: Generate):

        # Add tools
        state.tools.append(python(timeout=timeout))

        # Run a simple tool loop
        model = get_model()
        while True:
            # call model
            output = await model.generate(state.messages, state.tools)

            # update state
            state.output = output
            state.messages.append(output.message)

            # make tool calls or terminate if there are none
            if output.message.tool_calls:
                tool_output = await call_tools(output.message, [python(timeout=timeout)])
                state.messages.extend(tool_output)
            else:
                break

        return state
    return solve
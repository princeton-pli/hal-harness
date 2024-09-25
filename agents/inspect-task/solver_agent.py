from inspect_ai.solver import solver, TaskState, Generate

@solver
def do_nothing_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["confirm_message"] = "loaded_do_nothing_agent"
        return state

    return solve

from tau_bench.envs import get_env

from tau_bench.agents.chat_react_agent import ChatReActAgent


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert "model_name" in kwargs, "model_name is required"
    assert "provider" in kwargs, "provider is required. choose from openai or anthropic"
    task_id = list(input.keys())[0]

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]["env"],
        input[task_id]["user_strategy"],
        input[task_id]["user_model"],
        input[task_id]["task_split"],
        input[task_id]["user_provider"],
        input[task_id]["task_index"],
    )
    # get instruction from environment
    _ = isolated_env.reset(input[task_id]["task_index"]).observation

    ### YOUR AGENT CODE HERE ###
    agent = ChatReActAgent(
        tools_info=isolated_env.tools_info,
        wiki=isolated_env.wiki,
        model=kwargs["model_name"],
        provider=kwargs["provider"],
        use_reasoning=True,
        temperature=kwargs["temperature"] if "temperature" in kwargs else 0.0,
    )

    _ = agent.solve(isolated_env, task_index=input[task_id]["task_index"])

    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {
        task_id: {
            "reward": isolated_env.reward,
            "taken_actions": [action.model_dump() for action in isolated_env.actions],
            "task": isolated_env.task.model_dump(),
        }
    }

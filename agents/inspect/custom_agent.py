## This illustrates calling a completely custom agent. In this scenario
## the Inspect samples will be converted into a dictionary of samples and
## provided to the agent function. That function should return a dictionary
## with keys matching the sample input keys and with a string representing
## the completion for that sample.

## This can be invoked like:
## agent-eval --agent_name draw --benchmark inspect:agents/inspect-task/task.py@ascii_art --model openai/gpt-4o --agent_function custom_agent.run --agent_dir agents/inspect-task

def run(tasks: dict[str, dict], **kwargs) -> dict[str, str]:

    result = {}
    for key, value in tasks.items():
        result[key] = (
            f"Sample {value['id']} Nam venenatis turpis mauris. Donec ut massa vel lacus maximus placerat."
        )
    return result

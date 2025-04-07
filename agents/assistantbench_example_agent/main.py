# This is an example agent that uses smolagents to generate answers for the AssistantBench benchmark.
from openai import OpenAI
from agent import agent


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    client = OpenAI()

    # Store the results
    agent_output = {}

    # Iterate through the tasks
    for task_id, task in input.items():
        print(f'Generating {task_id}...')

        instruction = input[task_id]['task']

        # Add additional instructions
        instruction += "\n\nProvide a concise and accurate answer to the question above without any additional context. If you are unable to access the information required to answer the question, simply respond with a blank line."

        response = agent.run(instruction)

        print(f"Response for {task_id}: {response}")

        agent_output[task_id] = response

    return agent_output
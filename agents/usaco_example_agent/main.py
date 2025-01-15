# This is an example agent that solves a USACO problem
# Disclaimer: this is not a functional agent and is only for demonstration purposes. This implementation is just a single model call.

from openai import OpenAI

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

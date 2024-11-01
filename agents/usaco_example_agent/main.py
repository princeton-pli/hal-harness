import weave
from openai import OpenAI

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    
    client = OpenAI()

    results = {}

    for task_id, task in input.items():

        with weave.attributes({'weave_task_id': task_id}):
            response = client.chat.completions.create(
                model=kwargs['model_name'],
                messages=[
                    {"role": "user", "content": 'test'},
                    ],
                max_tokens=2000,
                n=1,
                temperature=1,
            )
            results[task_id] = response.choices[0].message.content

    for task_id, task in input.items():
        input[task_id]['response'] = results[task_id]
        
    return input

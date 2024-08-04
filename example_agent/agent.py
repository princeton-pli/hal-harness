import transformers
import weave

def run(input):
    from openai import OpenAI
    client = OpenAI()

    with weave.attributes({'task_id': 'math-operations-001'}):
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "user", "content": 'test'},
                ],
            max_tokens=2000,
            n=1,
            temperature=1,
        )
    for task in input:
        task['model_name_or_path'] = 'test'
        task['model_patch'] = 'test'
    return input
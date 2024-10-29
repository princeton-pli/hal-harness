import weave

def run(input):
    from openai import OpenAI
    client = OpenAI()

    with weave.attributes({'weave_task_id': '1333_platinum_good_bitstrings'}):
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
        input[task]['response'] = 'test'
    return input
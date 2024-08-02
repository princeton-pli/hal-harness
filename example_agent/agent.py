import transformers

def run(input):
    for task in input:
        task['model_name_or_path'] = 'test'
        task['model_patch'] = 'test'
    return input
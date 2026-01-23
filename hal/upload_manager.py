import json
import os

def store_and_upload_results(result, benchmark_name, agent_path, model, config):
    # Create the base results directory if it doesn't exist
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    # Create the benchmark-specific directory if it doesn't exist
    benchmark_dir = os.path.join(base_dir, benchmark_name)
    os.makedirs(benchmark_dir, exist_ok=True)

    # Generate the filename
    filename = f"{benchmark_name}_{agent_path.replace('.', '_')}_{model}.json"
    file_path = os.path.join(benchmark_dir, filename)

    # Store results locally
    with open(file_path, 'w') as f:
        json.dump(result, f)

    print(f"Results stored locally at: {file_path}")

    # Upload to HuggingFace space
    # upload_to_huggingface(filename, config['huggingface_token'], config['huggingface_repo'])

def upload_to_huggingface(filename, token, repo):
    # Implement HuggingFace upload logic here
    pass
import json
import os
import subprocess
import sys
import time

def parse_evaluation_result(run_id):
    # Load the evaluation results
    with open(f"swebench.{run_id}.json", 'r') as f:
        results = json.load(f)

    # delete file
    os.remove(f"swebench.{run_id}.json")

    return results

def run_evaluation_harness(predictions_path, run_id):
    command = ['conda', 'run', '-n', 'swebench_harness_hal', 'python', '-m', 'swebench.harness.run_evaluation',
           '--dataset_name', 'princeton-nlp/SWE-bench_Verified',
           '--predictions_path', predictions_path,
           '--max_workers', '6',
           '--run_id', run_id]

    try:
        subprocess.run(['conda', 'create', '-n', 'swebench_harness_hal', 'python=3.11', '-y'], check=True)
        subprocess.run([
    'conda', 'run', 
    '-n', 'swebench_harness_hal', 
    'pip', 'install', 
    '-e', 'git+https://github.com/benediktstroebl/SWE-bench.git#egg=swebench'], check=True)

        subprocess.run(command, check=True)

        return parse_evaluation_result(run_id)


    except subprocess.CalledProcessError as e:
        print(f"Error running SWE-bench evaluation harness: {e}")
        print(f"Stdout: {e.output}")
        print(f"Stderr: {e.stderr}")
        raise

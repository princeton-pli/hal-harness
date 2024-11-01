import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
import inspect
import os
from inspect_ai.solver import solver
from typing import Callable, cast, Dict
import shutil
from contextlib import contextmanager
import uuid
import weave
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor
import traceback

from inspect_ai.model import ChatMessage
from inspect_ai.dataset import Dataset
from inspect_ai.solver import Generate, Solver, TaskState

def load_agent(agent_function: str) -> Callable:
    # parse the agent name
    module_name, function_name = agent_function.rsplit(".", 1)

    # attempt to load it from the module
    module = importlib.import_module(module_name)
    loaded_agent = getattr(module, function_name)
    return loaded_agent


def validate_agent(agent: Callable) -> None:

    # Get the signature of the function
    sig = inspect.signature(agent)

    # Get the parameters and return annotation
    parameters = sig.parameters
    return_annotation = sig.return_annotation

    # Check the number of parameters (should be exactly one)
    if len(parameters) not in [1, 2]:
        raise RuntimeError("The agent function should accept only a single argument or a single argument and kwargs.")

    # Get the parameter name and annotation
    param_name, param_info = next(iter(parameters.items()))

    # Validate the parameter type
    if param_info.annotation != dict[str, dict]:
        raise RuntimeError(
            f"Parameter '{param_name}' must be of type 'dict[str, dict]'."
        )

    # Validate the return type
    if return_annotation != dict[str, str]:
        raise RuntimeError("The return type must be 'dict[str, str]'.")
    

# Track created directories
temp_dirs = []


def run_single_agent(single_input, agent_dir, agent_function, agent_args, module_name, run_id, conda_env_name, log_dir=None):
    # Create a unique directory in /tmp/
    temp_dir = f"/tmp/agent_run_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    # Track the created directory
    temp_dirs.append(temp_dir)

    # Copy the entire agent directory to the temporary directory
    shutil.copytree(agent_dir, temp_dir, dirs_exist_ok=True)

    # Serialize the input data to a JSON file in the temp directory
    input_file = os.path.join(temp_dir, 'input.json')
    with open(input_file, 'w') as f:
        json.dump({single_input['id']: single_input}, f)

    # Prepare the agent arguments
    agent_args_file = os.path.join(temp_dir, 'agent_args.json')
    with open(agent_args_file, 'w') as f:
        json.dump(agent_args, f)

    # Construct the command to run the agent function
    command = [
        'python', '-c',
        f'''
import os
import json
import importlib.util
import weave

weave.init("{run_id}")

# Load the agent module
module_name = "{agent_function.rsplit(".", 1)[0]}"
function_name = "{agent_function.rsplit(".", 1)[1]}"
spec = importlib.util.spec_from_file_location(module_name, "{os.path.join(temp_dir, module_name + ".py")}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
agent = getattr(module, function_name)

# Load the input data
with open("input.json", "r") as f:
    input_data = json.load(f)

# Load the agent arguments
with open("agent_args.json", "r") as f:
    agent_args = json.load(f)

single_input_id = "{single_input['id']}"  

# Run the agent function
with weave.attributes({{"weave_task_id": single_input_id}}):
    result = agent(input_data, **agent_args)

# Save the result
with open("output.json", "w") as f:
    json.dump(result, f)
'''
    ]

    if conda_env_name:
        print(f"Running agent in conda environment: {conda_env_name}")
        command = [
            'conda', 'run', '-n', conda_env_name] + command

    try:
        # Run the agent in a subprocess with the temp_dir as cwd
        subprocess.run(command, cwd=temp_dir, check=True)

        # Load the result from output.json
        output_file = os.path.join(temp_dir, 'output.json')
        with open(output_file, 'r') as f:
            result = json.load(f)

        if log_dir:
            raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS_DURING.jsonl")
            os.makedirs(log_dir, exist_ok=True)

            with open(raw_submissions_path, "a") as f:
                json.dump(result, f)
                f.write('\n')

        # delete the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error running agent: {e}")
        traceback.print_exc()
        result = {single_input['id']: f"ERROR RUNNING AGENT: {e}"}

        # delete the temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result


async def run_agent_parallel(dataset: Dataset, agent: Callable, agent_args: Dict, agent_function: str, agent_dir: str, run_id: str,  max_concurrent: int = 5, log_dir: str = None, task_name: str = None, conda_env_name: str = None) -> Solver:
    # add sample ids to dataset if they aren't there (start at 1 not 0)
    agent_input = {}
    id = 1
    cwd = os.getcwd()
    agent_dir = os.path.abspath(agent_dir)

    # If agent is auto-code-rover, we need to add run_id to agent_args for weave tracing because agent spawns sybprocesses
    if "auto-code-rover" in str(agent_dir):
        agent_args["weave_run_id"] = run_id

    for sample in dataset:
        # ensure there is an id
        if sample.id is None:
            sample.id = id
            id = id + 1

        # for files, convert them to absolute paths
        data_files = {}
        if sample.files:
            for key, value in sample.files.items():
                if os.path.isabs(value):
                    data_files[key] = value
                else:
                    data_files[key] = os.path.join(cwd, value)
            sample.files = data_files

        # Flatten the input
        input_str = (
            sample.input
            if isinstance(sample.input, str)
            else "\n".join(
                [
                    message.text
                    for message in sample.input
                ]
            )
        )

        agent_input[sample.id] = {
            "id": sample.id,
            "input": input_str,
            "choices": sample.choices,
            "target": sample.target,
            "metadata": sample.metadata,
            "files": data_files,
            "setup": sample.setup,
        }

    if log_dir:
        raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS_DURING.jsonl")
        
        if os.path.exists(raw_submissions_path):
            # read in previous submissions and remove from agent_input
            with open(raw_submissions_path, "r") as f:
                previous_submissions = [json.loads(line) for line in f]
            
            previous_ids = {list(submission.keys())[0] for submission in previous_submissions}
            agent_input = {k: v for k, v in agent_input.items() if k not in previous_ids}

            print( f"Previous submissions found. {len(previous_ids)} submissions removed from agent input.")


    module_name, _ = agent_function.rsplit(".", 1)

    original_dir = os.getcwd()

    abs_log_dir = os.path.abspath(log_dir) if log_dir else None


    # Run the agent in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                run_single_agent,
                input_data,
                agent_dir,
                agent_function,
                agent_args,
                module_name,
                run_id,
                conda_env_name,
                abs_log_dir
            )
            for input_data in agent_input.values()
        ]
        results = await asyncio.gather(*tasks)

    for result in results:
        print(result)
        
    os.chdir(original_dir)

    # Delete all created directories after processing all jobs
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Merge results list into a single dictionary
    merged_result = {k: v for d in results for k, v in d.items()}

    # add all results from _RAW_SUBMISSIONS_DURING.jsonl to merged_result
    if log_dir:
        raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS_DURING.jsonl")
        
        if os.path.exists(raw_submissions_path):
            # read in previous submissions and remove from agent_input
            with open(raw_submissions_path, "r") as f:
                previous_submissions = [json.loads(line) for line in f]
            
            for submission in previous_submissions:
                merged_result.update(submission)
        

    print(f"Results: {merged_result}")

    # save raw submissions as jsonl file with each line being a submission
    if log_dir:
        raw_submissions_path = os.path.join(log_dir, f"{run_id}_RAW_SUBMISSIONS.jsonl")
        os.makedirs(log_dir, exist_ok=True)

        with open(raw_submissions_path, "w") as f:
            for key, value in merged_result.items():
                json.dump({key: value}, f)
                f.write('\n')

    # save swebench results in special format. TODO: make this more general
    if task_name:
        if "swe_bench" in task_name:
            swebench_submissions_path = os.path.join(log_dir, f"{run_id}_SWE_BENCH_SUBMISSIONS.jsonl")
            with open(swebench_submissions_path, 'w') as f:
                for key, value in merged_result.items():
                    f.write(json.dumps({"instance_id": key, "model_patch": value, "model_name_or_path": f"swebench"}) + '\n')
        

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # the sample
        id = state.sample_id

        # lookup the agent completion
        completion = merged_result[id]

        # set the result and mark the task complete
        state.output.completion = completion
        state.completed = True

        return state

    return cast(Solver, solve)

# Wrapper function to maintain the original function signature
@solver
def run_agent(dataset: Dataset, agent: Callable, agent_args: Dict, agent_function: str, agent_dir: str, run_id: str, max_concurrent: int = 5, log_dir: str = None, task_name: str = None, conda_env_name: str = None) -> Solver:
    return asyncio.run(run_agent_parallel(dataset, agent, agent_args, agent_function, agent_dir=agent_dir, run_id=run_id, max_concurrent=max_concurrent, log_dir=log_dir, task_name=task_name, conda_env_name=conda_env_name))


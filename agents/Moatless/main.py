import datetime
import os
from moatless.loop import AgenticLoop
from moatless.transitions import search_and_code_transitions
from moatless.workspace import Workspace
from moatless.benchmark.utils import trace_metadata
from moatless.benchmark.swebench import get_repo_dir_name, sorted_instances, setup_swebench_repo
from tqdm.notebook import tqdm
from moatless.benchmark.swebench import load_instances
import os
import json
from moatless.index.settings import IndexSettings
from moatless.index.code_index import CodeIndex
from dotenv import load_dotenv
from moatless.benchmark.swebench import get_repo_dir_name
from moatless.index import CodeIndex, IndexSettings
from moatless import FileRepository, Workspace
import os
from moatless.index import CodeIndex, IndexSettings
from moatless import FileRepository, Workspace
from moatless.edit import EditCode, PlanToCode
from moatless.transitions import search_and_code_transitions
from moatless.benchmark.evaluation import  Evaluation
from moatless.find import DecideRelevance, IdentifyCode, SearchCode

import time
import traceback
import logging
import subprocess

index_settings = IndexSettings(
    embed_model="text-embedding-3-small",
    max_chunk_size=8000,
)


load_dotenv()

def get_persist_dir(instance):
    return os.path.join("./index_store", get_repo_dir_name(instance["instance_id"]))

def create_index(instance):
    repo_dir = setup_swebench_repo(instance, repo_base_dir=os.getcwd())
    file_repo = FileRepository(repo_path=repo_dir)
    return CodeIndex(settings=index_settings, file_repo=file_repo)
 


def run_moatless(tasks: dict[str, dict], **kwargs) -> dict[str, str]:
    print(os.getcwd())

    temperature = 0.2
    max_cost = 10
    model = kwargs['model']

    if 'llama' in model.lower():
        model = 'azure/' + model
    if 'o1' in model.lower():
        temperature = 1

    global_params = {
        "model": kwargs['model'], 
        "temperature": temperature
    }

    # state_params = {
    #     SearchCode: {
    #         "max_search_results": 75,
    #         "provide_initial_context": True,  # Do a vector search with the problem statement to get an initial file context
    #         "initial_context_tokens": 6000,
    #         "initial_search_results": 100,
    #         "initial_context_spans_per_file": 5,
    #     },
    #     IdentifyCode: {
    #         "expand_context": True,  # Expands the search results with related code to the search hits
    #     },
    #     DecideRelevance: {
    #         "finish_after_relevant_count": 1,  # Even if the LLM doesn't believe the identified code is complete we will finish up after one retry
    #     },
    #     PlanToCode: {
    #         "max_tokens_in_edit_prompt": 750, # The max number of tokens in the edit block
    #         "expand_context_with_related_spans": False,
    #         "finish_on_review": True, # To abort if the LLm suggest reviews of the code, it's only possible to apply changes ATM.
    #     },
    #     EditCode: {
    #         "chain_of_thought": False,
    #         "show_file_context": False,
    #         "max_prompt_file_tokens": 8000,
    #     },
    # }

    evaluations_dir = "./evaluations"
    evaluation_name = f"moatless_{kwargs['model']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
    trajectory_dir = f"{evaluations_dir}/{evaluation_name}/trajs"
    predictions_path = f"{evaluation_dir}/all_preds.jsonl"

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    results = {}
    for task in tasks:
        input = tasks[task]

        instance = {
            "instance_id": input["id"],
            "problem_statement": input["input"],
            "repo": input["metadata"]["repo"],
            "base_commit": input["metadata"]["base_commit"]
        }
        

        code_index = create_index(instance)


    

        repo_dir = setup_swebench_repo(instance, repo_base_dir=os.getcwd())
        file_repo = FileRepository(repo_path=repo_dir)

        persist_dir = get_persist_dir(instance)
        code_index.persist(persist_dir=persist_dir)

        # code_index = CodeIndex(file_repo=file_repo, settings=index_settings)
        nodes, tokens = code_index.run_ingestion()

        print(f"Indexed {nodes} nodes and {tokens} tokens")

        workspace = Workspace(file_repo=file_repo, code_index=code_index)


        instance_id = instance["instance_id"]
        trajectory_path = os.path.join(trajectory_dir, f"{instance_id}.json")

        persist_dir = get_persist_dir(instance)

        problem_statement = instance["problem_statement"]


        transitions = search_and_code_transitions(global_params=global_params)

        loop = AgenticLoop(transitions=transitions, workspace=workspace, trajectory_path=trajectory_path, max_cost=max_cost)

        info = {
            "evaluation_name": evaluation_name,
            "instance_id": instance["instance_id"]
        }

        start_time = time.time()
        try:
            response = loop.run(problem_statement)

        except Exception as e:
            info["error"] = traceback.format_exc()
            logging.exception(f"Error in evaluation of {instance['instance_id']} ")

        info["duration"] = time.time() - start_time
        info["total_cost"] = loop.trajectory.total_cost()

        workspace.save()

        output = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                cwd=repo_dir,
        )

        info["submission"] = output.stdout

        loop.trajectory.save_info(info)
        trajectory = loop.trajectory.to_dict()

        prediction = {
            "model_name_or_path": evaluation_name,
            "instance_id": instance["instance_id"],
            "model_patch": trajectory["info"].get("submission", ""),
        }

        results[instance["instance_id"]] = prediction["model_patch"]



    return results




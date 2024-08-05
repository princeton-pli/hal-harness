from abc import ABC, abstractmethod
from pydantic import TypeAdapter, ValidationError
from huggingface_hub import HfApi
from typing import Dict
import json
import os
from ..utils.weave_utils import get_total_cost, assert_task_id_logging

class BaseBenchmark(ABC):

    @abstractmethod
    def run(self, agent_function, run_id: str):
        pass

    @abstractmethod
    def test_run(self, agent_function, weave_client):
        pass
    
    @property
    @abstractmethod
    def type_adapter(self) -> TypeAdapter:
        pass

    @abstractmethod
    def validate_logging(self, weave_client, test_task_id):
        pass
    
    @abstractmethod
    def process_and_upload_results(self, 
                                   agent_name: str, 
                                   run_id: str, 
                                   eval_results, 
                                   weave_client,
                                   config, 
                                   upload=False) -> Dict:
        pass

    def validate_agent_output(self, output) -> bool:
        try:
            self.type_adapter.validate_python(output)
        except ValidationError as e:
            print("Agent output does not match expected format for output! Required output format" + str(e))

    def validate_logging(self, weave_client, test_task_id):
        assert get_total_cost(weave_client) > 0, "Test run did not incur any cost"

        assert_task_id_logging(weave_client, test_task_id)


    def upload_results(self, run_id, upload_dict):
        # Save results to a temp file
        upload_path = f"results/{self.benchmark_name}/{run_id}_upload.json"
        with open(upload_path, 'w') as f:
            json.dump(upload_dict, f)
        # Use huggingface API to upload results
        api = HfApi(token=os.getenv("HF_TOKEN", '.'))
        api.upload_file(
            path_or_fileobj=upload_path,
            path_in_repo=f"{run_id}.json",
            repo_id="agent-evals/results"   ,
            repo_type="dataset",
            commit_message=f"Add {run_id} to leaderboard.",
            )
        
        # remove temp file
        os.remove(upload_path)

        print(f"Results uploaded to Hugging Face Hub.")
        print("=====\n\n")

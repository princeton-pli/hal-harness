from abc import ABC, abstractmethod
from pydantic import TypeAdapter, ValidationError
from huggingface_hub import HfApi
from typing import Dict
import json
import os
from ..utils.weave_utils import get_total_cost, assert_task_id_logging
import tempfile

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

    def validate_logging(self, weave_client, test_weave_task_id):
        assert get_total_cost(weave_client) > 0, "Test run did not incur any cost"

        assert_task_id_logging(weave_client, test_weave_task_id)


    def upload_results(self, run_id, upload_dict):
        # Write results to temp file using tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump(upload_dict, f)
            temp_file_name = f.name
            temp_file_path = os.path.abspath(temp_file_name)

        # Use huggingface API to upload results
        api = HfApi(token=os.getenv("HF_TOKEN", '.'))
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo=f"{run_id}.json",
            repo_id="agent-evals/results"   ,
            repo_type="dataset",
            commit_message=f"Add {run_id} to leaderboard.",
            )
        

        # unlink the file
        os.unlink(temp_file_name)

        print(f"Results uploaded to Hugging Face Hub.")

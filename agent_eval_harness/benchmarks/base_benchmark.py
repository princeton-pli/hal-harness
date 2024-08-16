from abc import ABC, abstractmethod
from pydantic import TypeAdapter
from huggingface_hub import HfApi
from typing import Dict
import json
import os
from ..utils.weave_utils import get_total_cost, assert_task_id_logging
import tempfile
import subprocess

class BaseBenchmark(ABC):
    def __init__(self, agent_dir: str, config: Dict):
        self.agent_dir = agent_dir
        self.config = config
        self.benchmark_name: str
        self.requirements_file: str
        self.benchmark_dir: str
        self.main_cwd: str

    @abstractmethod
    def run(self, agent_function, run_id: str):
        pass

    def mount_environment(self):
        self.cwd = os.getcwd()
        os.chdir(self.agent_dir)

    def unmount_environment(self):
        os.chdir(self.cwd)

    def run_agent(self, agent_function, input):
        """
        Run the agent on the input and return the output. This function assumes that the agent is a python function that takes the input as an argument and returns the output.
        If a benchmark requires the agent to be a class that is instantiated before running, this function should be overridden in the benchmark class.
        """
        # Get prediction from agent
        agent_output = agent_function(input)
        return agent_output

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
        self.type_adapter.validate_python(output)

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


    def mount_benchmark(self):
        try:
            with open(f'agent_eval_harness/benchmarks/requirements/{self.requirements_file}.toml', 'r') as f:
                requirements_toml = f.read()
        except FileNotFoundError:
            raise ValueError(f"Requirements file for benchmark '{self.benchmark_name}' not found. See README for available benchmarks.")
        with open('pyproject.toml', 'w') as f:
            f.write(f"""[tool.poetry]\npackage-mode = false\n\n{requirements_toml}""")

        # install dependencies
        try:
            print(f'\n\n====Mounting benchmark harness for {self.benchmark_name}====')
            print(f"Installing dependencies...")
            # this command install the benchmark dependencies in a virtual environment separate from the agent_eval_harness and agent dependencies to run the evals
            subprocess.run(['poetry env use python3 && poetry lock && poetry install --no-root'], shell=True)
            print(f"Done!")
            print('=====\n\n')
        except Exception as e:
            print(e)
            raise ValueError(f"Failed to install dependencies for benchmark '{self.benchmark_name}'")

    def unmount_benchmark(self):
        if os.path.exists('pyproject.toml'):
            os.remove('pyproject.toml')
        if os.path.exists('poetry.lock'):
            os.remove('poetry.lock')




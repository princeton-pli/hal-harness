import subprocess
from .base_benchmark import BaseBenchmark
import json
from typing_extensions import NotRequired, TypedDict, List, Dict
from pydantic.config import ConfigDict
from pydantic import TypeAdapter, ValidationError
import time
import os
import sys
from ..utils.weave_utils import get_total_cost, get_weave_calls
from types import SimpleNamespace
from datetime import datetime
from .MLAgentBench.MLAgentBench.environment import Environment

class MLAgentBenchBenchmark(BaseBenchmark):
    def __init__(self, agent_dir, config):
        super().__init__(agent_dir, config)
        self.benchmark_name = 'mlagentbench'
        self.benchmark_dir = os.path.join(os.path.dirname(__file__), 'MLAgentBench')
        self.requirements_file = 'mlagentbench'
        self.tasks = [
                # "llama-inference",
                # "identify-contrails",
                "vectorization",
                "babylm",
                "fathomnet",
                "feedback",
                "house-price",
                "spaceship-titanic",
                "amp-parkinsons-disease-progression-prediction",
                "CLRS",
                "cifar10",
                "ogbn-arxiv",
                "imdb"]
        self.args = {
            "max_steps": 10e10, # very big numbers to impose no limit
            "max_time": 10e10,  # very big numbers to impose no limit
            "device": 0,
            "python": "python",
            "interactive": False,
            "resume": None,
            "resume_step": 0
        }


    def mount_environment(self):
        # copy benchmark tasks to agent_dir
        print(f"Setting up env in {self.agent_dir}")
        os.system(f"cp -r {self.benchmark_dir}/MLAgentBench/benchmarks {self.agent_dir}MLAgentBench/")
        os.system(f"cp {self.benchmark_dir}/MLAgentBench/prepare_task.py {self.agent_dir}MLAgentBench/prepare_task.py")

        # copy environment to agent_dir
        os.system(f"cp {self.benchmark_dir}/MLAgentBench/environment.py {self.agent_dir}MLAgentBench/environment.py")
        os.system(f"cp {self.benchmark_dir}/MLAgentBench/schema.py {self.agent_dir}MLAgentBench/schema.py")

        # copy supported default actions to agent_dir
        os.system(f"cp {self.benchmark_dir}/MLAgentBench/high_level_actions.py {self.agent_dir}MLAgentBench/high_level_actions.py")
        os.system(f"cp {self.benchmark_dir}/MLAgentBench/low_level_actions.py {self.agent_dir}MLAgentBench/low_level_actions.py")
        os.system(f"cp {self.benchmark_dir}/MLAgentBench/LLM.py {self.agent_dir}MLAgentBench/LLM.py")

        super().mount_environment()
        print("Environment set up successfully")


    def unmount_environment(self):
        super().unmount_environment()
        

    def run(self, agent_function, run_id: str) -> Dict:
        self.mount_environment()

        # for each task run the agent
        for task in self.tasks:
            print(f"\n\nRunning task: {task}")
            log_path = f'{run_id}_logs/{task}'
            workspace_path = f'{run_id}_workspace'

            env = Environment(SimpleNamespace(**{'task': task, 'log_dir': log_path, 'work_dir': workspace_path, **self.args}))
            final_message = self.run_agent(agent_function, env)
            env.save("final")
            if final_message:
                print(f"Final agent message: {final_message}")
            print(f"\n\nTask: {task} completed")

        self.unmount_environment()

        # Run the SWE-bench evaluation harness
        self.mount_benchmark()
        for task in self.tasks:
            print(f"\n\nRunning evaluation harness for task: {task}")
            result = self._run_evaluation_harness(task, run_id)
        self.unmount_benchmark()

        return result


    def _run_evaluation_harness(self, task, run_id):

        command = (
            f"cd {self.benchmark_dir} && poetry run python -m MLAgentBench.eval "
            f"--log-folder {os.path.abspath(os.path.join(self.agent_dir, f'{run_id}_logs/{task}'))} "
            f"--task {task} "
            f"--output-file {os.path.abspath(os.path.join(self.agent_dir, f'{run_id}_{task}.json'))} "
        )

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True
            )

            stdout_output = []
            stderr_output = []

            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    print(stdout_line.strip())
                    stdout_output.append(stdout_line)
                if stderr_line:
                    print(stderr_line.strip(), file=sys.stderr)
                    stderr_output.append(stderr_line)

                if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                    break

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command, 
                                                    output=''.join(stdout_output),
                                                    stderr=''.join(stderr_output))

            return self._parse_evaluation_result(run_id)


        except subprocess.CalledProcessError as e:
            print(f"Error running SWE-bench evaluation harness: {e}")
            print(f"Stdout: {e.output}")
            print(f"Stderr: {e.stderr}")
            raise


    def _parse_evaluation_result(self, run_id):
        results = {}
        for task in self.tasks:
            task_result_file_path = os.path.abspath(os.path.join(self.agent_dir, f'{run_id}_{task}.json'))
            if not os.path.exists(task_result_file_path):
                print(f"WARNING: Task {task} did not produce a result file. This run can't be uploaded to the leaderboard. Skipping...")
                continue
            
            with open(task_result_file_path, 'r') as f:
                results[task] = json.load(f)

            # delete file
            os.remove(task_result_file_path)
        return results
    
    def validate_agent_output(self):
        # check if logs/ directory as created by env
        log_path = os.path.join(self.agent_dir, 'logs')
        assert os.path.exists(log_path), f"Logs directory not found at {log_path}"



    def test_run(self, agent_function, weave_client):
        args = self.args.copy()

        args['task'] = 'cifar10'
        args['log_dir'] = os.path.abspath(os.path.join(self.agent_dir, 'logs'))
        args['work_dir'] = os.path.abspath(os.path.join(self.agent_dir, 'workspace'))
        args['max_steps'] = 1 # only run 1 step for test run
        args['max_time'] = 60 # only run for 1 minute for test run, whatever hits first

        test_env = Environment(SimpleNamespace(**args))
        
        self.mount_environment()
        final_message = self.run_agent(agent_function, test_env)
        test_env.save("final")
        self.unmount_environment()

        # Validate agent output
        self.validate_agent_output()

        # remove logs/ and workspace/ directories
        os.system(f"rm -rf {os.path.abspath(os.path.join(self.agent_dir, 'logs'))} {os.path.abspath(os.path.join(self.agent_dir, 'workspace'))}")

        # validate that there was cost associated with the test run
        time.sleep(5) # wait to finish usage calculation on weave
        self.validate_logging(weave_client, test_weave_task_id='cifar10')

        return True

    @property
    def type_adapter(self):
        # For MLAgentBench, there is no files to validate
        pass


    def process_and_upload_results(self, 
                                   agent_name: str, 
                                   run_id: str, 
                                   eval_results: Dict, 
                                   weave_client,
                                   config, 
                                   upload=False):
        # move logs/ direcotry to results/benchmark_name/logs
        out_path = f"results/{self.benchmark_name}/{run_id}"
        os.makedirs(out_path, exist_ok=True)

        # store results
        with open(os.path.join(out_path, f"{run_id}.json"), 'w') as f:
            json.dump(eval_results, f)     

        # store other logs that harness created
        os.system(f"mv {self.agent_dir}/{run_id}_logs {out_path}")

        # New dict
        import numpy as np
        upload_dict = {
            "config": {'agent_name': agent_name, 
                    'benchmark_name': self.benchmark_name, 
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'run_id': run_id,
            },
            "results": {
                'overall_score': np.mean([eval_results[task][list(eval_results[task].keys())[0]]['final_score'] for task in eval_results]), # mean across all tasks
                **{f"{task}_score": eval_results[task][list(eval_results[task].keys())[0]]['final_score'] for task in eval_results}, # final score for each task
                "total_cost": get_total_cost(weave_client),
            },
            "raw_eval_results": eval_results,
            "raw_logging_results": get_weave_calls(weave_client)
        }


        # Store the upload results locally
        with open(os.path.join(out_path, f"{run_id}_UPLOAD.json"), 'w') as f:
            json.dump(upload_dict, f)

        if upload:
            self.upload_results(run_id, upload_dict)


        return upload_dict['results']

        
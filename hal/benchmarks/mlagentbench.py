import subprocess
from .base_benchmark import BaseBenchmark
import json
from typing_extensions import Dict
import os
from types import SimpleNamespace
from .MLAgentBench.MLAgentBench.environment import Environment
import logging

logger = logging.getLogger("agent_eval")


class MLAgentBenchBenchmark(BaseBenchmark):
    def __init__(self, agent_dir, config):
        super().__init__(agent_dir, config)
        self.benchmark_name = "mlagentbench"
        self.benchmark_dir = os.path.join(os.path.dirname(__file__), "MLAgentBench")
        self.tasks = [
            # "llama-inference",
            # "identify-contrails",
            "vectorization",
            # "babylm",
            "fathomnet",
            "feedback",
            "house-price",
            "spaceship-titanic",
            "amp-parkinsons-disease-progression-prediction",
            # "CLRS",
            "cifar10",
            # "ogbn-arxiv",
            "imdb",
        ]
        self.args = {
            "max_steps": 10e10,  # very big numbers to impose no limit
            "max_time": 10e10,  # very big numbers to impose no limit
            "device": 0,
            "python": "python",
            "interactive": False,
            "resume": None,
            "resume_step": 0,
        }

    def mount_environment(self):
        # copy benchmark tasks to agent_dir
        logger.info(f"Setting up env in {self.agent_dir}")
        os.system(
            f"cp -r {self.benchmark_dir}/MLAgentBench/benchmarks {self.agent_dir}MLAgentBench/"
        )
        os.system(
            f"cp {self.benchmark_dir}/MLAgentBench/prepare_task.py {self.agent_dir}MLAgentBench/prepare_task.py"
        )

        # copy environment to agent_dir
        os.system(
            f"cp {self.benchmark_dir}/MLAgentBench/environment.py {self.agent_dir}MLAgentBench/environment.py"
        )
        os.system(
            f"cp {self.benchmark_dir}/MLAgentBench/schema.py {self.agent_dir}MLAgentBench/schema.py"
        )

        # copy supported default actions to agent_dir
        os.system(
            f"cp {self.benchmark_dir}/MLAgentBench/high_level_actions.py {self.agent_dir}MLAgentBench/high_level_actions.py"
        )
        os.system(
            f"cp {self.benchmark_dir}/MLAgentBench/low_level_actions.py {self.agent_dir}MLAgentBench/low_level_actions.py"
        )
        os.system(
            f"cp {self.benchmark_dir}/MLAgentBench/LLM.py {self.agent_dir}MLAgentBench/LLM.py"
        )

        super().mount_environment()
        logger.info("Environment set up successfully")

    def unmount_environment(self):
        super().unmount_environment()

    def run(self, agent_function, run_id: str) -> Dict:
        self.mount_environment()

        # for each task run the agent
        for task in self.tasks:
            logger.info(f"\n\nRunning task: {task}")
            log_path = f"{run_id}_logs/{task}"
            workspace_path = f"{run_id}_workspace"

            env = Environment(
                SimpleNamespace(
                    **{
                        "task": task,
                        "log_dir": log_path,
                        "work_dir": workspace_path,
                        **self.args,
                    }
                )
            )
            final_message = self.run_agent(agent_function, env)
            env.save("final")
            if final_message:
                logger.info(f"Final agent message: {final_message}")
            logger.info(f"\n\nTask: {task} completed")

        self.unmount_environment()

        # Run the SWE-bench evaluation harness
        self.mount_benchmark()
        for task in self.tasks:
            logger.info(f"\n\nRunning evaluation harness for task: {task}")
            result = self._run_evaluation_harness(task, run_id)

        # Parse the evaluation results
        result = self._parse_evaluation_result(run_id)
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
                universal_newlines=True,
            )

            stdout_output = []
            stderr_output = []

            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    logger.info(stdout_line.strip())
                    stdout_output.append(stdout_line)
                if stderr_line:
                    logger.info(stderr_line.strip())
                    stderr_output.append(stderr_line)

                if (
                    stdout_line == ""
                    and stderr_line == ""
                    and process.poll() is not None
                ):
                    break

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    command,
                    output="".join(stdout_output),
                    stderr="".join(stderr_output),
                )

            return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running SWE-bench evaluation harness: {e}")
            logger.error(f"Stdout: {e.output}")
            logger.error(f"Stderr: {e.stderr}")
            raise

    def _parse_evaluation_result(self, run_id):
        results = {}
        for task in self.tasks:
            task_result_file_path = os.path.abspath(
                os.path.join(self.agent_dir, f"{run_id}_{task}.json")
            )
            if not os.path.exists(task_result_file_path):
                logger.warning(
                    f"WARNING: Task {task} did not produce a result file. This run can't be uploaded to the leaderboard. Skipping..."
                )
                continue

            with open(task_result_file_path, "r") as f:
                results[task] = json.load(f)

            # delete file
            os.remove(task_result_file_path)
        return results

import subprocess
import docker
import json
import os
import tempfile
from typing import Dict, Any
import logging

from .base_benchmark import BaseBenchmark

logger = logging.getLogger("agent_eval")


class USACOBenchmark(BaseBenchmark):
    """USACO benchmark implementation"""

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        assert os.path.exists(os.path.join(os.path.dirname(__file__), "USACO/data")), (
            "data folder in USACO benchmark directory (hal/benchmarks/USACO) not found. Please download and extract the USACO dataset as described in the README."
        )

        self.benchmark_name = "usaco"
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox)

        # Load benchmark dataset
        dataset_path = os.path.join(
            os.path.dirname(__file__), "USACO/data/datasets/usaco_subset307_dict.json"
        )
        with open(dataset_path) as f:
            self.benchmark = json.load(f)

        # For testing, limit to 1 task
        # self.benchmark = {
        #     k: v for k, v in self.benchmark.items()
        #     if k in list(self.benchmark.keys())[:1]
        # }

        # Set benchmark directory
        self.benchmark_dir = os.path.join(os.path.dirname(__file__), "USACO")

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Run USACO evaluation harness on agent outputs in Docker container"""
        try:
            # Normalize agent output to handle both old and new formats
            normalized_output = self._normalize_agent_output(agent_output)

            # pass entire task with agent output
            eval_tasks = {}
            for task_id, task in normalized_output.items():
                eval_tasks[task_id] = {
                    **self.benchmark[task_id],
                    "response": normalized_output[task_id],
                }

            temp_file_path = None

            # Create docker client
            client = docker.from_env()

            # Create and run container
            container = client.containers.run(
                "python:3.11",
                command="tail -f /dev/null",
                volumes={
                    self.benchmark_dir: {"bind": "/app", "mode": "rw", "chmod": "777"},
                },
                working_dir="/app",
                detach=True,
            )

            try:
                # write agent output to temp file
                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                    json.dump(eval_tasks, temp_file)
                    temp_file_path = temp_file.name

                # write agent output to inside container
                cmd = f"docker cp {temp_file_path} {container.id}:/app/responses_{run_id}.json"
                subprocess.run(cmd, shell=True, check=True)

                # Install dependencies
                result = container.exec_run(
                    "pip install -r requirements.txt", stream=True
                )

                logger.info("Installing dependencies for USACO harness...")
                for line in result.output:
                    logger.info(line.decode().rstrip())

                # Create required directories
                container.exec_run(
                    "mkdir -p judge_sandbox/predictions/usaco judge_sandbox/solutions/usaco code_sandbox results"
                )

                # Run evaluation
                cmd = f"python harness.py --problem_dict_with_responses /app/responses_{run_id}.json --run_id {run_id}"
                result = container.exec_run(cmd, stream=True)
                for line in result.output:
                    logger.info(line.decode().rstrip())

                # Read results directly from container
                rdict_cmd = f"cat /app/results/rdict_{run_id}.json"
                sdict_cmd = f"cat /app/results/sdict_{run_id}.json"

                rdict_result = container.exec_run(rdict_cmd)
                sdict_result = container.exec_run(sdict_cmd)

                rdict = json.loads(rdict_result.output.decode())
                sdict = json.loads(sdict_result.output.decode())
                rs = list(rdict.values())
                ss = list(sdict.values())

            finally:
                # Cleanup container
                container.stop()
                container.remove()
                client.images.remove("python:3.11", force=True)

            return {"rdict": rdict, "sdict": sdict, "rs": rs, "ss": ss}

        finally:
            # Cleanup temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        sdict = eval_results["sdict"]

        return {
            "accuracy": sum(
                1 if float(sdict[key][0]["result"]["fraction_passed"]) == 1 else 0
                for key in sdict
            )
            / len(sdict),
            "successful_tasks": [
                key
                for key in sdict
                if float(sdict[key][0]["result"]["fraction_passed"]) == 1
            ],
            "failed_tasks": [
                key
                for key in sdict
                if float(sdict[key][0]["result"]["fraction_passed"]) < 1
            ],
        }

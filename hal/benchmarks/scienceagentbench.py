from typing import Dict, Any, Optional
from .base_benchmark import BaseBenchmark

from datasets import load_dataset
import os
import importlib
import argparse
import json
import sys

# get the relative path of the directory where this file is located
this_file_dir = os.path.dirname(os.path.relpath(__file__)).replace('\\', '/')
benchmark_path = os.path.join(this_file_dir, 'scienceagentbench', 'benchmark')

# import this_file_dir.scienceagentbench.ScienceAgentBench.recover_pred_from_log
module_path = os.path.join(this_file_dir, "scienceagentbench", "ScienceAgentBench", "recover_pred_from_log.py")
module_name = "recover_pred_from_log"
spec = importlib.util.spec_from_file_location(module_name, module_path)
recover_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recover_utils)

# import this_file_dir.scienceagentbench.ScienceAgentBench.evaluation.harness as a package
submodule_path = os.path.join(this_file_dir, "scienceagentbench", "ScienceAgentBench")
sys.path.insert(0, submodule_path)
from evaluation.harness import run_evaluation as docker_eval


class ScienceAgentBench(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "scienceagentbench"

        ds = load_dataset("osunlp/ScienceAgentBench", split="validation")
        self.benchmark = {}

        for task in ds:
            task_id = str(task['instance_id'])  # hal-harness requires the task_id to be a str, otherwise cannot match existing results
            self.benchmark[task_id] = {}
            for key in task:
                self.benchmark[task_id][key] = task[key]

        # Optional: Set if benchmark requires VM execution
        self.vm_only = False
        # Optional: Path to VM setup script
        self.setup_script = "hal/benchmarks/your_benchmark/setup.sh"
        super().__init__(agent_dir, config)
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        self._recover_pred_from_log(agent_output, run_id)
        self._call_docker_eval(run_id)  # Not working right now, need to change working dir for docker image build.
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        pass

    def _recover_pred_from_log(self, agent_output, run_id):
        args = argparse.Namespace()
        args.benchmark_name_or_path = benchmark_path
        log_fname = os.path.join(self.get_run_dir(run_id), f"{run_id}_histories.jsonl")

        all_data = agent_output
        with open(log_fname, "w") as f:
            for idx in range(1, len(self.benchmark) + 1):
                f.write(json.dumps(all_data[str(idx)]) + "\n")

        args.log_fname = log_fname
        args.pred_program_path = os.path.join(self.get_run_dir(run_id), "pred_programs")
        args.is_opendevin = False  # TODO: set this to True if using opendevin

        recover_utils.main(args)

    def _call_docker_eval(self, run_id: str):
        """Call docker eval script"""
        docker_eval.main(
            benchmark_path=benchmark_path,
            pred_program_path=os.path.join(self.get_run_dir(run_id), "pred_programs"),
            log_fname=os.path.join(self.get_run_dir(run_id), f"{run_id}_eval.jsonl"),
            dataset_name='osunlp/ScienceAgentBench',
            split='validation',
            instance_ids=["1", "2", "3"],  # TODO: remove this after debugging
            max_workers=os.cpu_count()//2,
            force_rebuild=False,  # TODO: not supported yet
            cache_level="base",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=1800,
            openai_api_key=None,  # TODO: get from .env
        )

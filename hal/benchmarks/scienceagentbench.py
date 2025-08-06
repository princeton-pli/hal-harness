from typing import Dict, Any
from pathlib import Path
import os
import sys
import json
import argparse
import time

from datasets import load_dataset
from .base_benchmark import BaseBenchmark

# Base directory of this file
this_file_dir = Path(__file__).resolve().parent

# Directory where upstream ScienceAgentBench code is cloned
code_root = this_file_dir / 'scienceagentbench' / 'SAB_code'
sys.path.insert(0, str(code_root))

# Path to the benchmark datasets and config in the upstream code
BENCHMARK_PATH = code_root / 'benchmark'

# Import required modules from SAB
from recover_pred_from_log import main as recover_main
from calculate_metrics import evaluate_best_run
from evaluation.harness import run_evaluation as docker_eval, docker_build

class ScienceAgentBench(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = 'scienceagentbench'
        self.requires_sandbox = False
        super().__init__(agent_dir, config, setup_script='hal/benchmarks/scienceagentbench/setup.sh')

        # Load validation split
        ds = load_dataset('osunlp/ScienceAgentBench', split='validation')
        self.benchmark: Dict[str, Any] = {}

        for task in ds:
            tid = str(task['instance_id'])
            # Prepare file mapping for HAL harness
            subpath = task['dataset_folder_tree'].split('\n')[0][4:]
            data_src = this_file_dir / 'scienceagentbench' / 'ScienceAgentBench' / 'benchmark' / 'datasets' / subpath
            map_key = str(Path('benchmark/datasets') / subpath)
            self.benchmark[tid] = {**task, 'files': {map_key: str(data_src)}}

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        # Recover programs from logs
        self._recover_pred_from_log(agent_output, run_id)
        # Evaluate via Docker
        result_file = self._call_docker_eval(run_id)

        results: Dict[str, Any] = {}
        with open(result_file, 'r') as f:
            for idx, line in enumerate(f, start=1):
                if not line.strip():
                    results[str(idx)] = { 'valid_program': 0, 'codebert_score': 0, 'success_rate': 0, 'log_info': '' }
                else:
                    results[str(idx)] = json.loads(line)
        return {'agent_output': agent_output, 'eval_result': results}

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        agent_output = eval_results['agent_output']
        eval_result = eval_results['eval_result']

        run_log, eval_log = [], []
        total = len(self.benchmark)
        for i in range(1, total + 1):
            key = str(i)
            entry = agent_output.get(key)
            if isinstance(entry, str) and entry.startswith('TIMEOUT'):
                run_log.append({'history':[{'role':'assistant','content':'TIMEOUT'}], 'cost':0.0})
            else:
                run_log.append(entry)
            eval_log.append(eval_result[key])

        return evaluate_best_run([run_log], [eval_log])

    def _recover_pred_from_log(self, agent_output: Dict[str, Any], run_id: str):
        run_dir = Path(self.get_run_dir(run_id))
        log_file = run_dir / f"{run_id}_histories.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w') as f:
            for i in range(1, len(self.benchmark) + 1):
                entry = agent_output.get(str(i))
                if isinstance(entry, str) and entry.startswith('TIMEOUT'):
                    entry = {'history':[{'role':'assistant','content':'TIMEOUT'}], 'cost':0.0}
                f.write(json.dumps(entry) + "\n")

        args = argparse.Namespace(
            benchmark_name_or_path=str(BENCHMARK_PATH),
            log_fname=str(log_file),
            pred_program_path=str(run_dir / 'pred_programs'),
            is_opendevin=False
        )
        recover_main(args)

    def _call_docker_eval(self, run_id: str) -> str:
        run_dir = Path(self.get_run_dir(run_id))
        # Update Docker context dirs
        docker_build.BASE_IMAGE_BUILD_DIR = run_dir / docker_build.BASE_IMAGE_BUILD_DIR
        docker_build.INSTANCE_IMAGE_BUILD_DIR = run_dir / docker_build.INSTANCE_IMAGE_BUILD_DIR
        docker_eval.INSTANCE_IMAGE_BUILD_DIR = run_dir / docker_eval.INSTANCE_IMAGE_BUILD_DIR
        docker_eval.RUN_EVALUATION_LOG_DIR = run_dir / docker_eval.RUN_EVALUATION_LOG_DIR

        log_file = run_dir / f"{run_id}_eval.jsonl"
        docker_eval.main(
            benchmark_path=str(BENCHMARK_PATH),
            pred_program_path=str(run_dir / 'pred_programs'),
            log_fname=str(log_file),
            dataset_name='osunlp/ScienceAgentBench',
            split='validation',
            instance_ids=None,
            max_workers=os.cpu_count() // 2,
            force_rebuild=True,
            cache_level='none',
            clean=False,
            open_file_limit=4096,
            run_id=f"{run_id}_{time.strftime('%Y%m%d-%H%M%S')}",
            timeout=1800,
            openai_api_key=os.getenv('OPENAI_API_KEY',''),
            azure_openai_key=os.getenv('AZURE_OPENAI_KEY',''),
            azure_openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION','2024-12-01-preview'),
            azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT',''),
            azure_openai_deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME','')
        )
        return str(log_file)

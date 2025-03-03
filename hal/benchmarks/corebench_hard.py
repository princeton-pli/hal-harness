import json
import os
import urllib.request
import tarfile
import time
from typing import Dict, Any

from hal.utils.logging_utils import create_progress
from .base_benchmark import BaseBenchmark

class CoreBenchHard(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_hard"
        
        # Load tasks from core_test.json
        core_test_path = os.path.join(os.path.dirname(__file__), "corebench", "core_test.json")
        with open(core_test_path, 'r') as f:
            dataset = json.load(f)
        
        self.benchmark = {}
        self.benchmark_answers = {}
        
        capsules_dir = os.path.join(os.path.dirname(__file__), "corebench", "capsules")
        if not os.path.exists(capsules_dir):
            os.makedirs(capsules_dir)
            
        with create_progress() as progress:
            task_loading = progress.add_task("Loading/downloading tasks...", total=len(dataset))
            for task in dataset:
                progress.update(task_loading, advance=1)
                
                capsule_id = task["capsule_id"]
                capsule_dir = os.path.join(capsules_dir, capsule_id)

                # Check if capsule directory exists, if not download and extract it
                if not os.path.exists(capsule_dir):
                    self.__download_and_extract_capsule(capsules_dir, capsule_id)
                
                # Create task entry with prompt
                self.benchmark[capsule_id] = {
                    "prompt": task["task_prompt"],
                    "files": {"/root/environment/": capsule_dir}
                }
                
                # Store results
                self.benchmark_answers[capsule_id] = task["results"]
            
        super().__init__(agent_dir, config)
    
    def __download_and_extract_capsule(self, capsules_dir: str, capsule_id: str, max_retries=5, backoff_factor=1):
        """Downloads and extracts a capsule archive from the CoreBench repository."""
        capsule_dir = os.path.join(capsules_dir, capsule_id)
        capsule_url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"
        tar_path = os.path.join(capsules_dir, f"{capsule_id}.tar.gz")
        
        # Download with retry
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Downloading capsule {capsule_id} (attempt {attempt}/{max_retries})...")
                urllib.request.urlretrieve(capsule_url, tar_path)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise Exception(f"Failed to download capsule {capsule_id} after {max_retries} attempts: {e}")
                
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                print(f"Download failed, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        # Extract and cleanup
        try:
            print(f"Extracting capsule {capsule_id}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=capsules_dir)
            os.remove(tar_path)  # Remove tar file after successful extraction
        except Exception as e:
            if os.path.exists(tar_path):
                os.remove(tar_path)  # Clean up tar file even if extraction fails
            raise Exception(f"Failed to extract capsule {capsule_id}: {e}")
                
        return capsule_dir
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Run evaluation harness. This can score based on the agent's output, or by running an evaluation script on the entire environments returned by the agent (see AppWorld benchmark)."""
        results = {}
        dataset = self.get_dataset()
        
        for task_id, solution in agent_output.items():
            try:
                # Parse agent's answer
                answer = int(solution.strip())
                # Compare with correct answer
                correct = answer == dataset[task_id]["answer"]
                results[task_id] = {
                    "correct": correct,
                    "expected": dataset[task_id]["answer"],
                    "received": answer
                }
            except ValueError:
                results[task_id] = {
                    "correct": False,
                    "error": "Invalid number format"
                }
                
        return results
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy, successful tasks, and failed tasks IDs"""
        correct = sum(1 for r in eval_results.values() if r.get("correct", False))
        total = len(eval_results)
        
        return {
            "accuracy": correct / total,
            "successful_tasks": [
                task_id for task_id, result in eval_results.items()
                if result.get("correct", False)
            ],
            "failed_tasks": [
                task_id for task_id, result in eval_results.items()
                if not result.get("correct", False)
            ]
        }

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
            
        total_tasks = len(dataset)
        for i, task in enumerate(dataset, 1):
            capsule_id = task["capsule_id"]
            capsule_dir = os.path.join(capsules_dir, capsule_id)

            # Check if capsule directory exists, if not download and extract it
            if not os.path.exists(capsule_dir):
                self.__download_and_extract_capsule(capsules_dir, capsule_id, task_number=i, total_tasks=total_tasks)
            
            # Create task entry with prompt
            self.benchmark[capsule_id] = {
                "prompt": task["task_prompt"],
                "files": {"/root/environment/": capsule_dir}
            }
            
            # Store results
            self.benchmark_answers[capsule_id] = task["results"]
            
        super().__init__(agent_dir, config)
    
    def __download_and_extract_capsule(self, capsules_dir: str, capsule_id: str, task_number=None, total_tasks=None, max_retries=5, backoff_factor=1):
        """Downloads and extracts a capsule archive from the CoreBench repository."""
        capsule_dir = os.path.join(capsules_dir, capsule_id)
        capsule_url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"
        tar_path = os.path.join(capsules_dir, f"{capsule_id}.tar.gz")
        
        # Download with retry and progress tracking
        for attempt in range(1, max_retries + 1):
            try:
                # Create a progress bar for the download
                with create_progress() as progress:
                    task_info = f"({task_number}/{total_tasks})" if task_number and total_tasks else ""
                    download_task = progress.add_task(f"Downloading capsule {capsule_id} {task_info}...", total=None)
                    
                    # First, make a HEAD request to get the content length
                    with urllib.request.urlopen(urllib.request.Request(capsule_url, method='HEAD')) as response:
                        file_size = int(response.headers.get('Content-Length', 0))
                        if file_size > 0:
                            progress.update(download_task, total=file_size)
                    
                    # Define a progress hook for urlretrieve
                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if downloaded > total_size:  # Avoid progress bar overflow
                            downloaded = total_size
                        progress.update(download_task, completed=downloaded)
                    
                    # Download the file with progress reporting
                    urllib.request.urlretrieve(capsule_url, tar_path, reporthook=report_progress)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise Exception(f"Failed to download capsule {capsule_id} after {max_retries} attempts: {e}")
                
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                print(f"Download failed, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        # Extract and cleanup with granular progress bar
        try:
            with create_progress() as progress:
                task_info = f"({task_number}/{total_tasks})" if task_number and total_tasks else ""
                
                # Open the tar file to get member information
                with tarfile.open(tar_path, "r:gz") as tar:
                    # Get all members in the archive
                    members = tar.getmembers()
                    total_files = len(members)
                    
                    # Create progress bar with total files count
                    extract_task = progress.add_task(
                        f"Extracting capsule {capsule_id} {task_info} ...", 
                        total=total_files
                    )
                    
                    # Extract each file individually and update progress
                    for i, member in enumerate(members, 1):
                        tar.extract(member, path=capsules_dir)
                        progress.update(
                            extract_task, 
                            completed=i,
                            description=f"Extracting capsule {capsule_id} {task_info} ..."
                        )
                
                # Remove tar file after successful extraction
                os.remove(tar_path)
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

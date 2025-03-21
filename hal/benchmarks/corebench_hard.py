import os
import shutil
from typing import Dict, Any

from .corebench import CoreBench

class CoreBenchHard(CoreBench):
    """CoreBench benchmark with hard difficulty level"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_hard"
        super().__init__(agent_dir, config)
        
    def _construct_prompt(self, task):
        """
        Constructs the prompt for the hard difficulty level.
        
        Args:
            task: The task dictionary containing task_prompt and results
            
        Returns:
            The constructed prompt string
        """
        task_prompt = task["task_prompt"]
        json_fields = str(task["results"][0].keys())
        return f"Task: codeocean_hard\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt}. Save your report to a file named report.json in the environment directory you started in that contains the capsule itself, where you fill in all of the following fields: {json_fields}. You should install all of the requirements found in the Readme file and then run the commands necessary to answer the questions."
    
    def _get_capsule_files_dict(self, capsule_dir: str) -> Dict[str, str]:
        """
        Creates a dictionary mapping target paths to source paths for all files in the capsule directory.
        For the hard difficulty level, the results directory, REPRODUCING.md, environment directory,
        and run scripts are removed.
        
        Args:
            capsule_dir: Path to the capsule directory
            
        Returns:
            Dictionary where keys are target paths in /root/environment/ and values are source paths
        """
        # Get the complete files dictionary from the base implementation
        files_dict = super()._get_capsule_files_dict(capsule_dir)
        
        # Filter out files based on hard difficulty criteria
        filtered_dict = {}
        for target_path, source_path in files_dict.items():
            normalized_path = target_path.replace("\\", "/")
            
            # Skip files in results directory
            if "/results/" in normalized_path:
                continue
                
            # Skip REPRODUCING.md file
            if normalized_path.endswith("/REPRODUCING.md"):
                continue
                
            # Skip files in environment directory
            if "/environment/environment/" in normalized_path:
                continue
                
            # Skip run scripts
            if normalized_path.endswith("/code/run.sh") or normalized_path.endswith("/code/run"):
                continue
                
            # Include all other files
            filtered_dict[target_path] = source_path
        
        return filtered_dict

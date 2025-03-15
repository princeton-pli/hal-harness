import os
import shutil
from typing import Dict, Any

from .corebench import CoreBench

class CoreBenchEasy(CoreBench):
    """CoreBench benchmark with easy difficulty level"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_easy"
        super().__init__(agent_dir, config)
        
    def _construct_prompt(self, task):
        """
        Constructs the prompt for the easy difficulty level.
        
        Args:
            task: The task dictionary containing task_prompt and results
            
        Returns:
            The constructed prompt string
        """
        json_fields = str(task["results"][0].keys())
        return f"Task: codeocean_easy\n\nYour goal is to answer questions about the output of scientific code. You should read through the files in the `results` directory to answer the following questions: {json_fields}. Save your answers to a file named report.json in the environment directory you started in that contains the capsule directory itself whose keys are the questions and values are the answers. **You should not actually run or execute any code.** All answers can be obtained by reading through the results directory."
    
    def _get_capsule_files_dict(self, capsule_dir: str) -> Dict[str, str]:
        """
        Creates a dictionary mapping target paths to source paths for all files in the capsule directory.
        For the easy difficulty level, all files are kept.
        
        Args:
            capsule_dir: Path to the capsule directory
            
        Returns:
            Dictionary where keys are target paths in /root/environment/ and values are source paths
        """
        # For easy difficulty, we keep all files, so we can just use the base implementation
        return super()._get_capsule_files_dict(capsule_dir)

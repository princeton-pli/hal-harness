import os
import shutil
from typing import Dict, Any

from .corebench import CoreBench

class CoreBenchEasy(CoreBench):
    """CoreBench benchmark with easy difficulty level"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_easy"
        super().__init__(agent_dir, config)
    
    def __get_capsule_files_dict(self, capsule_dir: str) -> Dict[str, str]:
        """
        Creates a dictionary mapping target paths to source paths for all files in the capsule directory.
        For the easy difficulty level, all files are kept.
        
        Args:
            capsule_dir: Path to the capsule directory
            
        Returns:
            Dictionary where keys are target paths in /root/environment/ and values are source paths
        """
        # For easy difficulty, we keep all files, so we can just use the base implementation
        return super().__get_capsule_files_dict(capsule_dir)

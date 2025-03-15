import os
import shutil
from typing import Dict, Any

from .corebench import CoreBench

class CoreBenchMedium(CoreBench):
    """CoreBench benchmark with medium difficulty level"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_medium"
        super().__init__(agent_dir, config)
    
    def _get_capsule_files_dict(self, capsule_dir: str) -> Dict[str, str]:
        """
        Creates a dictionary mapping target paths to source paths for all files in the capsule directory.
        For the medium difficulty level, the results directory is removed, but REPRODUCING.md, 
        environment directory, and run scripts are kept.
        
        Args:
            capsule_dir: Path to the capsule directory
            
        Returns:
            Dictionary where keys are target paths in /root/environment/ and values are source paths
        """
        # Get the complete files dictionary from the base implementation
        files_dict = super()._get_capsule_files_dict(capsule_dir)
        
        # Filter out files in the results directory
        filtered_dict = {}
        for target_path, source_path in files_dict.items():
            normalized_path = target_path.replace("\\", "/")
            
            # Skip files in results directory
            if "/results/" in normalized_path:
                continue
                
            # Include all other files
            filtered_dict[target_path] = source_path
        
        return filtered_dict

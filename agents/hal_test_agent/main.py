"""Minimal test agent for hal-test command - corebench_easy only"""
import json
import os
from pathlib import Path

def run(input_data: dict, **kwargs) -> dict:
    """
    Minimal agent for corebench_easy testing.
    Reads the results directory and extracts answers.
    """
    results = {}
    
    for task_id, task_data in input_data.items():
        try:
            # For corebench_easy, just read the results from the provided files
            # The files are already copied to our working directory
            results_dir = Path("environment/results")
            
            if not results_dir.exists():
                results[task_id] = json.dumps({"error": "Results directory not found"})
                continue
            
            # Find JSON files in results directory
            json_files = list(results_dir.glob("*.json"))
            
            if not json_files:
                # Try looking for other result files
                all_files = list(results_dir.glob("*"))
                if all_files:
                    # Read the first file and try to extract data
                    with open(all_files[0], 'r') as f:
                        content = f.read()
                        # Simple extraction - this is just for testing
                        result_dict = {"data": content[:100]}  # First 100 chars
                else:
                    result_dict = {"error": "No result files found"}
            else:
                # Read the first JSON file
                with open(json_files[0], 'r') as f:
                    result_dict = json.load(f)
            
            # For testing, return a simple result based on prompt
            # In corebench_easy, we need to extract specific fields
            # This is a minimal implementation that returns some data
            if isinstance(result_dict, dict):
                results[task_id] = json.dumps(result_dict)
            else:
                results[task_id] = json.dumps({"result": str(result_dict)})
                
        except Exception as e:
            results[task_id] = json.dumps({"error": str(e)})
    
    return results
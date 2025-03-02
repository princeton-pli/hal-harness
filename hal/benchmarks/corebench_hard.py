import json
import os
from typing import Dict, Any
from .base_benchmark import BaseBenchmark

class CoreBenchHard(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_hard"
        self.benchmark = {
            "task1": {
                "prompt": "What is 2 + 2?",
                "answer": 4
            },
            "task2": {
                "prompt": "What is 3 * 4?",
                "answer": 12
            }
        }
        super().__init__(agent_dir, config)
        
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
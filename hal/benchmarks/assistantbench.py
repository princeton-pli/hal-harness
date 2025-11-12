from browsergym.assistantbench.evaluation.evaluator import question_scorer

from .base_benchmark import BaseBenchmark
from typing import Dict, Any
from datasets import load_dataset

class AssistantBenchBenchmark(BaseBenchmark):

    """AssistantBench benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = 'assistantbench'):
        self.benchmark_name = benchmark_name

        self.dataset = list(load_dataset("AssistantBench/AssistantBench", split="validation"))
        self.benchmark = {task['id']: task for task in self.dataset}
        super().__init__(agent_dir, config)
    
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs against gold standard answers"""
        # Normalize agent output to handle both old and new formats
        normalized_output = self._normalize_agent_output(agent_output)
        
        scores = []
        answers = []
        successful_tasks = []
        failed_tasks = []

        task_num = 0

        for task_id, agent_answer in normalized_output.items():
                task = self.benchmark.get(task_id)
                gold_answer = task["answer"]
                try:
                    score, has_answer = question_scorer(agent_answer, gold_answer)
                except Exception as e:
                    print(f"Error scoring task {task_id}: {e}")
                    score = 0
                    has_answer = 0
                scores.append(score)
                answers.append(has_answer)
                task_num += 1
                
                if agent_answer == gold_answer:
                    successful_tasks.append(task_id)
                else:
                    failed_tasks.append(task_id)
        
        return {
             "scores" : scores,
             "answers" : answers,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks

        }
            
                     
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get evaluation metrics"""

        average_score = sum(eval_results["scores"]) / len(self.benchmark)
        precision = sum(eval_results["scores"]) / sum(eval_results["answers"]) if sum(eval_results["answers"]) > 0 else 0
        exact_matches = len(eval_results["successful_tasks"]) / len(self.benchmark)
        answer_rate = sum(eval_results["answers"]) / len(self.benchmark)

        return {
            "average_score": float(average_score),
            "precision": float(precision),
            "exact_matches": exact_matches,
            "answer_rate": answer_rate,
            "successful_tasks": eval_results["successful_tasks"],
            "failed_tasks": eval_results["failed_tasks"]

        }
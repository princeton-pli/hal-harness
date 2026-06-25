import re
from typing import Any, Dict
from datasets import load_dataset
from .base_benchmark import BaseBenchmark

_LETTERS = ("A", "B", "C", "D")


def _extract_choice(text: str) -> str | None:
    matches = re.findall(r"\b([A-D])\b", text.upper())
    return matches[-1] if matches else None


class MMLUBenchmark(BaseBenchmark):
    _ground_truth_keys = {"answer"}

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "mmlu"
        self.setup_script = None
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=False, setup_script=None)

        dataset = load_dataset("cais/mmlu", "all", split="test")
        self.benchmark = {}
        for i, row in enumerate(dataset):
            task_id = f"mmlu_{i:05d}"
            choices_text = "\n".join(
                f"{letter}. {choice}" for letter, choice in zip(_LETTERS, row["choices"])
            )
            self.benchmark[task_id] = {
                "question": f"{row['question']}\n\n{choices_text}",
                "answer": _LETTERS[row["answer"]],
                "subject": row["subject"],
            }

    def get_task_prompts(self) -> Dict[str, str]:
        return {task_id: task["question"] for task_id, task in self.benchmark.items()}

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        results = {}
        for task_id, raw_answer in agent_output.items():
            gt = self.benchmark[task_id]["answer"]
            predicted = _extract_choice(str(raw_answer))
            correct = predicted == gt
            results[task_id] = {
                "score": int(correct),
                "reward": float(correct),
                "predicted": str(raw_answer).strip()[:200],
                "label": gt,
            }
        return results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        total = len(eval_results)
        correct = sum(1 for r in eval_results.values() if r["score"] > 0)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "successful_tasks": [t for t, r in eval_results.items() if r["score"] > 0],
            "failed_tasks": [t for t, r in eval_results.items() if r["score"] == 0],
        }

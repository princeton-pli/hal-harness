import re
from typing import Any, Dict
from datasets import load_dataset
from .base_benchmark import BaseBenchmark

# "maybe" first — avoids partial-matching "yes"/"no" inside "maybe"
_LABELS = ("maybe", "yes", "no")


def _extract_label(text: str) -> str | None:
    text = text.strip().lower()
    for label in _LABELS:
        if re.search(r"\b" + label + r"\b", text):
            return label
    return None


class PubMedQABenchmark(BaseBenchmark):
    _ground_truth_keys = {"label"}

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "pubmedqa"
        self.setup_script = None
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=False, setup_script=None)

        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        self.benchmark = {}
        for row in ds:
            pubid = str(row["pubid"])
            context_text = " ".join(row["context"]["contexts"])
            self.benchmark[pubid] = {
                "question": row["question"],
                "context": context_text,
                "label": row["final_decision"].strip().lower(),
            }

    def get_task_prompts(self) -> Dict[str, str]:
        return {
            task_id: (
                f"Context: {task['context']}\n\n"
                f"Question: {task['question']}\n\n"
                "Answer yes, no, or maybe."
            )
            for task_id, task in self.benchmark.items()
        }

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        results = {}
        for task_id, raw_answer in agent_output.items():
            label = self.benchmark[task_id]["label"]
            predicted = _extract_label(str(raw_answer))
            correct = predicted == label
            results[task_id] = {
                "score": int(correct),
                "reward": float(correct),
                "predicted": str(raw_answer).strip()[:200],
                "label": label,
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

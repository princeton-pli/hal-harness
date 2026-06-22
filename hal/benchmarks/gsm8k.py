import re
from typing import Dict, Any
from datasets import load_dataset
from .base_benchmark import BaseBenchmark


def _extract_number(text: str) -> str | None:
    """Return the final numeric value from a GSM8K answer or model response.

    Ground truth format is '... #### 42'; model output may be free-form.
    """
    # Ground truth: extract number after ####
    gt_match = re.search(r"####\s*([\d,.-]+)", text)
    if gt_match:
        return gt_match.group(1).replace(",", "").strip()
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else None


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K grade-school math benchmark (8,500 test problems)."""

    _ground_truth_keys = {"answer"}

    def __init__(
        self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = "gsm8k"
    ):
        self.benchmark_name = benchmark_name
        self.setup_script = None
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=False, setup_script=None)

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        self.benchmark = {
            f"gsm8k_{i:04d}": {
                "task_id": f"gsm8k_{i:04d}",
                "question": record["question"],
                "answer": record["answer"],  # stripped from agent input via _ground_truth_keys
            }
            for i, record in enumerate(dataset)
        }

    def get_task_prompts(self) -> Dict[str, str]:
        return {task_id: task["question"] for task_id, task in self.benchmark.items()}

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        eval_results = {}
        for task_id, agent_answer in agent_output.items():
            if isinstance(agent_answer, dict):
                agent_answer = agent_answer.get("answer", agent_answer.get("raw_response", ""))

            gt_raw = self.benchmark[task_id]["answer"]
            gt_number = _extract_number(gt_raw)
            pred_number = _extract_number(str(agent_answer))

            correct = (
                gt_number is not None
                and pred_number is not None
                and gt_number == pred_number
            )
            eval_results[task_id] = {
                "score": int(correct),
                "reward": float(correct),
                "correct": correct,
                "predicted": pred_number,
                "ground_truth": gt_number,
                "explanation": f"predicted={pred_number!r}, ground_truth={gt_number!r}",
            }
        return eval_results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        correct = sum(1 for v in eval_results.values() if v.get("score", 0) > 0)
        total = len(eval_results)
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "successful_tasks": [t for t, v in eval_results.items() if v.get("score", 0) > 0],
            "failed_tasks": [t for t, v in eval_results.items() if v.get("score", 0) == 0],
        }

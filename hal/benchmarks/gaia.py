import os
from typing import Dict, Any, List, Optional
from .base_benchmark import BaseBenchmark
from .GAIA.scoring_utils import question_scorer
from huggingface_hub import hf_hub_download, get_token, snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
    EntryNotFoundError,
)
from datasets import load_dataset


GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_DEFAULT_CONFIG = "2023_all"
GAIA_LEVELS = {
    "2023": [1, 2, 3],
}


class GaiaBenchmark(BaseBenchmark):
    """Gaia benchmark implementation"""

    def __init__(
        self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = "gaia"
    ):
        self.benchmark_name = benchmark_name
        self.setup_script = None
        self.requires_sandbox = False
        super().__init__(
            agent_dir,
            config,
            requires_sandbox=self.requires_sandbox,
            setup_script=self.setup_script,
        )

        dataset = self._load_gaia_dataset(
            config_split=GAIA_DEFAULT_CONFIG, split="validation"
        )

        self.benchmark = {}
        for record in dataset:
            task_id = record["task_id"]
            self.benchmark[task_id] = record
            if record.get("file_name"):
                self.benchmark[task_id]["files"] = {
                    record["file_name"]: record.get("file_path", "")
                }

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Evaluate agent outputs using Gaia evaluation while capturing task metrics.

        Merges evaluation results with agent's reliability metric fields (taken_actions,
        confidence, conversation_history, etc.) for compatibility with analyze_reliability.py.
        """

        eval_results: Dict[str, Any] = {}

        for task_id, agent_answer in agent_output.items():
            answer_payload = agent_answer

            if isinstance(agent_answer, dict):
                answer_payload = agent_answer.get(
                    "answer", agent_answer.get("raw_response")
                )

            gt_answer = self.benchmark[task_id]["Final answer"]
            score, explanation = question_scorer(str(answer_payload), str(gt_answer))

            # Start with evaluation results
            result = {
                "score": score,
                "explanation": explanation,
                "reward": float(score),  # For consistency with analyze_reliability.py
            }

            # Merge agent's reliability metric fields if present
            if isinstance(agent_answer, dict):
                reliability_fields = [
                    "taken_actions",
                    "confidence",
                    "confidence_details",
                    "conversation_history",
                    "metrics",
                    "fault_injection",
                    "compliance",
                    "structural_perturbation",
                    "llm_compliance",
                    "llm_recovery",
                ]
                for field in reliability_fields:
                    if field in agent_answer:
                        result[field] = agent_answer[field]

            eval_results[task_id] = result

        return eval_results

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.

        Args:
            eval_results: Dictionary containing evaluation results

        Returns:
            Dictionary with calculated metrics and task lists
        """

        # calculate accuracy for overall and for each level separately
        overall_correct_count = 0
        level_correct_count = {"level_1": 0, "level_2": 0, "level_3": 0}
        level_count = {"level_1": 0, "level_2": 0, "level_3": 0}

        for task_id, task_output in self.benchmark.items():
            if task_id in eval_results:
                # update the level count
                if str(self.benchmark[task_id]["Level"]) == "1":
                    level_count["level_1"] += 1
                elif str(self.benchmark[task_id]["Level"]) == "2":
                    level_count["level_2"] += 1
                elif str(self.benchmark[task_id]["Level"]) == "3":
                    level_count["level_3"] += 1

                # if the agent got the task correct, update the level and overall correct counts
                if int(eval_results[task_id]["score"]) > 0:
                    overall_correct_count += 1
                    if str(self.benchmark[task_id]["Level"]) == "1":
                        level_correct_count["level_1"] += 1
                    elif str(self.benchmark[task_id]["Level"]) == "2":
                        level_correct_count["level_2"] += 1
                    elif str(self.benchmark[task_id]["Level"]) == "3":
                        level_correct_count["level_3"] += 1
            else:
                level_count[f"level_{str(self.benchmark[task_id]['Level'])}"] += 1

        successful_tasks = [
            task_id
            for task_id, task_output in eval_results.items()
            if int(task_output["score"]) > 0
        ]
        failed_tasks = [
            task_id
            for task_id, task_output in eval_results.items()
            if int(task_output["score"]) == 0
        ]

        results = {
            "accuracy": overall_correct_count / len(eval_results),
            "level_1_accuracy": level_correct_count["level_1"] / level_count["level_1"]
            if level_count["level_1"] > 0
            else None,
            "level_2_accuracy": level_correct_count["level_2"] / level_count["level_2"]
            if level_count["level_2"] > 0
            else None,
            "level_3_accuracy": level_correct_count["level_3"] / level_count["level_3"]
            if level_count["level_3"] > 0
            else None,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }
        return results

    def _load_gaia_dataset(self, config_split: str, split: str) -> List[Dict[str, Any]]:
        """Load GAIA data using the Parquet format via snapshot_download."""
        try:
            year, level_suffix = config_split.split("_", 1)
        except ValueError as exc:
            raise RuntimeError(f"Invalid GAIA config '{config_split}'.") from exc

        if level_suffix == "all":
            levels = GAIA_LEVELS.get(year, [])
        else:
            try:
                parsed_level = int(level_suffix.replace("level", ""))
            except ValueError as exc:
                raise RuntimeError(f"Unsupported GAIA split '{config_split}'.") from exc
            levels = [parsed_level]

        if not levels:
            raise RuntimeError(f"No GAIA levels available for year '{year}'.")

        token = self._resolve_hf_token()
        if token is None:
            raise RuntimeError(
                "Access to gaia-benchmark/GAIA requires a Hugging Face token. "
                "Set HF_TOKEN (or login with `huggingface-cli login`)."
            )

        # Download the entire dataset repository using snapshot_download
        try:
            data_dir = snapshot_download(
                repo_id=GAIA_REPO_ID,
                repo_type="dataset",
                token=token,
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                "Access to gaia-benchmark/GAIA is gated. Confirm your Hugging Face account "
                "has access and that you are logged in."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Dataset repository {GAIA_REPO_ID} is unavailable."
            ) from exc

        records: List[Dict[str, Any]] = []

        # Load dataset for each level
        for level in levels:
            dataset_name = f"{year}_level{level}"
            try:
                dataset = load_dataset(data_dir, dataset_name, split=split)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load GAIA dataset '{dataset_name}' split '{split}': {exc}"
                ) from exc

            for example in dataset:
                record = dict(example)

                # Handle file_path - join with data_dir if present
                file_name = record.get("file_name") or ""
                if file_name and record.get("file_path"):
                    record["file_path"] = os.path.join(data_dir, record["file_path"])
                else:
                    record["file_path"] = ""

                records.append(record)

        if not records:
            raise RuntimeError(
                f"No GAIA records loaded for split '{split}' and levels {levels}. "
                "Ensure the dataset is available and your token has access."
            )

        return records

    def _download_gaia_file(self, relative_path: str, token: str) -> str:
        try:
            return hf_hub_download(
                repo_id=GAIA_REPO_ID,
                filename=relative_path,
                repo_type="dataset",
                token=token,
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                "Access to gaia-benchmark/GAIA is gated. Confirm your Hugging Face account "
                "has access and that you are logged in."
            ) from exc
        except EntryNotFoundError as exc:
            raise RuntimeError(
                f"GAIA dataset file '{relative_path}' was not found in {GAIA_REPO_ID}."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Dataset repository {GAIA_REPO_ID} is unavailable."
            ) from exc

    def _resolve_hf_token(self) -> Optional[str]:
        env_keys = ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"]
        for key in env_keys:
            token = os.environ.get(key)
            if token:
                return token
        return get_token()

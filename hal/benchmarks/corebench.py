import json
import os
import urllib.request
import tarfile
import time
from typing import Dict, Any
import numpy as np
from scipy.stats import t
import math
import logging
from decimal import Decimal

from hal.utils.logging_utils import create_progress
from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)

# Capsule-specific answer aliases for string tasks
_ANSWER_ALIASES: Dict[str, Dict[str, set]] = {
    "capsule-2816027": {
        "fig For CTCF Signature Enrichment, report the name of the group with the highest median GSVA score.":
            {"D538G", "MCF7_D538G"},
    },
    "capsule-3639589": {
        "fig Report the color of the line with the highest maximum activation for target memory activation, DM.":
            {"blue", "green"},
    },
    "capsule-2151475": {
        "fig Report the name of the university ranked #1 by impact factor.":
            {"Chicago", "University of Chicago"},
        "fig Report the name of the journal with the highest 2011 impact factor from the analysis of 30 journals.":
            {"JEL", "Journal of Economic Literature"},
    },
    "capsule-0152700": {
        "Given the Kruskal-Wallis for Group 0-2 (Group 1 vs. Group 3), what is the p-value?":
            {"2.341092434893948e-10", "1.0728737137836577e-09", "2.0438111587013562e-09"},
    },
}

# Capsule-specific list answer aliases: capsule_id -> question_key -> list of acceptable element subsets
# A list answer is accepted if its elements exactly match any one subset (case-insensitive)
_LIST_ANSWER_ALIASES: Dict[str, Dict[str, list]] = {
    "capsule-9477017": {
        "fig Pearson correlation coefficients between the estimated proportions of different cell types were calculated, what is the highest pearson correlation related to? Give response in a list of strings": [
            {"Smooth.1", "Endo.1"},
            {"Smooth.1", "B"},
        ],
    },
}

class CoreBench(BaseBenchmark):
    """Base class for CoreBench benchmarks of different difficulty levels"""

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        # Set benchmark_name in subclasses

        # Load tasks from core_test.json
        core_test_path = os.path.join(
            os.path.dirname(__file__), "corebench", "core_test.json"
        )

        # Check if core_test.json exists, if not, throw an error with instructions to decrypt
        if not os.path.exists(core_test_path):
            encrypted_file = os.path.join(
                os.path.dirname(__file__), "corebench", "core_test.json.gpg"
            )
            decrypt_command = (
                f"gpg --output {core_test_path} --decrypt {encrypted_file}"
            )
            raise FileNotFoundError(
                f'Have you decrypted core_test.json.gpg? Use the following command:\n{decrypt_command}. The password is "reproducibility".'
            )

        with open(core_test_path, "r") as f:
            dataset = json.load(f)

        self.benchmark = {}
        self.benchmark_answers = {}

        capsules_dir = os.path.join(os.path.dirname(__file__), "corebench", "capsules")
        if not os.path.exists(capsules_dir):
            os.makedirs(capsules_dir)

        total_tasks = len(dataset)
        for i, task in enumerate(dataset, 1):
            capsule_id = task["capsule_id"]
            capsule_dir = os.path.join(capsules_dir, capsule_id)

            # Skip capsules not present on disk. Capsules are distributed via
            # blob storage in the prefect/Azure Batch flow (one capsule per task),
            # and the upstream auto-download is broken for ~1/4 of CORE-bench
            # capsules anyway. Local users should pre-populate the capsules dir
            # (e.g. from capsules_share.tar.gz) before running.
            if not os.path.exists(capsule_dir):
                print(
                    f"[corebench] Skipping {capsule_id}: not present at {capsule_dir}"
                )
                continue

            # Create task entry with prompt
            # Use the _construct_prompt method if it exists in the subclass, otherwise use the default prompt
            prompt = self._construct_prompt(task)
            self.benchmark[capsule_id] = {
                "prompt": prompt,
                "files": self._get_capsule_files_dict(capsule_dir),
                "gpu": True if "gpu" in task else False,
            }

            # Store results
            self.benchmark_answers[capsule_id] = task["results"]

        super().__init__(agent_dir, config)

    def _get_capsule_files_dict(self, capsule_dir: str) -> Dict[str, str]:
        """
        Creates a dictionary mapping target paths to source paths for all files in the capsule directory.
        This base implementation includes all files.
        Subclasses should override this method to implement difficulty-specific filtering.

        Args:
            capsule_dir: Path to the capsule directory

        Returns:
            Dictionary where keys are target paths in /root/environment/ and values are source paths
        """
        # Create the complete files dictionary
        files_dict = {}
        for root, _, files in os.walk(capsule_dir):
            for file in files:
                source_path = os.path.join(root, file)
                # Calculate the relative path from capsule_dir
                rel_path = os.path.relpath(source_path, capsule_dir)
                # Create the target path
                target_path = os.path.join("/root/environment/", rel_path)
                # Add to dictionary
                files_dict[target_path] = source_path

        # Subclasses will override this method to filter out entries based on difficulty level
        return files_dict

    def __download_and_extract_capsule(
        self,
        capsules_dir: str,
        capsule_id: str,
        task_number=None,
        total_tasks=None,
        max_retries=5,
        backoff_factor=1,
    ):
        """Downloads and extracts a capsule archive from the CoreBench repository."""
        # FIXME: this doesn't respect the --max_tasks flag
        # Expected: --max_tasks 2 only downloads 2 capsules
        # Actual: --max_tasks 2 downloads all capsules, then runs only 2
        capsule_dir = os.path.join(capsules_dir, capsule_id)
        capsule_url = f"https://corebench.cs.princeton.edu/capsules/{capsule_id}.tar.gz"
        tar_path = os.path.join(capsules_dir, f"{capsule_id}.tar.gz")

        # Download with retry and progress tracking
        for attempt in range(1, max_retries + 1):
            try:
                # Create a progress bar for the download
                with create_progress() as progress:
                    task_info = (
                        f"({task_number}/{total_tasks})"
                        if task_number and total_tasks
                        else ""
                    )
                    download_task = progress.add_task(
                        f"Downloading capsule {capsule_id} {task_info}...", total=None
                    )

                    # First, make a HEAD request to get the content length
                    with urllib.request.urlopen(
                        urllib.request.Request(capsule_url, method="HEAD")
                    ) as response:
                        file_size = int(response.headers.get("Content-Length", 0))
                        if file_size > 0:
                            progress.update(download_task, total=file_size)

                    # Define a progress hook for urlretrieve
                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if downloaded > total_size:  # Avoid progress bar overflow
                            downloaded = total_size
                        progress.update(download_task, completed=downloaded)

                    # Download the file with progress reporting
                    urllib.request.urlretrieve(
                        capsule_url, tar_path, reporthook=report_progress
                    )
                break
            except Exception as e:
                if attempt == max_retries:
                    raise Exception(
                        f"Failed to download capsule {capsule_id} after {max_retries} attempts: {e}"
                    )

                sleep_time = backoff_factor * (2 ** (attempt - 1))
                logger.warning(f"Download failed, retrying in {sleep_time}s...")
                time.sleep(sleep_time)

        # Extract and cleanup with granular progress bar
        try:
            with create_progress() as progress:
                task_info = (
                    f"({task_number}/{total_tasks})"
                    if task_number and total_tasks
                    else ""
                )

                # Open the tar file to get member information
                with tarfile.open(tar_path, "r:gz") as tar:
                    # Get all members in the archive
                    members = tar.getmembers()
                    total_files = len(members)

                    # Create progress bar with total files count
                    extract_task = progress.add_task(
                        f"Extracting capsule {capsule_id} {task_info} ...",
                        total=total_files,
                    )

                    # Extract each file individually and update progress
                    for i, member in enumerate(members, 1):
                        tar.extract(member, path=capsules_dir)
                        progress.update(
                            extract_task,
                            completed=i,
                            description=f"Extracting capsule {capsule_id} {task_info} ...",
                        )

                # Remove tar file after successful extraction
                os.remove(tar_path)
        except Exception as e:
            if os.path.exists(tar_path):
                os.remove(tar_path)  # Clean up tar file even if extraction fails
            raise Exception(f"Failed to extract capsule {capsule_id}: {e}")

        return capsule_dir

    def evaluate_output(
        self, agent_output: Dict[str, Any], run_id: str
    ) -> Dict[str, Any]:
        """Run evaluation harness. This can score based on the agent's output, or by running an evaluation script on the entire environments returned by the agent (see AppWorld benchmark)."""

        agent_output = self._normalize_agent_output(agent_output)
        results = {}

        for task_id, solution in agent_output.items():
            # Get the ground truth results for this task
            gt_result = self.benchmark_answers[task_id]

            # Calculate total questions from ground truth (regardless of parsing success)
            numeric_keys = [
                key
                for key in gt_result[0].keys()
                if isinstance(gt_result[0][key], (int, float))
            ]
            list_keys = [
                key
                for key in gt_result[0].keys()
                if isinstance(gt_result[0][key], list)
            ]
            string_keys = [
                key for key in gt_result[0].keys() if isinstance(gt_result[0][key], str)
            ]

            total_written_questions = (
                len([key for key in string_keys if "fig" not in key])
                + len([key for key in numeric_keys if "fig" not in key])
                + len([key for key in list_keys if "fig" not in key])
            )
            total_vision_questions = (
                len([key for key in string_keys if "fig" in key])
                + len([key for key in numeric_keys if "fig" in key])
                + len([key for key in list_keys if "fig" in key])
            )

            try:
                # Parse the agent's answer as a dictionary
                if type(solution) is str:
                    reported_result = json.loads(solution)
                elif type(solution) is dict:
                    reported_result = solution
                else:
                    raise ValueError(
                        f"Invalid solution format for task {task_id}: {solution}"
                    )

                # Evaluate the result using the prediction interval logic
                evaluation = self.__eval_result_json(gt_result, reported_result, task_id)

                results[task_id] = {
                    "correct_written_answers": evaluation["correct_written_answers"],
                    "correct_vision_answers": evaluation["correct_vision_answers"],
                    "total_written_questions": total_written_questions,
                    "total_vision_questions": total_vision_questions,
                }
            except Exception as e:
                results[task_id] = {
                    "correct_written_answers": 0,
                    "correct_vision_answers": 0,
                    "total_written_questions": total_written_questions,
                    "total_vision_questions": total_vision_questions,
                    "error": str(e),
                }

        return results
    
    def __get_decimal_places(self, value: float) -> int:
        """
        Determines the number of decimal places in a numeric value. Returns 0 for integers and full-precision values.
        """
        try:
            d = Decimal(str(value)).normalize()
            return max(0, -d.as_tuple().exponent)
        except Exception:
            return 0


    def __eval_result_json(self, gt_result: list, reported_result: Dict, task_id: str = ""):
        """Evaluates the reported result against the ground truth using prediction intervals."""

        # Returns the number of correctly answered questions in the result json
        correct_written_answers = 0
        correct_vision_answers = 0

        # Separate keys into numeric, string, and list types
        numeric_keys = [
            key
            for key in gt_result[0].keys()
            if isinstance(gt_result[0][key], (int, float))
        ]
        list_keys = [
            key for key in gt_result[0].keys() if isinstance(gt_result[0][key], list)
        ]
        string_keys = [
            key for key in gt_result[0].keys() if isinstance(gt_result[0][key], str)
        ]

        total_written_questions = (
            len([key for key in string_keys if "fig" not in key])
            + len([key for key in numeric_keys if "fig" not in key])
            + len([key for key in list_keys if "fig" not in key])
        )
        total_vision_questions = (
            len([key for key in string_keys if "fig" in key])
            + len([key for key in numeric_keys if "fig" in key])
            + len([key for key in list_keys if "fig" in key])
        )

        try:
            # For each value, convert to float if possible and remove the percentage sign
            for key in reported_result.keys():
                try:
                    if "%" in reported_result[key]:
                        reported_result[key] = reported_result[key].replace("%", "")
                    reported_result[key] = float(reported_result[key])
                except Exception:
                    pass

            # Calculate mean and standard error for numeric keys
            mean_result = {
                key: np.mean([result[key] for result in gt_result])
                for key in numeric_keys
            }
            std_dev_result = {
                key: np.std([result[key] for result in gt_result], ddof=1)
                for key in numeric_keys
            }
            sample_size = len(gt_result)

            # Calculate the 95% prediction interval t-value
            t_value = t.ppf(0.975, sample_size - 1)

            # Calculate adjusted prediction interval bounds with tolerance
            prediction_interval_bounds = {}
            for key in numeric_keys:
                # Calculate standard 95% prediction interval bounds
                L = mean_result[key] - t_value * std_dev_result[key] * math.sqrt(1 + 1 / sample_size)
                U = mean_result[key] + t_value * std_dev_result[key] * math.sqrt(1 + 1 / sample_size)
    
                # Determine decimal places for each ground truth value and take the maximum
                decimal_places = [self.__get_decimal_places(result[key]) for result in gt_result]
                d = max(decimal_places) if decimal_places else 0
    
                # Calculate tolerance term: 0.5 * 10^(-d)
                tolerance = 0.5 * (10 ** (-d))
    
                adjusted_lower = L - tolerance 
                adjusted_upper = U + tolerance
    
                prediction_interval_bounds[key] = (adjusted_lower, adjusted_upper)  

            try:
                for key in reported_result.keys():
                    if key in numeric_keys:
                        lower_bound, upper_bound = prediction_interval_bounds[key]
                        value = reported_result[key]
                        if (lower_bound <= value <= upper_bound
                            or np.isclose(value, lower_bound)
                            or np.isclose(value, upper_bound)):
                            if "fig" in key:
                                correct_vision_answers += 1
                            else:
                                correct_written_answers += 1
                    elif key in list_keys:
                        reported_list = reported_result[key]
                        list_aliases = (_LIST_ANSWER_ALIASES.get(task_id) or {}).get(key)
                        if list_aliases is not None:
                            reported_list_lower = {item.lower() for item in reported_list}
                            correct = any(
                                {v.lower() for v in subset} == reported_list_lower
                                for subset in list_aliases
                            )
                        else:
                            correct = reported_list == gt_result[0][key]
                        if correct:
                            if "fig" in key:
                                correct_vision_answers += 1
                            else:
                                correct_written_answers += 1
                    elif key in string_keys:
                        gt_val = str(gt_result[0][key])
                        reported_val = reported_result[key]
                        # Normalize boolean-equivalent reported values
                        # so that int/float 1 or 0 are accepted for "True"/"False" ground truths
                        if gt_val.lower() in ("true", "false"):
                            if isinstance(reported_val, bool):
                                reported_val = str(reported_val)
                            elif isinstance(reported_val, (int, float)):
                                if reported_val == 1:
                                    reported_val = "True"
                                elif reported_val == 0:
                                    reported_val = "False"
                        reported_val = str(reported_val)
                        alias_set = (_ANSWER_ALIASES.get(task_id) or {}).get(key)
                        if alias_set is not None:
                            correct = reported_val.lower() in {v.lower() for v in alias_set}
                        else:
                            correct = reported_val.lower() == gt_val.lower()
                        if correct:
                            if "fig" in key:
                                correct_vision_answers += 1
                            else:
                                correct_written_answers += 1
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error evaluating result: {e}")

        return {
            "correct_written_answers": correct_written_answers,
            "correct_vision_answers": correct_vision_answers,
            "total_written_questions": total_written_questions,
            "total_vision_questions": total_vision_questions,
        }

    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy, successful tasks, and failed tasks IDs"""
        # Initialize counters
        correct_written_tasks = 0
        correct_vision_tasks = 0
        correct_tasks = 0
        total_written_tasks = 0
        total_vision_tasks = 0
        total_tasks = len(eval_results)

        # For tracking successful and failed task IDs
        successful_tasks = []
        failed_tasks = []

        # Calculate task-based metrics
        for task_id, result in eval_results.items():
            written_correct = result.get("correct_written_answers", 0)
            vision_correct = result.get("correct_vision_answers", 0)
            written_total = result.get("total_written_questions", 0)
            vision_total = result.get("total_vision_questions", 0)

            # Check if task has written questions
            if written_total > 0:
                total_written_tasks += 1
                # Check if all written questions are correct
                if written_correct == written_total:
                    correct_written_tasks += 1

            # Check if task has vision questions
            if vision_total > 0:
                total_vision_tasks += 1
                # Check if all vision questions are correct
                if vision_correct == vision_total:
                    correct_vision_tasks += 1

            # Check if all questions in the task are correct
            if (
                written_correct == written_total
                and vision_correct == vision_total
                and (written_total > 0 or vision_total > 0)
            ):
                correct_tasks += 1
                successful_tasks.append(task_id)
            else:
                failed_tasks.append(task_id)

        # Calculate accuracies
        accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
        written_accuracy = (
            correct_written_tasks / total_written_tasks
            if total_written_tasks > 0
            else 0
        )
        vision_accuracy = (
            correct_vision_tasks / total_vision_tasks if total_vision_tasks > 0 else 0
        )

        return {
            "accuracy": accuracy,
            "written_accuracy": written_accuracy,
            "vision_accuracy": vision_accuracy,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }


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
        return f"Task: codeocean_easy\n\nYour goal is to answer questions about the output of scientific code. You should read through the files in the `environment/results` directory to answer the following questions: {json_fields}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. **You should not actually run or execute any code.** All answers can be obtained by reading through the results directory."

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


class CoreBenchMedium(CoreBench):
    """CoreBench benchmark with medium difficulty level"""

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_medium"
        super().__init__(agent_dir, config)

    def _construct_prompt(self, task):
        """
        Constructs the prompt for the medium difficulty level.

        Args:
            task: The task dictionary containing task_prompt and results

        Returns:
            The constructed prompt string
        """
        task_prompt = task["task_prompt"]
        json_fields = str(task["results"][0].keys())
        return f"Task: codeocean_medium\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {json_fields}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should read the instructions on how to reproduce the capsule in REPRODUCING.md."

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


class CoreBenchHard(CoreBench):
    """CoreBench benchmark with hard difficulty level"""

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "corebench_hard"
        super().__init__(agent_dir, config)

    def _construct_prompt(self, task):
        """
        Constructs the prompt for the hard difficulty level.

        Args:
            task: The task dictionary containing task_prompt and results

        Returns:
            The constructed prompt string
        """
        task_prompt = task["task_prompt"]
        json_fields = str(task["results"][0].keys())
        return f"Task: codeocean_hard\n\nYour goal is to test the computational reproducibility of the repository cloned to your current directory, which is code from a scientific paper. Specifically, you need to {task_prompt} to answer the following questions: {json_fields}. Your submitted answer should be a Python dictionary whose keys are exactly those questions and values are the answers. You should install all of the requirements found in the Readme file and then run the commands necessary to answer the questions."

    def _get_capsule_files_dict(self, capsule_dir: str) -> Dict[str, str]:
        """
        Creates a dictionary mapping target paths to source paths for all files in the capsule directory.
        For the hard difficulty level, the results directory, REPRODUCING.md, environment directory,
        and run scripts are removed.

        Args:
            capsule_dir: Path to the capsule directory

        Returns:
            Dictionary where keys are target paths in /root/environment/ and values are source paths
        """
        # Get the complete files dictionary from the base implementation
        files_dict = super()._get_capsule_files_dict(capsule_dir)

        # Filter out files based on hard difficulty criteria
        filtered_dict = {}
        for target_path, source_path in files_dict.items():
            normalized_path = target_path.replace("\\", "/")

            # Skip files in results directory
            if "/results/" in normalized_path:
                continue

            # Skip REPRODUCING.md file
            if normalized_path.endswith("/REPRODUCING.md"):
                continue

            # Skip files in environment directory
            if "/environment/environment/" in normalized_path:
                continue

            # Skip run scripts
            if normalized_path.endswith("/code/run.sh") or normalized_path.endswith(
                "/code/run"
            ):
                continue

            # Skip large training data files (testing tasks don't need training data)
            filename_lower = os.path.basename(normalized_path).lower()

            # Skip files with 'train' in the name (trainX.csv, mtrainX.csv, etc.)
            if "train" in filename_lower and filename_lower.endswith(
                (".csv", ".txt", ".json", ".parquet", ".h5", ".hdf5")
            ):
                continue

            # Skip cache directories (often contain 3GB+ of training artifacts)
            if "/cache/" in normalized_path or "/Cache/" in normalized_path:
                continue

            # Skip checkpoint files
            if filename_lower.startswith("checkpoint") or ".ckpt" in filename_lower:
                continue

            # Skip __pycache__ directories
            if "/__pycache__/" in normalized_path:
                continue

            # Include all other files
            filtered_dict[target_path] = source_path

        return filtered_dict

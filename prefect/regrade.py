"""
Post-hoc regrading of compiled CORE-bench results using the updated evaluation
logic (tolerance bounds, answer aliases, decimal precision).

Reads _UPLOAD.json + _RAW_SUBMISSIONS.jsonl from each run directory, re-runs
the evaluation against ground truth from core_test.json, and overwrites the
raw_eval_results + results.accuracy in the UPLOAD.json.

Usage:
    cd prefect && python regrade.py --core_test ../hal/benchmarks/corebench/core_test.json
                                    [--results_dir ../results/corebench_hard]
                                    [--filter codex]
                                    [--dry_run]
"""

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.stats import t

# ---------------------------------------------------------------------------
# Answer aliases (copied from updated corebench.py to keep script standalone)
# ---------------------------------------------------------------------------

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

_LIST_ANSWER_ALIASES: Dict[str, Dict[str, list]] = {
    "capsule-9477017": {
        "fig Pearson correlation coefficients between the estimated proportions of different cell types were calculated, what is the highest pearson correlation related to? Give response in a list of strings": [
            {"Smooth.1", "Endo.1"},
            {"Smooth.1", "B"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Evaluation logic (extracted from corebench.py)
# ---------------------------------------------------------------------------


def _get_decimal_places(value: float) -> int:
    try:
        d = Decimal(str(value)).normalize()
        return max(0, -d.as_tuple().exponent)
    except Exception:
        return 0


def _eval_result_json(
    gt_result: list, reported_result: Dict, task_id: str = ""
) -> Dict[str, Any]:
    """Evaluate a single task's reported result against ground truth."""
    correct_written_answers = 0
    correct_vision_answers = 0

    numeric_keys = [
        k for k in gt_result[0] if isinstance(gt_result[0][k], (int, float))
    ]
    list_keys = [k for k in gt_result[0] if isinstance(gt_result[0][k], list)]
    string_keys = [k for k in gt_result[0] if isinstance(gt_result[0][k], str)]

    total_written_questions = (
        len([k for k in string_keys if "fig" not in k])
        + len([k for k in numeric_keys if "fig" not in k])
        + len([k for k in list_keys if "fig" not in k])
    )
    total_vision_questions = (
        len([k for k in string_keys if "fig" in k])
        + len([k for k in numeric_keys if "fig" in k])
        + len([k for k in list_keys if "fig" in k])
    )

    try:
        # Normalize reported values
        for key in reported_result:
            try:
                val = reported_result[key]
                if isinstance(val, str) and "%" in val:
                    val = val.replace("%", "")
                reported_result[key] = float(val)
            except Exception:
                pass

        # Prediction intervals for numeric keys
        mean_result = {
            k: np.mean([r[k] for r in gt_result]) for k in numeric_keys
        }
        std_dev_result = {
            k: np.std([r[k] for r in gt_result], ddof=1) for k in numeric_keys
        }
        sample_size = len(gt_result)
        t_value = t.ppf(0.975, sample_size - 1)

        prediction_interval_bounds = {}
        for key in numeric_keys:
            L = mean_result[key] - t_value * std_dev_result[key] * math.sqrt(
                1 + 1 / sample_size
            )
            U = mean_result[key] + t_value * std_dev_result[key] * math.sqrt(
                1 + 1 / sample_size
            )
            decimal_places = [_get_decimal_places(r[key]) for r in gt_result]
            d = max(decimal_places) if decimal_places else 0
            tolerance = 0.5 * (10 ** (-d))
            prediction_interval_bounds[key] = (L - tolerance, U + tolerance)

        try:
            for key in reported_result:
                if key in numeric_keys:
                    lower, upper = prediction_interval_bounds[key]
                    value = reported_result[key]
                    if (
                        lower <= value <= upper
                        or np.isclose(value, lower)
                        or np.isclose(value, upper)
                    ):
                        if "fig" in key:
                            correct_vision_answers += 1
                        else:
                            correct_written_answers += 1
                elif key in list_keys:
                    reported_list = reported_result[key]
                    list_aliases = (_LIST_ANSWER_ALIASES.get(task_id) or {}).get(
                        key
                    )
                    if list_aliases is not None:
                        reported_lower = {item.lower() for item in reported_list}
                        correct = any(
                            {v.lower() for v in subset} == reported_lower
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
                        correct = reported_val.lower() in {
                            v.lower() for v in alias_set
                        }
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
        print(f"    Error evaluating {task_id}: {e}")

    return {
        "correct_written_answers": correct_written_answers,
        "correct_vision_answers": correct_vision_answers,
        "total_written_questions": total_written_questions,
        "total_vision_questions": total_vision_questions,
    }


def _normalize_agent_output(agent_output: dict) -> dict:
    """Handle both old ({task_id: response}) and new ({task_id: {answer: response}}) formats."""
    normalized = {}
    for task_id, task_data in agent_output.items():
        if isinstance(task_data, dict) and "answer" in task_data:
            normalized[task_id] = task_data["answer"]
        else:
            normalized[task_id] = task_data
    return normalized


def evaluate_run(
    agent_answers: dict, ground_truth: dict[str, list]
) -> dict[str, dict]:
    """Evaluate all tasks in a run against ground truth."""
    agent_answers = _normalize_agent_output(agent_answers)
    results = {}

    for task_id, solution in agent_answers.items():
        if task_id not in ground_truth:
            print(f"    WARN: {task_id} not in ground truth, skipping")
            continue

        gt_result = ground_truth[task_id]

        # Count total questions from ground truth
        numeric_keys = [
            k for k in gt_result[0] if isinstance(gt_result[0][k], (int, float))
        ]
        list_keys = [k for k in gt_result[0] if isinstance(gt_result[0][k], list)]
        string_keys = [k for k in gt_result[0] if isinstance(gt_result[0][k], str)]
        total_written = (
            len([k for k in string_keys if "fig" not in k])
            + len([k for k in numeric_keys if "fig" not in k])
            + len([k for k in list_keys if "fig" not in k])
        )
        total_vision = (
            len([k for k in string_keys if "fig" in k])
            + len([k for k in numeric_keys if "fig" in k])
            + len([k for k in list_keys if "fig" in k])
        )

        try:
            if isinstance(solution, str):
                reported_result = json.loads(solution)
            elif isinstance(solution, dict):
                reported_result = solution
            else:
                raise ValueError(f"Invalid solution format: {type(solution)}")

            evaluation = _eval_result_json(gt_result, reported_result, task_id)
            results[task_id] = evaluation
        except Exception as e:
            results[task_id] = {
                "correct_written_answers": 0,
                "correct_vision_answers": 0,
                "total_written_questions": total_written,
                "total_vision_questions": total_vision,
                "error": str(e),
            }

    return results


def compute_metrics(eval_results: dict) -> dict:
    """Compute accuracy and task lists from eval results."""
    correct_tasks = 0
    successful_tasks = []
    failed_tasks = []

    for task_id, result in eval_results.items():
        cw = result.get("correct_written_answers", 0)
        tw = result.get("total_written_questions", 0)
        cv = result.get("correct_vision_answers", 0)
        tv = result.get("total_vision_questions", 0)

        if cw == tw and cv == tv and (tw + tv) > 0:
            correct_tasks += 1
            successful_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)

    total = len(eval_results)
    return {
        "accuracy": correct_tasks / total if total else 0,
        "successful_tasks": sorted(successful_tasks),
        "failed_tasks": sorted(failed_tasks),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_raw_submissions(run_dir: Path) -> dict:
    """Load agent answers from _RAW_SUBMISSIONS.jsonl."""
    answers = {}
    for jsonl_path in run_dir.glob("*_RAW_SUBMISSIONS.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    answers.update(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return answers


def main():
    parser = argparse.ArgumentParser(description="Post-hoc regrade CORE-bench results")
    parser.add_argument(
        "--core_test",
        type=str,
        required=True,
        help="Path to core_test.json with ground truth",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results" / "corebench_hard"),
    )
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Load ground truth
    with open(args.core_test) as f:
        tasks = json.load(f)
    ground_truth = {t["capsule_id"]: t["results"] for t in tasks}
    print(f"Loaded {len(ground_truth)} tasks from {args.core_test}")

    results_dir = Path(args.results_dir)
    run_dirs = sorted(
        d
        for d in results_dir.iterdir()
        if d.is_dir() and (args.filter is None or args.filter in d.name)
    )
    print(f"Processing {len(run_dirs)} runs...")

    for run_dir in run_dirs:
        upload_files = list(run_dir.glob("*_UPLOAD.json"))
        if not upload_files:
            continue

        upload_path = upload_files[0]
        with open(upload_path) as f:
            data = json.load(f)

        # Load agent answers
        agent_answers = load_raw_submissions(run_dir)
        if not agent_answers:
            print(f"\n{run_dir.name}: no RAW_SUBMISSIONS, skipping")
            continue

        # Filter to tasks in ground truth
        relevant = {k: v for k, v in agent_answers.items() if k in ground_truth}
        if not relevant:
            print(f"\n{run_dir.name}: no matching tasks in ground truth, skipping")
            continue

        # Get old accuracy for comparison
        old_acc = data.get("results", {}).get("accuracy", 0)

        # Regrade
        new_eval = evaluate_run(relevant, ground_truth)
        new_metrics = compute_metrics(new_eval)
        new_acc = new_metrics["accuracy"]

        # Preserve existing fields in raw_eval_results (confidence, reward, etc.)
        old_eval = data.get("raw_eval_results", {})
        for task_id, new_result in new_eval.items():
            if task_id in old_eval and isinstance(old_eval[task_id], dict):
                # Keep confidence/reward/etc, update grading fields
                old_eval[task_id].update(new_result)
                # Update reward to match new grading
                cw = new_result.get("correct_written_answers", 0)
                tw = new_result.get("total_written_questions", 0)
                cv = new_result.get("correct_vision_answers", 0)
                tv = new_result.get("total_vision_questions", 0)
                old_eval[task_id]["reward"] = (
                    1.0 if (cw == tw and cv == tv and (tw + tv) > 0) else 0.0
                )
            else:
                old_eval[task_id] = new_result

        data["raw_eval_results"] = old_eval
        data["results"]["accuracy"] = new_acc
        data["results"]["successful_tasks"] = new_metrics["successful_tasks"]
        data["results"]["failed_tasks"] = new_metrics["failed_tasks"]

        delta = new_acc - old_acc
        marker = "+" if delta > 0 else "" if delta == 0 else ""
        print(
            f"  {run_dir.name}: {old_acc:.1%} -> {new_acc:.1%} ({marker}{delta:+.1%}) "
            f"[{len(relevant)} tasks]"
        )

        if not args.dry_run:
            with open(upload_path, "w") as f:
                json.dump(data, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()

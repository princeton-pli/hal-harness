"""Report generation: markdown, JSON, LaTeX."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List

from reliability_eval.loaders.agent_names import (
    get_model_metadata,
    sort_agents_by_provider_and_date,
    strip_agent_prefix,
)
from reliability_eval.metrics.consistency import compute_weighted_r_con
from reliability_eval.types import ReliabilityMetrics

# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive markdown report."""
    report = []
    report.append("# Reliability Evaluation Report\n\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("## Overview\n\n")
    report.append(f"- **Agents Analyzed**: {len(df)}\n\n")

    # Metrics summary table
    report.append("## Complete Metrics Summary\n\n")
    report.append(
        "| Agent | Acc | consistency_outcome | consistency_trajectory_distribution | consistency_trajectory_sequence | consistency_confidence | consistency_resource | predictability_rate_confidence_correlation | predictability_calibration | predictability_roc_auc | predictability_brier_score | robustness_fault_injection | robustness_structural | robustness_prompt_variation | safety_harm_severity | safety_compliance | safety_score |\n"
    )
    report.append(
        "|-------|-----|-------|----------|----------|--------|-------|------|-------|---------|---------|---------|----------|----------|--------|--------|----------|\n"
    )

    def fmt(v):
        return f"{v:.2f}" if not np.isnan(v) else "-"

    for _, row in df.iterrows():
        report.append(
            f"| {row['agent'][:15]} | {fmt(row['accuracy'])} | "
            f"{fmt(row['consistency_outcome'])} | {fmt(row.get('consistency_trajectory_distribution', np.nan))} | {fmt(row.get('consistency_trajectory_sequence', np.nan))} | "
            f"{fmt(row.get('consistency_confidence', np.nan))} | {fmt(row['consistency_resource'])} | "
            f"{fmt(row['predictability_rate_confidence_correlation'])} | {fmt(row['predictability_calibration'])} | {fmt(row.get('predictability_roc_auc', np.nan))} | {fmt(row.get('predictability_brier_score', np.nan))} | "
            f"{fmt(row['robustness_fault_injection'])} | {fmt(row['robustness_structural'])} | {fmt(row.get('robustness_prompt_variation', np.nan))} | "
            f"{fmt(row['safety_harm_severity'])} | {fmt(row['safety_compliance'])} | {fmt(row['safety_score'])} |\n"
        )

    # Dimension-level aggregates
    report.append("\n## Dimension-Level Scores (§3.7)\n\n")
    report.append(
        "*Note: Overall reliability is a uniform average of consistency, predictability, and robustness. Safety (reliability_safety) is reported separately.*\n\n"
    )
    report.append("| Agent | reliability_consistency | reliability_robustness | reliability_predictability | reliability_safety | Overall |\n")
    report.append("|-------|-------|-------|--------|-------|--------|\n")

    for _, row in df.iterrows():
        reliability_consistency = compute_weighted_r_con(
            row["consistency_outcome"],
            row.get("consistency_trajectory_distribution", np.nan),
            row.get("consistency_trajectory_sequence", np.nan),
            row["consistency_resource"],
        )
        reliability_robustness = np.nanmean(
            [row["robustness_fault_injection"], row["robustness_structural"], row.get("robustness_prompt_variation", np.nan)]
        )
        reliability_predictability = row.get("predictability_brier_score", np.nan)
        reliability_safety = row["safety_score"]
        Overall = np.nanmean([reliability_consistency, reliability_robustness, reliability_predictability])

        report.append(
            f"| {row['agent'][:15]} | {fmt(reliability_consistency)} | {fmt(reliability_robustness)} | "
            f"{fmt(reliability_predictability)} | {fmt(reliability_safety)} | {fmt(Overall)} |\n"
        )

    # Metrics explanation
    report.append("\n## Metrics Reference\n\n")

    report.append("### Consistency (§3.2)\n")
    report.append(
        "- **reliability_consistency**: Category-weighted aggregate = (1/3)·consistency_outcome + (1/3)·mean(consistency_trajectory_distribution, consistency_trajectory_sequence) + (1/3)·consistency_resource\n"
    )
    report.append("- **consistency_outcome**: Outcome consistency = 1 - Var(y)/(p(1-p)+ε)\n")
    report.append(
        "- **consistency_trajectory_distribution**: Trajectory distribution consistency (1 - JSD of action frequencies)\n"
    )
    report.append(
        "- **consistency_trajectory_sequence**: Trajectory sequence consistency (normalized edit distance)\n"
    )
    report.append(
        "- **consistency_confidence**: Confidence consistency = exp(-CV) of confidence scores\n"
    )
    report.append("- **consistency_resource**: Resource consistency = exp(-CV) across all runs\n\n")

    # Add CV breakdown table if data is available
    cv_cols = [
        "mean_time_cv",
        "mean_api_calls_cv",
        "mean_actions_cv",
        "mean_call_latency_cv",
    ]
    if any(col in df.columns for col in cv_cols):
        report.append("#### Resource CV Breakdown (lower = more consistent)\n\n")
        report.append("| Agent | Time CV | API Calls CV | Actions CV | Latency CV |\n")
        report.append("|-------|---------|--------------|------------|------------|\n")
        for _, row in df.iterrows():
            report.append(
                f"| {row['agent'][:15]} | "
                f"{fmt(row.get('mean_time_cv', np.nan))} | "
                f"{fmt(row.get('mean_api_calls_cv', np.nan))} | "
                f"{fmt(row.get('mean_actions_cv', np.nan))} | "
                f"{fmt(row.get('mean_call_latency_cv', np.nan))} |\n"
            )
        report.append("\n")

    report.append("### Predictability (§3.4)\n")
    report.append(
        "- **reliability_predictability** = predictability_brier_score (Brier score is a proper scoring rule capturing both calibration and discrimination)\n"
    )
    report.append("- **predictability_brier_score**: 1 - Brier Score (proper scoring rule)\n")
    report.append("- **predictability_calibration**: Calibration = 1 - ECE (reported separately)\n")
    report.append("- **predictability_roc_auc**: Discrimination = AUC-ROC (reported separately)\n\n")

    report.append("### Robustness (§3.3)\n")
    report.append("- **robustness_fault_injection**: Acc(fault)/Acc(baseline), clamped to [0,1]\n")
    report.append("- **robustness_structural**: Acc(perturbed)/Acc(baseline), clamped to [0,1]\n")
    report.append(
        "- **robustness_prompt_variation**: Acc(prompt_variation)/Acc(baseline), clamped to [0,1]\n\n"
    )

    report.append("### Safety (§3.5)\n")
    report.append(
        "- **safety_harm_severity**: Harm score = 1/(1 + mean_severity/H_ref), LLM-judged error severity\n"
    )
    report.append(
        "- **safety_compliance**: Compliance = Mean(1 - ViolationRate) across constraints, LLM-judged\n"
    )
    report.append(
        "- **safety_score**: 1 - Risk, where Risk = (1 - safety_compliance) * (1 - safety_harm_severity); severity weights: low=0.25, medium=0.5, high=1.0\n\n"
    )

    output_path = output_dir / "reliability_report.md"
    with open(output_path, "w") as f:
        f.writelines(report)
    print(f"📄 Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def save_detailed_json(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Save detailed JSON files containing the data behind each detailed plot.
    These can be used to recreate plots or for further analysis.
    """
    agent_to_metrics = {m.agent_name: m for m in all_metrics}
    df_sorted = sort_agents_by_provider_and_date(df)
    agent_names_full = df_sorted["agent"].tolist()

    def _safe(v):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                return None
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, np.ndarray):
            return [_safe(x) for x in v.tolist()]
        if isinstance(v, dict):
            return {k: _safe(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_safe(x) for x in v]
        return v

    # =========================================================================
    # 1. PREDICTABILITY: calibration, selective prediction, discrimination
    # =========================================================================
    predictability_data = {}
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if not m:
            continue
        display = strip_agent_prefix(agent_name)
        provider = get_model_metadata(agent_name).get("provider", "Unknown")
        entry = {
            "agent": agent_name,
            "display_name": display,
            "provider": provider,
            "predictability_calibration": _safe(m.predictability_calibration),
            "predictability_roc_auc": _safe(m.predictability_roc_auc),
            "predictability_brier_score": _safe(m.predictability_brier_score),
            "predictability_rate_confidence_correlation": _safe(m.predictability_rate_confidence_correlation),
            "mean_confidence": _safe(m.mean_confidence),
            "calibration_bins": _safe(m.extra.get("calibration_bins", [])),
            "aurc_data": {
                "coverages": _safe(m.extra.get("aurc_data", {}).get("coverages")),
                "risks": _safe(m.extra.get("aurc_data", {}).get("risks")),
                "optimal_risks": _safe(
                    m.extra.get("aurc_data", {}).get("optimal_risks")
                ),
                "aurc": _safe(m.extra.get("aurc_data", {}).get("aurc")),
                "e_aurc": _safe(m.extra.get("aurc_data", {}).get("e_aurc")),
            }
            if m.extra.get("aurc_data")
            else None,
            "auroc_data": _safe(m.extra.get("auroc_data")),
            "brier_data": _safe(m.extra.get("brier_data")),
        }
        predictability_data[agent_name] = entry

    with open(output_dir / "predictability_detailed.json", "w") as f:
        json.dump(predictability_data, f, indent=2)
    print(f"💾 Saved: {output_dir / 'predictability_detailed.json'}")

    # =========================================================================
    # 2. ABSTENTION: rates, precision, recall, selective accuracy, categories
    # =========================================================================
    abstention_data_out = {}
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if not m:
            continue
        display = strip_agent_prefix(agent_name)
        provider = get_model_metadata(agent_name).get("provider", "Unknown")
        abstention_extra = m.extra.get("abstention_data", {})
        entry = {
            "agent": agent_name,
            "display_name": display,
            "provider": provider,
            "accuracy": _safe(m.accuracy),
            "abstention_rate": _safe(m.abstention_rate),
            "abstention_precision": _safe(m.abstention_precision),
            "abstention_recall": _safe(m.abstention_recall),
            "abstention_selective_accuracy": _safe(m.abstention_selective_accuracy),
            "abstention_calibration": _safe(m.abstention_calibration),
            "confusion_matrix": _safe(abstention_extra.get("confusion_matrix", {})),
            "type_breakdown": _safe(abstention_extra.get("type_breakdown", {})),
            "n_tasks": _safe(abstention_extra.get("n_tasks", 0)),
            "n_abstained": _safe(abstention_extra.get("n_abstained", 0)),
        }
        abstention_data_out[agent_name] = entry

    with open(output_dir / "abstention_detailed.json", "w") as f:
        json.dump(abstention_data_out, f, indent=2)
    print(f"💾 Saved: {output_dir / 'abstention_detailed.json'}")

    # =========================================================================
    # 3. CONSISTENCY: consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource
    # =========================================================================
    consistency_data = {}
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if not m:
            continue
        display = strip_agent_prefix(agent_name)
        provider = get_model_metadata(agent_name).get("provider", "Unknown")
        # Extract per-task success rates from task_df
        task_df = m.extra.get("consistency_task_df", pd.DataFrame())
        task_outcomes = {}
        task_costs = {}
        task_times = {}
        task_api_calls = {}
        task_actions = {}
        task_confidences = {}
        if not task_df.empty and "task_id" in task_df.columns:
            for _, row in task_df.iterrows():
                tid = str(row["task_id"])
                if "success_rate" in task_df.columns:
                    task_outcomes[tid] = _safe(row["success_rate"])
                if "mean_cost" in task_df.columns:
                    task_costs[tid] = _safe(row["mean_cost"])
                if "mean_time" in task_df.columns:
                    task_times[tid] = _safe(row["mean_time"])
                if "mean_api_calls" in task_df.columns:
                    task_api_calls[tid] = _safe(row["mean_api_calls"])
                if "mean_actions" in task_df.columns:
                    task_actions[tid] = _safe(row["mean_actions"])
                if "mean_confidence" in task_df.columns:
                    task_confidences[tid] = _safe(row["mean_confidence"])
        entry = {
            "agent": agent_name,
            "display_name": display,
            "provider": provider,
            "consistency_outcome": _safe(m.consistency_outcome),
            "consistency_trajectory_distribution": _safe(m.consistency_trajectory_distribution),
            "consistency_trajectory_sequence": _safe(m.consistency_trajectory_sequence),
            "consistency_confidence": _safe(m.consistency_confidence),
            "consistency_resource": _safe(m.consistency_resource),
            "cv_breakdown": _safe(m.extra.get("cv_breakdown", {})),
            "conf_breakdown": _safe(m.extra.get("conf_breakdown", {})),
            "task_outcomes": task_outcomes,
            "task_levels": _safe(m.extra.get("task_levels", {})),
            "task_costs": task_costs,
            "task_times": task_times,
            "task_api_calls": task_api_calls,
            "task_actions": task_actions,
            "task_confidences": task_confidences,
        }
        consistency_data[agent_name] = entry

    with open(output_dir / "consistency_detailed.json", "w") as f:
        json.dump(consistency_data, f, indent=2)
    print(f"💾 Saved: {output_dir / 'consistency_detailed.json'}")

    # =========================================================================
    # 4. ROBUSTNESS: robustness_fault_injection, robustness_structural, robustness_prompt_variation with accuracies
    # =========================================================================
    robustness_data = {}
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if not m:
            continue
        display = strip_agent_prefix(agent_name)
        provider = get_model_metadata(agent_name).get("provider", "Unknown")
        entry = {
            "agent": agent_name,
            "display_name": display,
            "provider": provider,
            "robustness_fault_injection": _safe(m.robustness_fault_injection),
            "robustness_structural": _safe(m.robustness_structural),
            "robustness_prompt_variation": _safe(m.robustness_prompt_variation),
            "baseline_acc": _safe(m.extra.get("baseline_acc")),
            "fault_acc": _safe(m.extra.get("fault_acc")),
            "struct_acc": _safe(m.extra.get("struct_acc")),
            "prompt_acc": _safe(m.extra.get("prompt_acc")),
        }
        robustness_data[agent_name] = entry

    with open(output_dir / "robustness_detailed.json", "w") as f:
        json.dump(robustness_data, f, indent=2)
    print(f"💾 Saved: {output_dir / 'robustness_detailed.json'}")

    # =========================================================================
    # 5. SAFETY: safety_harm_severity, safety_compliance, violations, per-constraint
    # =========================================================================
    safety_data = {}
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if not m:
            continue
        display = strip_agent_prefix(agent_name)
        provider = get_model_metadata(agent_name).get("provider", "Unknown")
        entry = {
            "agent": agent_name,
            "display_name": display,
            "provider": provider,
            "safety_harm_severity": _safe(m.safety_harm_severity),
            "safety_compliance": _safe(m.safety_compliance),
            "safety_score": _safe(m.safety_score),
            "mean_severity": _safe(m.extra.get("safety_mean_severity")),
            "max_severity": _safe(m.extra.get("safety_max_severity")),
            "analysis_model": m.extra.get("safety_analysis_model"),
            "per_constraint": _safe(m.extra.get("safety_per_constraint", {})),
            "violations": _safe(m.extra.get("safety_violations", [])),
        }
        safety_data[agent_name] = entry

    with open(output_dir / "safety_detailed.json", "w") as f:
        json.dump(safety_data, f, indent=2)
    print(f"💾 Saved: {output_dir / 'safety_detailed.json'}")

    # =========================================================================
    # 6. LEVEL-STRATIFIED (GAIA only, if available)
    # =========================================================================
    has_level_data = any(m.extra.get("level_metrics") for m in all_metrics)
    if has_level_data:
        level_data = {}
        for agent_name in agent_names_full:
            m = agent_to_metrics.get(agent_name)
            if not m:
                continue
            display = strip_agent_prefix(agent_name)
            provider = get_model_metadata(agent_name).get("provider", "Unknown")
            entry = {
                "agent": agent_name,
                "display_name": display,
                "provider": provider,
                "level_metrics": _safe(m.extra.get("level_metrics", {})),
                "consistency_by_level": _safe(m.extra.get("consistency_by_level", {})),
                "fault_robustness_by_level": _safe(
                    m.extra.get("fault_robustness_by_level", {})
                ),
                "struct_robustness_by_level": _safe(
                    m.extra.get("struct_robustness_by_level", {})
                ),
                "prompt_robustness_by_level": _safe(
                    m.extra.get("prompt_robustness_by_level", {})
                ),
            }
            level_data[agent_name] = entry

        with open(output_dir / "level_stratified_detailed.json", "w") as f:
            json.dump(level_data, f, indent=2)
        print(f"💾 Saved: {output_dir / 'level_stratified_detailed.json'}")


def generate_full_latex_table(benchmark_data: list, output_dir: Path):
    """Generate a full LaTeX table with all reliability sub-metrics for multiple benchmarks.

    Uses per-column cell coloring (red-white-green gradient) for visual scanning.
    Best value per benchmark is bolded.

    Required LaTeX packages: colortbl, xcolor (with table option), multirow, booktabs, graphicx.
    """
    benchmark_display = {
        "gaia": "GAIA",
        "taubench_airline": r"$\tau$-bench Airline",
        "taubench_airline_original": r"$\tau$-bench Airline (original)",
        "taubench_retail": r"$\tau$-bench Retail",
    }

    def _value_to_rgb(val):
        """Map a value on [0, 1] to an RGB color: red → orange → green.

        <=0.5 -> red, 0.75 -> orange, 1.0 -> green.  Fixed scale.
        Returns (r, g, b) each in [0, 1].
        """
        if pd.isna(val):
            return None
        t = max(0.0, min(1.0, float(val)))
        # Anchor colors (pastel for readability)
        RED = (1.00, 0.60, 0.60)  # <=0.5
        ORANGE = (1.00, 0.85, 0.55)  # 0.75
        GREEN = (0.55, 0.88, 0.55)  # 1.0
        if t <= 0.5:
            return RED
        elif t <= 0.75:
            # Interpolate red -> orange over [0.5, 0.75]
            s = (t - 0.5) / 0.25
            return tuple(RED[i] + s * (ORANGE[i] - RED[i]) for i in range(3))
        else:
            # Interpolate orange -> green over [0.75, 1.0]
            s = (t - 0.75) / 0.25
            return tuple(ORANGE[i] + s * (GREEN[i] - ORANGE[i]) for i in range(3))

    def _cellcolor_cmd(rgb):
        """Return \\cellcolor command for an RGB tuple, or empty string."""
        if rgb is None:
            return ""
        return f"\\cellcolor[rgb]{{{rgb[0]:.3f},{rgb[1]:.3f},{rgb[2]:.3f}}}"

    def fmt(v, bold=False, rgb=None):
        if pd.isna(v):
            return "--"
        s = f"{v:.2f}"
        if bold:
            s = r"\textbf{" + s + "}"
        cc = _cellcolor_cmd(rgb)
        if cc:
            return cc + s
        return s

    def prepare_df(df):
        df_s = sort_agents_by_provider_and_date(df)
        if "reliability_consistency" not in df_s.columns:
            df_s["reliability_consistency"] = compute_weighted_r_con(
                df_s["consistency_outcome"], df_s["consistency_trajectory_distribution"], df_s["consistency_trajectory_sequence"], df_s["consistency_resource"]
            )
        if "reliability_predictability" not in df_s.columns:
            df_s["reliability_predictability"] = df_s["predictability_brier_score"]
        if "reliability_robustness" not in df_s.columns:
            df_s["reliability_robustness"] = df_s[["robustness_fault_injection", "robustness_structural", "robustness_prompt_variation"]].mean(
                axis=1, skipna=True
            )
        if "reliability_safety" not in df_s.columns:
            df_s["reliability_safety"] = df_s["safety_score"]
        if "reliability_overall" not in df_s.columns:
            df_s["reliability_overall"] = df_s[["reliability_consistency", "reliability_predictability", "reliability_robustness"]].mean(
                axis=1, skipna=True
            )
        df_s["display_name"] = df_s["agent"].map(strip_agent_prefix)
        return df_s

    # Sub-metric columns in order
    consistency_cols = ["consistency_outcome", "consistency_trajectory_distribution", "consistency_trajectory_sequence", "consistency_resource", "reliability_consistency"]
    predictability_cols = ["predictability_calibration", "predictability_roc_auc", "predictability_brier_score", "reliability_predictability"]
    robustness_cols = ["robustness_fault_injection", "robustness_structural", "robustness_prompt_variation", "reliability_robustness"]
    safety_cols = ["safety_harm_severity", "safety_compliance", "safety_score", "reliability_safety"]
    all_metric_cols = (
        ["accuracy"]
        + consistency_cols
        + predictability_cols
        + robustness_cols
        + safety_cols
        + ["reliability_overall"]
    )

    all_dfs = [prepare_df(df) for _, df in benchmark_data]

    # LaTeX column headers
    consistency_headers = [
        r"$C_\text{out}$",
        r"$C_\text{traj\_d}$",
        r"$C_\text{traj\_s}$",
        r"$C_\text{res}$",
        r"$\mathcal{R}_\text{Con}$",
    ]
    predictability_headers = [
        r"$P_\text{cal}$",
        r"$P_\text{auroc}$",
        r"$P_\text{brier}$",
        r"$\mathcal{R}_\text{Pred}$",
    ]
    robustness_headers = [
        r"$R_\text{fault}$",
        r"$R_\text{struct}$",
        r"$R_\text{prompt}$",
        r"$\mathcal{R}_\text{Rob}$",
    ]
    safety_headers = [
        r"$S_\text{harm}$",
        r"$S_\text{comp}$",
        r"$S_\text{safety}$",
        r"$\mathcal{R}_\text{Saf}$",
    ]

    # Build LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Full reliability metrics across benchmarks. "
        r"Best per benchmark shown in \textbf{bold}. "
        r"Cells shaded \colorbox[rgb]{1.000,0.600,0.600}{red} (${\leq}0.5$), "
        r"\colorbox[rgb]{1.000,0.850,0.550}{orange} ($0.75$), "
        r"\colorbox[rgb]{0.550,0.880,0.550}{green} ($1.0$).}"
    )
    lines.append(r"\label{tab:full_reliability}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    # 20 data cols + 1 benchmark label col = 21 cols total
    # Cols: benchmark | model | acc | 5 consistency | 4 predictability | 4 robustness | 4 safety | R_Rel
    lines.append(r"\begin{tabular}{lccccccccccccccccccccc}")
    lines.append(r"\toprule")

    # Header row 1: category spans (col 1=benchmark rowspan, col 2=model, col 3=acc, then groups)
    lines.append(
        r" & & "
        r" & \multicolumn{5}{c}{Consistency}"
        r" & \multicolumn{4}{c}{Predictability}"
        r" & \multicolumn{4}{c}{Robustness}"
        r" & \multicolumn{4}{c}{Safety}"
        r" & \\"
    )
    # cmidrules: consistency cols 4-8, predictability 9-12, robustness 13-16, safety 17-20
    lines.append(
        r"\cmidrule(lr){4-8}"
        r" \cmidrule(lr){9-12}"
        r" \cmidrule(lr){13-16}"
        r" \cmidrule(lr){17-20}"
    )

    # Header row 2: individual metric names
    header2_parts = ["", "Model", "Acc"]
    header2_parts.extend(consistency_headers)
    header2_parts.extend(predictability_headers)
    header2_parts.extend(robustness_headers)
    header2_parts.extend(safety_headers)
    header2_parts.append(r"$\mathcal{R}$")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    for bm_idx, (bm_name, df) in enumerate(benchmark_data):
        df_s = all_dfs[bm_idx]
        n_agents = len(df_s)
        display_bm = benchmark_display.get(bm_name, bm_name)

        # Find best (max) for each metric in this benchmark section
        best = {}
        for col in all_metric_cols:
            if col in df_s.columns:
                best[col] = df_s[col].max()
            else:
                best[col] = np.nan

        for i, (_, row) in enumerate(df_s.iterrows()):
            parts = []
            # Benchmark label column (multirow on first row)
            if i == 0:
                parts.append(
                    r"\multirow{"
                    + str(n_agents)
                    + r"}{*}{\rotatebox[origin=c]{90}{\textsc{"
                    + display_bm
                    + r"}}}"
                )
            else:
                parts.append("")

            # Model name
            parts.append(row["display_name"])

            # All metric columns with coloring and bolding
            for col in all_metric_cols:
                val = row.get(col, np.nan)
                is_best = (
                    not pd.isna(val)
                    and not pd.isna(best.get(col))
                    and abs(val - best[col]) < 1e-6
                )
                rgb = _value_to_rgb(val) if col != "accuracy" else None
                parts.append(fmt(val, bold=is_best, rgb=rgb))

            lines.append(" & ".join(parts) + r" \\")

        # Separator between benchmark sections (but not after the last one)
        if bm_idx < len(benchmark_data) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)

    # Save
    out_path = output_dir / "full_reliability_table.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)
    print(f"💾 Saved LaTeX table: {out_path}")

    # Also print to console
    print("\n" + "=" * 80)
    print("LaTeX Table Output:")
    print("=" * 80)
    print(latex_str)

    return latex_str

"""Level-stratified and provider-level analysis plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from reliability_eval.constants import (
    PROVIDER_COLORS,
)
from reliability_eval.loaders.agent_names import (
    get_provider,
    sort_agents_by_provider_and_date,
    strip_agent_prefix,
)
from reliability_eval.types import ReliabilityMetrics


def plot_level_stratified_analysis(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create level-stratified analysis plots for GAIA benchmark.

    Shows key reliability submetrics across difficulty levels (1, 2, 3).
    Layout: 5x2 grid (10 panels):

    Row 0: Accuracy, Mean Actions
    Row 1: consistency_outcome, consistency_resource
    Row 2: predictability_rate_confidence_correlation, predictability_calibration
    Row 3: predictability_roc_auc, robustness_fault_injection
    Row 4: robustness_structural, robustness_prompt_variation
    """
    # Check if any agent has level data
    has_level_data = False
    for m in all_metrics:
        if "level_metrics" in m.extra and m.extra["level_metrics"]:
            has_level_data = True
            break

    if not has_level_data:
        print("📊 Skipping level-stratified plot (no GAIA level data available)")
        return

    df_sorted = sort_agents_by_provider_and_date(df)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    agent_names_full = df_sorted["agent"].tolist()
    x_pos = np.arange(len(agents))
    levels = ["1", "2", "3"]
    level_colors = {
        "1": "#4CAF50",
        "2": "#FF9800",
        "3": "#F44336",
    }  # Green, Orange, Red
    level_labels = {"1": "L1 (Easy)", "2": "L2 (Med)", "3": "L3 (Hard)"}
    bar_width = 0.25

    fig, axes = plt.subplots(5, 2, figsize=(9, 10.5))

    def plot_metric_by_level(
        ax,
        metric_getter,
        ylabel,
        title=None,
        ylim=(0, 1.15),
        clamp_at=None,
        se_getter=None,
        show_xticks=False,
    ):
        """Helper to plot a metric grouped by level with optional error bars."""
        for i, level in enumerate(levels):
            vals = []
            ses = []
            for agent in agent_names_full:
                m = agent_to_metrics.get(agent)
                val = metric_getter(m, level) if m else np.nan
                if clamp_at is not None and not np.isnan(val):
                    val = min(val, clamp_at)
                vals.append(val)
                se = se_getter(m, level) if (se_getter and m) else 0.0
                ses.append(se if se and not np.isnan(se) else 0.0)
            offset = (i - 1) * bar_width
            vals_arr = np.array(vals)
            yerr = np.array(ses)
            # Clip error bars at [0, 1] for bounded metrics
            if clamp_at is not None or ylim[1] <= 1.5:
                upper_bound = clamp_at if clamp_at else 1.0
                yerr = np.where(np.isclose(vals_arr, upper_bound, atol=1e-9), 0, yerr)
                yerr = np.minimum(yerr, np.maximum(upper_bound - vals_arr, 0))
            has_se = np.any(yerr > 0)
            ax.bar(
                x_pos + offset,
                vals,
                bar_width,
                label=level_labels[level],
                color=level_colors[level],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                yerr=yerr if has_se else None,
                capsize=2,
                error_kw={"linewidth": 0.8, "color": "black"},
            )
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        if show_xticks:
            ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, axis="y")

    # ===== ROW 0: Accuracy, Mean Actions =====

    # 0.0 Accuracy by Level
    plot_metric_by_level(
        axes[0, 0],
        lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("accuracy_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$\mathrm{Accuracy}$",
        se_getter=lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("accuracy_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # 0.1 Mean Actions by Level
    max_traj = 0
    for m in all_metrics:
        traj_dict = m.extra.get("level_metrics", {}).get("trajectory_complexity", {})
        for v in traj_dict.values():
            if v and not np.isnan(v) and v > max_traj:
                max_traj = v
    plot_metric_by_level(
        axes[0, 1],
        lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("trajectory_complexity", {})
            .get(lvl, np.nan)
        ),
        r"$\mathrm{Mean\ Actions}$",
        ylim=(0, max(max_traj * 1.1, 10)),
        se_getter=lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("trajectory_complexity_se", {})
            .get(lvl, 0.0)
        ),
    )

    # ===== ROW 1: consistency_outcome, consistency_resource =====

    # 1.0 consistency_outcome (Outcome Consistency) by Level
    plot_metric_by_level(
        axes[1, 0],
        lambda m, lvl: (
            m.extra.get("consistency_by_level", {})
            .get("consistency_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$C_{\mathrm{out}}$",
        se_getter=lambda m, lvl: (
            m.extra.get("consistency_by_level", {})
            .get("consistency_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # 1.1 consistency_resource (Resource Consistency) by Level
    plot_metric_by_level(
        axes[1, 1],
        lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("consistency_resource_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$C_{\mathrm{res}}$",
        se_getter=lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("consistency_resource_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # ===== ROW 2: predictability_brier_score, predictability_calibration =====

    # 2.0 predictability_brier_score (Brier Score) by Level
    plot_metric_by_level(
        axes[2, 0],
        lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("predictability_brier_score_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$P_{\mathrm{Brier}}$",
        se_getter=lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("predictability_brier_score_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # 2.1 predictability_calibration (Calibration = 1-ECE) by Level
    plot_metric_by_level(
        axes[2, 1],
        lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("predictability_calibration_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$P_{\mathrm{cal}}$",
        se_getter=lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("predictability_calibration_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # ===== ROW 3: predictability_roc_auc, robustness_fault_injection =====

    # 3.0 predictability_roc_auc (Discrimination) by Level
    plot_metric_by_level(
        axes[3, 0],
        lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("predictability_roc_auc_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$P_{\mathrm{AUROC}}$",
        se_getter=lambda m, lvl: (
            m.extra.get("level_metrics", {})
            .get("predictability_roc_auc_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # 3.1 robustness_fault_injection (Fault Robustness) by Level
    plot_metric_by_level(
        axes[3, 1],
        lambda m, lvl: (
            m.extra.get("fault_robustness_by_level", {})
            .get("robustness_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$R_{\mathrm{fault}}$",
        clamp_at=1.0,
        se_getter=lambda m, lvl: (
            m.extra.get("fault_robustness_by_level", {})
            .get("robustness_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # ===== ROW 4: robustness_structural, robustness_prompt_variation =====

    # 4.0 robustness_structural (Structural Robustness) by Level
    plot_metric_by_level(
        axes[4, 0],
        lambda m, lvl: (
            m.extra.get("struct_robustness_by_level", {})
            .get("robustness_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$R_{\mathrm{env}}$",
        clamp_at=1.0,
        show_xticks=True,
        se_getter=lambda m, lvl: (
            m.extra.get("struct_robustness_by_level", {})
            .get("robustness_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # 4.1 robustness_prompt_variation (Prompt Robustness) by Level
    plot_metric_by_level(
        axes[4, 1],
        lambda m, lvl: (
            m.extra.get("prompt_robustness_by_level", {})
            .get("robustness_by_level", {})
            .get(lvl, np.nan)
        ),
        r"$R_{\mathrm{prompt}}$",
        clamp_at=1.0,
        show_xticks=True,
        se_getter=lambda m, lvl: (
            m.extra.get("prompt_robustness_by_level", {})
            .get("robustness_by_level_se", {})
            .get(lvl, 0.0)
        ),
    )

    # Add global legend at top center (where title used to be)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=level_colors[lvl],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        for lvl in levels
    ]
    labels = [level_labels[lvl] for lvl in levels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space at top for legend
    output_path = output_dir / "level_stratified_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_confidence_difficulty_alignment(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create a plot showing confidence-difficulty alignment.

    Analyzes whether models appropriately express lower confidence on harder tasks.
    Shows:
    1. Confidence vs Accuracy by Level (scatter with trend lines)
    2. Confidence-Accuracy Gap by Level
    """
    # Check if any agent has level data
    has_level_data = False
    for m in all_metrics:
        if "level_metrics" in m.extra and m.extra["level_metrics"]:
            has_level_data = True
            break

    if not has_level_data:
        print(
            "📊 Skipping confidence-difficulty alignment plot (no GAIA level data available)"
        )
        return

    df_sorted = sort_agents_by_provider_and_date(df)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    agents = df_sorted["agent"].tolist()
    levels = ["1", "2", "3"]
    level_names = {
        "1": "Level 1 (Easy)",
        "2": "Level 2 (Medium)",
        "3": "Level 3 (Hard)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Confidence vs Accuracy by Level (all agents)
    ax = axes[0]
    for level in levels:
        confs = []
        accs = []
        for agent in agents:
            m = agent_to_metrics.get(agent)
            if m and "level_metrics" in m.extra:
                lm = m.extra["level_metrics"]
                conf = lm.get("confidence_by_level", {}).get(level)
                acc = lm.get("accuracy_by_level", {}).get(level)
                if (
                    conf is not None
                    and acc is not None
                    and not np.isnan(conf)
                    and not np.isnan(acc)
                ):
                    confs.append(conf)
                    accs.append(acc)

        if confs:
            color = {"1": "#4CAF50", "2": "#FF9800", "3": "#F44336"}[level]
            ax.scatter(
                confs,
                accs,
                label=level_names[level],
                color=color,
                s=80,
                alpha=0.7,
                edgecolors="black",
            )

    # Perfect calibration line
    ax.plot(
        [0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Perfect Calibration"
    )
    ax.set_xlabel("Mean Confidence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Confidence vs Accuracy by Difficulty Level\n(each point = one agent at one level)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Calibration Gap by Level (confidence - accuracy)
    ax = axes[1]
    level_gaps = {level: [] for level in levels}
    for agent in agents:
        m = agent_to_metrics.get(agent)
        if m and "level_metrics" in m.extra:
            lm = m.extra["level_metrics"]
            for level in levels:
                conf = lm.get("confidence_by_level", {}).get(level)
                acc = lm.get("accuracy_by_level", {}).get(level)
                if (
                    conf is not None
                    and acc is not None
                    and not np.isnan(conf)
                    and not np.isnan(acc)
                ):
                    level_gaps[level].append(conf - acc)

    # Box plot of calibration gaps
    gap_data = [level_gaps[lvl] for lvl in levels if level_gaps[lvl]]
    gap_labels = [level_names[lvl] for lvl in levels if level_gaps[lvl]]
    colors = ["#4CAF50", "#FF9800", "#F44336"][: len(gap_data)]

    if gap_data:
        bp = ax.boxplot(gap_data, labels=gap_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Confidence - Accuracy (Gap)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Calibration Gap by Difficulty Level\n(positive = overconfident, negative = underconfident)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "confidence_difficulty_alignment.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_performance_drop_analysis(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Analyze how performance drops from Level 1 to Level 3.

    Shows:
    1. Absolute accuracy by level (line plot per model)
    2. Relative performance drop (L3/L1 ratio) - who degrades most?
    3. Performance drop ranking
    """
    has_level_data = any(
        "level_metrics" in m.extra and m.extra["level_metrics"] for m in all_metrics
    )
    if not has_level_data:
        print("📊 Skipping performance drop analysis (no GAIA level data available)")
        return

    df_sorted = sort_agents_by_provider_and_date(df)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}
    agents = df_sorted["agent"].tolist()
    levels = ["1", "2", "3"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Collect data
    agent_level_acc = {}
    for agent in agents:
        m = agent_to_metrics.get(agent)
        if m and "level_metrics" in m.extra:
            acc_by_level = m.extra["level_metrics"].get("accuracy_by_level", {})
            if acc_by_level:
                agent_level_acc[agent] = acc_by_level

    if not agent_level_acc:
        plt.close()
        return

    # 1. Accuracy trajectories by level (line plot)
    ax = axes[0]
    for agent, acc_by_level in agent_level_acc.items():
        provider = get_provider(agent)
        color = PROVIDER_COLORS.get(provider, "#999999")
        accs = [acc_by_level.get(lvl, np.nan) for lvl in levels]
        if not all(np.isnan(a) for a in accs):
            ax.plot(
                levels,
                accs,
                "o-",
                color=color,
                alpha=0.7,
                linewidth=2,
                label=strip_agent_prefix(agent)[:15],
                markersize=8,
            )

    ax.set_xlabel("Difficulty Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Accuracy Trajectory by Difficulty\n(steeper drop = worse scaling)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    # 2. Relative performance drop (L3/L1 ratio)
    ax = axes[1]
    drops = []
    agent_names = []
    colors = []
    for agent, acc_by_level in agent_level_acc.items():
        l1_acc = acc_by_level.get("1", np.nan)
        l3_acc = acc_by_level.get("3", np.nan)
        if not np.isnan(l1_acc) and not np.isnan(l3_acc) and l1_acc > 0:
            ratio = l3_acc / l1_acc
            drops.append(ratio)
            agent_names.append(strip_agent_prefix(agent))
            colors.append(PROVIDER_COLORS.get(get_provider(agent), "#999999"))

    if drops:
        # Sort by drop ratio
        sorted_idx = np.argsort(drops)[::-1]  # Best (highest ratio) first
        sorted_drops = [drops[i] for i in sorted_idx]
        sorted_names = [agent_names[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]

        y_pos = np.arange(len(sorted_drops))
        bars = ax.barh(
            y_pos,
            sorted_drops,
            color=sorted_colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("L3/L1 Accuracy Ratio", fontsize=12, fontweight="bold")
        ax.set_title(
            "Performance Retention (L3 vs L1)\n(higher = better scaling to hard tasks)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(0, 1.5)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, sorted_drops):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
            )

    # 3. Absolute drop (L1 - L3) ranking
    ax = axes[2]
    abs_drops = []
    agent_names = []
    colors = []
    for agent, acc_by_level in agent_level_acc.items():
        l1_acc = acc_by_level.get("1", np.nan)
        l3_acc = acc_by_level.get("3", np.nan)
        if not np.isnan(l1_acc) and not np.isnan(l3_acc):
            drop = l1_acc - l3_acc
            abs_drops.append(drop)
            agent_names.append(strip_agent_prefix(agent))
            colors.append(PROVIDER_COLORS.get(get_provider(agent), "#999999"))

    if abs_drops:
        # Sort by absolute drop (smallest drop first = best)
        sorted_idx = np.argsort(abs_drops)  # Smallest drop first
        sorted_drops = [abs_drops[i] for i in sorted_idx]
        sorted_names = [agent_names[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]

        y_pos = np.arange(len(sorted_drops))
        bars = ax.barh(
            y_pos,
            sorted_drops,
            color=sorted_colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Accuracy Drop (L1 - L3)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Absolute Performance Drop\n(smaller = more robust to difficulty)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, sorted_drops):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    output_path = output_dir / "level_performance_drop.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_provider_level_heatmap(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create heatmaps showing provider performance patterns across difficulty levels.

    Shows:
    1. Accuracy heatmap (provider x level)
    2. Confidence heatmap (provider x level)
    3. Calibration gap heatmap (provider x level)
    """
    has_level_data = any(
        "level_metrics" in m.extra and m.extra["level_metrics"] for m in all_metrics
    )
    if not has_level_data:
        print("📊 Skipping provider-level heatmap (no GAIA level data available)")
        return

    levels = ["1", "2", "3"]
    providers = ["OpenAI", "Google", "Anthropic"]

    # Aggregate by provider
    provider_acc = {p: {lvl: [] for lvl in levels} for p in providers}
    provider_conf = {p: {lvl: [] for lvl in levels} for p in providers}
    provider_gap = {p: {lvl: [] for lvl in levels} for p in providers}

    for m in all_metrics:
        provider = get_provider(m.agent_name)
        if provider not in providers:
            continue
        if "level_metrics" not in m.extra:
            continue

        lm = m.extra["level_metrics"]
        for level in levels:
            acc = lm.get("accuracy_by_level", {}).get(level)
            conf = lm.get("confidence_by_level", {}).get(level)
            if acc is not None and not np.isnan(acc):
                provider_acc[provider][level].append(acc)
            if conf is not None and not np.isnan(conf):
                provider_conf[provider][level].append(conf)
            if (
                acc is not None
                and conf is not None
                and not np.isnan(acc)
                and not np.isnan(conf)
            ):
                provider_gap[provider][level].append(conf - acc)

    # Create matrices
    acc_matrix = np.array(
        [
            [
                np.mean(provider_acc[p][lvl]) if provider_acc[p][lvl] else np.nan
                for lvl in levels
            ]
            for p in providers
        ]
    )
    conf_matrix = np.array(
        [
            [
                np.mean(provider_conf[p][lvl]) if provider_conf[p][lvl] else np.nan
                for lvl in levels
            ]
            for p in providers
        ]
    )
    gap_matrix = np.array(
        [
            [
                np.mean(provider_gap[p][lvl]) if provider_gap[p][lvl] else np.nan
                for lvl in levels
            ]
            for p in providers
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    level_labels = ["L1 (Easy)", "L2 (Medium)", "L3 (Hard)"]

    # 1. Accuracy heatmap
    ax = axes[0]
    im = ax.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(level_labels, fontsize=10)
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers, fontsize=11)
    ax.set_title("Accuracy by Provider & Level", fontsize=12, fontweight="bold")
    for i in range(len(providers)):
        for j in range(len(levels)):
            val = acc_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Confidence heatmap
    ax = axes[1]
    im = ax.imshow(conf_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(level_labels, fontsize=10)
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers, fontsize=11)
    ax.set_title("Confidence by Provider & Level", fontsize=12, fontweight="bold")
    for i in range(len(providers)):
        for j in range(len(levels)):
            val = conf_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Calibration gap heatmap (confidence - accuracy)
    ax = axes[2]
    max_abs = max(0.3, np.nanmax(np.abs(gap_matrix)))
    im = ax.imshow(
        gap_matrix, cmap="RdBu_r", aspect="auto", vmin=-max_abs, vmax=max_abs
    )
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(level_labels, fontsize=10)
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers, fontsize=11)
    ax.set_title(
        "Overconfidence Gap by Provider & Level\n(red=overconfident, blue=underconfident)",
        fontsize=12,
        fontweight="bold",
    )
    for i in range(len(providers)):
        for j in range(len(levels)):
            val = gap_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    output_path = output_dir / "level_provider_heatmap.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_level_consistency_patterns(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Analyze consistency patterns across difficulty levels.

    Shows:
    1. Consistency vs Accuracy scatter by level (are hard tasks also inconsistent?)
    2. Variance heatmap (model x level)
    3. "Difficulty frontier" - models that maintain consistency on hard tasks
    """
    has_level_data = any("consistency_by_level" in m.extra for m in all_metrics)
    if not has_level_data:
        print(
            "📊 Skipping level consistency patterns (no consistency-by-level data available)"
        )
        return

    df_sorted = sort_agents_by_provider_and_date(df)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}
    agents = df_sorted["agent"].tolist()
    levels = ["1", "2", "3"]
    level_colors = {"1": "#4CAF50", "2": "#FF9800", "3": "#F44336"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Consistency vs Accuracy by level
    ax = axes[0]
    for level in levels:
        consis = []
        accs = []
        for agent in agents:
            m = agent_to_metrics.get(agent)
            if m and "consistency_by_level" in m.extra and "level_metrics" in m.extra:
                c = (
                    m.extra["consistency_by_level"]
                    .get("consistency_by_level", {})
                    .get(level)
                )
                a = m.extra["level_metrics"].get("accuracy_by_level", {}).get(level)
                if (
                    c is not None
                    and a is not None
                    and not np.isnan(c)
                    and not np.isnan(a)
                ):
                    consis.append(c)
                    accs.append(a)
        if consis:
            ax.scatter(
                accs,
                consis,
                label=f"Level {level}",
                color=level_colors[level],
                s=80,
                alpha=0.7,
                edgecolors="black",
            )

    ax.set_xlabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        r"Outcome Consistency ($C_{\mathrm{out}}$)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Accuracy vs Consistency by Level\n(each point = one model at one level)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Variance heatmap (model x level)
    ax = axes[1]
    variance_data = []
    model_names = []
    for agent in agents:
        m = agent_to_metrics.get(agent)
        if m and "consistency_by_level" in m.extra:
            var_by_level = m.extra["consistency_by_level"].get("variance_by_level", {})
            if var_by_level:
                row = [var_by_level.get(lvl, np.nan) for lvl in levels]
                variance_data.append(row)
                model_names.append(strip_agent_prefix(agent))

    if variance_data:
        variance_matrix = np.array(variance_data)
        im = ax.imshow(variance_matrix, cmap="Reds", aspect="auto", vmin=0)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(["L1", "L2", "L3"], fontsize=10)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=8)
        ax.set_title(
            "Outcome Variance by Level\n(darker = more inconsistent)",
            fontsize=12,
            fontweight="bold",
        )
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 3. Consistency retention (L3 consistency / L1 consistency)
    ax = axes[2]
    retention = []
    model_names_ret = []
    colors = []
    for agent in agents:
        m = agent_to_metrics.get(agent)
        if m and "consistency_by_level" in m.extra:
            c_by_level = m.extra["consistency_by_level"].get("consistency_by_level", {})
            c1 = c_by_level.get("1")
            c3 = c_by_level.get("3")
            if (
                c1 is not None
                and c3 is not None
                and not np.isnan(c1)
                and not np.isnan(c3)
                and c1 > 0
            ):
                retention.append(c3 / c1)
                model_names_ret.append(strip_agent_prefix(agent))
                colors.append(PROVIDER_COLORS.get(get_provider(agent), "#999999"))

    if retention:
        sorted_idx = np.argsort(retention)[::-1]
        sorted_ret = [retention[i] for i in sorted_idx]
        sorted_names = [model_names_ret[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]

        y_pos = np.arange(len(sorted_ret))
        bars = ax.barh(
            y_pos,
            sorted_ret,
            color=sorted_colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Consistency Retention (L3/L1)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Consistency Retention on Hard Tasks\n(>1 = more consistent on hard tasks)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, sorted_ret):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    output_path = output_dir / "level_consistency_patterns.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_action_efficiency_by_level(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Analyze action efficiency across difficulty levels.

    Shows:
    1. Actions per task by level and outcome (success vs failure)
    2. Action "waste" - extra actions on failures vs successes
    3. Efficiency frontier - models that use fewer actions on hard tasks
    """
    has_level_data = any(
        "level_metrics" in m.extra and m.extra["level_metrics"] for m in all_metrics
    )
    if not has_level_data:
        print("📊 Skipping action efficiency analysis (no GAIA level data available)")
        return

    # This requires per-task action counts split by outcome, which we need to compute
    # For now, just show trajectory complexity patterns

    df_sorted = sort_agents_by_provider_and_date(df)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}
    agents = df_sorted["agent"].tolist()
    levels = ["1", "2", "3"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Trajectory complexity growth (actions at L3 / actions at L1)
    ax = axes[0]
    growth = []
    model_names = []
    colors = []
    for agent in agents:
        m = agent_to_metrics.get(agent)
        if m and "level_metrics" in m.extra:
            traj = m.extra["level_metrics"].get("trajectory_complexity", {})
            t1 = traj.get("1")
            t3 = traj.get("3")
            if (
                t1 is not None
                and t3 is not None
                and not np.isnan(t1)
                and not np.isnan(t3)
                and t1 > 0
            ):
                growth.append(t3 / t1)
                model_names.append(strip_agent_prefix(agent))
                colors.append(PROVIDER_COLORS.get(get_provider(agent), "#999999"))

    if growth:
        sorted_idx = np.argsort(growth)
        sorted_growth = [growth[i] for i in sorted_idx]
        sorted_names = [model_names[i] for i in sorted_idx]
        sorted_colors = [colors[i] for i in sorted_idx]

        y_pos = np.arange(len(sorted_growth))
        bars = ax.barh(
            y_pos,
            sorted_growth,
            color=sorted_colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Action Growth (L3/L1)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Action Count Growth on Hard Tasks\n(<1 = fewer actions on hard, >1 = more)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")

        for bar, val in zip(bars, sorted_growth):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
            )

    # 2. Actions vs Accuracy trade-off by level
    ax = axes[1]
    level_colors_local = {"1": "#4CAF50", "2": "#FF9800", "3": "#F44336"}
    for level in levels:
        actions_list = []
        accs = []
        for agent in agents:
            m = agent_to_metrics.get(agent)
            if m and "level_metrics" in m.extra:
                lm = m.extra["level_metrics"]
                traj = lm.get("trajectory_complexity", {}).get(level)
                acc = lm.get("accuracy_by_level", {}).get(level)
                if (
                    traj is not None
                    and acc is not None
                    and not np.isnan(traj)
                    and not np.isnan(acc)
                ):
                    actions_list.append(traj)
                    accs.append(acc)
        if actions_list:
            ax.scatter(
                actions_list,
                accs,
                label=f"Level {level}",
                color=level_colors_local[level],
                s=80,
                alpha=0.7,
                edgecolors="black",
            )

    ax.set_xlabel("Mean Actions per Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Actions vs Accuracy by Level\n(ideal: high accuracy, few actions)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "level_action_efficiency.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_level_reliability_summary(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create a summary radar chart showing reliability dimensions by level.

    For each level, shows average across all models:
    - Accuracy
    - Consistency
    - Calibration
    - Confidence
    """
    has_level_data = any(
        "level_metrics" in m.extra and m.extra["level_metrics"] for m in all_metrics
    )
    if not has_level_data:
        print("📊 Skipping level reliability summary (no GAIA level data available)")
        return

    levels = ["1", "2", "3"]
    dimensions = ["Accuracy", "Consistency", "Calibration", "Confidence"]

    # Aggregate data
    level_data = {lvl: {d: [] for d in dimensions} for lvl in levels}

    for m in all_metrics:
        if "level_metrics" not in m.extra:
            continue
        lm = m.extra["level_metrics"]

        for level in levels:
            acc = lm.get("accuracy_by_level", {}).get(level)
            cal = lm.get("calibration_by_level", {}).get(level)
            conf = lm.get("confidence_by_level", {}).get(level)

            if acc is not None and not np.isnan(acc):
                level_data[level]["Accuracy"].append(acc)
            if cal is not None and not np.isnan(cal):
                level_data[level]["Calibration"].append(cal)
            if conf is not None and not np.isnan(conf):
                level_data[level]["Confidence"].append(conf)

        if "consistency_by_level" in m.extra:
            for level in levels:
                cons = (
                    m.extra["consistency_by_level"]
                    .get("consistency_by_level", {})
                    .get(level)
                )
                if cons is not None and not np.isnan(cons):
                    level_data[level]["Consistency"].append(cons)

    # Compute means
    level_means = {lvl: {} for lvl in levels}
    for level in levels:
        for dim in dimensions:
            vals = level_data[level][dim]
            level_means[level][dim] = np.mean(vals) if vals else np.nan

    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    level_colors = {"1": "#4CAF50", "2": "#FF9800", "3": "#F44336"}
    level_names = {
        "1": "Level 1 (Easy)",
        "2": "Level 2 (Medium)",
        "3": "Level 3 (Hard)",
    }

    for level in levels:
        values = [level_means[level].get(d, 0) for d in dimensions]
        values += values[:1]  # Complete the loop
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=level_names[level],
            color=level_colors[level],
            markersize=8,
        )
        ax.fill(angles, values, alpha=0.15, color=level_colors[level])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Reliability Profile by Difficulty Level\n(average across all models)",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "level_reliability_radar.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

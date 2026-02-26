"""Dashboard plots: plot_reliability_dashboard, plot_metric_heatmap, plot_dimension_radar."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from reliability_eval.constants import (
    PROVIDER_COLORS,
)
from reliability_eval.loaders.agent_names import (
    sort_agents_by_provider_and_date,
    strip_agent_prefix,
)
from reliability_eval.metrics.consistency import compute_weighted_r_con
from reliability_eval.types import ReliabilityMetrics
from reliability_eval.plots.helpers import (
    generate_shaded_colors,
)


def plot_reliability_dashboard(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create comprehensive reliability dashboard with ALL metrics.

    Layout:
    - Row 0: Overall reliability score (bar chart + spider/radar chart)
    - Row 1: Consistency metrics (reliability_consistency summary + consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource)
    - Row 2: Predictability metrics (reliability_predictability summary + predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score)
    - Row 3: Robustness metrics (reliability_robustness summary + robustness_fault_injection, robustness_structural, robustness_prompt_variation)
    - Row 4: Safety metrics (reliability_safety summary + safety_harm_severity, safety_compliance, safety_score) — not included in reliability_overall

    Colors are based on model provider (OpenAI, Google, Anthropic) with shades for release date.
    Models are ordered by provider first, then by release date within each provider.
    """
    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)
    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    x_pos = np.arange(len(agents))

    # Generate provider-based colors with shades
    bar_colors = generate_shaded_colors(df_sorted)

    # Compute dimension-level scores
    df_sorted["reliability_consistency"] = compute_weighted_r_con(
        df_sorted["consistency_outcome"],
        df_sorted["consistency_trajectory_distribution"],
        df_sorted["consistency_trajectory_sequence"],
        df_sorted["consistency_resource"],
    )
    df_sorted["reliability_predictability"] = df_sorted[
        "predictability_brier_score"
    ]  # Brier score captures both calibration and discrimination
    df_sorted["reliability_robustness"] = df_sorted[["robustness_fault_injection", "robustness_structural", "robustness_prompt_variation"]].mean(
        axis=1, skipna=True
    )
    df_sorted["reliability_safety"] = df_sorted["safety_score"]
    # Overall reliability = uniform average of consistency, predictability, robustness
    df_sorted["reliability_overall"] = df_sorted[["reliability_consistency", "reliability_predictability", "reliability_robustness"]].mean(
        axis=1, skipna=True
    )

    # Create figure with GridSpec layout
    # Row 0: 2 plots (bar + radar for overall)
    # Rows 1-4: 6 plots each (1 summary + 5 submetrics for consistency, 1+4 for others)
    fig = plt.figure(figsize=(28, 26))
    gs = gridspec.GridSpec(
        5, 6, figure=fig, hspace=0.45, wspace=0.35, height_ratios=[1.2, 1, 1, 1, 1]
    )

    def plot_bar(
        ax, data, ylabel, title, colors_to_use, show_labels=True, ylim_max=1.05
    ):
        """Helper to create bar chart with provider-based colors."""
        valid_data = data.fillna(0)
        bars = ax.bar(
            x_pos,
            valid_data,
            color=colors_to_use,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, ylim_max)
        ax.grid(True, alpha=0.3, axis="y")
        if show_labels:
            for bar, val in zip(bars, data):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
        return bars

    # Add legend for providers
    def add_provider_legend(ax):
        """Add a legend showing provider colors."""
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=PROVIDER_COLORS["OpenAI"], edgecolor="black", label="OpenAI"
            ),
            Patch(
                facecolor=PROVIDER_COLORS["Google"], edgecolor="black", label="Google"
            ),
            Patch(
                facecolor=PROVIDER_COLORS["Anthropic"],
                edgecolor="black",
                label="Anthropic",
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9
        )

    # =========================================================================
    # ROW 0: OVERALL RELIABILITY (Bar Chart + Spider Chart)
    # =========================================================================

    # Overall reliability bar chart (spans 3 columns)
    ax = fig.add_subplot(gs[0, 0:3])
    plot_bar(
        ax,
        df_sorted["reliability_overall"],
        r"$R_{\mathrm{Overall}}$",
        r"Overall Reliability Score (mean of $R_{\mathrm{Con}}$, $R_{\mathrm{Pred}}$, $R_{\mathrm{Rob}}$)",
        bar_colors,
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Moderate")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Good")
    add_provider_legend(ax)

    # Spider/Radar chart for dimension-level comparison (spans 3 columns)
    ax = fig.add_subplot(gs[0, 3:6], polar=True)
    dimensions = ["reliability_consistency", "reliability_predictability", "reliability_robustness"]
    dim_labels = ["Consistency", "Predictability", "Robustness"]

    num_vars = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        values = [row[d] if not np.isnan(row[d]) else 0 for d in dimensions]
        values += values[:1]  # Close the polygon
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=1.5,
            label=row["agent"][:15],
            color=bar_colors[idx],
            alpha=0.7,
        )
        ax.fill(angles, values, alpha=0.1, color=bar_colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Reliability Dimension Profile", fontsize=11, fontweight="bold", pad=15
    )
    # Only show legend if few agents
    if len(agents) <= 6:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7)

    # =========================================================================
    # ROW 1: CONSISTENCY METRICS (reliability_consistency summary + consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_confidence, consistency_resource)
    # =========================================================================

    # reliability_consistency summary (aggregate)
    ax = fig.add_subplot(gs[1, 0])
    plot_bar(
        ax,
        df_sorted["reliability_consistency"],
        r"$R_{\mathrm{Con}}$",
        "Consistency\n(Aggregate)",
        bar_colors,
    )
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)

    # consistency_outcome
    ax = fig.add_subplot(gs[1, 1])
    plot_bar(
        ax,
        df_sorted["consistency_outcome"],
        r"$C_{\mathrm{out}}$",
        "Outcome\nConsistency",
        bar_colors,
    )

    # consistency_trajectory_distribution
    ax = fig.add_subplot(gs[1, 2])
    plot_bar(
        ax,
        df_sorted["consistency_trajectory_distribution"],
        r"$C^{d}_{\mathrm{traj}}$",
        "Trajectory\nDistribution",
        bar_colors,
    )

    # consistency_trajectory_sequence
    ax = fig.add_subplot(gs[1, 3])
    plot_bar(
        ax,
        df_sorted["consistency_trajectory_sequence"],
        r"$C^{s}_{\mathrm{traj}}$",
        "Trajectory\nSequence",
        bar_colors,
    )

    # consistency_confidence
    ax = fig.add_subplot(gs[1, 4])
    plot_bar(
        ax,
        df_sorted["consistency_confidence"],
        r"$C_{\mathrm{conf}}$",
        "Confidence\nConsistency",
        bar_colors,
    )

    # consistency_resource
    ax = fig.add_subplot(gs[1, 5])
    plot_bar(
        ax,
        df_sorted["consistency_resource"],
        r"$C_{\mathrm{res}}$",
        "Resource\nConsistency",
        bar_colors,
    )

    # =========================================================================
    # ROW 2: PREDICTABILITY METRICS (reliability_predictability summary + predictability_rate_confidence_correlation, predictability_calibration, predictability_roc_auc, predictability_brier_score)
    # =========================================================================

    # reliability_predictability summary
    ax = fig.add_subplot(gs[2, 0])
    plot_bar(
        ax,
        df_sorted["reliability_predictability"],
        r"$R_{\mathrm{Pred}}$",
        "Predictability\n(Aggregate)",
        bar_colors,
    )
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)

    # predictability_rate_confidence_correlation
    ax = fig.add_subplot(gs[2, 1])
    plot_bar(
        ax, df_sorted["predictability_rate_confidence_correlation"], r"$P_{\mathrm{rc}}$", "Risk-Coverage\nScore", bar_colors
    )

    # predictability_calibration
    ax = fig.add_subplot(gs[2, 2])
    plot_bar(
        ax,
        df_sorted["predictability_calibration"],
        r"$P_{\mathrm{cal}}$",
        "Calibration\n(1-ECE)",
        bar_colors,
    )

    # predictability_roc_auc
    ax = fig.add_subplot(gs[2, 3])
    plot_bar(
        ax,
        df_sorted["predictability_roc_auc"],
        r"$P_{\mathrm{AUROC}}$",
        "Discrimination\n(AUC-ROC)",
        bar_colors,
    )

    # predictability_brier_score
    ax = fig.add_subplot(gs[2, 4])
    plot_bar(
        ax,
        df_sorted["predictability_brier_score"],
        r"$P_{\mathrm{Brier}}$",
        "Quality\n(1-Brier)",
        bar_colors,
    )

    # Capability (accuracy) for context
    ax = fig.add_subplot(gs[2, 5])
    plot_bar(
        ax, df_sorted["accuracy"], "Accuracy", "Capability\n(Accuracy)", bar_colors
    )
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)

    # =========================================================================
    # ROW 3: ROBUSTNESS METRICS (reliability_robustness summary + robustness_fault_injection, robustness_structural, robustness_prompt_variation + extra)
    # =========================================================================

    # reliability_robustness summary
    ax = fig.add_subplot(gs[3, 0])
    plot_bar(
        ax,
        df_sorted["reliability_robustness"],
        r"$R_{\mathrm{Rob}}$",
        "Robustness\n(Aggregate)",
        bar_colors,
        ylim_max=1.15,
    )
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)

    # robustness_fault_injection
    ax = fig.add_subplot(gs[3, 1])
    plot_bar(
        ax,
        df_sorted["robustness_fault_injection"],
        r"$R_{\mathrm{fault}}$",
        "Fault\nRobustness",
        bar_colors,
        ylim_max=1.15,
    )
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    # robustness_structural
    ax = fig.add_subplot(gs[3, 2])
    plot_bar(
        ax,
        df_sorted["robustness_structural"],
        r"$R_{\mathrm{env}}$",
        "Environment\nRobustness",
        bar_colors,
        ylim_max=1.15,
    )
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    # robustness_prompt_variation
    ax = fig.add_subplot(gs[3, 3])
    plot_bar(
        ax,
        df_sorted["robustness_prompt_variation"],
        r"$R_{\mathrm{prompt}}$",
        "Prompt\nRobustness",
        bar_colors,
        ylim_max=1.15,
    )
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    # Risk-Coverage Curves (spans 2 columns)
    ax = fig.add_subplot(gs[3, 4:6])
    # Match all_metrics order to df_sorted order
    agent_to_metrics = {m.agent_name: m for m in all_metrics}
    for idx, agent in enumerate(agents):
        m = agent_to_metrics.get(agent)
        if (
            m
            and "aurc_data" in m.extra
            and m.extra["aurc_data"].get("coverages") is not None
        ):
            d = m.extra["aurc_data"]
            if len(d.get("coverages", [])) > 0:
                ax.plot(
                    d["coverages"],
                    d["risks"],
                    label=strip_agent_prefix(m.agent_name)[:12],
                    linewidth=2,
                    color=bar_colors[idx],
                    alpha=0.8,
                )
    ax.set_xlabel("Coverage", fontweight="bold", fontsize=9)
    ax.set_ylabel("Risk", fontweight="bold", fontsize=9)
    ax.set_title("Risk-Coverage Curves", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if len(agents) <= 8:
        ax.legend(fontsize=6, loc="best")

    # =========================================================================
    # ROW 4: SAFETY METRICS (reliability_safety summary + safety_harm_severity, safety_compliance, safety_score + calibration)
    # =========================================================================

    # reliability_safety summary
    ax = fig.add_subplot(gs[4, 0])
    plot_bar(
        ax, df_sorted["reliability_safety"], r"$R_{\mathrm{Saf}}$", "Safety\n(Aggregate)", bar_colors
    )
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Good")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5)

    # safety_harm_severity
    ax = fig.add_subplot(gs[4, 1])
    plot_bar(
        ax,
        df_sorted["safety_harm_severity"],
        r"$S_{\mathrm{harm}}$",
        "Harm Score\n(exp(-severity))",
        bar_colors,
    )

    # safety_compliance
    ax = fig.add_subplot(gs[4, 2])
    plot_bar(
        ax,
        df_sorted["safety_compliance"],
        r"$S_{\mathrm{comp}}$",
        "Compliance\n(1-violation)",
        bar_colors,
    )

    # safety_score
    ax = fig.add_subplot(gs[4, 3])
    plot_bar(
        ax, df_sorted["safety_score"], r"$S_{\mathrm{safety}}$", "Safety Score", bar_colors
    )

    # Calibration diagram (spans 2 columns)
    ax = fig.add_subplot(gs[4, 4:6])
    for idx, agent in enumerate(agents):
        m = agent_to_metrics.get(agent)
        if m and "calibration_bins" in m.extra and m.extra["calibration_bins"]:
            bins = m.extra["calibration_bins"]
            confs = [b["avg_confidence"] for b in bins if b.get("count", 0) > 0]
            accs = [b["avg_accuracy"] for b in bins if b.get("count", 0) > 0]
            if confs:
                ax.scatter(
                    confs,
                    accs,
                    s=60,
                    color=bar_colors[idx],
                    alpha=0.7,
                    label=strip_agent_prefix(m.agent_name)[:12],
                )
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Confidence", fontweight="bold", fontsize=9)
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=9)
    ax.set_title("Reliability Diagram (Calibration)", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if len(agents) <= 8:
        ax.legend(fontsize=6, loc="best")

    plt.suptitle(
        "Comprehensive Reliability Evaluation Dashboard",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    output_path = output_dir / "reliability_dashboard.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_metric_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of ALL metrics, sorted by provider and release date."""
    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)

    metrics_cols = [
        "accuracy",
        "consistency_outcome",
        "consistency_trajectory_distribution",
        "consistency_trajectory_sequence",
        "consistency_confidence",
        "consistency_resource",
        "predictability_rate_confidence_correlation",
        "predictability_calibration",
        "predictability_roc_auc",
        "predictability_brier_score",
        "robustness_fault_injection",
        "robustness_structural",
        "robustness_prompt_variation",
        "safety_harm_severity",
        "safety_compliance",
        "safety_score",
    ]
    labels = [
        "Accuracy",
        "consistency_outcome",
        "consistency_trajectory_distribution",
        "consistency_trajectory_sequence",
        "consistency_confidence",
        "consistency_resource",
        "predictability_rate_confidence_correlation",
        "predictability_calibration",
        "predictability_roc_auc",
        "predictability_brier_score",
        "robustness_fault_injection",
        "robustness_structural",
        "robustness_prompt_variation",
        "safety_harm_severity",
        "safety_compliance",
        "safety_score",
    ]

    available = [
        c
        for c in metrics_cols
        if c in df_sorted.columns and not df_sorted[c].isna().all()
    ]
    avail_labels = [labels[metrics_cols.index(c)] for c in available]

    if not available:
        print("⚠️  No metrics available for heatmap")
        return

    matrix = df_sorted[available].values

    fig, ax = plt.subplots(figsize=(14, max(6, len(df_sorted) * 0.7)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels(
        avail_labels, fontsize=10, fontweight="bold", rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(len(df_sorted)))

    # Add provider color indicators to y-axis labels
    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    providers = df_sorted["provider"].tolist()
    ax.set_yticklabels(agents, fontsize=10)

    # Color the y-axis labels by provider
    for idx, (tick_label, provider) in enumerate(zip(ax.get_yticklabels(), providers)):
        tick_label.set_color(PROVIDER_COLORS.get(provider, "#999999"))
        tick_label.set_fontweight("bold")

    for i in range(len(df_sorted)):
        for j in range(len(available)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 or val > 0.8 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                )

    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
    ax.set_title(
        "Reliability Metrics Heatmap\n(sorted by provider and release date)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add provider legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PROVIDER_COLORS["OpenAI"], edgecolor="black", label="OpenAI"),
        Patch(facecolor=PROVIDER_COLORS["Google"], edgecolor="black", label="Google"),
        Patch(
            facecolor=PROVIDER_COLORS["Anthropic"], edgecolor="black", label="Anthropic"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        fontsize=9,
    )

    plt.tight_layout()
    output_path = output_dir / "reliability_heatmap.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_dimension_radar(df: pd.DataFrame, output_dir: Path):
    """Create radar chart with DIMENSION-LEVEL aggregates (as per paper §3.7)."""
    # Sort by provider and release date
    df_dims = sort_agents_by_provider_and_date(df)

    # reliability_consistency = weighted consistency aggregate (outcome & resource weighted > trajectory)
    df_dims["reliability_consistency"] = compute_weighted_r_con(
        df_dims["consistency_outcome"], df_dims["consistency_trajectory_distribution"], df_dims["consistency_trajectory_sequence"], df_dims["consistency_resource"]
    )

    # reliability_robustness = mean of all robustness metrics (robustness_fault_injection, robustness_structural, robustness_prompt_variation)
    robustness_cols = [
        c for c in ["robustness_fault_injection", "robustness_structural", "robustness_prompt_variation"] if c in df_dims.columns
    ]
    if robustness_cols:
        df_dims["reliability_robustness"] = df_dims[robustness_cols].mean(axis=1, skipna=True)
    else:
        df_dims["reliability_robustness"] = np.nan

    # reliability_predictability = Brier score (proper scoring rule capturing calibration + discrimination)
    df_dims["reliability_predictability"] = df_dims["predictability_brier_score"]

    # reliability_safety = safety_score (lambda-weighted safety score)
    df_dims["reliability_safety"] = df_dims["safety_score"]

    dimensions = ["reliability_consistency", "reliability_robustness", "reliability_predictability"]
    dim_labels = ["Consistency", "Robustness", "Predictability"]

    available = [d for d in dimensions if not df_dims[d].isna().all()]
    avail_labels = [dim_labels[dimensions.index(d)] for d in available]

    if len(available) < 3:
        print("⚠️  Not enough dimensions for radar chart")
        return

    # Generate provider-based colors
    bar_colors = generate_shaded_colors(df_dims)

    num_vars = len(available)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for idx, (_, row) in enumerate(df_dims.iterrows()):
        values = [row[d] if not np.isnan(row[d]) else 0 for d in available]
        values += values[:1]
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=row["agent"],
            color=bar_colors[idx],
            alpha=0.7,
        )
        ax.fill(angles, values, alpha=0.15, color=bar_colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(avail_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Reliability Dimension Profile\n(sorted by provider and release date)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Show legend only for small number of agents
    if len(df_dims) <= 8:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "reliability_radar.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

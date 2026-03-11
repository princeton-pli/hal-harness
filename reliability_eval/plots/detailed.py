"""Detailed dimension plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from reliability_eval.constants import (
    PROVIDER_COLORS,
    PROVIDER_MARKERS,
)
from reliability_eval.loaders.agent_names import (
    get_model_metadata,
    get_provider,
    sort_agents_by_provider_and_date,
    strip_agent_prefix,
)
from reliability_eval.metrics.consistency import compute_weighted_r_con
from reliability_eval.types import ReliabilityMetrics
from reliability_eval.plots.helpers import (
    _add_bar_labels_ci,
    _bar_with_ci,
    _get_aggregate_yerr,
    _get_weighted_r_con_yerr,
    _get_yerr,
    filter_oldest_and_newest_per_provider,
    generate_shaded_colors,
)


def plot_consistency_detailed(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create detailed consistency plots - vertical layout with bar plots.
    Shows: reliability_consistency (overall), consistency_outcome, consistency_trajectory_distribution, consistency_trajectory_sequence, consistency_resource
    """
    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)
    bar_colors = generate_shaded_colors(df_sorted)

    # Vertical layout: 5 rows, 1 column
    fig, axes = plt.subplots(5, 1, figsize=(5, 12))

    # Extract just model names (remove scaffold prefixes)
    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    x_pos = np.arange(len(agents))

    # 1. reliability_consistency (Overall Consistency) - aggregate with propagated SE
    ax = axes[0]
    reliability_consistency = pd.Series(
        compute_weighted_r_con(
            df_sorted["consistency_outcome"],
            df_sorted["consistency_trajectory_distribution"],
            df_sorted["consistency_trajectory_sequence"],
            df_sorted["consistency_resource"],
        ),
        index=df_sorted.index,
    ).fillna(0)
    yerr_agg = _get_weighted_r_con_yerr(df_sorted, values=reliability_consistency)
    bars = ax.bar(
        x_pos,
        reliability_consistency,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        yerr=yerr_agg,
        capsize=3,
        error_kw={"linewidth": 1.0, "color": "black"},
    )
    ax.set_ylabel(r"$R_{\mathrm{Con}}$", fontsize=14, fontweight="bold")
    ax.set_title("Overall Consistency", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, reliability_consistency, yerr_agg)

    # 2. consistency_outcome (Outcome Consistency)
    ax = axes[1]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["consistency_outcome"].fillna(0),
        bar_colors,
        df_sorted,
        "consistency_outcome",
    )
    ax.set_ylabel(r"$C_{\mathrm{out}}$", fontsize=14, fontweight="bold")
    ax.set_title("Outcome Consistency", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["consistency_outcome"], yerr)

    # 3. consistency_trajectory_distribution (Trajectory Distribution Consistency)
    ax = axes[2]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["consistency_trajectory_distribution"].fillna(0),
        bar_colors,
        df_sorted,
        "consistency_trajectory_distribution",
    )
    ax.set_ylabel(r"$C^{d}_{\mathrm{traj}}$", fontsize=14, fontweight="bold")
    ax.set_title("Trajectory Distribution Consistency", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["consistency_trajectory_distribution"], yerr)

    # 4. consistency_trajectory_sequence (Trajectory Sequence Consistency)
    ax = axes[3]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["consistency_trajectory_sequence"].fillna(0),
        bar_colors,
        df_sorted,
        "consistency_trajectory_sequence",
    )
    ax.set_ylabel(r"$C^{s}_{\mathrm{traj}}$", fontsize=14, fontweight="bold")
    ax.set_title("Trajectory Sequence Consistency", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["consistency_trajectory_sequence"], yerr)

    # 5. consistency_resource (Resource Consistency) - at the bottom with x labels
    ax = axes[4]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["consistency_resource"].fillna(0),
        bar_colors,
        df_sorted,
        "consistency_resource",
    )
    ax.set_ylabel(r"$C_{\mathrm{res}}$", fontsize=14, fontweight="bold")
    ax.set_title("Resource Consistency", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["consistency_resource"], yerr)

    plt.tight_layout()
    output_path = output_dir / "consistency_detailed.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_predictability_detailed(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create detailed predictability plots - vertical layout with bar plots.
    Shows: reliability_predictability (overall), predictability_calibration, predictability_roc_auc, predictability_brier_score
    """
    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)
    bar_colors = generate_shaded_colors(df_sorted)

    # Vertical layout: 4 rows, 1 column (matching consistency plot style)
    fig, axes = plt.subplots(4, 1, figsize=(5, 10))

    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    x_pos = np.arange(len(agents))

    # 1. reliability_predictability (Overall Predictability) = predictability_brier_score
    ax = axes[0]
    reliability_predictability = df_sorted["predictability_brier_score"].fillna(0)
    yerr_pred = _get_yerr(
        df_sorted, "predictability_brier_score", values=reliability_predictability
    )
    bars = ax.bar(
        x_pos,
        reliability_predictability,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        yerr=yerr_pred,
        capsize=3,
        error_kw={"linewidth": 1.0, "color": "black"},
    )
    ax.set_ylabel(r"$R_{\mathrm{Pred}}$", fontsize=14, fontweight="bold")
    ax.set_title("Overall Predictability", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, reliability_predictability, yerr_pred)

    # 2. predictability_calibration (Calibration)
    ax = axes[1]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["predictability_calibration"].fillna(0),
        bar_colors,
        df_sorted,
        "predictability_calibration",
    )
    ax.set_ylabel(r"$P_{\mathrm{cal}}$", fontsize=14, fontweight="bold")
    ax.set_title("Calibration", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["predictability_calibration"], yerr)

    # 3. predictability_roc_auc (Discrimination)
    ax = axes[2]
    p_auroc_vals = (
        df_sorted["predictability_roc_auc"].fillna(0)
        if "predictability_roc_auc" in df_sorted.columns
        else pd.Series([0] * len(df_sorted))
    )
    bars, yerr = _bar_with_ci(
        ax, x_pos, p_auroc_vals, bar_colors, df_sorted, "predictability_roc_auc"
    )
    ax.set_ylabel(r"$P_{\mathrm{AUROC}}$", fontsize=14, fontweight="bold")
    ax.set_title("Discrimination", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, p_auroc_vals, yerr)

    # 4. predictability_brier_score (Overall Quality) - at the bottom with x labels
    ax = axes[3]
    p_brier_vals = (
        df_sorted["predictability_brier_score"].fillna(0)
        if "predictability_brier_score" in df_sorted.columns
        else pd.Series([0] * len(df_sorted))
    )
    bars, yerr = _bar_with_ci(
        ax, x_pos, p_brier_vals, bar_colors, df_sorted, "predictability_brier_score"
    )
    ax.set_ylabel(r"$P_{\mathrm{Brier}}$", fontsize=14, fontweight="bold")
    ax.set_title("Overall Quality", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, p_brier_vals, yerr)

    plt.tight_layout()
    output_path = output_dir / "predictability_detailed.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_accuracy_coverage_by_model(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create accuracy-coverage plots for each model in a 3x4 grid (provider x model).
    Rows: OpenAI, Google, Anthropic
    Cols: 4 models per provider (sorted by release date)
    Excludes gpt_5_2_xhigh reasoning model.
    Works with any benchmark by dynamically detecting scaffold prefixes.
    """
    # Define model order per provider (excluding reasoning variants like xhigh)
    provider_models = {
        "OpenAI": ["gpt_4_turbo", "gpt_4o_mini", "gpt_o1", "gpt_5_2", "gpt_5_4"],
        "Google": [
            "gemini_2_flash",
            "gemini_2_5_flash",
            "gemini_2_5_pro",
            "gemini_3_pro",
        ],
        "Anthropic": [
            "claude_haiku_3_5",
            "claude_sonnet_3_7",
            "claude_sonnet_4_5",
            "claude_opus_4_5",
        ],
    }

    provider_order = ["OpenAI", "Google", "Anthropic"]

    # Build mapping from model key to agent name
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    # Detect scaffold prefixes dynamically from actual agent names
    # e.g., 'taubench_toolcalling_gpt_4_turbo' -> prefix is 'taubench_toolcalling_'
    # e.g., 'gaia_generalist_claude_haiku_3_5' -> prefix is 'gaia_generalist_'
    all_model_keys = [m for models in provider_models.values() for m in models]
    detected_prefixes = set()
    for agent_name in agent_to_metrics.keys():
        for model_key in all_model_keys:
            if agent_name.endswith(model_key):
                prefix = agent_name[: -len(model_key)]
                if prefix:
                    detected_prefixes.add(prefix)
    detected_prefixes = (
        list(detected_prefixes)
        if detected_prefixes
        else ["taubench_toolcalling_", "taubench_fewshot_"]
    )

    # Model display name mapping for well-formatted titles
    model_display_names = {
        "gpt_4_turbo": "GPT-4 Turbo",
        "gpt_4o_mini": "GPT-4o mini",
        "gpt_o1": "o1",
        "gpt_5_2": "GPT-5.2",
        "gpt_5_4": "GPT-5.4",
        "gemini_2_flash": "Gemini 2.0 Flash",
        "gemini_2_5_flash": "Gemini 2.5 Flash",
        "gemini_2_5_pro": "Gemini 2.5 Pro",
        "gemini_3_pro": "Gemini 3 Pro",
        "claude_haiku_3_5": "Claude 3.5 Haiku",
        "claude_sonnet_3_7": "Claude 3.7 Sonnet",
        "claude_sonnet_4_5": "Claude Sonnet 4.5",
        "claude_opus_4_5": "Claude Opus 4.5",
    }

    # Dynamic grid: 3 rows x max_models columns
    max_models = max(len(models) for models in provider_models.values())
    fig, axes = plt.subplots(3, max_models, figsize=(2.5 * max_models, 7.5))

    # Common ticks for both axes
    axis_ticks = [0, 0.25, 0.5, 0.75, 1.0]

    for row_idx, provider in enumerate(provider_order):
        models = provider_models[provider]
        provider_color = PROVIDER_COLORS.get(provider, "#999999")

        # Hide unused axes for providers with fewer models
        for col_idx in range(len(models), max_models):
            axes[row_idx, col_idx].set_visible(False)

        for col_idx, model_key in enumerate(models):
            ax = axes[row_idx, col_idx]

            # Find agent with this model (try all detected scaffolds)
            agent_name = None
            for prefix in detected_prefixes:
                candidate = f"{prefix}{model_key}"
                if candidate in agent_to_metrics:
                    agent_name = candidate
                    break

            # Get display name
            display_name = model_display_names.get(
                model_key, model_key.replace("_", " ").title()
            )

            if agent_name is None:
                ax.text(
                    0.5,
                    0.5,
                    f"{display_name}\n(no data)",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                ax.set_xticks(axis_ticks)
                ax.set_yticks(axis_ticks)
                ax.set_aspect("equal")
                if row_idx == 2:
                    ax.set_xlabel("Coverage", fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel("Accuracy", fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_title(display_name, fontsize=11, fontweight="bold")
                continue

            m = agent_to_metrics[agent_name]

            # Get accuracy-coverage data (convert from risk-coverage)
            if "aurc_data" in m.extra and m.extra["aurc_data"]:
                d = m.extra["aurc_data"]
                if d.get("coverages") is not None and len(d.get("coverages", [])) > 0:
                    coverages = np.array(d["coverages"])
                    # Convert risk to accuracy
                    accuracies = 1 - np.array(d["risks"])
                    optimal_accuracies = (
                        1 - np.array(d["optimal_risks"])
                        if d.get("optimal_risks")
                        else None
                    )

                    # Plot model's accuracy-coverage curve
                    ax.plot(
                        coverages,
                        accuracies,
                        color=provider_color,
                        linewidth=2,
                        label="Model",
                        alpha=0.9,
                    )

                    # Plot ideal/optimal bound
                    if optimal_accuracies is not None:
                        ax.plot(
                            coverages,
                            optimal_accuracies,
                            "k--",
                            linewidth=1.5,
                            alpha=0.7,
                            label="Ideal",
                        )
                        # Fill the gap
                        ax.fill_between(
                            coverages,
                            accuracies,
                            optimal_accuracies,
                            alpha=0.2,
                            color=provider_color,
                        )

                    # Plot random baseline (horizontal line at overall accuracy)
                    overall_accuracy = accuracies[-1]  # Accuracy at full coverage
                    ax.axhline(
                        y=overall_accuracy,
                        color="red",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.6,
                        label="Random",
                    )

                    # Add predictability_roc_auc annotation at bottom right
                    ax.annotate(
                        r"$P_{\mathrm{AUROC}}$" + f"={m.predictability_roc_auc:.2f}",
                        xy=(0.97, 0.03),
                        xycoords="axes fraction",
                        ha="right",
                        va="bottom",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                        ),
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"{model_key}\n(no curve data)",
                        ha="center",
                        va="center",
                        fontsize=10,
                        transform=ax.transAxes,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{model_key}\n(no AURC data)",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )

            # Format subplot - square with equal ticks
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Panel title
            ax.set_title(display_name, fontsize=11, fontweight="bold")

            if row_idx == 2:
                ax.set_xlabel("Coverage", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Accuracy", fontsize=11)

    # Add legend (horizontal, top center)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, label="Model"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Ideal"),
        Line2D([0], [0], color="red", linewidth=1.5, linestyle=":", label="Random"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=9,
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
    )

    plt.tight_layout()
    output_path = output_dir / "accuracy_coverage_by_model.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_calibration_by_model(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create calibration/reliability diagram plots for each model in a grid (provider x model).
    Rows: OpenAI, Google, Anthropic
    Cols: dynamic based on max models per provider
    Excludes reasoning variants (xhigh, medium).
    Works with any benchmark by dynamically detecting scaffold prefixes.
    """
    # Define model order per provider (excluding reasoning variants)
    provider_models = {
        "OpenAI": ["gpt_4_turbo", "gpt_4o_mini", "gpt_o1", "gpt_5_2", "gpt_5_4"],
        "Google": [
            "gemini_2_flash",
            "gemini_2_5_flash",
            "gemini_2_5_pro",
            "gemini_3_pro",
        ],
        "Anthropic": [
            "claude_haiku_3_5",
            "claude_sonnet_3_7",
            "claude_sonnet_4_5",
            "claude_opus_4_5",
        ],
    }

    provider_order = ["OpenAI", "Google", "Anthropic"]

    # Model display name mapping for well-formatted titles
    model_display_names = {
        "gpt_4_turbo": "GPT-4 Turbo",
        "gpt_4o_mini": "GPT-4o mini",
        "gpt_o1": "o1",
        "gpt_5_2": "GPT-5.2",
        "gpt_5_4": "GPT-5.4",
        "gemini_2_flash": "Gemini 2.0 Flash",
        "gemini_2_5_flash": "Gemini 2.5 Flash",
        "gemini_2_5_pro": "Gemini 2.5 Pro",
        "gemini_3_pro": "Gemini 3 Pro",
        "claude_haiku_3_5": "Claude 3.5 Haiku",
        "claude_sonnet_3_7": "Claude 3.7 Sonnet",
        "claude_sonnet_4_5": "Claude Sonnet 4.5",
        "claude_opus_4_5": "Claude Opus 4.5",
    }

    # Build mapping from model key to agent name
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    # Detect scaffold prefixes dynamically from actual agent names
    all_model_keys = [m for models in provider_models.values() for m in models]
    detected_prefixes = set()
    for agent_name in agent_to_metrics.keys():
        for model_key in all_model_keys:
            if agent_name.endswith(model_key):
                prefix = agent_name[: -len(model_key)]
                if prefix:
                    detected_prefixes.add(prefix)
    detected_prefixes = (
        list(detected_prefixes)
        if detected_prefixes
        else ["taubench_toolcalling_", "taubench_fewshot_"]
    )

    # Dynamic grid: 3 rows x max_models columns
    max_models = max(len(models) for models in provider_models.values())
    fig, axes = plt.subplots(3, max_models, figsize=(2.5 * max_models, 7.5))

    # Common ticks for both axes
    axis_ticks = [0, 0.25, 0.5, 0.75, 1.0]

    for row_idx, provider in enumerate(provider_order):
        models = provider_models[provider]
        provider_color = PROVIDER_COLORS.get(provider, "#999999")

        # Hide unused axes for providers with fewer models
        for col_idx in range(len(models), max_models):
            axes[row_idx, col_idx].set_visible(False)

        for col_idx, model_key in enumerate(models):
            ax = axes[row_idx, col_idx]

            # Find agent with this model (try all detected scaffolds)
            agent_name = None
            for prefix in detected_prefixes:
                candidate = f"{prefix}{model_key}"
                if candidate in agent_to_metrics:
                    agent_name = candidate
                    break

            # Get display name
            display_name = model_display_names.get(
                model_key, model_key.replace("_", " ").title()
            )

            if agent_name is None:
                ax.text(
                    0.5,
                    0.5,
                    f"{display_name}\n(no data)",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                ax.set_xticks(axis_ticks)
                ax.set_yticks(axis_ticks)
                ax.set_aspect("equal")
                if row_idx == 2:
                    ax.set_xlabel("Confidence", fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel("Accuracy", fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_title(display_name, fontsize=11, fontweight="bold")
                continue

            m = agent_to_metrics[agent_name]
            bins = m.extra.get("calibration_bins", [])

            if bins:
                valid_bins = [b for b in bins if b.get("count", 0) > 0]
                if valid_bins:
                    confs = [b["avg_confidence"] for b in valid_bins]
                    accs = [b["avg_accuracy"] for b in valid_bins]
                    counts = [b["count"] for b in valid_bins]
                    max_count = max(counts)
                    sizes = [c / max_count * 300 + 50 for c in counts]

                    # Plot calibration points
                    ax.scatter(
                        confs,
                        accs,
                        s=sizes,
                        alpha=0.7,
                        color=provider_color,
                        edgecolors="black",
                        linewidth=1,
                    )

                    # Perfect calibration line
                    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)

                    # Gap lines showing miscalibration
                    for conf, acc in zip(confs, accs):
                        ax.plot(
                            [conf, conf],
                            [conf, acc],
                            color="red",
                            alpha=0.4,
                            linewidth=1,
                        )

                    # ECE annotation at top left
                    ece = (
                        1 - m.predictability_calibration
                        if not np.isnan(m.predictability_calibration)
                        else np.nan
                    )
                    if not np.isnan(ece):
                        ax.annotate(
                            f"ECE={ece:.3f}",
                            xy=(0.03, 0.97),
                            xycoords="axes fraction",
                            ha="left",
                            va="top",
                            fontsize=10,
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.8
                            ),
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"{display_name}\n(no valid bins)",
                        ha="center",
                        va="center",
                        fontsize=10,
                        transform=ax.transAxes,
                    )
                    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{display_name}\n(no calibration data)",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)

            # Format subplot - square with equal ticks
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Panel title
            ax.set_title(display_name, fontsize=11, fontweight="bold")

            if row_idx == 2:
                ax.set_xlabel("Confidence", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Accuracy", fontsize=11)

    # Add global legend for circle sizes (scaled down for legend display)
    from matplotlib.lines import Line2D

    legend_marker_sizes = [5, 9, 13]  # scaled down for legend
    legend_labels = ["Few samples", "Medium", "Many samples"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=s,
            markeredgecolor="black",
            markeredgewidth=1,
            label=label,
            linestyle="None",
        )
        for s, label in zip(legend_marker_sizes, legend_labels)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        fontsize=9,
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        title="Sample count",
        title_fontsize=9,
    )

    plt.tight_layout()
    output_path = output_dir / "calibration_by_model.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_robustness_detailed(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create detailed robustness plots - vertical layout with bar plots.
    Shows: reliability_robustness (overall), robustness_fault_injection, robustness_structural, robustness_prompt_variation
    """
    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)
    bar_colors = generate_shaded_colors(df_sorted)

    # Vertical layout: 4 rows, 1 column
    fig, axes = plt.subplots(4, 1, figsize=(5, 10))

    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    x_pos = np.arange(len(agents))

    # 1. reliability_robustness (Overall Robustness) - aggregate with propagated SE
    ax = axes[0]
    robustness_cols = ["robustness_fault_injection", "robustness_structural"]
    if "robustness_prompt_variation" in df_sorted.columns:
        robustness_cols.append("robustness_prompt_variation")
    reliability_robustness = (
        df_sorted[robustness_cols].mean(axis=1, skipna=True).fillna(0)
    )
    rob_se_cols = ["robustness_fault_injection_se", "robustness_structural_se"]
    if "robustness_prompt_variation" in df_sorted.columns:
        rob_se_cols.append("robustness_prompt_variation_se")
    yerr_agg = _get_aggregate_yerr(
        df_sorted, rob_se_cols, values=reliability_robustness
    )
    bars = ax.bar(
        x_pos,
        reliability_robustness,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        yerr=yerr_agg,
        capsize=3,
        error_kw={"linewidth": 1.0, "color": "black"},
    )
    ax.set_ylabel(r"$R_{\mathrm{Rob}}$", fontsize=14, fontweight="bold")
    ax.set_title("Overall Robustness", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, reliability_robustness, yerr_agg)

    # 2. robustness_fault_injection (Fault Robustness)
    ax = axes[1]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["robustness_fault_injection"].fillna(0),
        bar_colors,
        df_sorted,
        "robustness_fault_injection",
    )
    ax.set_ylabel(r"$R_{\mathrm{fault}}$", fontsize=14, fontweight="bold")
    ax.set_title("Fault Robustness", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["robustness_fault_injection"], yerr)

    # 3. robustness_structural (Structural Robustness)
    ax = axes[2]
    bars, yerr = _bar_with_ci(
        ax,
        x_pos,
        df_sorted["robustness_structural"].fillna(0),
        bar_colors,
        df_sorted,
        "robustness_structural",
    )
    ax.set_ylabel(r"$R_{\mathrm{env}}$", fontsize=14, fontweight="bold")
    ax.set_title("Environment Robustness", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, df_sorted["robustness_structural"], yerr)

    # 4. robustness_prompt_variation (Prompt Robustness) - at the bottom with x labels
    ax = axes[3]
    r_prompt_vals = (
        df_sorted["robustness_prompt_variation"].fillna(0)
        if "robustness_prompt_variation" in df_sorted.columns
        else pd.Series([0] * len(agents))
    )
    bars, yerr = _bar_with_ci(
        ax, x_pos, r_prompt_vals, bar_colors, df_sorted, "robustness_prompt_variation"
    )
    ax.set_ylabel(r"$R_{\mathrm{prompt}}$", fontsize=14, fontweight="bold")
    ax.set_title("Prompt Robustness", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    _add_bar_labels_ci(ax, bars, r_prompt_vals, yerr)

    plt.tight_layout()
    output_path = output_dir / "robustness_detailed.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_safety_detailed(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create detailed safety plots - vertical layout with bar plots.
    Shows: reliability_safety (overall weighted score), Severity Distribution (stacked bars),
           Violation Rate, and per-constraint violation rates (grouped bars).
    """
    df_sorted = sort_agents_by_provider_and_date(df)
    bar_colors = generate_shaded_colors(df_sorted)

    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    agent_names_full = df_sorted["agent"].tolist()
    x_pos = np.arange(len(agents))
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    # Collect per-constraint data
    all_constraints = set()
    per_constraint_by_agent = {}
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if m:
            pc = m.extra.get("safety_per_constraint", {})
            per_constraint_by_agent[agent_name] = pc
            all_constraints.update(pc.keys())

    has_constraints = len(all_constraints) > 0
    n_rows = 4 if has_constraints else 3
    fig = plt.figure(figsize=(10, 3 * n_rows))

    # First row: two side-by-side subplots for safety_compliance and safety_harm_severity
    ax_comp = fig.add_subplot(n_rows, 2, 1)
    ax_harm = fig.add_subplot(n_rows, 2, 2)
    # Remaining rows span full width
    axes_rest = []
    for i in range(1, n_rows):
        axes_rest.append(fig.add_subplot(n_rows, 1, i + 1))

    def add_bar_labels(ax, bars, values):
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # 1a. safety_compliance (Compliance: fraction of tasks with no violations)
    s_comp = df_sorted["safety_compliance"].fillna(0)
    bars = ax_comp.bar(
        x_pos, s_comp, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax_comp.set_ylabel(r"$S_{\mathrm{comp}}$", fontsize=14, fontweight="bold")
    ax_comp.set_title(
        "Compliance (Violation-Free Rate)", fontsize=14, fontweight="bold"
    )
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax_comp.set_ylim(0, 1.15)
    ax_comp.grid(True, alpha=0.3, axis="y")
    ax_comp.tick_params(axis="y", labelsize=11)
    add_bar_labels(ax_comp, bars, s_comp)

    # 1b. safety_harm_severity (Conditional severity: 1 - E[severity | violation])
    s_harm = df_sorted["safety_harm_severity"].fillna(1.0)
    bars = ax_harm.bar(
        x_pos, s_harm, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax_harm.set_ylabel(r"$S_{\mathrm{harm}}$", fontsize=14, fontweight="bold")
    ax_harm.set_title("Harm (Conditional Severity)", fontsize=14, fontweight="bold")
    ax_harm.set_xticks(x_pos)
    ax_harm.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax_harm.set_ylim(0, 1.15)
    ax_harm.grid(True, alpha=0.3, axis="y")
    ax_harm.tick_params(axis="y", labelsize=11)
    add_bar_labels(ax_harm, bars, s_harm)

    # 2. Severity Distribution (stacked bars: low/medium/high)
    ax = axes_rest[0]
    severity_data = {
        agent: {"low": 0.0, "medium": 0.0, "high": 0.0} for agent in agent_names_full
    }
    for agent_name in agent_names_full:
        m = agent_to_metrics.get(agent_name)
        if not m:
            continue
        violations = m.extra.get("safety_violations", [])
        num_runs = max(m.num_runs, 1)
        for v in violations:
            sev = v.get("severity", "medium")
            if sev in severity_data[agent_name]:
                severity_data[agent_name][sev] += 1
        for sev in severity_data[agent_name]:
            severity_data[agent_name][sev] /= num_runs

    severity_levels = ["low", "medium", "high"]
    severity_colors = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
    bottom = np.zeros(len(agents))
    for sev in severity_levels:
        counts = [severity_data[a][sev] for a in agent_names_full]
        display_label = "Med" if sev == "medium" else sev.capitalize()
        ax.bar(
            x_pos,
            counts,
            bottom=bottom,
            label=display_label,
            color=severity_colors[sev],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += np.array(counts)
    ax.set_ylabel("Violations", fontsize=14, fontweight="bold")
    ax.set_title("Severity Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax.legend(title="Severity", fontsize=9, title_fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)

    # 3. Violation Rate (fraction of tasks with any violation = 1 - safety_compliance)
    ax = axes_rest[1]
    viol_rate = (1 - df_sorted["safety_compliance"].fillna(1)).clip(lower=0)
    bars = ax.bar(
        x_pos, viol_rate, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax.set_ylabel("Rate", fontsize=14, fontweight="bold")
    ax.set_title(
        "Violation Rate (Fraction of Tasks with Violations)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
    ax.set_ylim(0, max(viol_rate.max() * 1.3, 0.1) if viol_rate.max() > 0 else 0.1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=11)
    add_bar_labels(ax, bars, viol_rate)

    # 4. Per-constraint violation rates (grouped bars)
    if has_constraints:
        ax = axes_rest[2]
        constraint_list = sorted(all_constraints)
        n_c = len(constraint_list)
        bar_width = 0.8 / n_c
        constraint_colors = plt.cm.Set2(np.linspace(0, 1, n_c))

        def shorten_constraint(name):
            name = name.replace("_customer_service", "").replace("_gaia", "")
            name = name.replace("_", " ").title()
            if len(name) > 22:
                name = name[:20] + ".."
            return name

        for i, constraint in enumerate(constraint_list):
            # Violation rate = 1 - pass rate
            vals = []
            for agent_name in agent_names_full:
                pc = per_constraint_by_agent.get(agent_name, {})
                pass_rate = pc.get(constraint, 1.0)
                vals.append(1.0 - pass_rate)
            offset = (i - n_c / 2 + 0.5) * bar_width
            ax.bar(
                x_pos + offset,
                vals,
                bar_width,
                label=shorten_constraint(constraint),
                color=constraint_colors[i],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_ylabel("Violation Rate", fontsize=14, fontweight="bold")
        ax.set_title("Per-Constraint Violation Rates", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=11)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(fontsize=8, loc="upper center", ncol=2)

    plt.tight_layout()
    output_path = output_dir / "safety_detailed.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_safety_severity_violations(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create a side-by-side plot showing severity distribution and violation types.

    Left subplot: Severity distribution by model (stacked bars for low/medium/high)
    Right subplot: Violation types by model (grouped bars for each constraint type)

    Each subplot is roughly the size of combined_overall_reliability.pdf panels.
    """
    df_sorted = sort_agents_by_provider_and_date(df)
    # Only include oldest and newest model per provider
    df_sorted = filter_oldest_and_newest_per_provider(df_sorted)

    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    agent_names_full = df_sorted["agent"].tolist()
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    # Collect severity and violation data per agent (averaged across runs)
    severity_data = {
        agent: {"low": 0.0, "medium": 0.0, "high": 0.0} for agent in agent_names_full
    }
    violation_data = {}  # constraint_type -> {agent: avg_count}
    all_constraint_types = set()

    for agent_name in agent_names_full:
        if agent_name not in agent_to_metrics:
            continue
        m = agent_to_metrics[agent_name]
        violations = m.extra.get("safety_violations", [])
        num_runs = max(m.num_runs, 1)  # Avoid division by zero

        for v in violations:
            # Count severity levels (will normalize after)
            sev = v.get("severity", "medium")
            if sev in severity_data[agent_name]:
                severity_data[agent_name][sev] += 1

            # Count constraint types (will normalize after)
            constraint = v.get("constraint", "unknown")
            all_constraint_types.add(constraint)
            if constraint not in violation_data:
                violation_data[constraint] = {a: 0.0 for a in agent_names_full}
            violation_data[constraint][agent_name] += 1

        # Normalize by number of runs to get average per run
        for sev in severity_data[agent_name]:
            severity_data[agent_name][sev] /= num_runs
        for constraint in violation_data:
            if agent_name in violation_data[constraint]:
                violation_data[constraint][agent_name] /= num_runs

    # Check if we have any data to plot
    total_violations = sum(sum(severity_data[a].values()) for a in agent_names_full)
    if total_violations == 0:
        print("📊 Skipping safety_severity_violations.pdf (no violation data)")
        return

    # Shorten constraint names for display
    _constraint_abbrevs = {
        "authentication_bypass": "Auth Byp.",
        "financial_accuracy": "Fin. Acc.",
        "destructive_ops": "Destr. Ops",
        "policy_circumvention": "Pol. Circ.",
        "no_unauthorized_access": "Unauth. Acc.",
        "no_destructive_ops": "Destr. Ops",
        "data_minimization": "Data Min.",
        "commitment_overreach": "Commit. Over.",
    }

    def shorten_constraint(name):
        name = name.replace("_customer_service", "").replace("_gaia", "")
        if name in _constraint_abbrevs:
            return _constraint_abbrevs[name]
        name = name.replace("_", " ").title()
        if len(name) > 22:
            name = name[:20] + ".."
        return name

    # Constraints to exclude from the violation types plot
    excluded_constraints = {
        "data_minimization_customer_service",
        "commitment_overreach_customer_service",
    }

    # Create figure with 2 stacked subplots
    fig, axes = plt.subplots(
        2, 1, figsize=(4.25, 4.8), gridspec_kw={"height_ratios": [0.7, 1]}
    )

    x_pos = np.arange(len(agents))

    # === Top subplot: Severity Distribution ===
    ax = axes[0]
    severity_levels = ["low", "medium", "high"]
    severity_colors = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}

    bottom = np.zeros(len(agents))
    for sev in severity_levels:
        counts = [severity_data[a][sev] for a in agent_names_full]
        display_label = "Med" if sev == "medium" else sev.capitalize()
        ax.bar(
            x_pos,
            counts,
            bottom=bottom,
            label=display_label,
            color=severity_colors[sev],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += np.array(counts)

    ax.set_ylabel("Violations", fontsize=11, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])  # No x-axis labels on top plot
    ax.set_xlim(-0.6, len(agents) - 0.4)
    ax.set_title("Harm Severity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=10)

    # === Bottom subplot: Violation Types ===
    ax = axes[1]

    # Sort constraint types by total count, excluding specified constraints
    constraint_types = sorted(
        [c for c in all_constraint_types if c not in excluded_constraints],
        key=lambda c: sum(violation_data.get(c, {}).values()),
        reverse=True,
    )

    if len(constraint_types) > 0:
        n_constraints = min(len(constraint_types), 6)
        constraint_types = constraint_types[:n_constraints]

        bar_width = 0.8 / n_constraints
        constraint_colors = plt.cm.Set2(np.linspace(0, 1, n_constraints))

        for i, constraint in enumerate(constraint_types):
            counts = [
                violation_data.get(constraint, {}).get(a, 0) for a in agent_names_full
            ]
            offset = (i - n_constraints / 2 + 0.5) * bar_width
            ax.bar(
                x_pos + offset,
                counts,
                bar_width,
                label=shorten_constraint(constraint),
                color=constraint_colors[i],
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_ylabel("Violations", fontsize=11, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.6, len(agents) - 0.4)
        ax.set_title("Compliance Violations", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, bbox_to_anchor=(0.38, 1.0), loc="upper center", ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=10)
    else:
        ax.text(
            0.5,
            0.5,
            "No constraint data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_path = output_dir / "safety_severity_violations.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_safety_deep_analysis(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create comprehensive safety analysis with deeper insights.

    2x2 figure with:
    - Top-left: Constraint violation heatmap (model x constraint type)
    - Top-right: Severity distribution by constraint type (stacked bars)
    - Bottom-left: Per-constraint pass rates by model (grouped bars)
    - Bottom-right: Violation count vs accuracy scatter (with provider colors)
    """
    df_sorted = sort_agents_by_provider_and_date(df)

    agents = [strip_agent_prefix(a) for a in df_sorted["agent"].tolist()]
    agent_names_full = df_sorted["agent"].tolist()
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    # Collect all data (averaged across runs)
    violation_matrix = {}  # constraint -> {agent: avg_count}
    severity_by_constraint = {}  # constraint -> {severity: avg_count}
    per_constraint_rates = {}  # constraint -> {agent: pass_rate}
    violations_per_agent = {}  # agent -> avg total violations
    accuracy_per_agent = {}  # agent -> accuracy
    all_constraints = set()
    # Track total severity counts per constraint before averaging
    _severity_raw = {}

    for agent_name in agent_names_full:
        if agent_name not in agent_to_metrics:
            continue
        m = agent_to_metrics[agent_name]
        violations = m.extra.get("safety_violations", [])
        per_constraint = m.extra.get("safety_per_constraint", {})
        num_runs = max(m.num_runs, 1)  # Avoid division by zero

        violations_per_agent[agent_name] = len(violations) / num_runs
        accuracy_per_agent[agent_name] = m.accuracy if not np.isnan(m.accuracy) else 0

        # Per-constraint pass rates (already averaged in compute_safety_metrics)
        for constraint, rate in per_constraint.items():
            all_constraints.add(constraint)
            if constraint not in per_constraint_rates:
                per_constraint_rates[constraint] = {}
            per_constraint_rates[constraint][agent_name] = rate

        # Violation matrix and severity breakdown
        for v in violations:
            constraint = v.get("constraint", "unknown")
            severity = v.get("severity", "medium")
            all_constraints.add(constraint)

            if constraint not in violation_matrix:
                violation_matrix[constraint] = {a: 0.0 for a in agent_names_full}
            violation_matrix[constraint][agent_name] += 1

            if constraint not in _severity_raw:
                _severity_raw[constraint] = {"low": 0.0, "medium": 0.0, "high": 0.0}
            if severity in _severity_raw[constraint]:
                _severity_raw[constraint][severity] += 1

        # Normalize violation_matrix counts by num_runs for this agent
        for constraint in violation_matrix:
            if (
                agent_name in violation_matrix[constraint]
                and violation_matrix[constraint][agent_name] > 0
            ):
                violation_matrix[constraint][agent_name] /= num_runs

    # Normalize severity_by_constraint by total number of runs across all agents
    total_runs = sum(
        max(agent_to_metrics[a].num_runs, 1)
        for a in agent_names_full
        if a in agent_to_metrics
    )
    num_agents = sum(1 for a in agent_names_full if a in agent_to_metrics)
    avg_runs = total_runs / max(num_agents, 1)
    severity_by_constraint = {}
    for constraint, sevs in _severity_raw.items():
        severity_by_constraint[constraint] = {k: v / avg_runs for k, v in sevs.items()}

    # Check if we have data
    total_violations = sum(violations_per_agent.get(a, 0) for a in agent_names_full)
    if total_violations == 0:
        print("📊 Skipping safety_deep_analysis.pdf (no violation data)")
        return

    # Sort constraints by total violations
    constraint_list = sorted(
        all_constraints,
        key=lambda c: sum(violation_matrix.get(c, {}).values()),
        reverse=True,
    )
    # Limit to top 7 for readability
    constraint_list = constraint_list[:7]

    # Shorten constraint names for display
    def shorten_constraint(name):
        # Remove common suffixes and shorten
        name = name.replace("_customer_service", "")
        name = name.replace("_", " ").title()
        if len(name) > 18:
            name = name[:16] + ".."
        return name

    short_constraints = [shorten_constraint(c) for c in constraint_list]

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # === Top-left: Constraint violation heatmap ===
    ax = axes[0, 0]
    heatmap_data = np.zeros((len(constraint_list), len(agents)))
    for i, constraint in enumerate(constraint_list):
        for j, agent in enumerate(agent_names_full):
            heatmap_data[i, j] = violation_matrix.get(constraint, {}).get(agent, 0)

    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(agents)))
    ax.set_yticks(np.arange(len(constraint_list)))
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_constraints, fontsize=9)
    ax.set_title(
        "Avg Violation Count by Model & Constraint", fontsize=11, fontweight="bold"
    )

    # Add text annotations
    for i in range(len(constraint_list)):
        for j in range(len(agents)):
            val = heatmap_data[i, j]
            if val > 0:
                text_color = "white" if val > heatmap_data.max() * 0.5 else "black"
                label = f"{val:.1f}" if val != int(val) else str(int(val))
                ax.text(
                    j, i, label, ha="center", va="center", fontsize=7, color=text_color
                )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Avg Violations", fontsize=9)

    # === Top-right: Severity by constraint type ===
    ax = axes[0, 1]
    x_pos = np.arange(len(constraint_list))
    bar_width = 0.6
    severity_colors = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}

    bottom = np.zeros(len(constraint_list))
    for sev in ["low", "medium", "high"]:
        counts = [
            severity_by_constraint.get(c, {}).get(sev, 0) for c in constraint_list
        ]
        ax.bar(
            x_pos,
            counts,
            bar_width,
            bottom=bottom,
            label=sev.capitalize(),
            color=severity_colors[sev],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += np.array(counts)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_constraints, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Avg Violations per Run", fontsize=10, fontweight="bold")
    ax.set_title(
        "Violation Severity by Constraint Type", fontsize=11, fontweight="bold"
    )
    ax.legend(title="Severity", fontsize=8, title_fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # === Bottom-left: Per-constraint pass rates ===
    ax = axes[1, 0]

    if per_constraint_rates:
        n_constraints = min(len(constraint_list), 5)
        top_constraints = constraint_list[:n_constraints]
        bar_width = 0.8 / n_constraints
        constraint_colors = plt.cm.Set2(np.linspace(0, 1, n_constraints))

        x_pos = np.arange(len(agents))
        for i, constraint in enumerate(top_constraints):
            rates = [
                per_constraint_rates.get(constraint, {}).get(a, 1.0)
                for a in agent_names_full
            ]
            offset = (i - n_constraints / 2 + 0.5) * bar_width
            ax.bar(
                x_pos + offset,
                rates,
                bar_width,
                label=shorten_constraint(constraint),
                color=constraint_colors[i],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pass Rate (1 = no violations)", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Per-Constraint Compliance Rate by Model", fontsize=11, fontweight="bold"
        )
        ax.legend(
            title="Constraint", fontsize=7, title_fontsize=8, loc="lower right", ncol=2
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, linewidth=1)
    else:
        ax.text(
            0.5,
            0.5,
            "No per-constraint data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    # === Bottom-right: Violations vs Accuracy scatter ===
    ax = axes[1, 1]

    # Get provider colors
    provider_colors = {"openai": "#10A37F", "anthropic": "#D97706", "google": "#4285F4"}
    scatter_colors = []
    for agent in agent_names_full:
        provider = get_provider(agent)
        scatter_colors.append(provider_colors.get(provider, "#888888"))

    x_vals = [violations_per_agent.get(a, 0) for a in agent_names_full]
    y_vals = [accuracy_per_agent.get(a, 0) for a in agent_names_full]

    for i, agent in enumerate(agent_names_full):
        ax.scatter(
            x_vals[i],
            y_vals[i],
            c=scatter_colors[i],
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.annotate(
            agents[i],
            (x_vals[i], y_vals[i]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
            alpha=0.8,
        )

    # Add legend for providers
    for provider, color in provider_colors.items():
        ax.scatter(
            [],
            [],
            c=color,
            s=80,
            label=provider.capitalize(),
            edgecolors="black",
            linewidth=0.5,
        )
    ax.legend(title="Provider", fontsize=8, title_fontsize=9, loc="upper right")

    ax.set_xlabel("Avg Violations per Run", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
    ax.set_title("Avg Violations vs Task Accuracy", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add trend line if enough data
    if len(x_vals) >= 3 and max(x_vals) > 0:
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, max(x_vals), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=1.5, label="Trend")

    plt.tight_layout()
    output_path = output_dir / "safety_deep_analysis.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_safety_lambda_sensitivity(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Plot safety_score vs release date for several choices of lambda.

    safety_score = max(1 - lambda * P(violation), 0) * safety_harm_severity

    Creates a 1-row x N-col grid where each panel shows one value of lambda.
    Agents are scatter-plotted by provider with a linear trend line.
    """
    from scipy import stats
    import matplotlib.dates as mdates

    lambda_values = [1.0, 5.0, 10.0, 20.0]

    df_sorted = sort_agents_by_provider_and_date(df)

    if "release_timestamp" not in df_sorted.columns:
        df_sorted["release_timestamp"] = pd.to_datetime(
            df_sorted["agent"].map(
                lambda x: get_model_metadata(x).get("date", "2024-01-01")
            )
        )
    if "provider" not in df_sorted.columns:
        df_sorted["provider"] = df_sorted["agent"].map(
            lambda x: get_model_metadata(x).get("provider", "Unknown")
        )

    # Check that safety_harm_severity and safety_compliance are available
    if (
        "safety_harm_severity" not in df_sorted.columns
        or "safety_compliance" not in df_sorted.columns
    ):
        print(
            "📊 Skipping safety_lambda_sensitivity.pdf (no safety_harm_severity/safety_compliance data)"
        )
        return

    has_data = (
        df_sorted["safety_harm_severity"].notna().any()
        and df_sorted["safety_compliance"].notna().any()
    )
    if not has_data:
        print(
            "📊 Skipping safety_lambda_sensitivity.pdf (safety_harm_severity/safety_compliance all NaN)"
        )
        return

    # Compute safety_score = max(1 - lambda * P(violation), 0) * safety_harm_severity for each lambda
    P_violation = 1.0 - df_sorted["safety_compliance"]
    for lam in lambda_values:
        col = f"safety_score_lam{lam}"
        df_sorted[col] = (1.0 - lam * P_violation).clip(lower=0.0) * df_sorted[
            "safety_harm_severity"
        ]

    fig, axes = plt.subplots(1, len(lambda_values), figsize=(4 * len(lambda_values), 4))
    if len(lambda_values) == 1:
        axes = [axes]

    for idx, lam in enumerate(lambda_values):
        ax = axes[idx]
        col = f"safety_score_lam{lam}"
        y = df_sorted[col]
        x = df_sorted["release_timestamp"]
        providers = df_sorted["provider"]

        valid = x.notna() & y.notna()
        if valid.sum() < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"$\\lambda$ = {lam}")
            continue

        x_v, y_v, prov_v = x[valid], y[valid], providers[valid]

        for provider in ["OpenAI", "Google", "Anthropic"]:
            mask = prov_v == provider
            if mask.sum() == 0:
                continue
            ax.scatter(
                x_v[mask],
                y_v[mask],
                c=PROVIDER_COLORS.get(provider, "#999999"),
                marker=PROVIDER_MARKERS.get(provider, "o"),
                s=50,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.6,
                label=provider,
            )

        # Trend line
        x_num = (x_v - x_v.min()).dt.days.values
        slope, intercept, r_value, p_value, _ = stats.linregress(x_num, y_v.values)
        x_range = np.array([x_num.min(), x_num.max()])
        x_dates = [x_v.min() + pd.Timedelta(days=d) for d in x_range]
        y_trend = slope * x_range + intercept
        ax.plot(x_dates, y_trend, "k--", linewidth=1.5, alpha=0.7)

        slope_yr = slope * 365
        ax.annotate(
            f"r={r_value:+.2f}\nslope={slope_yr:+.2f}/yr\np={p_value:.2f}",
            xy=(0.975, 0.04),
            xycoords="axes fraction",
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
                linewidth=0.5,
            ),
        )

        lam_label = {1.0: "baseline"}.get(lam, f"{lam:.0f}x penalty")
        ax.set_title(
            f"$S_{{\\mathrm{{safety}}}}$  ($\\lambda$={lam:.0f}, {lam_label})",
            fontsize=11,
        )
        ax.set_ylabel(r"$S_{\mathrm{safety}}$" if idx == 0 else "")
        ax.set_xlabel("Release Date")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(by_label),
        framealpha=0.95,
        edgecolor="gray",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / "safety_lambda_sensitivity.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_abstention_detailed(
    df: pd.DataFrame, all_metrics: List[ReliabilityMetrics], output_dir: Path
):
    """
    Create detailed abstention plots showing abstention rate and calibration metrics.

    Plots include:
    1. Abstention Rate (abstention_rate) - fraction of tasks where model abstained
    2. Abstention Precision (abstention_precision) - P(fail | abstain)
    3. Abstention Recall (abstention_recall) - P(abstain | fail)
    4. Selective Accuracy (abstention_selective_accuracy) - accuracy when NOT abstaining
    5. Confusion Matrix - abstained vs succeeded/failed
    6. Abstention Type Breakdown - by type (inability, uncertainty, etc.)
    """
    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)
    bar_colors = generate_shaded_colors(df_sorted)
    agent_to_metrics = {m.agent_name: m for m in all_metrics}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    agent_names_full = df_sorted["agent"].tolist()
    agents = [strip_agent_prefix(a) for a in agent_names_full]
    x_pos = np.arange(len(agents))
    sorted_metrics = [agent_to_metrics.get(a) for a in agent_names_full]

    # 1. Abstention Rate
    ax = axes[0, 0]
    a_rate_vals = df_sorted["abstention_rate"].fillna(0)
    bars = ax.bar(
        x_pos,
        a_rate_vals,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Abstention Rate\nP(abstain)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, df_sorted["abstention_rate"]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 2. Abstention Precision - P(fail | abstain)
    ax = axes[0, 1]
    a_prec_vals = df_sorted["abstention_precision"].fillna(0)
    bars = ax.bar(
        x_pos,
        a_prec_vals,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel(
        "Abstention Precision\nP(would fail | abstained)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, df_sorted["abstention_precision"]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 3. Abstention Recall - P(abstain | fail)
    ax = axes[0, 2]
    a_rec_vals = df_sorted["abstention_recall"].fillna(0)
    bars = ax.bar(
        x_pos, a_rec_vals, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )
    ax.set_ylabel(
        "Abstention Recall\nP(abstained | failed)", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, df_sorted["abstention_recall"]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 4. Selective Accuracy - accuracy when NOT abstaining
    ax = axes[1, 0]
    a_sel_vals = df_sorted["abstention_selective_accuracy"].fillna(0)
    accuracy_vals = df_sorted["accuracy"].fillna(0)
    width = 0.35
    ax.bar(
        x_pos - width / 2,
        accuracy_vals,
        width,
        label="Overall Accuracy",
        alpha=0.8,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x_pos + width / 2,
        a_sel_vals,
        width,
        label="Selective Accuracy",
        alpha=0.8,
        color="tab:green",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Stacked bar: Confusion matrix breakdown
    ax = axes[1, 1]
    confusion_data = {
        "abstained_failed": [],
        "abstained_succeeded": [],
        "proceeded_failed": [],
        "proceeded_succeeded": [],
    }

    for m in sorted_metrics:
        if not m:
            confusion_data["abstained_failed"].append(0)
            confusion_data["abstained_succeeded"].append(0)
            confusion_data["proceeded_failed"].append(0)
            confusion_data["proceeded_succeeded"].append(0)
            continue

        abstention_data = m.extra.get("abstention_data", {})
        cm = abstention_data.get("confusion_matrix", {})
        n_tasks = abstention_data.get("n_tasks", 0)

        if n_tasks > 0:
            confusion_data["abstained_failed"].append(
                cm.get("abstained_and_failed", 0) / n_tasks
            )
            confusion_data["abstained_succeeded"].append(
                cm.get("abstained_and_succeeded", 0) / n_tasks
            )
            confusion_data["proceeded_failed"].append(
                cm.get("proceeded_and_failed", 0) / n_tasks
            )
            confusion_data["proceeded_succeeded"].append(
                cm.get("proceeded_and_succeeded", 0) / n_tasks
            )
        else:
            confusion_data["abstained_failed"].append(0)
            confusion_data["abstained_succeeded"].append(0)
            confusion_data["proceeded_failed"].append(0)
            confusion_data["proceeded_succeeded"].append(0)

    # Stacked bar chart
    bottom = np.zeros(len(agents))
    ax.bar(
        x_pos,
        confusion_data["proceeded_succeeded"],
        label="Proceeded + Succeeded",
        color="tab:green",
        alpha=0.8,
        bottom=bottom,
    )
    bottom += confusion_data["proceeded_succeeded"]
    ax.bar(
        x_pos,
        confusion_data["proceeded_failed"],
        label="Proceeded + Failed",
        color="tab:red",
        alpha=0.8,
        bottom=bottom,
    )
    bottom += confusion_data["proceeded_failed"]
    ax.bar(
        x_pos,
        confusion_data["abstained_succeeded"],
        label="Abstained + Succeeded",
        color="tab:orange",
        alpha=0.8,
        bottom=bottom,
    )
    bottom += confusion_data["abstained_succeeded"]
    ax.bar(
        x_pos,
        confusion_data["abstained_failed"],
        label="Abstained + Failed",
        color="tab:purple",
        alpha=0.8,
        bottom=bottom,
    )

    ax.set_ylabel("Fraction of Tasks", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Abstention type breakdown (grouped bar)
    ax = axes[1, 2]
    all_types = set()
    for m in sorted_metrics:
        if not m:
            continue
        abstention_data = m.extra.get("abstention_data", {})
        type_breakdown = abstention_data.get("type_breakdown", {})
        all_types.update(type_breakdown.keys())

    # Remove 'none' from types and sort with 'inability' first
    other_types = sorted([t for t in all_types if t != "none" and t != "inability"])
    abstention_types = (["inability"] if "inability" in all_types else []) + other_types

    if abstention_types:
        n_types = len(abstention_types)
        width = 0.8 / len(agents) if len(agents) > 1 else 0.4

        for i, m in enumerate(sorted_metrics):
            if not m:
                continue
            abstention_data = m.extra.get("abstention_data", {})
            type_breakdown = abstention_data.get("type_breakdown", {})
            n_tasks = abstention_data.get("n_tasks", 1) or 1

            # Get counts for each type (as fraction of total)
            type_fractions = [
                type_breakdown.get(t, {}).get("count", 0) / n_tasks
                for t in abstention_types
            ]

            x_type = np.arange(n_types)
            ax.bar(
                x_type + i * width,
                type_fractions,
                width,
                label=strip_agent_prefix(m.agent_name),
                alpha=0.8,
                color=bar_colors[i],
            )

        ax.set_ylabel("Fraction of Tasks", fontsize=12, fontweight="bold")
        ax.set_xticks(x_type + width * len(sorted_metrics) / 2)
        ax.set_xticklabels(abstention_types, rotation=45, ha="right", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No abstention type data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
    plt.tight_layout()
    output_path = output_dir / "abstention_detailed.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

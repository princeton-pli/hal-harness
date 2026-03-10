"""Epoch AI-style charts for social media sharing."""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from reliability_eval.loaders.agent_names import (
    get_model_metadata,
    sort_agents_by_provider_and_date,
    strip_agent_prefix,
)
from reliability_eval.metrics.consistency import compute_weighted_r_con
from reliability_eval.plots.helpers import (
    _CI_Z,
    _clip_yerr,
    _get_aggregate_yerr,
    _get_weighted_r_con_yerr,
    generate_shaded_colors,
)

# ── Static assets ─────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent.parent / "website" / "static"
_HAL_LOGO_PNG = _STATIC_DIR / "logo.png"
_PRINCETON_LOGO_PNG = _STATIC_DIR / "princeton-light.png"

# ── Style constants ───────────────────────────────────────────────────
_FONT_FAMILY = ["DM Sans", "Helvetica", "DejaVu Sans"]
_COLOR_TEXT = "#1a1a1a"
_COLOR_SUBTLE = "#6b7280"
_BG_COLOR = "#fafaf8"
_BAR_HEIGHT = 0.6


def _place_logo(fig, png_path: Path, rect: list, anchor: str = "W"):
    """Place a pre-rasterised PNG logo on *fig* at *rect* [l, b, w, h].

    *anchor* controls alignment inside the rect when aspect-ratio
    doesn't match: 'W' = left-aligned, 'E' = right-aligned, 'C' = centred.
    """
    if not png_path.exists():
        return None
    img = mpimg.imread(str(png_path))
    logo_ax = fig.add_axes(rect)
    logo_ax.imshow(img)
    logo_ax.set_anchor(anchor)
    logo_ax.axis("off")
    return logo_ax


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived reliability columns if missing."""
    df_sorted = sort_agents_by_provider_and_date(df)

    if "reliability_consistency" not in df_sorted.columns:
        df_sorted["reliability_consistency"] = compute_weighted_r_con(
            df_sorted["consistency_outcome"],
            df_sorted["consistency_trajectory_distribution"],
            df_sorted["consistency_trajectory_sequence"],
            df_sorted["consistency_resource"],
        )
    if "reliability_predictability" not in df_sorted.columns:
        df_sorted["reliability_predictability"] = df_sorted["predictability_brier_score"]
    if "reliability_robustness" not in df_sorted.columns:
        df_sorted["reliability_robustness"] = df_sorted[
            ["robustness_fault_injection", "robustness_structural", "robustness_prompt_variation"]
        ].mean(axis=1, skipna=True)
    if "reliability_overall" not in df_sorted.columns:
        df_sorted["reliability_overall"] = df_sorted[
            ["reliability_consistency", "reliability_predictability", "reliability_robustness"]
        ].mean(axis=1, skipna=True)
    if "provider" not in df_sorted.columns:
        df_sorted["provider"] = df_sorted["agent"].map(
            lambda x: get_model_metadata(x).get("provider", "Unknown")
        )
    return df_sorted


def _draw_panel(ax, labels, values, colors, title, show_xticks=False, xerr=None):
    """Draw one horizontal-bar panel on *ax* in Epoch AI style."""
    n = len(labels)
    y_pos = np.arange(n)

    for j in range(n):
        ax.barh(
            y_pos[j], values[j], height=_BAR_HEIGHT,
            color=colors[j], edgecolor="black", linewidth=1, zorder=2,
        )

    # Draw error bars on top
    if xerr is not None:
        ax.errorbar(
            values, y_pos, xerr=xerr, fmt="none",
            ecolor="#333333", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, color=_COLOR_TEXT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=4)

    if show_xticks:
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=_COLOR_SUBTLE)
    else:
        ax.set_xticklabels([])

    # Light vertical grid only
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#a0a0a0", linewidth=1.0)
    ax.yaxis.grid(False)
    # Prominent zero line
    ax.axvline(0, color="black", linewidth=1.5, zorder=1.5)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor(_BG_COLOR)


def plot_social_overall_reliability(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Generate an Epoch AI-style horizontal bar chart for social media.

    One panel per benchmark, showing Overall Reliability per model,
    with provider-shaded colors and logos.

    Output: social_overall_reliability.pdf

    Args:
        benchmark_data: List of (benchmark_name, dataframe) tuples.
        output_dir: Directory to save the output image.
        padding: Uniform padding around the plot as a figure fraction
            (0.0 = no padding, 0.06 = 6% on each side).
    """
    if not benchmark_data:
        print("  No benchmark data for social media plot")
        return

    # Temporarily override font for this plot only
    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    # Prepare data for each benchmark panel
    panels = []
    for bm_name, bm_df in benchmark_data:
        df_sorted = _prepare_dataframe(bm_df)
        # Restrict to OpenAI models only
        df_sorted = df_sorted[df_sorted["provider"] == "OpenAI"].reset_index(drop=True)
        valid = df_sorted["reliability_overall"].notna()
        df_valid = df_sorted[valid].copy()
        if len(df_valid) == 0:
            continue

        # Sort by reliability (highest at top of chart)
        df_valid = df_valid.sort_values("reliability_overall", ascending=True).reset_index(drop=True)

        labels = [strip_agent_prefix(a) for a in df_valid["agent"]]
        values = df_valid["reliability_overall"].values
        presorted = sort_agents_by_provider_and_date(df_valid)
        colors = generate_shaded_colors(presorted)
        color_map = dict(zip(presorted["agent"], colors))
        colors_sorted = [color_map.get(a, "#999999") for a in df_valid["agent"]]

        # Propagate SE for reliability_overall = mean(R_con, R_pred, R_rob)
        # SE(mean) = sqrt(se_con^2 + se_pred^2 + se_rob^2) / 3
        se_con = _get_weighted_r_con_yerr(df_valid) / _CI_Z  # undo CI scaling to get raw SE
        se_pred_col = "predictability_brier_score_se"
        se_pred = df_valid[se_pred_col].values if se_pred_col in df_valid.columns else np.zeros(len(df_valid))
        se_pred = np.where(np.isnan(se_pred), 0, se_pred)
        rob_se_cols = ["robustness_fault_injection_se", "robustness_structural_se", "robustness_prompt_variation_se"]
        rob_existing = [c for c in rob_se_cols if c in df_valid.columns]
        if rob_existing:
            rob_sq = np.zeros(len(df_valid))
            for c in rob_existing:
                se = np.where(np.isnan(df_valid[c].values), 0, df_valid[c].values)
                rob_sq += se**2
            se_rob = np.sqrt(rob_sq) / len(rob_existing)
        else:
            se_rob = np.zeros(len(df_valid))
        xerr = _CI_Z * np.sqrt(se_con**2 + se_pred**2 + se_rob**2) / 3
        xerr = _clip_yerr(xerr, values)

        display_name = _display.get(bm_name, bm_name)
        panels.append((display_name, labels, values, colors_sorted, xerr))

    if not panels:
        print("  No valid panels for social media plot")
        return

    n_panels = len(panels)
    panel_sizes = [len(p[1]) for p in panels]

    # Size: ~0.38in per bar row, plus space for title header and footer logos
    header_height = 1.1   # title + subtitle
    footer_height = 0.6   # logos
    panel_gap = 0.6       # gap between panels
    bar_row_height = 0.38
    body_height = sum(s * bar_row_height for s in panel_sizes) + panel_gap * (n_panels - 1)
    fig_height = header_height + body_height + footer_height + 0.3
    fig_width = 7

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)

    # Use gridspec: header row (fixed), one row per panel, footer row (fixed)
    height_ratios = [header_height] + [s * bar_row_height for s in panel_sizes] + [footer_height]
    gs = gridspec.GridSpec(
        n_panels + 2, 1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.28, right=0.97, top=0.97, bottom=0.02,
    )

    # Header: title + subtitle. We render a dummy panel first to find where
    # the y-tick labels start, then align header/footer to that x position.
    # For now, place header text using axes transAxes of the first bar panel
    # (drawn later), so fall back to fig.text at a matching left margin.
    _LEFT_MARGIN = 0.05  # figure-fraction; tuned to align with y-tick labels
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_bbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, header_bbox.y1 - 0.01,
        "More capable models are not more reliable",
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, header_bbox.y0 + header_bbox.height * 0.35,
        "Overall Reliability by model, sorted by score.",
        fontsize=12, color=_COLOR_SUBTLE,
        va="top", ha="left", wrap=True,
    )

    # Bar panels
    for i, (title, labels, values, colors, xerr) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i])
        is_last = i == n_panels - 1
        _draw_panel(ax, labels, values, colors, title, show_xticks=is_last, xerr=xerr)
        # Rotated benchmark label on the far left
        ax_bbox = ax.get_position()
        fig.text(
            0.01, ax_bbox.y0 + ax_bbox.height / 2,
            title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # Footer: logos
    footer_ax = fig.add_subplot(gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)

    # Position logos within the footer row using inset axes in figure coords
    # Get footer bbox in figure coordinates
    fig.canvas.draw()
    footer_bbox = footer_ax.get_position()

    logo_h = footer_bbox.height * 0.8
    logo_y = footer_bbox.y0 + footer_bbox.height * 0.1
    _RIGHT_EDGE = 0.97  # match gridspec right edge
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")

    # Centered URL in figure coords
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, footer_bbox.y0 + footer_bbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "overall_reliability.pdf"
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    # Restore previous font settings
    for k, v in prev_rc.items():
        plt.rcParams[k] = v


def plot_social_openai_overall(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Epoch AI-style horizontal bar chart of overall reliability for all OpenAI models.

    Output: social_openai_overall.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for GPT 5.2 vs 5.4 social plot")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }
    _OPENAI_GREEN = "#10A37F"

    panels = []
    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        df_prep = df_prep[df_prep["provider"] == "OpenAI"].reset_index(drop=True)
        valid = df_prep["reliability_overall"].notna()
        df_valid = df_prep[valid].copy()
        if len(df_valid) == 0:
            continue

        df_valid = df_valid.sort_values("reliability_overall", ascending=True).reset_index(drop=True)

        labels = [strip_agent_prefix(a) for a in df_valid["agent"]]
        values = df_valid["reliability_overall"].values
        colors = [
            _OPENAI_GREEN if "5_4" in a else _BG_COLOR
            for a in df_valid["agent"]
        ]

        xerrs = []
        for _, row in df_valid.iterrows():
            se_con = _se_for_metric(row, df_valid, "reliability_consistency")
            se_pred = _se_for_metric(row, df_valid, "reliability_predictability")
            se_rob = _se_for_metric(row, df_valid, "reliability_robustness")
            xerrs.append(_CI_Z * np.sqrt(se_con**2 + se_pred**2 + se_rob**2) / 3)

        values = np.array(values)
        xerrs = np.array(xerrs)
        xerrs = _clip_yerr(xerrs, values)

        display_name = _display.get(bm_name, bm_name)
        panels.append((display_name, labels, values, colors, xerrs))

    if not panels:
        print("  No valid panels for OpenAI overall reliability social plot")
        for k, v in prev_rc.items():
            plt.rcParams[k] = v
        return

    n_panels = len(panels)
    panel_sizes = [len(p[1]) for p in panels]

    header_height = 1.1
    footer_height = 0.6
    bar_row_height = 0.45
    body_height = sum(s * bar_row_height for s in panel_sizes) + 0.6 * (n_panels - 1)
    fig_height = header_height + body_height + footer_height + 0.3
    fig_width = 7

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)

    height_ratios = (
        [header_height]
        + [s * bar_row_height for s in panel_sizes]
        + [footer_height]
    )
    gs = gridspec.GridSpec(
        n_panels + 2, 1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.28, right=0.97, top=0.97, bottom=0.02,
    )

    _LEFT_MARGIN = 0.05
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_bbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, header_bbox.y1 - 0.01,
        "Overall Reliability across OpenAI Models",
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, header_bbox.y0 + header_bbox.height * 0.35,
        "All OpenAI models compared across benchmarks. GPT 5.4 highlighted.",
        fontsize=12, color=_COLOR_SUBTLE,
        va="top", ha="left", wrap=True,
    )

    for i, (title, labels, values, colors, xerr) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i])
        is_last = i == n_panels - 1
        _draw_panel(ax, labels, values, colors, title, show_xticks=is_last, xerr=xerr)
        # Rotated benchmark label on the far left
        ax_bbox = ax.get_position()
        fig.text(
            0.01, ax_bbox.y0 + ax_bbox.height / 2,
            title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)

    fig.canvas.draw()
    footer_bbox = footer_ax.get_position()

    logo_h = footer_bbox.height * 0.8
    logo_y = footer_bbox.y0 + footer_bbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")

    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, footer_bbox.y0 + footer_bbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "openai_overall.pdf"
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── Shared model specs & colors for 5.2-vs-5.4 plots ─────────────────
_BENCHMARK_MODELS_52_54 = {
    "gaia": [
        ("gpt_5_2", "GPT 5.2"),
        ("gpt_5_4", "GPT 5.4"),
        ("gpt_5_2_medium", "GPT 5.2 (medium)"),
        ("gpt_5_4_medium", "GPT 5.4 (medium)"),
    ],
    "taubench_airline": [
        ("gpt_5_2", "GPT 5.2"),
        ("gpt_5_4", "GPT 5.4"),
        ("gpt_5_2_xhigh", "GPT 5.2 (xhigh)"),
        ("gpt_5_4_xhigh", "GPT 5.4 (xhigh)"),
    ],
}


_MODEL_COLORS_52_54 = {
    "gpt_5_2": _BG_COLOR,        # background color (5.2 = "empty" bar)
    "gpt_5_2_medium": _BG_COLOR,
    "gpt_5_2_xhigh": _BG_COLOR,
    "gpt_5_4": "#10A37F",        # solid OpenAI green (5.4)
    "gpt_5_4_medium": "#10A37F",
    "gpt_5_4_xhigh": "#10A37F",
}

_REASONING_SUFFIXES_52_54 = {
    "gpt_5_2_medium", "gpt_5_2_xhigh",
    "gpt_5_4_medium", "gpt_5_4_xhigh",
}

_BM_DISPLAY = {
    "taubench_airline": r"$\tau$-bench",
    "gaia": "GAIA",
}


def _get_model_row(df_prep, suffix):
    """Return the first row whose agent name ends with *suffix*, or None."""
    matches = df_prep[df_prep["agent"].str.endswith(suffix)]
    return matches.iloc[0] if not matches.empty else None


def _se_for_metric(row, df_prep, metric):
    """Return the raw SE (not CI-scaled) for a single model row and metric."""
    if metric == "accuracy":
        se_col = "accuracy_se"
        return 0 if se_col not in df_prep.columns else (
            0 if np.isnan(row.get(se_col, np.nan)) else row[se_col]
        )
    if metric == "reliability_consistency":
        single = df_prep[df_prep["agent"] == row["agent"]]
        return float(_get_weighted_r_con_yerr(single) / _CI_Z)
    if metric == "reliability_predictability":
        se_col = "predictability_brier_score_se"
        return 0 if se_col not in df_prep.columns else (
            0 if np.isnan(row.get(se_col, np.nan)) else row[se_col]
        )
    if metric == "reliability_robustness":
        rob_se_cols = [
            "robustness_fault_injection_se",
            "robustness_structural_se",
            "robustness_prompt_variation_se",
        ]
        existing = [c for c in rob_se_cols if c in df_prep.columns]
        if not existing:
            return 0
        sq = sum(
            (0 if np.isnan(row.get(c, np.nan)) else row[c]) ** 2
            for c in existing
        )
        return np.sqrt(sq) / len(existing)
    return 0


def plot_social_openai_detailed(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Horizontal bar grid showing reliability breakdown for all OpenAI models.

    Layout: rows = benchmarks, columns = metrics.  Each cell shows
    horizontal bars for all OpenAI models with a shared y-axis
    (model labels only on the leftmost column).  Column headers give
    the metric name.

    Output: social_openai_detailed.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for detailed OpenAI social plot")
        return

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }
    _OPENAI_GREEN = "#10A37F"

    metrics = [
        ("accuracy", "Accuracy"),
        ("reliability_consistency", "Consistency"),
        ("reliability_predictability", "Predictability"),
        ("reliability_robustness", "Robustness"),
    ]

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    # ── Collect per-benchmark data ────────────────────────────────────
    # panels[i] = (bm_display, entries)
    # entries[j] = (agent_name, label, [val_per_metric], [yerr_per_metric])
    panels = []
    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        df_prep = df_prep[df_prep["provider"] == "OpenAI"].reset_index(drop=True)
        if len(df_prep) == 0:
            continue

        entries = []
        for _, row in df_prep.iterrows():
            agent = row["agent"]
            display_label = strip_agent_prefix(agent)
            vals, yerrs = [], []
            for col, _ in metrics:
                v = row.get(col, np.nan)
                vals.append(v if not np.isnan(v) else 0)
                yerrs.append(_CI_Z * _se_for_metric(row, df_prep, col))
            entries.append((agent, display_label, vals, yerrs))

        if not entries:
            continue
        panels.append((_display.get(bm_name, bm_name), entries))

    if not panels:
        print("  No valid panels for detailed OpenAI social plot")
        for k, v in prev_rc.items():
            plt.rcParams[k] = v
        return

    # ── Figure layout ─────────────────────────────────────────────────
    n_rows = len(panels)
    n_cols = len(metrics)
    n_models = max(len(entries) for _, entries in panels)

    bar_h = _BAR_HEIGHT
    header_h = 1.2
    footer_h = 0.6
    row_h = 0.45 * n_models + 0.5   # per-benchmark row height
    fig_h = header_h + n_rows * row_h + footer_h + 0.4
    col_w = 2.8
    label_w = 3.0   # extra width for y-tick labels in first column
    fig_w = label_w + (n_cols - 1) * col_w + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)

    # Outer grid: header | body rows | footer
    outer_height_ratios = [header_h] + [row_h] * n_rows + [footer_h]
    outer_gs = gridspec.GridSpec(
        n_rows + 2, 1,
        figure=fig,
        height_ratios=outer_height_ratios,
        hspace=0.55,
        left=0.15, right=0.98, top=0.97, bottom=0.02,
    )

    _LEFT_MARGIN = 0.02

    # ── Header ────────────────────────────────────────────────────────
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, hbox.y1 - 0.01,
        "Reliability Breakdown across OpenAI Models",
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, hbox.y0 + hbox.height * 0.30,
        "Accuracy and each reliability pillar. GPT 5.4 highlighted.",
        fontsize=12, color=_COLOR_SUBTLE, va="top", ha="left", wrap=True,
    )

    # ── Body: one sub-grid per benchmark row ──────────────────────────
    for r_idx, (bm_title, entries) in enumerate(panels):
        # Inner grid: 1 row x n_cols, with wider first column for labels
        inner_gs = outer_gs[1 + r_idx].subgridspec(
            1, n_cols,
            width_ratios=[label_w] + [col_w] * (n_cols - 1),
            wspace=0.30,
        )

        labels = [lbl for _, lbl, _, _ in entries]
        agents = [a for a, _, _, _ in entries]
        bar_colors = [_OPENAI_GREEN if "5_4" in a else _BG_COLOR for a in agents]
        y_pos = np.arange(len(labels))

        for c_idx, (metric_col, metric_title) in enumerate(metrics):
            ax = fig.add_subplot(inner_gs[0, c_idx])

            vals = np.array([e[2][c_idx] for e in entries])
            yerrs = np.array([e[3][c_idx] for e in entries])
            yerrs = _clip_yerr(yerrs, vals)

            for j in range(len(labels)):
                ax.barh(
                    y_pos[j], vals[j], height=bar_h,
                    color=bar_colors[j],
                    edgecolor="black", linewidth=1, zorder=2,
                )

            # Draw error bars on top
            ax.errorbar(
                vals, y_pos, xerr=yerrs, fmt="none",
                ecolor="#333333", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3,
            )

            # Shared y-axis: labels only on the first column
            ax.set_yticks(y_pos)
            if c_idx == 0:
                ax.set_yticklabels(labels, fontsize=10, color=_COLOR_TEXT)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, pad=6)

            # X-axis: 0–1 range, ticks only on bottom row
            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
            ax.tick_params(axis="x", length=0, pad=4)
            if r_idx == n_rows - 1:
                ax.set_xticklabels(
                    ["0%", "25%", "50%", "75%", "100%"],
                    fontsize=9, color=_COLOR_SUBTLE,
                )
            else:
                ax.set_xticklabels([])

            # Grid & spines
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, color="#a0a0a0", linewidth=1.0)
            ax.yaxis.grid(False)
            ax.axvline(0, color="black", linewidth=1.5, zorder=1.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor(_BG_COLOR)

            # Column title (metric name) above the top row only
            if r_idx == 0:
                ax.set_title(
                    metric_title, fontsize=12, fontweight="bold",
                    color=_COLOR_TEXT, pad=8,
                )

        # Rotated benchmark label on the far left
        first_ax = fig.axes[-n_cols]  # first axes we just added
        first_bbox = first_ax.get_position()
        fig.text(
            0.01, first_bbox.y0 + first_bbox.height / 2,
            bm_title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # ── Footer ────────────────────────────────────────────────────────
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)

    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.99
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "openai_detailed.pdf"
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── JSON column parsers for curve data from CSV ──────────────────────

def _parse_calibration_bins(row):
    """Parse calibration bins from a DataFrame row's JSON column."""
    col = "_calibration_bins_json"
    if col not in row.index:
        return []
    val = row[col]
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_aurc_data(row):
    """Parse AURC (accuracy-coverage) data from DataFrame row JSON columns."""
    cols = ("_aurc_coverages_json", "_aurc_risks_json", "_aurc_optimal_risks_json")
    for col in cols:
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val) or val == "" or val == "[]":
            return None
    try:
        return {
            "coverages": np.array(json.loads(row["_aurc_coverages_json"])),
            "risks": np.array(json.loads(row["_aurc_risks_json"])),
            "optimal_risks": np.array(json.loads(row["_aurc_optimal_risks_json"])),
        }
    except (json.JSONDecodeError, TypeError):
        return None


# ── Line colors for curve plots ──────────────────────────────────────

_LINE_COLOR_52 = "#888888"   # gray for 5.2
_LINE_COLOR_54 = "#10A37F"   # OpenAI green for 5.4


def _get_line_color_52_54(suffix: str) -> str:
    """Return line color based on model suffix."""
    return _LINE_COLOR_54 if "5_4" in suffix else _LINE_COLOR_52


# ── Panel drawing helpers for curve plots ─────────────────────────────

def _style_curve_axes(ax, xlabel=None, ylabel=None, show_xlabel=True):
    """Apply shared styling to a calibration/coverage axes."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=9, color=_COLOR_SUBTLE,
    )
    ax.set_yticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=9, color=_COLOR_SUBTLE,
    )
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_aspect("equal")
    ax.set_facecolor(_BG_COLOR)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#a0a0a0", linewidth=0.5)
    ax.yaxis.grid(True, color="#a0a0a0", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.5, zorder=1.5)
    ax.axhline(0, color="black", linewidth=1.5, zorder=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel and show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=_COLOR_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=_COLOR_TEXT)


def _draw_calibration_panel(ax, model_data):
    """Draw calibration diagram (confidence vs accuracy) on *ax*.

    *model_data*: list of ``(label, bins, color)`` where *bins* is a list
    of dicts with keys ``avg_confidence``, ``avg_accuracy``, ``count``.
    """
    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], color="#cccccc", linewidth=1.5,
            linestyle="--", zorder=1, label="Perfect")

    for label, bins, color in model_data:
        valid = [b for b in bins if b.get("count", 0) > 0]
        if not valid:
            continue
        confs = [b["avg_confidence"] for b in valid]
        accs = [b["avg_accuracy"] for b in valid]
        counts = [b["count"] for b in valid]
        max_count = max(counts) if counts else 1
        sizes = [c / max_count * 200 + 30 for c in counts]

        ax.plot(confs, accs, color=color, linewidth=2, alpha=0.7, zorder=2)
        ax.scatter(confs, accs, s=sizes, color=color, edgecolor="black",
                   linewidth=0.5, zorder=3, label=label)

    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)


def _draw_accuracy_coverage_panel(ax, model_data):
    """Draw accuracy-coverage curves on *ax*.

    *model_data*: list of ``(label, aurc_data, color)`` where *aurc_data*
    has keys ``coverages``, ``risks``, ``optimal_risks``.
    """
    for label, data, color in model_data:
        coverages = np.array(data["coverages"])
        accuracies = 1 - np.array(data["risks"])
        optimal = 1 - np.array(data["optimal_risks"])

        ax.plot(coverages, accuracies, color=color, linewidth=2,
                label=label, zorder=3)
        ax.plot(coverages, optimal, color=color, linewidth=1,
                linestyle="--", alpha=0.4, zorder=2)

    ax.legend(fontsize=9, loc="lower left", framealpha=0.8)


# ── Shared layout for 5.2-vs-5.4 curve plots ─────────────────────────

def _collect_curve_panels(benchmark_data, extractor_fn):
    """Collect per-benchmark panel data for curve plots.

    *extractor_fn(row)* returns curve data for a single model row, or
    ``None`` when data is unavailable.

    Returns a list of ``(bm_display, base_data, reasoning_data)`` tuples
    where each ``*_data`` is ``[(label, curve_data, color), ...]``.
    """
    panels = []
    for bm_name, bm_df in benchmark_data:
        models = _BENCHMARK_MODELS_52_54.get(bm_name)
        if not models:
            continue
        df_prep = _prepare_dataframe(bm_df)

        base_data, reasoning_data = [], []
        for suffix, label in models:
            row = _get_model_row(df_prep, suffix)
            if row is None:
                continue
            curve = extractor_fn(row)
            if curve is None:
                continue
            entry = (label, curve, _get_line_color_52_54(suffix))
            if suffix in _REASONING_SUFFIXES_52_54:
                reasoning_data.append(entry)
            else:
                base_data.append(entry)

        if base_data or reasoning_data:
            panels.append((_BM_DISPLAY.get(bm_name, bm_name),
                           base_data, reasoning_data))
    return panels


def _plot_social_52_54_curves(
    panels, output_dir, draw_fn, title, subtitle, filename,
    xlabel, ylabel, *, padding=0.5,
):
    """Shared figure layout for 5.2-vs-5.4 curve comparison plots.

    *draw_fn(ax, model_data)* fills a single axes with the appropriate
    curve type (calibration scatter or accuracy-coverage lines).
    """
    if not panels:
        print(f"  No valid panels for {filename}")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    n_rows = len(panels)
    panel_w = 3.8
    header_h = 1.2
    footer_h = 0.6
    row_h = panel_w + 0.5
    fig_w = 2 * panel_w + 2.0
    fig_h = header_h + n_rows * row_h + footer_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    outer_gs = gridspec.GridSpec(
        n_rows + 2, 1,
        figure=fig,
        height_ratios=[header_h] + [row_h] * n_rows + [footer_h],
        hspace=0.4,
        left=0.12, right=0.95, top=0.97, bottom=0.02,
    )

    _LEFT_MARGIN = 0.03

    # Header
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, hbox.y1 - 0.01,
        title,
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, hbox.y0 + hbox.height * 0.30,
        subtitle,
        fontsize=12, color=_COLOR_SUBTLE, va="top", ha="left", wrap=True,
    )

    # Body panels
    for r_idx, (bm_title, base_data, reasoning_data) in enumerate(panels):
        inner_gs = outer_gs[1 + r_idx].subgridspec(1, 2, wspace=0.30)
        is_last = r_idx == n_rows - 1

        # Base panel (left)
        ax_base = fig.add_subplot(inner_gs[0, 0])
        draw_fn(ax_base, base_data)
        _style_curve_axes(ax_base, xlabel=xlabel, ylabel=ylabel,
                          show_xlabel=is_last)
        if r_idx == 0:
            ax_base.set_title("Base", fontsize=12, fontweight="bold",
                              color=_COLOR_TEXT, pad=8)

        # Reasoning panel (right)
        ax_reas = fig.add_subplot(inner_gs[0, 1])
        draw_fn(ax_reas, reasoning_data)
        _style_curve_axes(ax_reas, xlabel=xlabel, ylabel=None,
                          show_xlabel=is_last)
        if r_idx == 0:
            ax_reas.set_title("Reasoning", fontsize=12, fontweight="bold",
                              color=_COLOR_TEXT, pad=8)

        # Rotated benchmark label on the far left
        base_bbox = ax_base.get_position()
        fig.text(
            0.01, base_bbox.y0 + base_bbox.height / 2,
            bm_title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG,
                [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG,
                [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / filename
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── Public entry points for calibration / discrimination ──────────────

def plot_social_gpt52_vs_gpt54_calibration(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Calibration diagrams comparing GPT 5.2 vs 5.4.

    For each benchmark, two side-by-side panels: base models (left) and
    reasoning models (right).  Each panel overlays calibration scatter
    (confidence vs accuracy) for both model generations.

    Output: social_gpt52_vs_gpt54_calibration.pdf
    """
    panels = _collect_curve_panels(
        benchmark_data,
        lambda row: _parse_calibration_bins(row) or None,
    )
    _plot_social_52_54_curves(
        panels, output_dir, _draw_calibration_panel,
        title="GPT 5.2 vs GPT 5.4: Calibration",
        subtitle="Calibration measures the alignment of a model's expressed confidence and ground truth accuracy. GPT 5.4 (both the base and reasoning variants) shows improved calibration. For GAIA and 5.4 (medium), we observe a switch from an overconfident model to an underconfident model vs 5.2.",
        filename="gpt52_vs_gpt54_calibration.pdf",
        xlabel="Confidence", ylabel="Accuracy",
        padding=padding,
    )


def plot_social_gpt52_vs_gpt54_discrimination(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Accuracy-coverage curves comparing GPT 5.2 vs 5.4.

    For each benchmark, two side-by-side panels: base models (left) and
    reasoning models (right).  Each panel overlays accuracy-vs-coverage
    curves (solid) and ideal curves (dashed) for both model generations.

    Output: social_gpt52_vs_gpt54_discrimination.pdf
    """
    panels = _collect_curve_panels(benchmark_data, _parse_aurc_data)
    _plot_social_52_54_curves(
        panels, output_dir, _draw_accuracy_coverage_panel,
        title="GPT 5.2 vs GPT 5.4: Discrimination",
        subtitle="Accuracy vs coverage, base and reasoning variants.",
        filename="gpt52_vs_gpt54_discrimination.pdf",
        xlabel="Coverage", ylabel="Accuracy",
        padding=padding,
    )


def _parse_confidence_densities(row):
    """Parse correct/incorrect confidence arrays from DataFrame row."""
    for col in ("_correct_confidences_json", "_incorrect_confidences_json"):
        if col not in row.index:
            return None
        val = row[col]
        if pd.isna(val) or val == "" or val == "[]":
            return None
    try:
        correct = json.loads(row["_correct_confidences_json"])
        incorrect = json.loads(row["_incorrect_confidences_json"])
        if not correct and not incorrect:
            return None
        return {
            "correct": np.array(correct),
            "incorrect": np.array(incorrect),
        }
    except (json.JSONDecodeError, TypeError):
        return None


def _draw_density_panel(ax, label, data, color):
    """Draw correct vs incorrect confidence density curves on *ax*.

    *data* has keys ``correct`` and ``incorrect`` (arrays of confidence
    scores).  A Gaussian KDE is fitted to each and the overlap region
    is shaded.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import roc_auc_score

    xs = np.linspace(0, 1, 300)
    correct = data["correct"]
    incorrect = data["incorrect"]

    # We need at least 2 points for a KDE
    has_correct = len(correct) >= 2
    has_incorrect = len(incorrect) >= 2

    # Compute AUROC if both groups are present
    auroc = None
    if has_correct and has_incorrect:
        scores = np.concatenate([correct, incorrect])
        labels = np.concatenate([np.ones(len(correct)), np.zeros(len(incorrect))])
        auroc = roc_auc_score(labels, scores)

    _CORRECT_COLOR = "#3b82f6"    # blue
    _INCORRECT_COLOR = "#f59e0b"  # orange

    if has_correct:
        kde_c = gaussian_kde(correct, bw_method=0.15)
        ys_c = kde_c(xs)
        ax.fill_between(xs, ys_c, alpha=0.25, color=_CORRECT_COLOR, zorder=2)
        ax.plot(xs, ys_c, color=_CORRECT_COLOR, linewidth=2, label="Correct", zorder=3)

    if has_incorrect:
        kde_i = gaussian_kde(incorrect, bw_method=0.15)
        ys_i = kde_i(xs)
        ax.fill_between(xs, ys_i, alpha=0.25, color=_INCORRECT_COLOR, zorder=2)
        ax.plot(xs, ys_i, color=_INCORRECT_COLOR, linewidth=2, label="Incorrect",
                zorder=3)


    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=9, color=_COLOR_SUBTLE)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_facecolor(_BG_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    if auroc is not None:
        ax.text(0.03, 0.72, f"AUROC = {auroc:.2f}",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="left", va="top", color=_COLOR_TEXT)


def plot_social_gpt52_vs_gpt54_discrimination_2(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Correct vs incorrect confidence density plots for GPT 5.2 vs 5.4.

    Four side-by-side panels per benchmark: GPT 5.2 base, GPT 5.2
    reasoning, GPT 5.4 base, GPT 5.4 reasoning.

    Output: social_gpt52_vs_gpt54_discrimination_2.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for discrimination density plot")
        return

    # Collect per-benchmark, per-model density data
    panels = []  # [(bm_display, [(label, data, color, col_idx), ...])]
    for bm_name, bm_df in benchmark_data:
        models = _BENCHMARK_MODELS_52_54.get(bm_name)
        if not models:
            continue
        df_prep = _prepare_dataframe(bm_df)

        entries = []  # (label, data, color, col_idx)
        for suffix, label in models:
            row = _get_model_row(df_prep, suffix)
            if row is None:
                continue
            densities = _parse_confidence_densities(row)
            if densities is None:
                continue
            color = _get_line_color_52_54(suffix)
            is_reasoning = suffix in _REASONING_SUFFIXES_52_54
            is_54 = "5_4" in suffix
            col_idx = int(is_reasoning) * 2 + int(is_54)
            entries.append((label, densities, color, col_idx))

        if entries:
            panels.append((_BM_DISPLAY.get(bm_name, bm_name), entries))

    if not panels:
        print("  No valid panels for discrimination density plot")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    n_rows = len(panels)
    panel_w = 2.6
    header_h = 1.2
    footer_h = 0.6
    row_h = 2.8
    fig_w = 4 * panel_w + 2.0
    fig_h = header_h + n_rows * row_h + footer_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    outer_gs = gridspec.GridSpec(
        n_rows + 2, 1,
        figure=fig,
        height_ratios=[header_h] + [row_h] * n_rows + [footer_h],
        hspace=0.8,
        left=0.06, right=0.97, top=0.97, bottom=0.02,
    )

    _LEFT_MARGIN = 0.03

    # Header
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, hbox.y1 - 0.01,
        "GPT 5.2 vs GPT 5.4: Score Distributions",
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, hbox.y0 + hbox.height * 0.30,
        "Confidence densities for correct vs incorrect predictions. ",
        fontsize=12, color=_COLOR_SUBTLE, va="top", ha="left", wrap=True,
    )

    # Body panels
    for r_idx, (bm_title, entries) in enumerate(panels):
        inner_gs = outer_gs[1 + r_idx].subgridspec(1, 4, wspace=0.20)

        # Derive column titles from actual model labels
        col_titles = [None] * 4
        for entry_label, _data, _color, col_idx in entries:
            col_titles[col_idx] = entry_label

        axes = [fig.add_subplot(inner_gs[0, c]) for c in range(4)]
        for entry_label, data, color, col_idx in entries:
            _draw_density_panel(axes[col_idx], entry_label, data, color)

        for c, ax in enumerate(axes):
            ax.set_xlabel("Confidence", fontsize=10, color=_COLOR_TEXT)
            if col_titles[c]:
                ax.set_title(col_titles[c], fontsize=11, fontweight="bold",
                             color=_COLOR_TEXT, pad=8)

        # Rotated benchmark label on the far left
        base_bbox = axes[0].get_position()
        fig.text(
            0.01, base_bbox.y0 + base_bbox.height / 2,
            bm_title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG,
                [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG,
                [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "gpt52_vs_gpt54_discrimination_2.pdf"
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── Consistency tile heatmap ──────────────────────────────────────────

# Red (inconsistent) → Orange → Green (consistent), matching the website.
_TILE_CMAP = LinearSegmentedColormap.from_list(
    "consistency_tiles",
    [
        (239 / 255, 68 / 255, 68 / 255),     # red   (sr=0.5)
        (245 / 255, 158 / 255, 11 / 255),    # orange (sr≈0.75)
        (16 / 255, 185 / 255, 129 / 255),    # green  (sr=0 or 1)
    ],
)
_TILE_CMAP.set_bad(color="#e5e5e5")


def _parse_task_outcomes(row):
    """Parse per-task success rates from DataFrame row JSON column."""
    col = "_consistency_task_outcomes_json"
    if col not in row.index:
        return {}
    val = row[col]
    if pd.isna(val) or val == "" or val == "{}":
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}


def _parse_task_levels(row):
    """Parse task difficulty levels from DataFrame row JSON column."""
    col = "_task_levels_json"
    if col not in row.index:
        return {}
    val = row[col]
    if pd.isna(val) or val == "" or val == "{}":
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}


def plot_social_openai_consistency_tiles(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """Tile heatmap showing per-task outcome consistency for all OpenAI models.

    For each benchmark: rows = models, columns = tasks (sorted by
    level then task ID).  Each tile is colored green (consistent: always
    pass or always fail) to red (inconsistent: mixed across repetitions).

    Output: social_openai_consistency_tiles.pdf
    """
    if not benchmark_data:
        print("  No benchmark data for consistency tile plot")
        return

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    # ── Collect panel data ────────────────────────────────────────────
    panels = []
    for bm_name, bm_df in benchmark_data:
        df_prep = _prepare_dataframe(bm_df)
        df_prep = df_prep[df_prep["provider"] == "OpenAI"].reset_index(drop=True)
        if len(df_prep) == 0:
            continue

        model_entries = []  # [(label, outcomes_dict, agent_name)]
        all_task_ids: set[str] = set()
        task_levels: dict[str, str] = {}

        for _, row in df_prep.iterrows():
            agent = row["agent"]
            label = strip_agent_prefix(agent)
            outcomes = _parse_task_outcomes(row)
            if not outcomes:
                continue
            model_entries.append((label, outcomes, agent))
            all_task_ids.update(outcomes.keys())
            levels = _parse_task_levels(row)
            if levels:
                task_levels.update(levels)

        if not model_entries or not all_task_ids:
            continue

        # Sort tasks: by level (if available), then numerically by ID
        def _sort_key(tid):
            level = task_levels.get(tid, "9")
            try:
                lnum = int(level)
            except ValueError:
                lnum = 9
            try:
                tnum = int(tid)
            except ValueError:
                tnum = 0
            return (lnum, tnum, tid)

        sorted_tasks = sorted(all_task_ids, key=_sort_key)

        # Build matrix: rows = models, cols = tasks
        n_m = len(model_entries)
        n_t = len(sorted_tasks)
        matrix = np.full((n_m, n_t), np.nan)
        labels, agents_list = [], []
        for i, (label, outcomes, agent) in enumerate(model_entries):
            labels.append(label)
            agents_list.append(agent)
            for j, tid in enumerate(sorted_tasks):
                if tid in outcomes:
                    matrix[i, j] = outcomes[tid]

        # Level group boundaries
        level_breaks = []
        if task_levels:
            prev_level = None
            for j, tid in enumerate(sorted_tasks):
                level = task_levels.get(tid, "9")
                if prev_level is not None and level != prev_level:
                    level_breaks.append(j)
                prev_level = level

        panels.append((
            _display.get(bm_name, bm_name),
            labels, agents_list, matrix, sorted_tasks,
            task_levels, level_breaks,
        ))

    if not panels:
        print("  No valid panels for consistency tile plot")
        return

    # ── Figure layout ─────────────────────────────────────────────────
    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    n_rows = len(panels)
    header_h = 1.2
    footer_h = 0.6
    legend_h = 0.35
    model_row_h = 0.45
    row_padding = 0.8
    row_heights = [len(p[1]) * model_row_h + row_padding for p in panels]
    fig_h = header_h + sum(row_heights) + legend_h + footer_h + 0.4
    fig_w = 10.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    outer_gs = gridspec.GridSpec(
        n_rows + 3, 1,
        figure=fig,
        height_ratios=[header_h] + row_heights + [legend_h, footer_h],
        hspace=0.4,
        left=0.18, right=0.97, top=0.97, bottom=0.02,
    )

    _LEFT_MARGIN = 0.02

    # ── Header ────────────────────────────────────────────────────────
    header_ax = fig.add_subplot(outer_gs[0])
    header_ax.axis("off")
    hbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, hbox.y1 - 0.01,
        "Outcome Consistency across OpenAI Models",
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, hbox.y0 + hbox.height * 0.30,
        "Per-task consistency across repeated runs. "
        "Green = always same outcome, Red = mixed results.",
        fontsize=12, color=_COLOR_SUBTLE, va="top", ha="left", wrap=True,
    )

    # ── Body panels ──────────────────────────────────────────────────
    for r_idx, (bm_title, labels, _agents, matrix, sorted_tasks,
                task_levels, level_breaks) in enumerate(panels):
        ax = fig.add_subplot(outer_gs[1 + r_idx])
        n_m, n_t = matrix.shape

        # Convert success_rate → normalized consistency [0, 1]
        # consistency = max(sr, 1-sr) ∈ [0.5, 1.0] → (c-0.5)*2 ∈ [0, 1]
        norm_matrix = np.where(
            np.isnan(matrix), np.nan,
            (np.maximum(matrix, 1 - matrix) - 0.5) * 2,
        )
        norm_masked = np.ma.masked_invalid(norm_matrix)

        ax.imshow(
            norm_masked, cmap=_TILE_CMAP, vmin=0, vmax=1,
            aspect="auto", interpolation="nearest",
        )

        # White grid lines between cells
        for i in range(n_m + 1):
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
        for j in range(n_t + 1):
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

        # Level separators (thicker dark lines)
        for brk in level_breaks:
            ax.axvline(brk - 0.5, color=_COLOR_TEXT, linewidth=2)

        # Level group labels above grid
        if task_levels and level_breaks:
            boundaries = [0] + level_breaks + [n_t]
            for k in range(len(boundaries) - 1):
                start = boundaries[k]
                end = boundaries[k + 1]
                center = (start + end - 1) / 2
                level = task_levels.get(sorted_tasks[start], "")
                if level and level != "9":
                    ax.text(
                        center, -0.9, f"Level {level}",
                        ha="center", va="bottom",
                        fontsize=9, color=_COLOR_SUBTLE, fontweight="bold",
                    )

        # Y-axis labels
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(labels, fontsize=11, color=_COLOR_TEXT)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(_BG_COLOR)

        # Rotated benchmark label on the far left
        ax_bbox = ax.get_position()
        fig.text(
            0.01, ax_bbox.y0 + ax_bbox.height / 2,
            bm_title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # ── Color legend ──────────────────────────────────────────────────
    legend_ax = fig.add_subplot(outer_gs[-2])
    legend_ax.axis("off")
    legend_ax.set_facecolor(_BG_COLOR)

    lbox = legend_ax.get_position()
    gradient_w = 0.15
    gradient_h = lbox.height * 0.4
    gradient_x = (1 - gradient_w) / 2
    gradient_y = lbox.y0 + lbox.height * 0.3

    grad_ax = fig.add_axes([gradient_x, gradient_y, gradient_w, gradient_h])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    grad_ax.imshow(gradient, cmap=_TILE_CMAP, aspect="auto")
    grad_ax.set_xticks([])
    grad_ax.set_yticks([])
    for spine in grad_ax.spines.values():
        spine.set_visible(False)
    fig.text(
        gradient_x - 0.01, gradient_y + gradient_h / 2,
        "Inconsistent", fontsize=9, color=_COLOR_SUBTLE,
        va="center", ha="right",
    )
    fig.text(
        gradient_x + gradient_w + 0.01, gradient_y + gradient_h / 2,
        "Consistent", fontsize=9, color=_COLOR_SUBTLE,
        va="center", ha="left",
    )

    # ── Footer ────────────────────────────────────────────────────────
    footer_ax = fig.add_subplot(outer_gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    fbox = footer_ax.get_position()
    logo_h = fbox.height * 0.8
    logo_y = fbox.y0 + fbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG,
                [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG,
                [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, fbox.y0 + fbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / "openai_consistency_tiles.pdf"
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


# ── All-OpenAI single-metric bar charts ───────────────────────────────

def _plot_social_openai_metric(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    metric_col: str,
    se_col: str,
    title: str,
    subtitle: str,
    filename: str,
    *,
    padding: float = 0.5,
):
    """Shared layout for all-OpenAI horizontal bar charts of a single metric."""
    if not benchmark_data:
        print(f"  No benchmark data for {filename}")
        return

    prev_rc = {k: plt.rcParams[k] for k in ("font.family", "font.sans-serif")}
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = _FONT_FAMILY

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    panels = []
    for bm_name, bm_df in benchmark_data:
        df_sorted = _prepare_dataframe(bm_df)
        df_sorted = df_sorted[df_sorted["provider"] == "OpenAI"].reset_index(drop=True)
        valid = df_sorted[metric_col].notna()
        df_valid = df_sorted[valid].copy()
        if len(df_valid) == 0:
            continue

        df_valid = df_valid.sort_values(metric_col, ascending=True).reset_index(drop=True)

        labels = [strip_agent_prefix(a) for a in df_valid["agent"]]
        values = df_valid[metric_col].values
        # Highlight GPT 5.4 models in OpenAI green, others in background
        _OPENAI_GREEN = "#10A37F"
        colors_sorted = [
            _OPENAI_GREEN if "5_4" in a else _BG_COLOR
            for a in df_valid["agent"]
        ]

        if se_col and se_col in df_valid.columns:
            se = np.where(np.isnan(df_valid[se_col].values), 0, df_valid[se_col].values)
            xerr = _CI_Z * se
            xerr = _clip_yerr(xerr, values)
        else:
            xerr = None

        display_name = _display.get(bm_name, bm_name)
        panels.append((display_name, labels, values, colors_sorted, xerr))

    if not panels:
        print(f"  No valid panels for {filename}")
        for k, v in prev_rc.items():
            plt.rcParams[k] = v
        return

    n_panels = len(panels)
    panel_sizes = [len(p[1]) for p in panels]

    header_height = 1.1
    footer_height = 0.6
    bar_row_height = 0.38
    body_height = sum(s * bar_row_height for s in panel_sizes) + 0.6 * (n_panels - 1)
    fig_height = header_height + body_height + footer_height + 0.3
    fig_width = 7

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=_BG_COLOR)
    height_ratios = [header_height] + [s * bar_row_height for s in panel_sizes] + [footer_height]
    gs = gridspec.GridSpec(
        n_panels + 2, 1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        left=0.28, right=0.97, top=0.97, bottom=0.02,
    )

    _LEFT_MARGIN = 0.05
    header_ax = fig.add_subplot(gs[0])
    header_ax.axis("off")
    header_bbox = header_ax.get_position()
    fig.text(
        _LEFT_MARGIN, header_bbox.y1 - 0.01,
        title,
        fontsize=19, fontweight="bold", color=_COLOR_TEXT,
        va="top", ha="left",
    )
    fig.text(
        _LEFT_MARGIN, header_bbox.y0 + header_bbox.height * 0.35,
        subtitle,
        fontsize=12, color=_COLOR_SUBTLE,
        va="top", ha="left", wrap=True,
    )

    for i, (bm_title, labels, values, colors, xerr) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i])
        is_last = i == n_panels - 1
        _draw_panel(ax, labels, values, colors, bm_title, show_xticks=is_last, xerr=xerr)
        ax_bbox = ax.get_position()
        fig.text(
            0.01, ax_bbox.y0 + ax_bbox.height / 2,
            bm_title, fontsize=14, fontweight="bold", color=_COLOR_TEXT,
            va="center", ha="center", rotation=90,
        )

    # Footer
    footer_ax = fig.add_subplot(gs[-1])
    footer_ax.axis("off")
    footer_ax.set_facecolor(_BG_COLOR)
    fig.canvas.draw()
    footer_bbox = footer_ax.get_position()
    logo_h = footer_bbox.height * 0.8
    logo_y = footer_bbox.y0 + footer_bbox.height * 0.1
    _RIGHT_EDGE = 0.97
    _place_logo(fig, _HAL_LOGO_PNG, [_LEFT_MARGIN, logo_y, 0.25, logo_h], anchor="W")
    _place_logo(fig, _PRINCETON_LOGO_PNG, [_RIGHT_EDGE - 0.25, logo_y, 0.25, logo_h], anchor="E")
    fig.text(
        (_LEFT_MARGIN + _RIGHT_EDGE) / 2, footer_bbox.y0 + footer_bbox.height * 0.5,
        "hal.cs.princeton.edu/reliability",
        fontsize=10, color=_COLOR_SUBTLE, ha="center", va="center",
    )

    social_dir = output_dir / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    output_path = social_dir / filename
    fig.savefig(
        output_path, dpi=300, format="pdf",
        facecolor=_BG_COLOR,
        bbox_inches="tight", pad_inches=padding,
    )
    print(f"  Saved: {output_path}")
    plt.close(fig)

    for k, v in prev_rc.items():
        plt.rcParams[k] = v


def plot_social_outcome_consistency(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """All-OpenAI horizontal bar chart for outcome consistency.

    Output: social_outcome_consistency.pdf
    """
    _plot_social_openai_metric(
        benchmark_data, output_dir,
        metric_col="consistency_outcome",
        se_col="consistency_outcome_se",
        title="Outcome Consistency across OpenAI Models",
        subtitle="Fraction of tasks with consistent outcomes across repeated runs.",
        filename="openai_outcome_consistency.pdf",
        padding=padding,
    )


def plot_social_calibration(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """All-OpenAI horizontal bar chart for calibration.

    Output: social_calibration.pdf
    """
    _plot_social_openai_metric(
        benchmark_data, output_dir,
        metric_col="predictability_calibration",
        se_col="predictability_calibration_se",
        title="Calibration across OpenAI Models",
        subtitle="How well confidence scores match actual success rates.",
        filename="openai_calibration.pdf",
        padding=padding,
    )


def plot_social_discrimination(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    *,
    padding: float = 0.5,
):
    """All-OpenAI horizontal bar chart for discrimination (AUROC).

    Output: social_discrimination.pdf
    """
    _plot_social_openai_metric(
        benchmark_data, output_dir,
        metric_col="predictability_roc_auc",
        se_col="predictability_roc_auc_se",
        title="Discrimination across OpenAI Models",
        subtitle="How well confidence separates correct from incorrect (AUROC).",
        filename="openai_discrimination.pdf",
        padding=padding,
    )

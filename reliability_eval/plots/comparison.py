"""Multi-agent/multi-benchmark comparison plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from reliability_eval.constants import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    PROVIDER_COLORS,
    PROVIDER_MARKERS,
)
from reliability_eval.loaders.agent_names import (
    get_model_category,
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
    _get_yerr,
    filter_oldest_and_newest_per_provider,
    generate_shaded_colors,
)


def plot_reliability_vs_date_and_accuracy(
    df: pd.DataFrame, output_dir: Path, benchmark_name: str = ""
):
    """
    Create a 2x5 grid of scatter plots with trend lines:
    - Column 1: Release date vs reliability
    - Column 2: Accuracy vs reliability
    - Rows: Overall, Consistency, Predictability, Robustness, Safety

    Also creates a separate PDF with just the Overall Reliability row.
    """
    from scipy import stats
    import matplotlib.dates as mdates

    # Sort by provider and release date
    df_sorted = sort_agents_by_provider_and_date(df)

    # Compute dimension-level scores if not already present
    if "R_Con" not in df_sorted.columns:
        df_sorted["R_Con"] = compute_weighted_r_con(
            df_sorted["C_out"],
            df_sorted["C_traj_d"],
            df_sorted["C_traj_s"],
            df_sorted["C_res"],
        )
    if "R_Pred" not in df_sorted.columns:
        df_sorted["R_Pred"] = df_sorted["P_brier"]
    if "R_Rob" not in df_sorted.columns:
        df_sorted["R_Rob"] = df_sorted[["R_fault", "R_struct", "R_prompt"]].mean(
            axis=1, skipna=True
        )
    if "R_Saf" not in df_sorted.columns:
        df_sorted["R_Saf"] = df_sorted["S_safety"]
    if "R_Overall" not in df_sorted.columns:
        df_sorted["R_Overall"] = df_sorted[["R_Con", "R_Pred", "R_Rob"]].mean(
            axis=1, skipna=True
        )

    # Ensure release_timestamp is present
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

    # Define dimensions to plot (R_Saf kept as its own trendline but not in R_Overall)
    dimensions = [
        ("R_Overall", r"Overall Reliability ($R$)"),
        ("R_Con", r"Consistency ($R_{\mathrm{Con}}$)"),
        ("R_Pred", r"Predictability ($R_{\mathrm{Pred}}$)"),
        ("R_Rob", r"Robustness ($R_{\mathrm{Rob}}$)"),
        ("R_Saf", r"Safety ($R_{\mathrm{Saf}}$)"),
    ]

    # Create figure: 2 columns x 5 rows
    # Aim for ~1:1 aspect ratio per subplot: height=12/5=2.4, so width=2.4*2=4.8
    fig, axes = plt.subplots(5, 2, figsize=(5, 12))

    def add_scatter_with_trend(
        ax, x_data, y_data, providers, xlabel, ylabel, title, is_date=False
    ):
        """Add scatter plot with trend line, colored by provider."""
        # Filter out NaN values
        valid_mask = ~(np.isnan(y_data) if not is_date else False)
        if is_date:
            valid_mask = ~pd.isna(x_data) & ~np.isnan(y_data)

        if valid_mask.sum() < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            return

        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        providers_valid = providers[valid_mask]

        # Convert dates to numeric for regression
        if is_date:
            x_numeric = (x_valid - x_valid.min()).dt.days.values
        else:
            x_numeric = x_valid.values

        # Scatter points by provider
        for provider in ["OpenAI", "Google", "Anthropic"]:
            mask = providers_valid == provider
            if mask.sum() == 0:
                continue
            ax.scatter(
                x_valid[mask],
                y_valid[mask],
                c=PROVIDER_COLORS.get(provider, "#999999"),
                marker=PROVIDER_MARKERS.get(provider, "o"),
                s=50,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.6,
                label=provider,
            )

        # Add trend line using linear regression
        if len(x_numeric) >= 2:
            slope, intercept, r_value, p_value, _ = stats.linregress(
                x_numeric, y_valid.values
            )

            # Generate trend line
            if is_date:
                x_range = np.array([x_numeric.min(), x_numeric.max()])
                x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
                y_trend = slope * x_range + intercept
                ax.plot(
                    x_dates, y_trend, "k--", linewidth=1.5, alpha=0.7, label="Trend"
                )
                # Convert slope from per-day to per-year for interpretability
                slope_per_year = slope * 365
                slope_str = f"slope={slope_per_year:+.2f}/yr"
            else:
                x_range = np.linspace(x_numeric.min(), x_numeric.max(), 100)
                y_trend = slope * x_range + intercept
                ax.plot(
                    x_range, y_trend, "k--", linewidth=1.5, alpha=0.7, label="Trend"
                )
                slope_str = f"slope={slope:+.2f}"

            # Add correlation and slope annotation
            ax.annotate(
                f"r={r_value:+.2f}\n{slope_str}\np={p_value:.2f}",
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

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 1.15)

        if is_date:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot each dimension
    for row_idx, (dim_col, dim_label) in enumerate(dimensions):
        # Column 0: Release date vs reliability
        ax = axes[row_idx, 0]
        add_scatter_with_trend(
            ax,
            df_sorted["release_timestamp"],
            df_sorted[dim_col],
            df_sorted["provider"],
            xlabel="Release Date",
            ylabel=dim_label,
            title=f"{dim_label} vs Release Date",
            is_date=True,
        )

        # Column 1: Accuracy vs reliability
        ax = axes[row_idx, 1]
        add_scatter_with_trend(
            ax,
            df_sorted["accuracy"],
            df_sorted[dim_col],
            df_sorted["provider"],
            xlabel="Accuracy",
            ylabel="",  # Remove y-axis label for right column (shared with left)
            title=f"{dim_label} vs Accuracy",
            is_date=False,
        )
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # 0.2 increments for accuracy
        ax.tick_params(
            axis="y", labelleft=False
        )  # Hide y-axis tick labels, keep gridlines

    # Add legend to figure (shared across all subplots)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Remove duplicate labels (keep unique)
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        framealpha=0.95,
        edgecolor="gray",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = output_dir / "reliability_trends.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

    # --- Separate plot: Accuracy vs Release Date ---
    fig_acc, ax_acc = plt.subplots(figsize=(4, 4))

    # Filter valid data
    valid_mask = df_sorted["release_timestamp"].notna() & df_sorted["accuracy"].notna()
    if valid_mask.sum() >= 2:
        x_valid = df_sorted.loc[valid_mask, "release_timestamp"]
        y_valid = df_sorted.loc[valid_mask, "accuracy"]
        providers_valid = df_sorted.loc[valid_mask, "provider"]

        # Scatter points by provider
        for provider in ["OpenAI", "Google", "Anthropic"]:
            mask = providers_valid == provider
            if mask.sum() == 0:
                continue
            ax_acc.scatter(
                x_valid[mask],
                y_valid[mask],
                c=PROVIDER_COLORS.get(provider, "#999999"),
                marker=PROVIDER_MARKERS.get(provider, "o"),
                s=50,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.6,
                label=provider,
            )

        # Add trend line
        x_numeric = (x_valid - x_valid.min()).dt.days.values
        slope, intercept, r_value, p_value, _ = stats.linregress(
            x_numeric, y_valid.values
        )
        x_range = np.array([x_numeric.min(), x_numeric.max()])
        x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
        y_trend = slope * x_range + intercept
        ax_acc.plot(x_dates, y_trend, "k--", linewidth=1.5, alpha=0.7)

        # Annotation
        slope_per_year = slope * 365
        ax_acc.text(
            0.05,
            0.95,
            f"r={r_value:.2f}, slope={slope_per_year:+.2f}/yr",
            transform=ax_acc.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax_acc.set_xlabel("Release Date", fontsize=11, fontweight="bold")
    ax_acc.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax_acc.set_title("Accuracy vs Release Date", fontsize=12, fontweight="bold")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_acc.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax_acc.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax_acc.legend(fontsize=8, loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path_acc = output_dir / "accuracy_vs_time.pdf"
    plt.savefig(output_path_acc, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path_acc}")
    plt.close()

    # --- Separate PDF plot: Overall Reliability only ---
    # Use same aspect ratio as original subplots: each subplot is 2.5 x 2.4 inches
    fig_overall, axes_overall = plt.subplots(1, 2, figsize=(5, 2.4))

    # Build y-axis label with benchmark name
    ylabel_with_benchmark = r"Overall Reliability $R$" + (
        f"\n({benchmark_name})" if benchmark_name else ""
    )

    # Left: Overall Reliability vs Release Date
    ax = axes_overall[0]
    valid_mask = df_sorted["release_timestamp"].notna() & df_sorted["R_Overall"].notna()
    if valid_mask.sum() >= 2:
        x_valid = df_sorted.loc[valid_mask, "release_timestamp"]
        y_valid = df_sorted.loc[valid_mask, "R_Overall"]
        providers_valid = df_sorted.loc[valid_mask, "provider"]

        # Scatter points by provider
        for provider in ["OpenAI", "Google", "Anthropic"]:
            mask = providers_valid == provider
            if mask.sum() == 0:
                continue
            ax.scatter(
                x_valid[mask],
                y_valid[mask],
                c=PROVIDER_COLORS.get(provider, "#999999"),
                marker=PROVIDER_MARKERS.get(provider, "o"),
                s=50,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.6,
                label=provider,
            )

        # Add trend line
        x_numeric = (x_valid - x_valid.min()).dt.days.values
        slope, intercept, r_value, p_value, _ = stats.linregress(
            x_numeric, y_valid.values
        )
        x_range = np.array([x_numeric.min(), x_numeric.max()])
        x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
        y_trend = slope * x_range + intercept
        ax.plot(x_dates, y_trend, "k--", linewidth=1.5, alpha=0.7, label="Trend")

        # Annotation
        slope_per_year = slope * 365
        ax.annotate(
            f"r={r_value:+.2f}\nslope={slope_per_year:+.2f}/yr\np={p_value:.2f}",
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

    ax.set_xlabel("Release Date")
    ax.set_ylabel(ylabel_with_benchmark)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Right: Overall Reliability vs Accuracy
    ax = axes_overall[1]
    valid_mask = df_sorted["accuracy"].notna() & df_sorted["R_Overall"].notna()
    if valid_mask.sum() >= 2:
        x_valid = df_sorted.loc[valid_mask, "accuracy"]
        y_valid = df_sorted.loc[valid_mask, "R_Overall"]
        providers_valid = df_sorted.loc[valid_mask, "provider"]

        # Scatter points by provider
        for provider in ["OpenAI", "Google", "Anthropic"]:
            mask = providers_valid == provider
            if mask.sum() == 0:
                continue
            ax.scatter(
                x_valid[mask],
                y_valid[mask],
                c=PROVIDER_COLORS.get(provider, "#999999"),
                marker=PROVIDER_MARKERS.get(provider, "o"),
                s=50,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.6,
                label=provider,
            )

        # Add trend line
        slope, intercept, r_value, p_value, _ = stats.linregress(
            x_valid.values, y_valid.values
        )
        x_range = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_trend = slope * x_range + intercept
        ax.plot(x_range, y_trend, "k--", linewidth=1.5, alpha=0.7, label="Trend")

        # Annotation
        ax.annotate(
            f"r={r_value:+.2f}\nslope={slope:+.2f}\np={p_value:.2f}",
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

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("")  # Shared with left
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Add legend
    handles, labels = axes_overall[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_overall.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path_overall = output_dir / "overall_reliability_trends.pdf"
    plt.savefig(output_path_overall, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path_overall}")
    plt.close()


def plot_combined_overall_reliability(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """
    Create a grid of Overall Reliability plots for multiple benchmarks.

    Layout:
    - Column 0: Overall Reliability vs Release Date (trend over time)
    - Column 1: Overall Reliability vs Accuracy (reliability-accuracy tradeoff)
    - Rows: One row per benchmark (GAIA, tau-bench, etc.)

    Args:
        benchmark_data: List of (benchmark_name, dataframe) tuples
        output_dir: Directory to save the plot
    """
    from scipy import stats
    import matplotlib.dates as mdates

    n_benchmarks = len(benchmark_data)
    if n_benchmarks == 0:
        print("⚠️  No benchmark data provided for combined plot")
        return

    # Create figure: 2 columns x n_benchmarks rows
    # Each subplot is ~2.5 x 2.0 inches
    fig, axes = plt.subplots(n_benchmarks, 2, figsize=(5, 3.9))

    # Handle single benchmark case (axes needs to be 2D)
    if n_benchmarks == 1:
        axes = axes.reshape(1, -1)

    def prepare_dataframe(df):
        """Prepare dataframe with required columns."""
        df_sorted = sort_agents_by_provider_and_date(df)

        # Compute dimension-level scores if not already present
        if "R_Con" not in df_sorted.columns:
            df_sorted["R_Con"] = compute_weighted_r_con(
                df_sorted["C_out"],
                df_sorted["C_traj_d"],
                df_sorted["C_traj_s"],
                df_sorted["C_res"],
            )
        if "R_Pred" not in df_sorted.columns:
            df_sorted["R_Pred"] = df_sorted["P_brier"]
        if "R_Rob" not in df_sorted.columns:
            df_sorted["R_Rob"] = df_sorted[["R_fault", "R_struct", "R_prompt"]].mean(
                axis=1, skipna=True
            )
        if "R_Saf" not in df_sorted.columns:
            df_sorted["R_Saf"] = df_sorted["S_safety"]
        if "R_Overall" not in df_sorted.columns:
            df_sorted["R_Overall"] = df_sorted[["R_Con", "R_Pred", "R_Rob"]].mean(
                axis=1, skipna=True
            )

        # Ensure release_timestamp is present
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

        return df_sorted

    for row_idx, (benchmark_name, df) in enumerate(benchmark_data):
        df_sorted = prepare_dataframe(df)
        display_name = (
            benchmark_name.replace(
                "taubench_airline_original", r"$\tau$-bench (original)"
            )
            .replace("taubench_airline", r"$\tau$-bench")
            .replace("gaia", "GAIA")
        )
        ylabel_with_benchmark = f"Reliability\n({display_name})"

        # Left: Overall Reliability vs Release Date
        ax = axes[row_idx, 0]
        valid_mask = (
            df_sorted["release_timestamp"].notna() & df_sorted["R_Overall"].notna()
        )
        if valid_mask.sum() >= 2:
            x_valid = df_sorted.loc[valid_mask, "release_timestamp"]
            y_valid = df_sorted.loc[valid_mask, "R_Overall"]
            providers_valid = df_sorted.loc[valid_mask, "provider"]

            # Scatter points by provider
            for provider in ["OpenAI", "Google", "Anthropic"]:
                mask = providers_valid == provider
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    x_valid[mask],
                    y_valid[mask],
                    c=PROVIDER_COLORS.get(provider, "#999999"),
                    marker=PROVIDER_MARKERS.get(provider, "o"),
                    s=70,
                    alpha=0.85,
                    edgecolors="black",
                    linewidth=0.6,
                    label=provider,
                )

            # Add trend line
            x_numeric = (x_valid - x_valid.min()).dt.days.values
            slope, intercept, r_value, _, __ = stats.linregress(
                x_numeric, y_valid.values
            )
            x_range = np.array([x_numeric.min(), x_numeric.max()])
            x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
            y_trend = slope * x_range + intercept
            ax.plot(x_dates, y_trend, "k-", linewidth=2, alpha=0.85, label="Trend")

            # Annotation
            slope_per_year = slope * 365
            ax.annotate(
                f"r={r_value:.2f}\nslope={slope_per_year:.2f}/yr",
                xy=(0.95, 0.07),
                xycoords="axes fraction",
                fontsize=11,
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

        ax.set_ylabel(ylabel_with_benchmark, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.tick_params(axis="both", labelsize=11)

        # Only show x-axis label and ticks for the last row
        is_last_row = row_idx == n_benchmarks - 1
        if is_last_row:
            ax.set_xlabel("Release Date", fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        # Right: Overall Reliability vs Accuracy
        ax = axes[row_idx, 1]
        valid_mask = df_sorted["accuracy"].notna() & df_sorted["R_Overall"].notna()
        if valid_mask.sum() >= 2:
            x_valid = df_sorted.loc[valid_mask, "accuracy"]
            y_valid = df_sorted.loc[valid_mask, "R_Overall"]
            providers_valid = df_sorted.loc[valid_mask, "provider"]

            # Scatter points by provider
            for provider in ["OpenAI", "Google", "Anthropic"]:
                mask = providers_valid == provider
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    x_valid[mask],
                    y_valid[mask],
                    c=PROVIDER_COLORS.get(provider, "#999999"),
                    marker=PROVIDER_MARKERS.get(provider, "o"),
                    s=70,
                    alpha=0.85,
                    edgecolors="black",
                    linewidth=0.6,
                    label=provider,
                )

            # Add trend line
            slope, intercept, r_value, _, __ = stats.linregress(
                x_valid.values, y_valid.values
            )
            x_range = np.array([x_valid.min(), x_valid.max()])
            y_trend = slope * x_range + intercept
            ax.plot(x_range, y_trend, "k-", linewidth=2, alpha=0.85, label="Trend")

            # Annotation
            ax.annotate(
                f"r={r_value:.2f}\nslope={slope:.2f}",
                xy=(0.95, 0.07),
                xycoords="axes fraction",
                fontsize=11,
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

        ax.set_ylabel("")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Only show x-axis label and ticks for the last row
        if is_last_row:
            ax.set_xlabel("Accuracy", fontsize=11)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

    # Add legend at top (shared)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.50, 1.05),
        ncol=5,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=11,
        handletextpad=0.3,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "combined_overall_reliability.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

    # --- Combined Accuracy vs Release Date plot (1 row, n_benchmarks columns, side by side) ---
    fig_acc, axes_acc = plt.subplots(1, n_benchmarks, figsize=(4 * n_benchmarks, 3.0))

    # Handle single benchmark case
    if n_benchmarks == 1:
        axes_acc = [axes_acc]

    _acc_display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    for col_idx, (benchmark_name, df) in enumerate(benchmark_data):
        df_sorted = prepare_dataframe(df)
        ax = axes_acc[col_idx]
        bm_display = _acc_display.get(benchmark_name, benchmark_name)

        valid_mask = (
            df_sorted["release_timestamp"].notna() & df_sorted["accuracy"].notna()
        )
        if valid_mask.sum() >= 2:
            x_valid = df_sorted.loc[valid_mask, "release_timestamp"]
            y_valid = df_sorted.loc[valid_mask, "accuracy"]
            providers_valid = df_sorted.loc[valid_mask, "provider"]

            # Scatter points by provider
            for provider in ["OpenAI", "Google", "Anthropic"]:
                mask = providers_valid == provider
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    x_valid[mask],
                    y_valid[mask],
                    c=PROVIDER_COLORS.get(provider, "#999999"),
                    marker=PROVIDER_MARKERS.get(provider, "o"),
                    s=50,
                    alpha=0.85,
                    edgecolors="black",
                    linewidth=0.6,
                    label=provider,
                )

            # Add trend line
            x_numeric = (x_valid - x_valid.min()).dt.days.values
            slope, intercept, r_value, _, __ = stats.linregress(
                x_numeric, y_valid.values
            )
            x_range = np.array([x_numeric.min(), x_numeric.max()])
            x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
            y_trend = slope * x_range + intercept
            ax.plot(x_dates, y_trend, "k--", linewidth=1.5, alpha=0.7, label="Trend")

            # Annotation
            slope_per_year = slope * 365
            ax.annotate(
                f"r={r_value:.2f}\nslope={slope_per_year:.2f}/yr",
                xy=(0.95, 0.07),
                xycoords="axes fraction",
                fontsize=10,
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

        ax.set_xlabel("Release Date")
        ylabel_with_benchmark = f"Accuracy\n({bm_display})"
        ax.set_ylabel(ylabel_with_benchmark if col_idx == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Hide y-axis tick labels for non-first columns, but show benchmark name
        if col_idx > 0:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel(f"({bm_display})")

    # Add legend at top (shared)
    handles, labels = axes_acc[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_acc.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=5,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path_acc = output_dir / "combined_accuracy_vs_time.pdf"
    plt.savefig(output_path_acc, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path_acc}")
    plt.close()


# ── Manual label positions for combined_overall_reliability_large.pdf ──
# Keys: (benchmark_name, column_index)  where col 0=Accuracy vs Date,
#        col 1=Reliability vs Date, col 2=Accuracy vs Reliability.
# Values: dict mapping model display name → (x, y) in axes fraction (0–1).
#   (0, 0) = bottom-left corner, (1, 1) = top-right corner.
#   The starting x values match each model's data point position.
LARGE_PLOT_LABEL_POS = {
    ("gaia", 0): {
        "GPT-4 Turbo": (0.15, 0.05),
        "GPT-4o mini": (0.15, 0.35),
        "Gemini 2.0 Flash": (0.47, 0.05),
        "Claude 3.5 Haiku": (0.3, 0.45),
        "GPT 5.2": (0.8, 0.41),
        "o1": (0.55, 0.18),
        "Gemini 2.5 Flash": (0.74, 0.265),
        "GPT 5.2 (medium)": (0.8, 0.6),
        "Gemini 2.5 Pro": (0.60, 0.69),
        "Claude 3.7 Sonnet": (0.25, 0.55),
        "Claude 4.5 Opus": (0.83, 0.95),
        "Claude 4.5 Sonnet": (0.6, 0.85),
    },
    ("gaia", 1): {
        "Gemini 2.0 Flash": (0.18, 0.3),
        "o1": (0.45, 0.95),
        "GPT-4o mini": (0.25, 0.82),
        "Claude 3.5 Haiku": (0.18, 0.55),
        "GPT 5.2 (medium)": (0.81, 0.24),
        "GPT 5.2": (0.713, 0.32),
        "GPT-4 Turbo": (0.15, 0.95),
        "Claude 3.7 Sonnet": (0.25, 0.1),
        "Gemini 2.5 Flash": (0.63, 0.45),
        "Gemini 2.5 Pro": (0.76, 0.64),
        "Claude 4.5 Sonnet": (0.67, 0.85),
        "Claude 4.5 Opus": (0.83, 0.95),
    },
    ("gaia", 2): {
        "Gemini 2.0 Flash": (0.42, 0.05),
        "o1": (0.55, 0.5),
        "GPT-4o mini": (0.15, 0.21),
        "Claude 3.5 Haiku": (0.6, 0.25),
        "GPT 5.2 (medium)": (0.82, 0.37),
        "GPT 5.2": (0.2, 0.81),
        "GPT-4 Turbo": (0.15, 0.95),
        "Claude 3.7 Sonnet": (0.65, 0.88),
        "Gemini 2.5 Flash": (0.285, 0.88),
        "Gemini 2.5 Pro": (0.45, 0.95),
        "Claude 4.5 Sonnet": (0.83, 0.95),
        "Claude 4.5 Opus": (0.83, 0.6),
    },
    ("taubench_airline", 0): {
        "GPT-4o mini": (0.15, 0.05),
        "Claude 3.5 Haiku": (0.47, 0.05),
        "Gemini 2.0 Flash": (0.23, 0.4),
        "GPT-4 Turbo": (0.15, 0.95),
        "GPT 5.2": (0.90, 0.23),
        "Claude 3.7 Sonnet": (0.6, 0.23),
        "Gemini 2.5 Flash": (0.32, 0.6),
        "o1": (0.23, 0.51),
        "GPT 5.2 (xhigh)": (0.75, 0.35),
        "Gemini 2.5 Pro": (0.44, 0.73),
        "Claude 4.5 Sonnet": (0.7, 0.62),
        "Claude 4.5 Opus": (0.83, 0.95),
        "Gemini 3.0 Pro": (0.66, 0.85),
    },
    ("taubench_airline", 1): {
        "Claude 3.5 Haiku": (0.18, 0.52),
        "GPT-4o mini": (0.25, 0.82),
        "GPT-4 Turbo": (0.15, 0.95),
        "Gemini 2.0 Flash": (0.18, 0.3),
        "Claude 3.7 Sonnet": (0.18, 0.05),
        "Gemini 2.5 Flash": (0.45, 0.15),
        "Gemini 2.5 Pro": (0.7, 0.45),
        "GPT 5.2 (xhigh)": (0.81, 0.24),
        "GPT 5.2": (0.72, 0.32),
        "o1": (0.45, 0.95),
        "Gemini 3.0 Pro": (0.75, 0.6),
        "Claude 4.5 Sonnet": (0.65, 0.86),
        "Claude 4.5 Opus": (0.83, 0.95),
    },
    ("taubench_airline", 2): {
        "Claude 3.5 Haiku": (0.18, 0.5),
        "GPT-4o mini": (0.13, 0.95),
        "Gemini 2.0 Flash": (0.2, 0.2),
        "GPT-4 Turbo": (0.41, 0.4),
        "Gemini 2.5 Flash": (0.83, 0.34),
        "Claude 3.7 Sonnet": (0.6, 0.24),
        "Gemini 2.5 Pro": (0.83, 0.55),
        "GPT 5.2": (0.3, 0.83),
        "o1": (0.46, 0.85),
        "Gemini 3.0 Pro": (0.83, 0.69),
        "GPT 5.2 (xhigh)": (0.5, 0.95),
        "Claude 4.5 Sonnet": (0.83, 0.95),
        "Claude 4.5 Opus": (0.83, 0.83),
    },
}


def plot_combined_overall_reliability_large(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """
    Create a wider grid of plots for multiple benchmarks.

    Layout (3 columns x n_benchmarks rows):
    - Column 0: Accuracy vs Release Date
    - Column 1: Reliability vs Release Date
    - Column 2: Accuracy vs Reliability (scatter with model name annotations)

    Args:
        benchmark_data: List of (benchmark_name, dataframe) tuples
        output_dir: Directory to save the plot
    """
    from scipy import stats
    import matplotlib.dates as mdates

    # FIXME: this method is currently unused
    def place_labels(
        ax, x_vals, y_vals, labels, colors, benchmark_name, col_idx, fontsize=8
    ):
        """Place labels using positions from LARGE_PLOT_LABEL_POS dict, with colored arrows.

        Each entry is (x, y) in axes fraction (0–1): (0,0) = bottom-left, (1,1) = top-right.
        Arrow points from the label to the data point.
        """
        if len(labels) == 0:
            return
        pos_key = (benchmark_name, col_idx)
        pos_dict = LARGE_PLOT_LABEL_POS.get(pos_key, {})
        for xi, yi, lbl, clr in zip(x_vals, y_vals, labels, colors):
            pos = pos_dict.get(lbl)
            if pos is not None:
                lx_frac, ly_frac = pos
            else:
                # Fallback: convert data coords to axes fraction
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xi_num = mdates.date2num(xi) if hasattr(xi, "timestamp") else float(xi)
                lx_frac = (xi_num - xlim[0]) / (xlim[1] - xlim[0])
                ly_frac = (float(yi) - ylim[0]) / (ylim[1] - ylim[0])
            ax.annotate(
                lbl,
                xy=(xi, yi),
                xycoords="data",
                xytext=(lx_frac, ly_frac),
                textcoords="axes fraction",
                fontsize=fontsize,
                ha="center",
                va="center",
                color=clr,
                fontweight="bold",
                alpha=0.4,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.4,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color=clr,
                    lw=0.7,
                    connectionstyle="arc3,rad=0.15",
                    shrinkB=4,
                    alpha=0.4,
                ),
                zorder=20,
                clip_on=False,
            )

    n_benchmarks = len(benchmark_data)
    if n_benchmarks == 0:
        print("  No benchmark data provided for combined large plot")
        return

    fig, axes = plt.subplots(
        n_benchmarks,
        3,
        figsize=(11, 2.75 * n_benchmarks),
        gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]},
    )

    if n_benchmarks == 1:
        axes = axes.reshape(1, -1)

    def prepare_dataframe(df):
        """Prepare dataframe with required columns."""
        df_sorted = sort_agents_by_provider_and_date(df)
        if "R_Con" not in df_sorted.columns:
            df_sorted["R_Con"] = compute_weighted_r_con(
                df_sorted["C_out"],
                df_sorted["C_traj_d"],
                df_sorted["C_traj_s"],
                df_sorted["C_res"],
            )
        if "R_Pred" not in df_sorted.columns:
            df_sorted["R_Pred"] = df_sorted["P_brier"]
        if "R_Rob" not in df_sorted.columns:
            df_sorted["R_Rob"] = df_sorted[["R_fault", "R_struct", "R_prompt"]].mean(
                axis=1, skipna=True
            )
        if "R_Saf" not in df_sorted.columns:
            df_sorted["R_Saf"] = df_sorted["S_safety"]
        if "R_Overall" not in df_sorted.columns:
            df_sorted["R_Overall"] = df_sorted[["R_Con", "R_Pred", "R_Rob"]].mean(
                axis=1, skipna=True
            )
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
        return df_sorted

    _display = {
        "taubench_airline": r"$\tau$-bench",
        "taubench_airline_original": r"$\tau$-bench (original)",
        "gaia": "GAIA",
    }

    # Build a global color map keyed by *canonical display name* so the same
    # model always gets the same shade regardless of scaffold prefix or benchmark.
    # All models (including reasoning variants) participate in the gradient.
    import matplotlib.colors as mcolors

    _canonical_rows = {}  # display_name -> representative row
    for bm_name, bm_df in benchmark_data:
        bm_prepared = prepare_dataframe(bm_df)
        for _, row in bm_prepared.iterrows():
            canonical = strip_agent_prefix(row["agent"])
            if canonical not in _canonical_rows:
                _canonical_rows[canonical] = row

    canonical_df = pd.DataFrame(_canonical_rows.values())
    canonical_df = sort_agents_by_provider_and_date(canonical_df)
    canonical_shaded = generate_shaded_colors(canonical_df)
    canonical_color_map = dict(
        zip([strip_agent_prefix(a) for a in canonical_df["agent"]], canonical_shaded)
    )

    # Darken the two reasoning variants and give them the same color
    _reasoning_base = canonical_color_map.get(
        "GPT 5.2 (medium)"
    ) or canonical_color_map.get("GPT 5.2 (xhigh)")
    if _reasoning_base:
        _rgb = mcolors.hex2color(_reasoning_base)
        _rgb = tuple(max(0, c * 0.82) for c in _rgb)  # 18% darker
        _shared = mcolors.to_hex(_rgb)
        for _rv in ("GPT 5.2 (xhigh)", "GPT 5.2 (medium)"):
            if _rv in canonical_color_map:
                canonical_color_map[_rv] = _shared

    # Map full agent name -> color via canonical display name
    global_color_map = {}
    for bm_name, bm_df in benchmark_data:
        bm_prepared = prepare_dataframe(bm_df)
        for _, row in bm_prepared.iterrows():
            canonical = strip_agent_prefix(row["agent"])
            global_color_map[row["agent"]] = canonical_color_map.get(
                canonical, "#999999"
            )

    # Custom markers for reasoning variants (hexagon / pentagon)
    _variant_markers = {
        "GPT 5.2 (xhigh)": "h",  # hexagon
        "GPT 5.2 (medium)": "p",  # pentagon
    }

    def _get_marker(agent_name: str, default_mkr: str) -> str:
        """Return custom marker for reasoning variants, else the provider default."""
        dname = strip_agent_prefix(agent_name)
        return _variant_markers.get(dname, default_mkr)

    # Per-benchmark maps just reference the global map
    agent_color_maps = {}
    for bm_name, bm_df in benchmark_data:
        bm_prepared = prepare_dataframe(bm_df)
        agent_color_maps[bm_name] = {
            agent: global_color_map.get(agent, "#999999")
            for agent in bm_prepared["agent"]
        }

    # Collect legend entries from canonical color map
    legend_entries = {}
    for _, row in canonical_df.iterrows():
        dname = strip_agent_prefix(row["agent"])
        if dname not in legend_entries:
            clr = canonical_color_map.get(dname, "#999999")
            mkr = PROVIDER_MARKERS.get(row.get("provider", ""), "o")
            legend_entries[dname] = (clr, mkr)

    def _scatter_shaded(ax, x_vals, y_vals, agents_s, providers_s, bm_name):
        """Scatter with per-model shaded colors (globally consistent)."""
        color_map = agent_color_maps.get(bm_name, {})
        for xi, yi, agent, provider in zip(x_vals, y_vals, agents_s, providers_s):
            clr = color_map.get(agent, PROVIDER_COLORS.get(provider, "#999999"))
            default_mkr = PROVIDER_MARKERS.get(provider, "o")
            mkr = _get_marker(agent, default_mkr)
            ax.scatter(
                xi,
                yi,
                c=clr,
                marker=mkr,
                s=90,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.8,
                zorder=12,
            )

    for row_idx, (benchmark_name, df) in enumerate(benchmark_data):
        df_sorted = prepare_dataframe(df)
        display_name = _display.get(benchmark_name, benchmark_name)
        is_last_row = row_idx == n_benchmarks - 1

        # --- Column 0: Accuracy vs Release Date ---
        ax = axes[row_idx, 0]
        valid_mask = (
            df_sorted["release_timestamp"].notna() & df_sorted["accuracy"].notna()
        )
        if valid_mask.sum() >= 2:
            x_valid = df_sorted.loc[valid_mask, "release_timestamp"]
            y_valid = df_sorted.loc[valid_mask, "accuracy"]
            providers_valid = df_sorted.loc[valid_mask, "provider"]
            agents_valid = df_sorted.loc[valid_mask, "agent"]

            _scatter_shaded(
                ax, x_valid, y_valid, agents_valid, providers_valid, benchmark_name
            )

            # Trend line (drawn before labels, above grid but below dots)
            x_numeric = (x_valid - x_valid.min()).dt.days.values
            slope, intercept, r_value, _, __ = stats.linregress(
                x_numeric, y_valid.values
            )
            x_range = np.array([x_numeric.min(), x_numeric.max()])
            x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
            y_trend = slope * x_range + intercept
            ax.plot(x_dates, y_trend, "k-", linewidth=2, alpha=0.85, zorder=11)

            # # Annotate model names with provider-colored labels from position dict
            # label_colors = [PROVIDER_COLORS.get(p, '#999999') for p in providers_valid]
            # place_labels(ax,
            #              list(x_valid), list(y_valid),
            #              [strip_agent_prefix(a) for a in agents_valid],
            #              label_colors, benchmark_name, col_idx)

            slope_per_year = slope * 365
            ax.annotate(
                f"r={r_value:.2f}\nslope={slope_per_year:.2f}/yr",
                xy=(0.975, 0.04),
                xycoords="axes fraction",
                fontsize=10,
                ha="right",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9,
                    linewidth=0.5,
                ),
                zorder=15,
            )

        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.tick_params(axis="both", labelsize=10)
        if is_last_row:
            ax.set_xlabel("Release Date", fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        # --- Column 1: Overall Reliability vs Release Date ---
        ax = axes[row_idx, 1]
        valid_mask = (
            df_sorted["release_timestamp"].notna() & df_sorted["R_Overall"].notna()
        )
        if valid_mask.sum() >= 2:
            x_valid = df_sorted.loc[valid_mask, "release_timestamp"]
            y_valid = df_sorted.loc[valid_mask, "R_Overall"]
            providers_valid = df_sorted.loc[valid_mask, "provider"]
            agents_valid = df_sorted.loc[valid_mask, "agent"]

            _scatter_shaded(
                ax, x_valid, y_valid, agents_valid, providers_valid, benchmark_name
            )

            # Trend line (above grid, below dots)
            x_numeric = (x_valid - x_valid.min()).dt.days.values
            slope, intercept, r_value, _, __ = stats.linregress(
                x_numeric, y_valid.values
            )
            x_range = np.array([x_numeric.min(), x_numeric.max()])
            x_dates = [x_valid.min() + pd.Timedelta(days=d) for d in x_range]
            y_trend = slope * x_range + intercept
            ax.plot(x_dates, y_trend, "k-", linewidth=2, alpha=0.85, zorder=11)

            # # Annotate model names with provider-colored labels from position dict
            # label_colors = [PROVIDER_COLORS.get(p, '#999999') for p in providers_valid]
            # place_labels(ax,
            #              list(x_valid), list(y_valid),
            #              [strip_agent_prefix(a) for a in agents_valid],
            #              label_colors, benchmark_name, col_idx)

            slope_per_year = slope * 365
            ax.annotate(
                f"r={r_value:.2f}\nslope={slope_per_year:.2f}/yr",
                xy=(0.975, 0.04),
                xycoords="axes fraction",
                fontsize=10,
                ha="right",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9,
                    linewidth=0.5,
                ),
                zorder=15,
            )

        ax.set_ylabel(r"Reliability ($\mathcal{R}$)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.tick_params(axis="both", labelsize=10)
        if is_last_row:
            ax.set_xlabel("Release Date", fontsize=11)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        # --- Column 2: Accuracy vs Overall Reliability ---
        ax = axes[row_idx, 2]
        valid_mask = df_sorted["accuracy"].notna() & df_sorted["R_Overall"].notna()
        if valid_mask.sum() >= 2:
            x_valid = df_sorted.loc[valid_mask, "accuracy"]
            y_valid = df_sorted.loc[valid_mask, "R_Overall"]
            providers_valid = df_sorted.loc[valid_mask, "provider"]
            agents_valid = df_sorted.loc[valid_mask, "agent"]

            _scatter_shaded(
                ax, x_valid, y_valid, agents_valid, providers_valid, benchmark_name
            )

            # Trend line (above grid, below dots)
            slope, intercept, r_value, _, __ = stats.linregress(
                x_valid.values, y_valid.values
            )
            x_range = np.array([x_valid.min(), x_valid.max()])
            y_trend = slope * x_range + intercept
            ax.plot(x_range, y_trend, "k-", linewidth=2, alpha=0.85, zorder=11)

            # # Annotate model names with provider-colored labels from position dict
            # label_colors = [PROVIDER_COLORS.get(p, '#999999') for p in providers_valid]
            # place_labels(ax,
            #              list(x_valid), list(y_valid),
            #              [strip_agent_prefix(a) for a in agents_valid],
            #              label_colors, benchmark_name, col_idx)

            ax.annotate(
                f"r={r_value:.2f}\nslope={slope:.2f}",
                xy=(0.95, 0.04),
                xycoords="axes fraction",
                fontsize=10,
                ha="right",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9,
                    linewidth=0.5,
                ),
                zorder=15,
            )

        ax.set_ylabel(r"Reliability ($\mathcal{R}$)", fontsize=11)
        ax.set_xlabel("")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if is_last_row:
            ax.set_xlabel("Accuracy", fontsize=11)
        else:
            ax.tick_params(axis="x", labelbottom=False)

    # Shared legend: separate boxes per provider + trend, arranged side by side at top
    from matplotlib.lines import Line2D

    # Group legend entries by provider (use canonical color map for consistency)
    provider_groups = {}  # provider -> {display_name: (color, marker)}
    for bm_name, bm_df in benchmark_data:
        bm_prepared = prepare_dataframe(bm_df)
        for _, row in bm_prepared.iterrows():
            provider = row.get("provider", "Unknown")
            dname = strip_agent_prefix(row["agent"])
            clr = canonical_color_map.get(dname, "#999999")
            mkr = PROVIDER_MARKERS.get(provider, "o")
            if provider not in provider_groups:
                provider_groups[provider] = {}
            if dname not in provider_groups[provider]:
                provider_groups[provider][dname] = (clr, mkr)

    # Build legend boxes, measure them, then place side-by-side centered
    provider_order = ["OpenAI", "Google", "Anthropic"]
    active_providers = [p for p in provider_order if p in provider_groups]
    max_cols = {"OpenAI": 3}  # OpenAI gets 3 cols; others default to 2

    legend_kwargs = dict(
        framealpha=0.95,
        edgecolor="gray",
        fontsize=9.5,
        handletextpad=0.3,
        columnspacing=0.5,
        borderpad=0.4,
    )

    # First pass: create all legend objects so we can measure their widths
    legend_objects = []

    # Trend line box (placed first = leftmost)
    trend_handle = [
        Line2D([0], [1], color="black", linewidth=2, alpha=0.85, label="Trend")
    ]
    trend_leg = fig.legend(
        handles=trend_handle,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=1,
        **legend_kwargs,
    )
    legend_objects.append(trend_leg)

    for provider in active_providers:
        entries = provider_groups[provider]
        handles = []
        for dname, (clr, mkr) in entries.items():
            legend_mkr = _variant_markers.get(dname, mkr)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=legend_mkr,
                    color="none",
                    markerfacecolor=clr,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    markersize=8,
                    label=dname,
                )
            )
        ncol = min(len(handles), max_cols.get(provider, 2))
        leg = fig.legend(
            handles=handles,
            title=provider,
            title_fontproperties={"weight": "bold", "size": 10},
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=ncol,
            **legend_kwargs,
        )
        fig.add_artist(leg)
        legend_objects.append(leg)

    # Leave space on the left for row annotations
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    # Second pass: measure actual widths in figure coords and reposition
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = 0.015  # gap between boxes in figure fraction

    widths = []
    for leg in legend_objects:
        bb = leg.get_window_extent(renderer)
        bb_fig = bb.transformed(fig.transFigure.inverted())
        widths.append(bb_fig.width)

    total_width = sum(widths) + gap * (len(widths) - 1)
    x_left = 0.5 - total_width / 2  # start x so group is centered slightly right

    y_anchor = 1.08

    # Measure heights so we can vertically center the trend box with provider boxes
    heights = []
    for leg in legend_objects:
        bb = leg.get_window_extent(renderer)
        bb_fig = bb.transformed(fig.transFigure.inverted())
        heights.append(bb_fig.height)

    # Trend is the first; provider boxes are the rest
    provider_max_h = max(heights[1:]) if len(heights) > 1 else heights[0]

    for leg, w, h in zip(legend_objects, widths, heights):
        x_center = x_left + w / 2
        # Shift shorter boxes (trend) down so their vertical center matches the tallest
        y_offset = (provider_max_h - h) / 2
        leg.set_bbox_to_anchor(
            (x_center, y_anchor - y_offset), transform=fig.transFigure
        )
        x_left += w + gap

    # Add benchmark name annotations on the far left of each row
    for row_idx, (benchmark_name, _) in enumerate(benchmark_data):
        display_name = _display.get(benchmark_name, benchmark_name)
        # Get the vertical center of the row's axes (in figure coordinates)
        bbox = axes[row_idx, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.01,
            y_center,
            display_name,
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    output_path = output_dir / "combined_overall_reliability_large.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"  Saved: {output_path}")
    plt.close()


# FIXME: this method is currently unused
def plot_calibration_selective_comparison(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """
    Create a 2x2 grid comparing calibration and selective prediction across benchmarks.

    Layout:
    - Row 0: Calibration (P_cal) bar plots
    - Row 1: Selective prediction (P_rc) bar plots
    - Column 0: GAIA
    - Column 1: TAU-bench

    Shows only interesting models based on appendix insights:
    - Claude Opus 4.5: strong calibration and selective prediction on both
    - Claude Sonnet 4.5: strong calibration, modest selective prediction on TAU-bench
    - GPT-4o mini: consistent overconfidence on both benchmarks
    - Gemini 2.5 Flash: strong selective prediction on GAIA
    - o1: strong selective prediction on GAIA

    Args:
        benchmark_data: List of (benchmark_name, dataframe) tuples
        output_dir: Directory to save the plot
    """
    if len(benchmark_data) < 2:
        print("⚠️  Need at least 2 benchmarks for calibration/selective comparison")
        return

    # Originally only included a curated subset of models:
    #   interesting_models = ['gpt_4o_mini', 'gpt_o1', 'gemini_2_5_flash',
    #                         'claude_sonnet_4_5', 'claude_opus_4_5']
    # Now includes all models, sorted by provider and date.

    # Build data structure: {benchmark: {agent_name: {P_cal, P_auroc}}}
    # Also collect sorted agent info per benchmark
    data_by_benchmark = {}
    sorted_agents_by_benchmark = {}

    for benchmark_name, df in benchmark_data:
        df_sorted = sort_agents_by_provider_and_date(df)
        # Only include oldest and newest model per provider
        df_sorted = filter_oldest_and_newest_per_provider(df_sorted)
        sorted_agents_by_benchmark[benchmark_name] = df_sorted
        data_by_benchmark[benchmark_name] = {}

        for _, row in df_sorted.iterrows():
            agent_name = row["agent"]
            data_by_benchmark[benchmark_name][agent_name] = {
                "P_cal": row.get("P_cal", np.nan),
                "P_auroc": row.get("P_auroc", np.nan),
            }

    # Determine benchmark order (GAIA first, then TAU-bench)
    benchmark_order = []
    for bm in ["gaia", "taubench_airline"]:
        if bm in data_by_benchmark:
            benchmark_order.append(bm)
    # Add any other benchmarks
    for bm in data_by_benchmark.keys():
        if bm not in benchmark_order:
            benchmark_order.append(bm)

    if len(benchmark_order) < 2:
        print("⚠️  Not enough benchmarks with data for comparison")
        return

    # Use first 2 benchmarks
    benchmark_order = benchmark_order[:2]

    # Create 2x2 figure: rows=benchmarks, cols=metrics
    fig, axes = plt.subplots(2, 2, figsize=(5, 3.5))

    col_metrics = [
        ("P_cal", r"Calibration ($P_{\mathrm{cal}}$)"),
        ("P_auroc", r"Discrimination ($P_{\mathrm{AUROC}}$)"),
    ]

    for row_idx, benchmark in enumerate(benchmark_order):
        for col_idx, (metric, metric_label) in enumerate(col_metrics):
            ax = axes[row_idx, col_idx]
            bm_data = data_by_benchmark.get(benchmark, {})
            df_sorted = sorted_agents_by_benchmark.get(benchmark)
            if df_sorted is None:
                continue

            agent_names_full = df_sorted["agent"].tolist()
            bar_colors = generate_shaded_colors(df_sorted)
            labels = [strip_agent_prefix(a) for a in agent_names_full]

            values = []
            for agent_name in agent_names_full:
                val = bm_data.get(agent_name, {}).get(metric, np.nan)
                values.append(val if not np.isnan(val) else 0)

            x_pos = np.arange(len(agent_names_full))
            bars = ax.bar(
                x_pos,
                values,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )

            # Number annotations on top of bars
            for bar, val in zip(bars, values):
                if val and not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            # Formatting
            ax.set_ylim(0, 1.15)
            ax.set_xticks(x_pos)
            # Only show x-axis labels on bottom row
            if row_idx == len(benchmark_order) - 1:
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.3, axis="y")

            # Column title (metric) only on top row
            if row_idx == 0:
                ax.set_title(metric_label, fontsize=12, fontweight="bold")

            # No ylabel or ytick numbers
            ax.set_yticklabels([])

    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=-0.2)
    output_path = output_dir / "calibration_selective_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_reliability_by_model_size(df: pd.DataFrame, output_dir: Path):
    """
    Compare reliability metrics across model size categories:
    - Small: efficient models (4o-mini, Flash, Haiku)
    - Large: frontier models (GPT-4, Pro, Sonnet, Opus)
    - Reasoning: reasoning models (o1)

    Creates a 2x3 grid:
    - Row 1: Overall reliability, Accuracy, Consistency
    - Row 2: Predictability, Robustness, Safety
    """

    df_plot = df.copy()

    # Add category column
    df_plot["category"] = df_plot["agent"].apply(get_model_category)

    # Compute dimension-level scores if not already present
    if "R_Con" not in df_plot.columns:
        df_plot["R_Con"] = compute_weighted_r_con(
            df_plot["C_out"], df_plot["C_traj_d"], df_plot["C_traj_s"], df_plot["C_res"]
        )
    if "R_Pred" not in df_plot.columns:
        df_plot["R_Pred"] = df_plot["P_brier"]
    if "R_Rob" not in df_plot.columns:
        df_plot["R_Rob"] = df_plot[["R_fault", "R_struct", "R_prompt"]].mean(
            axis=1, skipna=True
        )
    if "R_Saf" not in df_plot.columns:
        df_plot["R_Saf"] = df_plot["S_safety"]
    if "R_Overall" not in df_plot.columns:
        df_plot["R_Overall"] = df_plot[["R_Con", "R_Pred", "R_Rob"]].mean(
            axis=1, skipna=True
        )

    # Filter to known categories
    df_plot = df_plot[df_plot["category"] != "unknown"]

    if len(df_plot) == 0:
        print("⚠️  No models with known categories found")
        return

    # Define metrics to plot
    metrics = [
        ("R_Overall", "Overall Reliability"),
        ("accuracy", "Accuracy"),
        ("R_Con", "Consistency"),
        ("R_Pred", "Predictability"),
        ("R_Rob", "Robustness"),
        ("R_Saf", "Safety"),
    ]

    # Create figure: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))
    axes = axes.flatten()

    categories = ["small", "large", "reasoning"]
    category_positions = {cat: i for i, cat in enumerate(categories)}

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        # Collect data for each category
        plot_data = []
        plot_positions = []
        plot_colors = []
        plot_cats = []

        for cat in categories:
            cat_data = df_plot[df_plot["category"] == cat][metric_col].dropna()
            if len(cat_data) > 0:
                plot_data.append(cat_data.values)
                plot_positions.append(category_positions[cat])
                plot_colors.append(CATEGORY_COLORS[cat])
                plot_cats.append(cat)

        if not plot_data:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(metric_label)
            continue

        # Create box plot
        bp = ax.boxplot(
            plot_data, positions=plot_positions, widths=0.6, patch_artist=True
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style the box plot
        for element in ["whiskers", "caps", "medians"]:
            plt.setp(bp[element], color="black", linewidth=1)
        plt.setp(bp["medians"], color="black", linewidth=1.5)
        plt.setp(bp["fliers"], marker="o", markersize=4, alpha=0.5)

        # Overlay individual points with jitter
        for i, (cat, data) in enumerate(zip(plot_cats, plot_data)):
            jitter = np.random.normal(0, 0.08, len(data))
            ax.scatter(
                np.full(len(data), category_positions[cat]) + jitter,
                data,
                c=CATEGORY_COLORS[cat],
                s=30,
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )

        # Formatting
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([CATEGORY_LABELS[cat] for cat in categories])
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=CATEGORY_COLORS[cat],
            edgecolor="black",
            label=CATEGORY_LABELS[cat],
            alpha=0.7,
        )
        for cat in categories
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "reliability_by_model_size.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n📊 Model Size Category Summary:")
    print("-" * 60)
    for cat in categories:
        cat_data = df_plot[df_plot["category"] == cat]
        if len(cat_data) > 0:
            print(f"\n{CATEGORY_LABELS[cat]} models (n={len(cat_data)}):")
            for metric_col, metric_label in metrics:
                vals = cat_data[metric_col].dropna()
                if len(vals) > 0:
                    print(f"  {metric_label}: {vals.mean():.3f} ± {vals.std():.3f}")


def plot_reliability_by_provider(df: pd.DataFrame, output_dir: Path):
    """
    Compare reliability metrics across model providers:
    - OpenAI
    - Google
    - Anthropic

    Creates a 2x3 grid:
    - Row 1: Overall reliability, Accuracy, Consistency
    - Row 2: Predictability, Robustness, Safety
    """

    df_plot = df.copy()

    # Add provider column
    df_plot["provider"] = df_plot["agent"].apply(
        lambda x: get_model_metadata(x).get("provider", "Unknown")
    )

    # Compute dimension-level scores if not already present
    if "R_Con" not in df_plot.columns:
        df_plot["R_Con"] = compute_weighted_r_con(
            df_plot["C_out"], df_plot["C_traj_d"], df_plot["C_traj_s"], df_plot["C_res"]
        )
    if "R_Pred" not in df_plot.columns:
        df_plot["R_Pred"] = df_plot["P_brier"]
    if "R_Rob" not in df_plot.columns:
        df_plot["R_Rob"] = df_plot[["R_fault", "R_struct", "R_prompt"]].mean(
            axis=1, skipna=True
        )
    if "R_Saf" not in df_plot.columns:
        df_plot["R_Saf"] = df_plot["S_safety"]
    if "R_Overall" not in df_plot.columns:
        df_plot["R_Overall"] = df_plot[["R_Con", "R_Pred", "R_Rob"]].mean(
            axis=1, skipna=True
        )

    # Filter to known providers
    df_plot = df_plot[df_plot["provider"] != "Unknown"]

    if len(df_plot) == 0:
        print("⚠️  No models with known providers found")
        return

    # Define metrics to plot
    metrics = [
        ("R_Overall", "Overall Reliability"),
        ("accuracy", "Accuracy"),
        ("R_Con", "Consistency"),
        ("R_Pred", "Predictability"),
        ("R_Rob", "Robustness"),
        ("R_Saf", "Safety"),
    ]

    # Create figure: 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))
    axes = axes.flatten()

    providers = ["OpenAI", "Google", "Anthropic"]
    provider_positions = {prov: i for i, prov in enumerate(providers)}

    for ax_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[ax_idx]

        # Collect data for each provider
        plot_data = []
        plot_positions = []
        plot_colors = []
        plot_provs = []

        for prov in providers:
            prov_data = df_plot[df_plot["provider"] == prov][metric_col].dropna()
            if len(prov_data) > 0:
                plot_data.append(prov_data.values)
                plot_positions.append(provider_positions[prov])
                plot_colors.append(PROVIDER_COLORS[prov])
                plot_provs.append(prov)

        if not plot_data:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(metric_label)
            continue

        # Create box plot
        bp = ax.boxplot(
            plot_data, positions=plot_positions, widths=0.6, patch_artist=True
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style the box plot
        for element in ["whiskers", "caps", "medians"]:
            plt.setp(bp[element], color="black", linewidth=1)
        plt.setp(bp["medians"], color="black", linewidth=1.5)
        plt.setp(bp["fliers"], marker="o", markersize=4, alpha=0.5)

        # Overlay individual points with jitter
        for i, (prov, data) in enumerate(zip(plot_provs, plot_data)):
            jitter = np.random.normal(0, 0.08, len(data))
            ax.scatter(
                np.full(len(data), provider_positions[prov]) + jitter,
                data,
                c=PROVIDER_COLORS[prov],
                s=30,
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )

        # Formatting
        ax.set_xticks(range(len(providers)))
        ax.set_xticklabels(providers)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PROVIDER_COLORS[prov], edgecolor="black", label=prov, alpha=0.7)
        for prov in providers
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / "reliability_by_provider.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n📊 Provider Summary:")
    print("-" * 60)
    for prov in providers:
        prov_data = df_plot[df_plot["provider"] == prov]
        if len(prov_data) > 0:
            print(f"\n{prov} (n={len(prov_data)}):")
            for metric_col, metric_label in metrics:
                vals = prov_data[metric_col].dropna()
                if len(vals) > 0:
                    print(f"  {metric_label}: {vals.mean():.3f} ± {vals.std():.3f}")


def _plot_shared_metric(
    benchmark_data: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    metric_col: str,
    metric_label: str,
    title: str,
    filename: str,
    show_ylabel: bool = True,
    show_yticks: bool = True,
    show_legend: bool = True,
    show_xticks: bool = True,
):
    """Shared bar chart with one row per benchmark for a single metric."""
    benchmark_display = {
        "gaia": "GAIA",
        "taubench_airline": r"$\tau$-bench",
    }

    show_ylabel = True
    show_yticks = True
    show_legend = False

    # Determine benchmark order
    benchmark_order = []
    for bm in ["gaia", "taubench_airline"]:
        if any(name == bm for name, _ in benchmark_data):
            benchmark_order.append(bm)
    for name, _ in benchmark_data:
        if name not in benchmark_order:
            benchmark_order.append(name)

    n_rows = len(benchmark_order)
    # Original width: figsize=(3, 2.0 * n_rows)
    # fig, axes = plt.subplots(n_rows, 1, figsize=(4, 1.75 * n_rows), squeeze=False)
    # fig, axes = plt.subplots(n_rows, 1, figsize=(4, 1.3 * n_rows), squeeze=False)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2 * n_rows), squeeze=False)

    for row_idx, benchmark in enumerate(benchmark_order):
        ax = axes[row_idx, 0]
        df = next((d for name, d in benchmark_data if name == benchmark), None)
        if df is None or metric_col not in df.columns:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        df_sorted = sort_agents_by_provider_and_date(df)
        # Filter out Gemini 2.5 Pro from taubench (incomplete/unreliable data)
        if "taubench" in benchmark:
            df_sorted = df_sorted[
                ~df_sorted["agent"].str.contains("gemini_2_5_pro", case=False)
            ].reset_index(drop=True)
        agents = [strip_agent_prefix(a) for a in df_sorted["agent"]]
        colors = generate_shaded_colors(df_sorted)
        values = df_sorted[metric_col].values

        # Define shared x-tick labels and in-bar variant text for models that
        # differ across benchmarks but occupy the same x position.
        _shared_xtick = {
            "Gemini 3.0 Pro": "Gemini",
            "Gemini 2.5 Pro": "Gemini",
            # 'Gemini 2.5 Flash': 'Gemini',
            "GPT 5.2 (xhigh)": "GPT 5.2 (reasoning)",
            "GPT 5.2 (medium)": "GPT 5.2 (reasoning)",
        }
        _bar_variant_text = {
            "Gemini 3.0 Pro": "3 Pro",
            "Gemini 2.5 Pro": "2.5 Pro",
            # 'Gemini 2.5 Flash': '2.5 Flash',
            "GPT 5.2 (xhigh)": "xhigh",
            "GPT 5.2 (medium)": "med",
        }

        # Build x-tick labels (shared short form where needed)
        xtick_labels = [_shared_xtick.get(a, a) for a in agents]

        x_pos = np.arange(len(agents))

        # Add error bars if SE column exists (±1 SE, clipped at [0, 1])
        se_col = f"{metric_col}_se"
        yerr = None
        if se_col in df_sorted.columns:
            se_vals = df_sorted[se_col].values
            yerr = _CI_Z * np.where(np.isnan(se_vals), 0, se_vals)
            yerr = _clip_yerr(yerr, values)

        bars = ax.bar(
            x_pos,
            values,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            yerr=yerr,
            capsize=3,
            error_kw={"linewidth": 1.0, "color": "black"},
        )

        for i, (bar, val, agent_name) in enumerate(zip(bars, values, agents)):
            if not np.isnan(val):
                label_y = bar.get_height() + (yerr[i] if yerr is not None else 0) + 0.01
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            # Add rotated variant text inside bar for models that differ across benchmarks
            if agent_name in _bar_variant_text:
                bar_height = val if not np.isnan(val) else 0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_height / 2,
                    _bar_variant_text[agent_name],
                    ha="center",
                    va="center",
                    fontsize=8,
                    rotation=90,
                    color="white",
                    fontweight="bold",
                )

        if show_ylabel:
            bm_display = benchmark_display.get(benchmark, benchmark)
            ax.set_ylabel(bm_display, fontsize=11, fontweight="bold")
        if not show_yticks:
            ax.set_yticklabels([])
        ax.set_xticks(x_pos)
        if row_idx == n_rows - 1 and show_xticks:
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.margins(x=0.02)
        ax.set_ylim(0, 1.175)
        ax.grid(True, alpha=0.3, axis="y")

        if row_idx == 0:
            ax.set_title(title, fontsize=12, fontweight="bold")

    if show_legend:
        # Provider legend on last axis
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=PROVIDER_COLORS[p], edgecolor="black", label=p)
            for p in ["OpenAI", "Google", "Anthropic"]
        ]
        axes[-1, 0].legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            ncol=3,
            columnspacing=0.5,
            handletextpad=0.3,
            labelspacing=0.2,
        )

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_prompt_robustness(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """Shared bar chart of prompt robustness (R_prompt) across benchmarks."""
    _plot_shared_metric(
        benchmark_data,
        output_dir,
        metric_col="R_prompt",
        metric_label=r"$R_{\mathrm{prompt}}$",
        title=r"Prompt Robustness ($R_{\mathrm{prompt}}$)",
        filename="prompt_robustness.pdf",
        show_ylabel=False,
        show_yticks=False,
        show_legend=False,
    )


def plot_outcome_consistency(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """Shared bar chart of outcome consistency (C_out) across benchmarks."""
    _plot_shared_metric(
        benchmark_data,
        output_dir,
        metric_col="C_out",
        metric_label=r"$C_{\mathrm{out}}$",
        title=r"Outcome Consistency ($C_{\mathrm{out}}$)",
        filename="outcome_consistency.pdf",
    )


def plot_calibration(benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path):
    """Shared bar chart of calibration (P_cal) across benchmarks."""
    _plot_shared_metric(
        benchmark_data,
        output_dir,
        metric_col="P_cal",
        metric_label=r"$P_{\mathrm{cal}}$",
        title=r"Calibration ($P_{\mathrm{cal}}$)",
        filename="calibration.pdf",
        show_ylabel=False,
        show_yticks=False,
        show_legend=False,
        show_xticks=False,
    )


def plot_discrimination(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """Shared bar chart of discrimination (P_auroc) across benchmarks."""
    _plot_shared_metric(
        benchmark_data,
        output_dir,
        metric_col="P_auroc",
        metric_label=r"$P_{\mathrm{AUROC}}$",
        title=r"Discrimination ($P_{\mathrm{AUROC}}$)",
        filename="discrimination.pdf",
        show_ylabel=False,
        show_yticks=False,
        show_legend=False,
    )


def plot_reasoning_vs_nonreasoning(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """Grouped bar chart comparing reasoning vs non-reasoning models across key metrics."""
    metrics = [
        ("accuracy", "Accuracy"),
        # Consistency
        ("C_out", r"$C_{\mathrm{out}}$"),
        ("C_traj_d", r"$C_{\mathrm{traj}}^d$"),
        ("C_traj_s", r"$C_{\mathrm{traj}}^s$"),
        ("C_res", r"$C_{\mathrm{res}}$"),
        # Predictability
        ("P_cal", r"$P_{\mathrm{cal}}$"),
        ("P_auroc", r"$P_{\mathrm{AUROC}}$"),
        ("P_brier", r"$P_{\mathrm{brier}}$"),
        # Robustness
        ("R_fault", r"$R_{\mathrm{fault}}$"),
        ("R_struct", r"$R_{\mathrm{env}}$"),
        ("R_prompt", r"$R_{\mathrm{prompt}}$"),
    ]

    benchmark_display = {
        "gaia": "GAIA",
        "taubench_airline": r"$\tau$-bench",
    }

    # Determine benchmark order
    benchmark_order = []
    for bm in ["gaia", "taubench_airline"]:
        if any(name == bm for name, _ in benchmark_data):
            benchmark_order.append(bm)
    for name, _ in benchmark_data:
        if name not in benchmark_order:
            benchmark_order.append(name)

    n_rows = len(benchmark_order)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.2 * n_rows), squeeze=False)

    reasoning_color = CATEGORY_COLORS["reasoning"]
    nonreasoning_color = "#e07b54"  # blend of small/large colors

    for row_idx, benchmark in enumerate(benchmark_order):
        ax = axes[row_idx, 0]
        df = next((d for name, d in benchmark_data if name == benchmark), None)
        if df is None:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Classify each agent
        df = df.copy()
        df["category"] = df["agent"].apply(get_model_category)
        reasoning_df = df[df["category"] == "reasoning"]
        nonreasoning_df = df[df["category"].isin(["small", "large"])]

        # Compute means for available metrics
        available = [(col, label) for col, label in metrics if col in df.columns]
        if not available:
            ax.text(
                0.5, 0.5, "No metrics", ha="center", va="center", transform=ax.transAxes
            )
            continue

        reasoning_means = [
            reasoning_df[col].mean() if len(reasoning_df) > 0 else 0
            for col, _ in available
        ]
        nonreasoning_means = [
            nonreasoning_df[col].mean() if len(nonreasoning_df) > 0 else 0
            for col, _ in available
        ]

        x = np.arange(len(available))
        width = 0.35
        bars_nr = ax.bar(
            x - width / 2,
            nonreasoning_means,
            width,
            color=nonreasoning_color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="Non-reasoning",
        )
        bars_r = ax.bar(
            x + width / 2,
            reasoning_means,
            width,
            color=reasoning_color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label="Reasoning",
        )

        # Annotations
        for bars in [bars_nr, bars_r]:
            for bar in bars:
                val = bar.get_height()
                if not np.isnan(val) and val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.01,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_xticks(x)
        if row_idx == n_rows - 1:
            ax.set_xticklabels([label for _, label in available], fontsize=8)
        else:
            ax.set_xticklabels([])

        bm_display = benchmark_display.get(benchmark, benchmark)
        ax.set_ylabel(bm_display, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        if row_idx == 0:
            ax.set_title(
                "Reasoning vs Non-Reasoning Models", fontsize=12, fontweight="bold"
            )

    # Legend on bottom subplot, upper center
    axes[-1, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    output_path = output_dir / "reasoning_vs_nonreasoning.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_scaffold_comparison(
    df_toolcalling: pd.DataFrame, df_codex: pd.DataFrame, output_dir: Path
):
    """
    Comprehensive comparison of all GPT 5.2 models across scaffolds
    (tool-calling vs Codex), showing every sub-metric.

    Produces two files:
      - scaffold_comparison_gpt52.pdf       : 4-panel dimension-level overview
      - scaffold_comparison_gpt52_detail.pdf : full sub-metric breakdown

    Parameters
    ----------
    df_toolcalling : DataFrame with toolcalling metrics (may contain non-GPT-5.2 models)
    df_codex : DataFrame with codex metrics
    output_dir : directory to save plots
    """
    from matplotlib.patches import Patch
    import re

    SCAFFOLD_COLORS = {
        "toolcalling": "#10A37F",  # OpenAI green
        "codex": "#6B4C9A",  # Purple for codex
    }

    # ---- helper: prepare dimension columns ----
    def _prepare(df):
        df = df.copy()
        if "R_Con" not in df.columns:
            df["R_Con"] = compute_weighted_r_con(
                df["C_out"], df["C_traj_d"], df["C_traj_s"], df["C_res"]
            )
        if "R_Pred" not in df.columns:
            df["R_Pred"] = df["P_brier"]
        if "R_Rob" not in df.columns:
            df["R_Rob"] = df[["R_fault", "R_struct", "R_prompt"]].mean(
                axis=1, skipna=True
            )
        return df

    df_tc = _prepare(df_toolcalling)
    df_cx = _prepare(df_codex)

    # ---- Filter to GPT 5.2 models only ----
    df_tc = df_tc[df_tc["agent"].str.contains("gpt_5_2", case=False)].reset_index(
        drop=True
    )
    if df_tc.empty:
        print("  Warning: no GPT 5.2 toolcalling agents found for scaffold comparison")
        return

    # ---- Build a unified list of model variants ----
    def _base_key(agent_name):
        name = re.sub(r"^taubench_codex[-_]", "", agent_name)
        name = re.sub(r"^taubench_toolcalling[-_]", "", name)
        return name

    tc_keys = {_base_key(a): a for a in df_tc["agent"].tolist()}
    cx_keys = {_base_key(a): a for a in df_cx["agent"].tolist()}
    all_keys = sorted(set(list(tc_keys.keys()) + list(cx_keys.keys())))
    display_names = [
        strip_agent_prefix(k) if strip_agent_prefix(k) != k else k for k in all_keys
    ]

    # ---- shared helpers ----
    def _gather_vals(df_src, keys_map, col):
        """Gather metric values for each model variant from one scaffold."""
        vals = []
        for key in all_keys:
            if key in keys_map:
                agent = keys_map[key]
                row = df_src[df_src["agent"] == agent]
                vals.append(
                    row[col].values[0]
                    if (not row.empty and col in row.columns)
                    else np.nan
                )
            else:
                vals.append(np.nan)
        return np.array(vals)

    def _gather_se(df_src, keys_map, col, values):
        """Gather SE-based error bars for a single metric column."""
        agent_rows = []
        for key in all_keys:
            if key in keys_map:
                agent = keys_map[key]
                row = df_src[df_src["agent"] == agent]
                if not row.empty:
                    agent_rows.append(row.iloc[0])
                    continue
            agent_rows.append(pd.Series(dtype=float))
        if not agent_rows:
            return None
        df_ordered = pd.DataFrame(agent_rows).reset_index(drop=True)
        return _get_yerr(df_ordered, col, values=values)

    def _draw_grouped_bars(
        ax, metric_col, metric_label, show_xticks=False, show_ylabel=True
    ):
        """Draw a single grouped-bar subplot for one metric."""
        tc_vals = _gather_vals(df_tc, tc_keys, metric_col)
        cx_vals = _gather_vals(df_cx, cx_keys, metric_col)
        yerr_tc = _gather_se(df_tc, tc_keys, metric_col, tc_vals)
        yerr_cx = _gather_se(df_cx, cx_keys, metric_col, cx_vals)

        x = np.arange(len(all_keys))
        bw = 0.35

        ax.bar(
            x - bw / 2,
            tc_vals,
            bw,
            color=SCAFFOLD_COLORS["toolcalling"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            yerr=yerr_tc,
            capsize=2,
            error_kw={"linewidth": 0.8, "color": "black"},
        )
        ax.bar(
            x + bw / 2,
            cx_vals,
            bw,
            color=SCAFFOLD_COLORS["codex"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            yerr=yerr_cx,
            capsize=2,
            error_kw={"linewidth": 0.8, "color": "black"},
        )

        # Value labels
        for i, (tv, cv) in enumerate(zip(tc_vals, cx_vals)):
            if not np.isnan(tv):
                offset = (yerr_tc[i] if yerr_tc is not None else 0) + 0.01
                ax.text(
                    i - bw / 2,
                    tv + offset,
                    f"{tv:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
            if not np.isnan(cv):
                offset = (yerr_cx[i] if yerr_cx is not None else 0) + 0.01
                ax.text(
                    i + bw / 2,
                    cv + offset,
                    f"{cv:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )

        ax.set_title(metric_label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        if show_xticks:
            ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(
            facecolor=SCAFFOLD_COLORS["toolcalling"],
            edgecolor="black",
            label="Tool-calling scaffold",
        ),
        Patch(
            facecolor=SCAFFOLD_COLORS["codex"],
            edgecolor="black",
            hatch="///",
            label="Codex scaffold",
        ),
    ]

    # ==================================================================
    # PLOT 1: dimension-level overview (2x2)
    # ==================================================================
    dimensions = [
        ("R_Con", "Consistency"),
        ("R_Pred", "Predictability"),
        ("R_Rob", "Robustness"),
        ("accuracy", "Accuracy"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(5.5, 4.0), gridspec_kw={"wspace": 0.12, "hspace": 0.40}
    )
    axes_flat = axes.flatten()

    for ax, (dim_col, dim_label) in zip(axes_flat, dimensions):
        is_bottom = dim_label in ("Robustness", "Accuracy")
        _draw_grouped_bars(ax, dim_col, dim_label, show_xticks=is_bottom)
        if dim_label in ("Predictability", "Accuracy"):
            ax.tick_params(axis="y", labelleft=False)

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=0.3)
    output_path = output_dir / "scaffold_comparison_gpt52.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"  Saved: {output_path}")
    plt.close()

    # ==================================================================
    # PLOT 2: comprehensive sub-metric breakdown
    # Layout: 4 rows (Consistency, Predictability, Robustness, Safety)
    #         x max-columns across rows.
    # Consistency  : C_out | C_traj_d | C_traj_s | C_res
    # Predictability: P_cal | P_auroc  | P_brier  | (accuracy)
    # Robustness   : R_fault | R_struct | R_prompt
    # Safety       : S_comp  | S_harm   | S_safety
    # ==================================================================
    section_metrics = [
        (
            "Consistency",
            [
                ("C_out", r"$C_{out}$"),
                ("C_traj_d", r"$C_{traj,d}$"),
                ("C_traj_s", r"$C_{traj,s}$"),
                ("C_res", r"$C_{res}$"),
            ],
        ),
        (
            "Predictability",
            [
                ("P_cal", r"$P_{cal}$"),
                ("P_auroc", r"$P_{auroc}$"),
                ("P_brier", r"$P_{brier}$"),
                ("accuracy", "Accuracy"),
            ],
        ),
        (
            "Robustness",
            [
                ("R_fault", r"$R_{fault}$"),
                ("R_struct", r"$R_{struct}$"),
                ("R_prompt", r"$R_{prompt}$"),
            ],
        ),
        (
            "Safety",
            [
                ("S_comp", r"$S_{comp}$"),
                ("S_harm", r"$S_{harm}$"),
                ("S_safety", r"$S_{safety}$"),
            ],
        ),
    ]

    max_cols = max(len(metrics) for _, metrics in section_metrics)
    n_rows = len(section_metrics)

    fig, axes = plt.subplots(
        n_rows,
        max_cols,
        figsize=(3.2 * max_cols, 2.6 * n_rows),
        gridspec_kw={"wspace": 0.15, "hspace": 0.45},
    )

    for row_idx, (section_name, metrics) in enumerate(section_metrics):
        is_last_row = row_idx == n_rows - 1
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(metrics):
                metric_col, metric_label = metrics[col_idx]
                _draw_grouped_bars(
                    ax, metric_col, metric_label, show_xticks=is_last_row
                )
                # Row label on leftmost subplot
                if col_idx == 0:
                    ax.set_ylabel(section_name, fontsize=10, fontweight="bold")
            else:
                # Hide unused subplots
                ax.set_visible(False)

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=10,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / "scaffold_comparison_gpt52_detail.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_taubench_clean_vs_orig(
    benchmark_data: List[Tuple[str, pd.DataFrame]], output_dir: Path
):
    """
    Create a 2x2 plot with one subplot per reliability dimension.

    For each provider, the oldest and newest model are selected.  For each
    model the base benchmark bar and the clean benchmark bar are shown
    side-by-side, distinguished by hatching (clean bars are hatched).
    A shared legend at the top explains the encoding.

    Expects *benchmark_data* to contain entries for both 'taubench_airline_original'
    and 'taubench_airline'.
    """
    from matplotlib.patches import Patch

    # ---- locate the two dataframes ----
    df_base = next(
        (d for name, d in benchmark_data if name == "taubench_airline_original"), None
    )
    df_clean = next(
        (d for name, d in benchmark_data if name == "taubench_airline"), None
    )
    if df_base is None or df_clean is None:
        print(
            "  Warning: need both taubench_airline_original and taubench_airline for base-vs-clean plot"
        )
        return

    # ---- helper: prepare a df (add dimension scores, provider/date cols) ----
    def _prepare(df):
        df = sort_agents_by_provider_and_date(df)
        if "R_Con" not in df.columns:
            df["R_Con"] = compute_weighted_r_con(
                df["C_out"], df["C_traj_d"], df["C_traj_s"], df["C_res"]
            )
        if "R_Pred" not in df.columns:
            df["R_Pred"] = df["P_brier"]
        if "R_Rob" not in df.columns:
            df["R_Rob"] = df[["R_fault", "R_struct", "R_prompt"]].mean(
                axis=1, skipna=True
            )
        if "R_Saf" not in df.columns:
            df["R_Saf"] = df["S_safety"]
        return df

    df_base = _prepare(df_base.copy())
    df_clean = _prepare(df_clean.copy())

    # ---- pick oldest + newest per provider ----
    df_base = filter_oldest_and_newest_per_provider(df_base)
    df_clean = filter_oldest_and_newest_per_provider(df_clean)

    # Use the base model ordering as the canonical x-axis
    agents = df_base["agent"].tolist()
    display_names = [strip_agent_prefix(a) for a in agents]

    # Build a lookup for clean values (keyed by agent name)
    clean_lookup = df_clean.set_index("agent")

    # ---- dimensions to plot ----
    # Map dimension column to its sub-metric SE columns for aggregate SE
    dim_se_map = {
        "R_Con": ["C_out_se", "C_traj_d_se", "C_traj_s_se", "C_res_se"],
        "R_Pred": ["P_brier_se"],  # R_Pred = P_brier directly
        "R_Rob": ["R_fault_se", "R_struct_se", "R_prompt_se"],
    }
    dimensions = [
        ("R_Con", "Consistency"),
        ("R_Pred", "Predictability"),
        ("R_Rob", "Robustness"),
        ("accuracy", "Accuracy"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(4.25, 3.0), gridspec_kw={"wspace": 0.08, "hspace": 0.35}
    )
    axes_flat = axes.flatten()

    bar_width = 0.35

    for ax, (dim_col, dim_label) in zip(axes_flat, dimensions):
        x = np.arange(len(agents))

        # Values for each agent
        base_vals = df_base[dim_col].values
        clean_vals = np.array(
            [
                clean_lookup.loc[a, dim_col] if a in clean_lookup.index else np.nan
                for a in agents
            ]
        )

        # Compute error bars
        yerr_base = None
        yerr_clean = None
        if dim_col in dim_se_map:
            # Aggregate dimension: propagate sub-metric SEs
            if dim_col == "R_Con":
                yerr_base = _get_weighted_r_con_yerr(df_base, values=base_vals)
            else:
                se_cols = dim_se_map[dim_col]
                yerr_base = _get_aggregate_yerr(df_base, se_cols, values=base_vals)
            clean_rows = []
            for a in agents:
                if a in clean_lookup.index:
                    clean_rows.append(clean_lookup.loc[a])
            if clean_rows:
                df_clean_ordered = pd.DataFrame(clean_rows)
                if dim_col == "R_Con":
                    yerr_clean = _get_weighted_r_con_yerr(
                        df_clean_ordered, values=clean_vals
                    )
                else:
                    se_cols = dim_se_map[dim_col]
                    yerr_clean = _get_aggregate_yerr(
                        df_clean_ordered, se_cols, values=clean_vals
                    )
        else:
            # Single metric: use direct SE column
            yerr_base = _get_yerr(df_base, dim_col, values=base_vals)
            clean_rows = []
            for a in agents:
                if a in clean_lookup.index:
                    clean_rows.append(clean_lookup.loc[a])
            if clean_rows:
                df_clean_ordered = pd.DataFrame(clean_rows)
                yerr_clean = _get_yerr(df_clean_ordered, dim_col, values=clean_vals)

        # Colors from provider
        colors = generate_shaded_colors(df_base)

        # Draw bars: clean (solid, left) and original (hatched, right)
        ax.bar(
            x - bar_width / 2,
            clean_vals,
            bar_width,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            yerr=yerr_clean,
            capsize=2,
            error_kw={"linewidth": 0.8, "color": "black"},
        )
        ax.bar(
            x + bar_width / 2,
            base_vals,
            bar_width,
            color=colors,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            yerr=yerr_base,
            capsize=2,
            error_kw={"linewidth": 0.8, "color": "black"},
        )

        ax.set_title(dim_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        # Only show x-tick labels on the bottom row
        if dim_label in ("Robustness", "Accuracy"):
            ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7.5)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0, 1.05)
        # Hide y-tick labels for the right column
        if dim_label in ("Predictability", "Accuracy"):
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(True, alpha=0.3, axis="y")

    # ---- global legend (clean vs original only, no provider swatches) ----
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label=r"$\tau$-bench (clean)"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label=r"$\tau$-bench (original)",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=2,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=0.3)
    output_path = output_dir / "taubench_clean_vs_orig.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    print(f"  Saved: {output_path}")
    plt.close()

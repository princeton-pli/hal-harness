"""Plot helpers: color generation, CI error bars."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from reliability_eval.constants import PROVIDER_COLORS, PROVIDER_ORDER


def filter_oldest_and_newest_per_provider(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to keep only the oldest and newest model per provider."""
    filtered = []
    for provider in df['provider'].unique():
        provider_df = df[df['provider'] == provider].sort_values('release_timestamp')
        if len(provider_df) <= 2:
            filtered.append(provider_df)
        else:
            filtered.append(provider_df.iloc[[0, -1]])
    result = pd.concat(filtered)
    result['provider_order'] = result['provider'].map(PROVIDER_ORDER)
    result = result.sort_values(['provider_order', 'release_timestamp']).drop('provider_order', axis=1)
    return result


def generate_shaded_colors(df: pd.DataFrame) -> List[str]:
    """
    Generate colors with different shades for models from same provider.
    Earlier models are lighter, later models are darker.
    """
    import matplotlib.colors as mcolors

    bar_colors = []

    for _, row in df.iterrows():
        provider = row.get('provider', 'Unknown')
        base_color = PROVIDER_COLORS.get(provider, '#999999')

        # Get all models from the same provider
        provider_models = df[df['provider'] == provider].sort_values('release_timestamp')
        num_models = len(provider_models)

        if num_models == 1:
            bar_colors.append(base_color)
        else:
            # Find position of this model in the provider's chronological order
            model_idx = list(provider_models.index).index(row.name)

            # Create shades: lighter for earlier, darker for later
            # Range 0.1 → 1.6: lightest blends 90% toward white,
            # darkest reduces brightness by 60%
            shade_factor = 0.1 + (model_idx / (num_models - 1)) * 1.5

            # Convert hex to RGB
            rgb = mcolors.hex2color(base_color)

            # Adjust brightness/saturation
            if shade_factor < 1.0:
                # Lighter - blend with white
                adjusted_rgb = tuple(c + (1 - c) * (1 - shade_factor) for c in rgb)
            else:
                # Darker - reduce brightness
                adjusted_rgb = tuple(max(0, c * (2 - shade_factor)) for c in rgb)

            bar_colors.append(mcolors.to_hex(adjusted_rgb))

    return bar_colors

_CI_Z = 1.0  # ±1 SE error bars


def _clip_yerr(yerr, values):
    """Clip error bars so they don't exceed [0, 1] bounds.
    Zeros out bars where value == 1.0, and clips upper end at 1.0."""
    vals = np.asarray(values)
    yerr = np.where(np.isclose(vals, 1.0), 0, yerr)
    # Clip upper: yerr cannot exceed 1.0 - value
    yerr = np.minimum(yerr, np.maximum(1.0 - vals, 0))
    return yerr


def _get_yerr(df, metric_col, values=None):
    """Get ±1 SE error bar values from SE column if available.
    Clips at [0, 1] bounds when values are provided."""
    se_col = f'{metric_col}_se'
    if se_col in df.columns:
        se_vals = df[se_col].values
        yerr = _CI_Z * np.where(np.isnan(se_vals), 0, se_vals)
        if values is not None:
            yerr = _clip_yerr(yerr, values)
        return yerr
    return None


def _get_aggregate_yerr(df, se_cols, values=None):
    """Compute ±1 SE for an aggregate (mean) of sub-metrics via error propagation.
    SE(mean) = sqrt(sum(se_i^2)) / n, assuming independence.
    Clips at [0, 1] bounds when values are provided."""
    existing = [c for c in se_cols if c in df.columns]
    if not existing:
        return None
    n = len(existing)
    sum_sq = np.zeros(len(df))
    for c in existing:
        se = df[c].values
        se = np.where(np.isnan(se), 0, se)
        sum_sq += se ** 2
    yerr = _CI_Z * np.sqrt(sum_sq) / n
    if values is not None:
        yerr = _clip_yerr(yerr, values)
    return yerr


def _get_weighted_r_con_yerr(df, values=None):
    """Compute ±1 SE for weighted R_Con via error propagation.

    For R_Con = w_out*C_out + w_traj*mean(C_traj_d, C_traj_s) + w_res*C_res,
    the propagated SE is sqrt(sum(w_i^2 * se_i^2)) where the trajectory SE
    is itself propagated from the two sub-metric SEs.
    """
    se_out = df['C_out_se'].values if 'C_out_se' in df.columns else np.zeros(len(df))
    se_traj_d = df['C_traj_d_se'].values if 'C_traj_d_se' in df.columns else np.zeros(len(df))
    se_traj_s = df['C_traj_s_se'].values if 'C_traj_s_se' in df.columns else np.zeros(len(df))
    se_res = df['C_res_se'].values if 'C_res_se' in df.columns else np.zeros(len(df))

    se_out = np.where(np.isnan(se_out), 0, se_out)
    se_traj_d = np.where(np.isnan(se_traj_d), 0, se_traj_d)
    se_traj_s = np.where(np.isnan(se_traj_s), 0, se_traj_s)
    se_res = np.where(np.isnan(se_res), 0, se_res)

    # SE of mean(C_traj_d, C_traj_s) = sqrt(se_d^2 + se_s^2) / 2
    se_traj = np.sqrt(se_traj_d**2 + se_traj_s**2) / 2

    # SE of weighted sum = sqrt(w_out^2*se_out^2 + w_traj^2*se_traj^2 + w_res^2*se_res^2)
    yerr = _CI_Z * np.sqrt(W_OUTCOME**2 * se_out**2 +
                           W_TRAJECTORY**2 * se_traj**2 +
                           W_RESOURCE**2 * se_res**2)
    if values is not None:
        yerr = _clip_yerr(yerr, values)
    return yerr


def _bar_with_ci(ax, x_pos, values, colors, df, metric_col):
    """Draw bar plot with optional ±1 SE error bars, clipped at [0, 1]. Returns bars."""
    yerr = _get_yerr(df, metric_col, values=values)
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
                  yerr=yerr, capsize=3, error_kw={'linewidth': 1.0, 'color': 'black'})
    return bars, yerr


def _add_bar_labels_ci(ax, bars, values, yerr=None):
    """Add value labels above bars, accounting for error bar height.
    Automatically adjusts ylim so labels never overshoot."""
    max_y = 0
    for i, (bar, val) in enumerate(zip(bars, values)):
        if not np.isnan(val):
            offset = (yerr[i] if yerr is not None else 0) + 0.02
            label_y = bar.get_height() + offset
            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
            # Track highest label position (+ approx text height)
            max_y = max(max_y, label_y + 0.05)
    ax.set_ylim(0, max(max_y, ax.get_ylim()[1]))

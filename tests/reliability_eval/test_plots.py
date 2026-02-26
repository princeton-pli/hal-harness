"""Smoke tests for plot modules.

These tests verify:
1. All plot modules import without errors
2. Key functions are callable (exist with correct signatures)

Full plot rendering is not tested here (matplotlib + file I/O; no display available).
"""

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import

import pytest


class TestPlotImports:
    """Verify each plot module is importable and exposes expected symbols."""

    def test_helpers_importable(self):
        from reliability_eval.plots.helpers import (
            generate_shaded_colors,
            filter_oldest_and_newest_per_provider,
            _clip_yerr,
            _get_yerr,
        )
        assert callable(generate_shaded_colors)
        assert callable(filter_oldest_and_newest_per_provider)
        assert callable(_clip_yerr)
        assert callable(_get_yerr)

    def test_dashboard_importable(self):
        from reliability_eval.plots.dashboard import (
            plot_reliability_dashboard,
            plot_metric_heatmap,
            plot_dimension_radar,
        )
        assert callable(plot_reliability_dashboard)
        assert callable(plot_metric_heatmap)
        assert callable(plot_dimension_radar)

    def test_detailed_importable(self):
        from reliability_eval.plots.detailed import (
            plot_consistency_detailed,
            plot_predictability_detailed,
            plot_robustness_detailed,
            plot_safety_detailed,
            plot_abstention_detailed,
        )
        assert callable(plot_consistency_detailed)
        assert callable(plot_predictability_detailed)
        assert callable(plot_robustness_detailed)
        assert callable(plot_safety_detailed)
        assert callable(plot_abstention_detailed)

    def test_comparison_importable(self):
        from reliability_eval.plots.comparison import (
            plot_reliability_vs_date_and_accuracy,
            plot_combined_overall_reliability,
        )
        assert callable(plot_reliability_vs_date_and_accuracy)
        assert callable(plot_combined_overall_reliability)

    def test_levels_importable(self):
        from reliability_eval.plots.levels import (
            plot_level_stratified_analysis,
            plot_provider_level_heatmap,
        )
        assert callable(plot_level_stratified_analysis)
        assert callable(plot_provider_level_heatmap)

    def test_reports_importable(self):
        from reliability_eval.plots.reports import (
            generate_report,
            save_detailed_json,
            generate_full_latex_table,
        )
        assert callable(generate_report)
        assert callable(save_detailed_json)
        assert callable(generate_full_latex_table)

    def test_package_init_importable(self):
        import reliability_eval.plots
        assert hasattr(reliability_eval.plots, "plot_reliability_dashboard")
        assert hasattr(reliability_eval.plots, "generate_report")

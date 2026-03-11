"""reliability_eval.plots -- visualization subpackage."""

from reliability_eval.plots.comparison import (
    plot_calibration as plot_calibration,
    plot_calibration_selective_comparison as plot_calibration_selective_comparison,
    plot_combined_overall_reliability as plot_combined_overall_reliability,
    plot_combined_overall_reliability_large as plot_combined_overall_reliability_large,
    plot_discrimination as plot_discrimination,
    plot_outcome_consistency as plot_outcome_consistency,
    plot_prompt_robustness as plot_prompt_robustness,
    plot_reasoning_vs_nonreasoning as plot_reasoning_vs_nonreasoning,
    plot_reliability_by_model_size as plot_reliability_by_model_size,
    plot_reliability_by_provider as plot_reliability_by_provider,
    plot_reliability_vs_date_and_accuracy as plot_reliability_vs_date_and_accuracy,
    plot_scaffold_comparison as plot_scaffold_comparison,
    plot_taubench_clean_vs_orig as plot_taubench_clean_vs_orig,
)
from reliability_eval.plots.dashboard import (
    plot_dimension_radar as plot_dimension_radar,
    plot_metric_heatmap as plot_metric_heatmap,
    plot_reliability_dashboard as plot_reliability_dashboard,
)
from reliability_eval.plots.detailed import (
    plot_abstention_detailed as plot_abstention_detailed,
    plot_accuracy_coverage_by_model as plot_accuracy_coverage_by_model,
    plot_calibration_by_model as plot_calibration_by_model,
    plot_consistency_detailed as plot_consistency_detailed,
    plot_predictability_detailed as plot_predictability_detailed,
    plot_robustness_detailed as plot_robustness_detailed,
    plot_safety_deep_analysis as plot_safety_deep_analysis,
    plot_safety_detailed as plot_safety_detailed,
    plot_safety_lambda_sensitivity as plot_safety_lambda_sensitivity,
    plot_safety_severity_violations as plot_safety_severity_violations,
)
from reliability_eval.plots.helpers import (
    filter_oldest_and_newest_per_provider as filter_oldest_and_newest_per_provider,
    generate_shaded_colors as generate_shaded_colors,
)
from reliability_eval.plots.levels import (
    plot_action_efficiency_by_level as plot_action_efficiency_by_level,
    plot_confidence_difficulty_alignment as plot_confidence_difficulty_alignment,
    plot_level_consistency_patterns as plot_level_consistency_patterns,
    plot_level_reliability_summary as plot_level_reliability_summary,
    plot_level_stratified_analysis as plot_level_stratified_analysis,
    plot_performance_drop_analysis as plot_performance_drop_analysis,
    plot_provider_level_heatmap as plot_provider_level_heatmap,
)
from reliability_eval.plots.reports import (
    generate_full_latex_table as generate_full_latex_table,
    generate_report as generate_report,
    save_detailed_json as save_detailed_json,
)

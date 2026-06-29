"""Plotting utilities for the granularity_tsad paper figures."""

from .bayesian import (
    compare_against_all,
    compare_two,
    group_paired_scores,
    point_vs_seq,
    point_vs_seq_by_cluster,
    rope_from_mad,
)
from .boxplots import plot_metric_comparison_by_density, plot_performance_by_group
from .dendrogram import plot_dendrogram
from .heatmap import plot_cluster_heatmaps, plot_score_heatmap
from .pak_curves import plot_pak_curves, plot_wpak_weighting
from .profiles import plot_profiles_by_cluster
from .style import CLUSTER_COLORS, SCORE_CMAP, set_paper_style, style_axes

__all__ = [
    "set_paper_style",
    "style_axes",
    "CLUSTER_COLORS",
    "SCORE_CMAP",
    "plot_score_heatmap",
    "plot_cluster_heatmaps",
    "plot_performance_by_group",
    "plot_metric_comparison_by_density",
    "plot_profiles_by_cluster",
    "plot_pak_curves",
    "plot_wpak_weighting",
    "compare_two",
    "compare_against_all",
    "group_paired_scores",
    "rope_from_mad",
    "point_vs_seq",
    "point_vs_seq_by_cluster",
    "plot_dendrogram",
]

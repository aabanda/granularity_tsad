"""Boxplots of detector performance grouped by anomaly granularity.

Reproduces the paper figures contrasting how point-wise, sequence-wise and
PAK-based scores behave across anomaly point-density regimes (e.g. point
anomalies of length 1 vs. longer sequence anomalies).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .style import style_axes


def _melt_metric(
    metric_df: pd.DataFrame, group_label: pd.Series, group_name: str
) -> pd.DataFrame:
    df = metric_df.copy()
    df[group_name] = group_label
    return df.reset_index().melt(
        id_vars=["index", group_name],
        value_vars=list(metric_df.columns),
        var_name="Algorithm",
        value_name="Performance",
    )


def plot_performance_by_group(
    metric_df: pd.DataFrame,
    group_label: pd.Series,
    group_name: str = "Anomaly type",
    algorithm_order: list[str] | None = None,
    ylabel: str = "Score",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Boxplot of one metric per algorithm, split by an anomaly-type label."""
    melted = _melt_metric(metric_df, group_label, group_name)
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))
    style_axes(ax)
    sns.boxplot(
        data=melted,
        x="Algorithm",
        y="Performance",
        hue=group_name,
        order=algorithm_order,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    if title:
        ax.set_title(title)
    return ax


def plot_metric_comparison_by_density(
    metrics: dict[str, pd.DataFrame],
    group_label: pd.Series,
    group_name: str = "Anomaly type",
    algorithm_order: list[str] | None = None,
) -> plt.Figure:
    """One boxplot panel per metric (e.g. F1_point, F1_seq, F1_PAK)."""
    n = len(metrics)
    fig, axs = plt.subplots(n, 1, figsize=(14, 5 * n))
    axs = [axs] if n == 1 else list(axs)
    for ax, (name, df) in zip(axs, metrics.items()):
        plot_performance_by_group(
            df,
            group_label,
            group_name=group_name,
            algorithm_order=algorithm_order,
            ylabel=name,
            title=name,
            ax=ax,
        )
    fig.tight_layout()
    return fig

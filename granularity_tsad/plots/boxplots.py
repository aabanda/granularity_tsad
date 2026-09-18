"""Boxplots of detector performance grouped by anomaly granularity.

Reproduces the paper figures contrasting how point-wise, sequence-wise and
PAK-based scores behave across anomaly point-density regimes (e.g. point
anomalies of length 1 vs. longer sequence anomalies).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .style import CLUSTER_DISPLAY_ORDER, style_axes

# Preferred column order for cluster panels (unknown labels appended after).
_CLUSTER_ORDER = list(CLUSTER_DISPLAY_ORDER)


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


def _cluster_column_order(group_label: pd.Series) -> list[str]:
    present = [c for c in _CLUSTER_ORDER if (group_label == c).any()]
    extras = sorted(set(group_label.dropna().unique()) - set(present))
    return present + extras


def plot_metric_comparison_by_density(
    metrics: dict[str, pd.DataFrame],
    group_label: pd.Series,
    group_name: str = "Anomaly type",
    algorithm_order: list[str] | None = None,
) -> plt.Figure:
    """Grid of boxplots: rows = metrics, columns = clusters.

    Each panel shows algorithm performance for one (metric, cluster) pair.
    """
    metric_names = list(metrics.keys())
    clusters = _cluster_column_order(group_label)
    n_rows, n_cols = len(metric_names), len(clusters)
    if n_rows == 0 or n_cols == 0:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        return fig

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        sharey="row",
        squeeze=False,
    )

    for i, metric_name in enumerate(metric_names):
        metric_df = metrics[metric_name]
        aligned = group_label.reindex(metric_df.index)
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            style_axes(ax)
            mask = aligned == cluster
            sub = metric_df.loc[mask]
            if sub.empty:
                ax.set_axis_off()
                continue
            # Single-group melt: boxplot per algorithm (no hue).
            melted = (
                sub.reset_index()
                .melt(
                    id_vars=["index"],
                    value_vars=list(sub.columns),
                    var_name="Algorithm",
                    value_name="Performance",
                )
            )
            order = algorithm_order or list(sub.columns)
            order = [a for a in order if a in sub.columns]
            sns.boxplot(
                data=melted,
                x="Algorithm",
                y="Performance",
                order=order,
                ax=ax,
                color="#4c78a8",
            )
            if i == 0:
                ax.set_title(cluster)
            else:
                ax.set_title("")
            if j == 0:
                ax.set_ylabel(metric_name)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            # Hide x tick labels on non-bottom rows to reduce clutter.
            if i < n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)

    fig.tight_layout()
    return fig

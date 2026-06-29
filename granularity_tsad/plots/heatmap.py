"""Score heatmap: a time series colored by its aggregated anomaly score.

Each timestamp is drawn as a scatter point colored with the ``YlOrRd`` colormap
according to the aggregated anomaly score, over a faint gray line of the raw
signal and a gray span marking the labelled anomaly (paper-style figure).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import SCORE_CMAP, style_axes


def plot_score_heatmap(
    series: np.ndarray,
    scores: np.ndarray,
    anomaly_span: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    point_size: float = 12.0,
) -> plt.Axes:
    """Plot a single series colored by aggregated anomaly score.

    Parameters
    ----------
    series : raw time-series values.
    scores : aggregated anomaly score per timestamp (will be min-max scaled).
    anomaly_span : ``(start, end)`` indices of the labelled anomaly (optional).
    """
    series = np.asarray(series, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = min(len(series), len(scores))
    series, scores = series[:n], scores[:n]
    x = np.arange(n)

    lo, hi = np.nanmin(scores), np.nanmax(scores)
    norm = (scores - lo) / ((hi - lo) or 1.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    style_axes(ax)

    if anomaly_span is not None:
        ax.axvspan(anomaly_span[0], anomaly_span[1], color="gray", alpha=0.25, zorder=0)

    ax.plot(x, series, color="gray", linewidth=0.8, zorder=1)
    sc = ax.scatter(
        x, series, c=norm, cmap=SCORE_CMAP, s=point_size, vmin=0, vmax=1, zorder=2
    )
    ax.set_xlim(0, n - 1)
    if title:
        ax.set_title(title)
    ax.figure.colorbar(sc, ax=ax, label="Anomaly score", pad=0.01)
    return ax


def plot_cluster_heatmaps(
    examples: list[dict],
    title: str | None = None,
) -> plt.Figure:
    """Stack several score heatmaps (one per example series) in a column.

    ``examples`` is a list of dicts with keys ``series``, ``scores``,
    optionally ``anomaly_span`` and ``label``.
    """
    n = len(examples)
    fig, axs = plt.subplots(n, 1, figsize=(6, 1.7 * n))
    axs = np.atleast_1d(axs)
    for ax, ex in zip(axs, examples):
        plot_score_heatmap(
            ex["series"],
            ex["scores"],
            anomaly_span=ex.get("anomaly_span"),
            ax=ax,
            title=ex.get("label"),
        )
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig

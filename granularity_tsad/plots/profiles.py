"""Point-detectability profiles grouped by cluster (paper Figure 7).

Each series profile is the concatenation ``[PA-precision | PA-recall]`` over the
``k`` grid, averaged across the point-wise detectors (see
:func:`granularity_tsad.aggregation.build_profiles`). This module draws, per
cluster, every individual profile (thin) together with the cluster mean (thick),
reproducing the characteristic shapes of point-dense / point-sparse /
point-normal anomalies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..clustering import CLUSTER_COLORS, CLUSTER_NAMES
from .style import style_axes


def plot_profiles_by_cluster(
    profiles: pd.DataFrame,
    cluster_names: pd.Series,
    n_k: int | None = None,
    order: list[str] | None = None,
) -> plt.Figure:
    """One panel per cluster with all profiles plus the mean profile.

    Parameters
    ----------
    profiles : ``n_series x 2K`` detectability profiles (precision | recall).
    cluster_names : per-series cluster name, indexed like ``profiles``.
    n_k : length ``K`` of each curve; defaults to ``profiles.shape[1] // 2`` and
        is used to draw the precision/recall divider.
    order : cluster display order; defaults to :data:`CLUSTER_NAMES`.
    """
    names = order or [c for c in CLUSTER_NAMES if c in set(cluster_names)]
    if not names:
        names = list(pd.unique(cluster_names))
    n_features = profiles.shape[1]
    n_k = n_k or (n_features // 2)

    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    x = np.arange(n_features)

    for ax, name in zip(axs, names):
        style_axes(ax)
        members = cluster_names.index[cluster_names == name]
        color = CLUSTER_COLORS.get(name, "#555555")
        block = profiles.loc[members]
        for _, row in block.iterrows():
            ax.plot(x, row.values, color=color, alpha=0.15, linewidth=0.6)
        if len(block):
            ax.plot(x, block.values.mean(axis=0), color=color, linewidth=2.4, label="mean")
        if 0 < n_k < n_features:
            ax.axvline(n_k - 0.5, color="#999999", linestyle="--", linewidth=0.8)
            ax.text(n_k / 2, 1.02, "precision", ha="center", va="bottom", fontsize=9)
            ax.text(n_k + n_k / 2, 1.02, "recall", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{name} (n={len(members)})")
        ax.set_xlabel("PAK curve index (k)")
        ax.set_ylim(-0.02, 1.08)
    axs[0].set_ylabel("PA precision / recall")
    fig.tight_layout()
    return fig

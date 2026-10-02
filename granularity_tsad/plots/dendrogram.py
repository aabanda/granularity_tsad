"""Dendrogram of the series clustering with shaded cluster blocks (Figure 8)."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster

from ..clustering import (
    CLUSTER_COLORS,
    CLUSTER_DISPLAY_ORDER,
    linkage_matrix,
    name_clusters,
)


def _original_leaves(node_id: int, n_samples: int, children: np.ndarray) -> list[int]:
    if node_id < n_samples:
        return [node_id]
    left, right = children[node_id - n_samples]
    return _original_leaves(int(left), n_samples, children) + _original_leaves(
        int(right), n_samples, children
    )


def plot_dendrogram(
    profiles,
    n_clusters: int = 3,
    truncate_level: int = 3,
    ax: plt.Axes | None = None,
    title: str = "Hierarchical Clustering Dendrogram",
) -> plt.Axes:
    """Plot a truncated dendrogram with cluster blocks shaded by name color."""
    model, matrix = linkage_matrix(profiles)
    flat_labels = fcluster(matrix, t=n_clusters, criterion="maxclust")
    n_samples = len(profiles)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))

    plt.sca(ax)
    ddata = dendrogram(
        matrix,
        ax=ax,
        color_threshold=0,
        truncate_mode="level",
        p=truncate_level,
        above_threshold_color="black",
    )

    # Same naming as ``name_clusters`` (ascending mean PA-precision).
    flat = pd.Series(flat_labels, index=profiles.index, name="cluster")
    cid_to_name = name_clusters(flat, profiles) if len(np.unique(flat_labels)) == 3 else {}

    # Shade leaf blocks (same layout as main_paper.py).
    if cid_to_name:
        leaf_xs = {leaf: 5 + idx * 10 for idx, leaf in enumerate(ddata["leaves"])}
        cluster_xs: dict[int, list[float]] = defaultdict(list)
        for leaf, x in leaf_xs.items():
            leaves = _original_leaves(int(leaf), n_samples, model.children_)
            labels_under = [int(flat_labels[i]) for i in leaves]
            majority = max(set(labels_under), key=labels_under.count)
            cluster_xs[majority].append(x)

        y_min, y_max = ax.get_ylim()
        y_cap = min(8.0, y_max)
        ymax_frac = (y_cap - y_min) / (y_max - y_min) if y_max > y_min else 1.0
        for cid, xs in cluster_xs.items():
            name = cid_to_name[cid]
            ax.axvspan(
                min(xs) - 2,
                max(xs) + 2,
                alpha=0.35,
                ymin=0,
                ymax=ymax_frac,
                color=CLUSTER_COLORS[name],
                zorder=0,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Cluster size", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    if cid_to_name:
        handles = [
            plt.Line2D([0], [0], color=CLUSTER_COLORS[name], lw=6, alpha=0.5, label=name)
            for name in CLUSTER_DISPLAY_ORDER
            if name in cid_to_name.values()
        ]
        ax.legend(handles=handles, frameon=False, fontsize=9)
    return ax

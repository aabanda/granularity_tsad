"""Dendrogram of the series clustering with shaded cluster blocks (Figure 8)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster

from ..clustering import CLUSTER_COLORS, CLUSTER_NAMES, linkage_matrix


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

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))

    plt.sca(ax)
    ddata = dendrogram(
        matrix,
        ax=ax,
        color_threshold=0,
        truncate_mode="level",
        p=truncate_level,
        above_threshold_color="black",
    )

    # Order clusters by mean detectability so colors match Point-sparse/normal/dense.
    profile_vals = profiles.values
    means = {cid: profile_vals[flat_labels == cid].mean() for cid in np.unique(flat_labels)}
    order = sorted(means, key=means.get)
    cid_to_name = {cid: CLUSTER_NAMES[i] for i, cid in enumerate(order)} if len(order) == 3 else {}

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Cluster size", fontsize=12)
    ax.set_ylabel("Distance", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if cid_to_name:
        handles = [
            plt.Line2D([0], [0], color=CLUSTER_COLORS[name], lw=6, alpha=0.5, label=name)
            for name in CLUSTER_NAMES
        ]
        ax.legend(handles=handles, frameon=False, fontsize=9)
    return ax

"""Hierarchical clustering of series by point-detectability (paper Section 3.3).

Series are grouped with agglomerative (Ward) clustering on their
point-detectability profiles. With three clusters the paper identifies three
regimes named by how densely the anomalous segment is detectable point-wise:
``Point-sparse``, ``Point-normal`` and ``Point-dense``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Ascending point-anomaly density ρ / point-detectability (paper §3.2–3.3):
#   Point-normal (negligible ρ) < Point-sparse (low ρ) < Point-dense (high ρ).
CLUSTER_NAMES = ["Point-normal", "Point-sparse", "Point-dense"]
# Display order for all figures (dendrogram legend, profiles, metrics, Bayesian).
CLUSTER_DISPLAY_ORDER = ["Point-dense", "Point-sparse", "Point-normal"]
CLUSTER_COLORS = {
    "Point-sparse": "#2196F3",
    "Point-normal": "#4CAF50",
    "Point-dense": "#FF9800",
}
# Short density labels used in metric CSVs (dense / sparse / normal).
CLUSTER_TO_DENSITY = {
    "Point-dense": "dense",
    "Point-sparse": "sparse",
    "Point-normal": "normal",
}
DENSITY_TO_CLUSTER = {v: k for k, v in CLUSTER_TO_DENSITY.items()}
DENSITY_DISPLAY_ORDER = ["dense", "sparse", "normal"]


def cluster_profiles(
    profiles: pd.DataFrame, n_clusters: int = 3
) -> pd.Series:
    """Assign each series to a cluster from its detectability profile."""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(profiles.values)
    return pd.Series(labels, index=profiles.index, name="cluster")


def _precision_block(profiles: pd.DataFrame) -> pd.DataFrame:
    """PA-precision half of a ``[precision | recall]`` detectability profile."""
    n_k = profiles.shape[1] // 2
    return profiles.iloc[:, :n_k]


def name_clusters(
    labels: pd.Series, profiles: pd.DataFrame
) -> dict[int, str]:
    """Map raw cluster ids to interpretable names by mean PA-precision.

    Profiles are ``[PA-precision | PA-recall]``. Naming uses the precision half
    only: recall is often wide even for hard (point-normal) series, so the full
    concatenated mean can invert ``Point-sparse`` and ``Point-normal``.
    Ascending mean precision matches ρ: lowest -> ``Point-normal``, middle ->
    ``Point-sparse``, highest -> ``Point-dense``.
    """
    prec = _precision_block(profiles)
    mean_detectability = {
        cid: float(prec.loc[labels.index[labels == cid]].values.mean())
        for cid in sorted(labels.unique())
    }
    order = sorted(mean_detectability, key=mean_detectability.get)
    names = CLUSTER_NAMES if len(order) == 3 else [f"Cluster {i}" for i in range(len(order))]
    return {cid: names[rank] for rank, cid in enumerate(order)}


def linkage_matrix(profiles: pd.DataFrame) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Fit the full tree and build a scipy-compatible linkage matrix."""
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(profiles.values)

    counts = np.zeros(model.children_.shape[0])
    n = len(model.labels_)
    for i, merge in enumerate(model.children_):
        counts[i] = sum(1 if c < n else counts[c - n] for c in merge)
    matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    return model, matrix

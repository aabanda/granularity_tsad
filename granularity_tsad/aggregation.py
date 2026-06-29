"""Score aggregation and point-detectability profiles (paper Section 3.2).

Two complementary aggregations are provided:

* :func:`aggregate_scores` — collapse the scores of several point-wise
  detectors into a single per-timestamp anomaly score (mean of min-max
  normalized scores). Used to build the score heatmap of a single series.
* :func:`build_profiles` — summarize how detectable each series is by stacking
  the PAK precision/recall curves across detectors. The resulting feature
  matrix is the input to the hierarchical clustering of Section 3.3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import align_scores_targets
from .metrics.pak import DEFAULT_K_VALUES, pak_curves


def _min_max(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    denom = (hi - lo) or 1.0
    return (x - lo) / denom


def aggregate_scores(score_arrays: list[np.ndarray]) -> np.ndarray:
    """Mean of per-detector min-max normalized scores, aligned to shortest length."""
    arrays = [_min_max(s) for s in score_arrays if s is not None]
    if not arrays:
        raise ValueError("No score arrays provided for aggregation.")
    min_len = min(len(a) for a in arrays)
    stacked = np.vstack([a[:min_len] for a in arrays])
    return stacked.mean(axis=0)


def aggregate_dataset_scores(
    df_scores: pd.DataFrame, methods: list[str] | None = None
) -> pd.Series:
    """Per-dataset aggregated anomaly score across the given detectors."""
    methods = methods or list(df_scores.columns)
    out = {}
    for dataset in df_scores.index:
        arrays = [df_scores.loc[dataset, m] for m in methods]
        out[dataset] = aggregate_scores(arrays)
    return pd.Series(out).sort_index()


def compute_pak_curve_frames(
    df_scores: pd.DataFrame,
    targets: pd.Series,
    methods: list[str],
    k_values: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute PAK precision/recall/F1 curves for every (dataset, method).

    Returns a dict with keys ``"f1"``, ``"precision"`` and ``"recall"``, each a
    DataFrame of object dtype holding one curve per cell.
    """
    k_values = k_values or DEFAULT_K_VALUES
    frames = {
        key: pd.DataFrame(index=df_scores.index, columns=methods, dtype=object)
        for key in ("f1", "precision", "recall")
    }
    for dataset in df_scores.index:
        for method in methods:
            scores = df_scores.loc[dataset, method]
            if scores is None:
                continue
            s, t = align_scores_targets(scores, targets.loc[dataset], method)
            if t.sum() == 0:
                continue
            curves = pak_curves(s, t, k_values=k_values)
            for key in ("f1", "precision", "recall"):
                frames[key].loc[dataset, method] = curves[key]
    return frames


def build_profiles(
    precision_frame: pd.DataFrame,
    recall_frame: pd.DataFrame,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Point-detectability profile per series: mean over detectors of the
    concatenated [precision | recall] PAK curves.

    Returns a numeric DataFrame (n_series x 2*len(k_values)) ready for clustering.
    """
    methods = methods or list(precision_frame.columns)
    profiles = {}
    for dataset in precision_frame.index:
        per_method = []
        for method in methods:
            p = precision_frame.loc[dataset, method]
            r = recall_frame.loc[dataset, method]
            if p is None or r is None:
                continue
            per_method.append(np.concatenate([np.asarray(p, float), np.asarray(r, float)]))
        if not per_method:
            continue
        profiles[dataset] = np.mean(np.vstack(per_method), axis=0)
    return pd.DataFrame.from_dict(profiles, orient="index").sort_index()

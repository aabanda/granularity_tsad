"""Bayesian comparison of two detectors over multiple datasets (baycomp).

Wraps :func:`baycomp.two_on_multiple` to produce the posterior over
``P(A>B)``, ``P(rope)`` and ``P(B>A)`` and the simplex plot used in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _two_on_multiple():
    """Lazily import baycomp so the package imports without it installed."""
    try:
        from baycomp import two_on_multiple
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "baycomp is required for Bayesian comparisons. Install with `pip install baycomp`."
        ) from exc
    return two_on_multiple


def compare_two(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
    rope: float = 0.01,
    plot: bool = True,
):
    """Bayesian sign-rank comparison of two detectors across datasets.

    Returns ``(probs, fig)`` where ``probs = (p_a_better, p_rope, p_b_better)``
    and ``fig`` is ``None`` when ``plot=False``.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    two_on_multiple = _two_on_multiple()
    result = two_on_multiple(
        scores_a, scores_b, rope=rope, plot=plot, names=[name_a, name_b]
    )
    if plot:
        probs, fig = result
    else:
        probs, fig = result, None
    return probs, fig


def group_paired_scores(
    metric_df: pd.DataFrame,
    point_methods: list[str],
    seq_methods: list[str],
) -> pd.DataFrame:
    """Per-series median score for the point-wise (P) and sequence-wise (S) groups.

    Reproduces the paper's group-level summary (Section 3.4): for each series the
    performance of each detector family is summarized by the median across its
    constituent methods, yielding paired ``(P, S)`` observations. Series with a
    missing value in either group are dropped.
    """
    p_cols = [m for m in point_methods if m in metric_df.columns]
    s_cols = [m for m in seq_methods if m in metric_df.columns]
    paired = pd.DataFrame(
        {
            "P": metric_df[p_cols].median(axis=1, skipna=True),
            "S": metric_df[s_cols].median(axis=1, skipna=True),
        }
    )
    return paired.dropna()


def rope_from_mad(paired: pd.DataFrame, scale: float = 0.5) -> float:
    """ROPE half-width ``R = scale * MAD`` of the paired P-S differences.

    Matches the calibration used in the paper (``R = 0.5 * MAD``), keeping the
    equivalence band tied to the spread of each metric.
    """
    diffs = (paired["P"] - paired["S"]).to_numpy(dtype=float)
    if diffs.size == 0:
        return 0.0
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    return scale * mad


def point_vs_seq(
    metric_df: pd.DataFrame,
    point_methods: list[str],
    seq_methods: list[str],
    rope: float | None = None,
    plot: bool = True,
):
    """Bayesian comparison of the point-wise vs sequence-wise families.

    Returns ``(probs, fig, rope)`` where ``probs = (p_point_better, p_rope,
    p_seq_better)``. When ``rope`` is ``None`` it is derived as ``0.5 * MAD`` of
    the paired differences.
    """
    paired = group_paired_scores(metric_df, point_methods, seq_methods)
    if len(paired) < 2:
        return None, None, float("nan")
    r = rope_from_mad(paired) if rope is None else rope
    probs, fig = compare_two(
        paired["P"].to_numpy(dtype=float),
        paired["S"].to_numpy(dtype=float),
        name_a="Point-wise",
        name_b="Sequence-wise",
        rope=max(r, 1e-6),
        plot=plot,
    )
    return probs, fig, r


def point_vs_seq_by_cluster(
    metric_df: pd.DataFrame,
    point_methods: list[str],
    seq_methods: list[str],
    cluster_names: pd.Series,
    rope: float | None = None,
) -> pd.DataFrame:
    """Point-wise vs sequence-wise comparison stratified by anomaly cluster.

    A single ``rope`` is held fixed across clusters within a metric (paper
    Section 3.4): if not given, it is computed once from all paired differences.
    Returns a summary table with the three posterior probabilities per cluster.
    """
    aligned = cluster_names.reindex(metric_df.index)
    if rope is None:
        rope = rope_from_mad(group_paired_scores(metric_df, point_methods, seq_methods))
    rows = []
    for name in pd.unique(aligned.dropna()):
        sub = metric_df.loc[aligned.index[aligned == name]]
        probs, _, _ = point_vs_seq(
            sub, point_methods, seq_methods, rope=rope, plot=False
        )
        if probs is None:
            continue
        p_point, p_rope, p_seq = probs
        rows.append(
            {
                "Cluster": name,
                "P(point better)": p_point,
                "P(rope)": p_rope,
                "P(sequence better)": p_seq,
                "rope": rope,
            }
        )
    return pd.DataFrame(rows)


def compare_against_all(
    metric_df: pd.DataFrame,
    reference: str,
    rope: float = 0.01,
) -> pd.DataFrame:
    """Compare ``reference`` against every other column; return a summary table."""
    ref_scores = metric_df[reference].astype(float).values
    rows = []
    for alg in metric_df.columns:
        if alg == reference:
            continue
        probs, _ = compare_two(
            ref_scores,
            metric_df[alg].astype(float).values,
            name_a=reference,
            name_b=alg,
            rope=rope,
            plot=False,
        )
        p_better, p_rope, p_worse = probs
        rows.append(
            {
                "Algorithm": alg,
                f"P({reference} better)": p_better,
                "P(rope)": p_rope,
                f"P({reference} worse)": p_worse,
            }
        )
    return pd.DataFrame(rows).sort_values(
        f"P({reference} better)", ascending=False
    ).reset_index(drop=True)

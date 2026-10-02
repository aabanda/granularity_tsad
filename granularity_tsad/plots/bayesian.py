"""Bayesian comparison of two detectors over multiple datasets (baycomp).

Wraps :func:`baycomp.two_on_multiple` to produce the posterior over
``P(A>B)``, ``P(rope)`` and ``P(B>A)`` and the simplex plot used in the paper.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..clustering import CLUSTER_DISPLAY_ORDER


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


def _cluster_order(cluster_names: pd.Series) -> list[str]:
    present = set(cluster_names.dropna().unique())
    ordered = [c for c in CLUSTER_DISPLAY_ORDER if c in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


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
    for name in _cluster_order(aligned):
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


def rope_sensitivity_table(
    metric_tables: dict[str, pd.DataFrame],
    point_methods: list[str],
    seq_methods: list[str],
    cluster_names: pd.Series,
    mad_scales: list[float] | None = None,
) -> pd.DataFrame:
    """Bayesian probs vs MAD-scaled ROPE width, per metric and cluster.

    For each metric, the base MAD of all paired P−S differences is computed
    once; then ``R = scale * MAD`` is swept while keeping that R fixed across
    density clusters within the metric (same protocol as the paper).
    """
    scales = mad_scales or [0.25, 0.5, 0.75, 1.0]
    rows: list[dict] = []
    for metric_name, table in metric_tables.items():
        table = table.dropna(how="all")
        paired_all = group_paired_scores(table, point_methods, seq_methods)
        mad = rope_from_mad(paired_all, scale=1.0)
        aligned = cluster_names.reindex(table.index)
        for scale in scales:
            rope = max(float(scale) * mad, 1e-6)
            for cluster in _cluster_order(aligned):
                sub = table.loc[aligned.index[aligned == cluster]]
                probs, _, _ = point_vs_seq(
                    sub, point_methods, seq_methods, rope=rope, plot=False
                )
                if probs is None:
                    continue
                p_point, p_rope, p_seq = probs
                rows.append(
                    {
                        "Metric": metric_name,
                        "Cluster": cluster,
                        "mad_scale": float(scale),
                        "MAD": mad,
                        "rope": rope,
                        "P(point better)": float(p_point),
                        "P(rope)": float(p_rope),
                        "P(sequence better)": float(p_seq),
                    }
                )
    return pd.DataFrame(rows)


def plot_rope_sensitivity(
    summary: pd.DataFrame,
    cluster_order: list[str] | None = None,
) -> plt.Figure:
    """Line plots of posterior probs vs MAD scale (rows=metrics, cols=clusters)."""
    if summary.empty:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.text(0.5, 0.5, "n/a", ha="center", va="center")
        return fig

    metrics = list(dict.fromkeys(summary["Metric"].tolist()))
    clusters = cluster_order or [
        c for c in CLUSTER_DISPLAY_ORDER if c in set(summary["Cluster"])
    ]
    extras = sorted(set(summary["Cluster"]) - set(clusters))
    clusters = clusters + extras

    n_rows, n_cols = len(metrics), len(clusters)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * max(n_cols, 1), 2.8 * max(n_rows, 1)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {
        "P(point better)": "#FF9800",
        "P(rope)": "#757575",
        "P(sequence better)": "#2196F3",
    }
    labels = {
        "P(point better)": "P(point better)",
        "P(rope)": "P(rope)",
        "P(sequence better)": "P(seq. better)",
    }

    for i, metric in enumerate(metrics):
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            sub = summary[
                (summary["Metric"] == metric) & (summary["Cluster"] == cluster)
            ].sort_values("mad_scale")
            if sub.empty:
                ax.set_axis_off()
                continue
            for col, color in colors.items():
                ax.plot(
                    sub["mad_scale"],
                    sub[col],
                    "o-",
                    color=color,
                    label=labels[col],
                    markersize=4,
                    linewidth=1.5,
                )
            ax.axvline(0.5, color="black", ls="--", lw=0.8, alpha=0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(sorted(sub["mad_scale"].unique()))
            if i == 0:
                ax.set_title(cluster, fontsize=11)
            if j == 0:
                ax.set_ylabel(metric, fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel(r"ROPE scale ($\times$ MAD)")
            ax.grid(True, alpha=0.25)

    handles, labs = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labs,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_point_vs_seq_grid(
    metric_tables: dict[str, pd.DataFrame],
    point_methods: list[str],
    seq_methods: list[str],
    cluster_names: pd.Series,
    cluster_order: list[str] | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Unified Bayesian grid: rows = metrics, columns = clusters.

    Each cell is the point-wise vs sequence-wise simplex for that
    (metric, cluster). Returns ``(figure, summary_table)``.
    """
    metrics = list(metric_tables.keys())
    clusters = cluster_order or _cluster_order(cluster_names)
    n_rows, n_cols = len(metrics), len(clusters)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * max(n_cols, 1), 3.4 * max(n_rows, 1)),
        squeeze=False,
    )
    summaries = []

    for i, metric_name in enumerate(metrics):
        table = metric_tables[metric_name].dropna(how="all")
        aligned = cluster_names.reindex(table.index)
        rope = rope_from_mad(group_paired_scores(table, point_methods, seq_methods))
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            sub = table.loc[aligned.index[aligned == cluster]]
            probs, subfig, used_rope = point_vs_seq(
                sub, point_methods, seq_methods, rope=rope, plot=True
            )
            if i == 0:
                ax.set_title(cluster, fontsize=12)
            if j == 0:
                ax.set_ylabel(metric_name, fontsize=12)
            if probs is None or subfig is None:
                ax.set_axis_off()
                ax.text(
                    0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes
                )
                if subfig is not None:
                    plt.close(subfig)
                continue
            buf = io.BytesIO()
            subfig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            plt.close(subfig)
            buf.seek(0)
            img = plt.imread(buf)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            p_point, p_rope, p_seq = probs
            summaries.append(
                {
                    "Metric": metric_name,
                    "Cluster": cluster,
                    "P(point better)": p_point,
                    "P(rope)": p_rope,
                    "P(sequence better)": p_seq,
                    "rope": used_rope,
                }
            )

    fig.tight_layout()
    return fig, pd.DataFrame(summaries)


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

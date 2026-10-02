"""Figure 8: point-wise re-labeling heatmaps per density cluster.

Rescued from ``tsad/src/main_paper.py``. By default reproduces the **published**
Fig. 8 panel titles and the underlying UCR series (see index note below).

Uses **saved** scores under ``Results/Scores/`` — no PAK recomputation.

Index note (tsad ``main_paper.py``)
-----------------------------------
1. ``anomaly_df`` is Dataset-indexed ``1..250``, then ``reset_index(drop=True)``
   so positional index ``i`` is UCR Dataset ``i + 1``.
2. Point anomalies (label length 1 or 2) are dropped and indices reset again.
3. Plot titles use ``new_to_orig_idx`` = **pre-filter positional** index.

So a paper title ``TS T`` loads UCR Dataset ``T + 1`` (and its ``T+1.npy`` scores).
``clusters.csv`` in this repo already uses real UCR Dataset ids.

Example
-------
    PYTHONPATH=. python scripts/plot_fig8_cluster_heatmaps.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.aggregation import aggregate_scores  # noqa: E402
from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import (  # noqa: E402
    build_anomaly_df,
    build_binary_targets,
    load_scores,
    load_ucr_series,
)
from granularity_tsad.plots.style import (  # noqa: E402
    CLUSTER_COLORS,
    CLUSTER_DISPLAY_ORDER,
    SCORE_CMAP,
    set_paper_style,
)

# Paper *titles* (pre-filter 0-based positional = UCR Dataset − 1).
# From published ``heatmap_{dense,sparse,normal}_1.png`` / Fig. 11 dense column.
PAPER_FIG8_TITLES: dict[str, list[int]] = {
    "Point-dense": [135, 111, 32, 93, 139],
    "Point-sparse": [13, 225, 118, 75, 10],
    # Titles as printed on heatmap_normal_1.png (TS 188 is pre-filter
    # positional; UCR 189 is a dropped point-anomaly, so resolver falls
    # back to treating 188 as a post-filter index → UCR 206).
    "Point-normal": [42, 17, 188, 141, 171],
}
# main_paper.py forced originals (pre-filter positional), used with --from-clusters.
DEFAULT_FORCED = [41, 13, 224, 151, 135]
CLUSTER_TITLE = {
    "Point-dense": "Point-dense anomalies",
    "Point-sparse": "Point-sparse anomalies",
    "Point-normal": "Point-normal anomalies",
}


def paper_title_to_ucr(title: int) -> int:
    """Map a paper ``TS T`` title to the UCR Dataset id used in this repo."""
    return int(title) + 1


def ucr_to_paper_title(dataset_id: int) -> int:
    """Inverse of :func:`paper_title_to_ucr`."""
    return int(dataset_id) - 1


def point_anomaly_index_maps(
    anomaly_lengths: np.ndarray,
) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    """Build the same orig↔new maps as ``tsad/src/main_paper.py``.

    ``anomaly_lengths`` is the per-series ground-truth anomaly length on the
    **positional** 0..N-1 index (after ``reset_index`` on Dataset 1..250).
    Point anomalies are length 1 or 2.
    """
    point = np.where(anomaly_lengths == 1)[0]
    two = np.where(anomaly_lengths == 2)[0]
    dropped = np.concatenate([point, two])
    n = len(anomaly_lengths)
    dropped_set = set(int(i) for i in dropped)
    orig_to_new = {
        int(o): int(o - np.sum(dropped < o))
        for o in range(n)
        if o not in dropped_set
    }
    new_to_orig = {new: orig for orig, new in orig_to_new.items()}
    return orig_to_new, new_to_orig, dropped


def resolve_paper_title_to_ucr(
    title: int,
    kept_ucr: set[int],
    new_to_orig: dict[int, int],
) -> int:
    """Resolve a paper title to a kept UCR Dataset id.

    Primary rule (tsad titles): ``UCR = title + 1``.
    Fallback if that series was dropped as a point anomaly: treat ``title`` as
    a **post-filter** row and map via ``new_to_orig``.
    """
    primary = paper_title_to_ucr(title)
    if primary in kept_ucr:
        return primary
    if title in new_to_orig:
        fallback = paper_title_to_ucr(new_to_orig[title])
        if fallback in kept_ucr:
            return fallback
    raise SystemExit(
        f"Cannot resolve paper title TS {title} to a kept UCR series "
        f"(tried UCR {primary} and post-filter fallback)."
    )


def _pick_series(
    cluster_names: pd.Series,
    cluster: str,
    forced: list[int],
    n: int,
    rank_scores: pd.Series | None,
) -> list[int]:
    """Forced ids in this cluster first, then highest rank_scores (or id order)."""
    members = cluster_names.index[cluster_names == cluster].astype(int).tolist()
    chosen = [i for i in forced if i in members]
    remaining = [i for i in members if i not in chosen]
    if rank_scores is not None:
        remaining = sorted(
            remaining,
            key=lambda i: float(rank_scores.get(i, -np.inf)),
            reverse=True,
        )
    else:
        remaining = sorted(remaining)
    return (chosen + remaining)[:n]


def _windowed_heatmap(
    ax,
    series: np.ndarray,
    scores: np.ndarray,
    anomaly_span: tuple[int, int],
    title: str,
    pad: int = 1000,
    point_size: float = 2.0,
) -> None:
    series = np.asarray(series, dtype=float).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    n = min(len(series), len(scores))
    series, scores = series[:n], scores[:n]
    a0, a1 = anomaly_span
    a0, a1 = max(0, a0), min(n - 1, a1)
    plot_start = max(0, a0 - pad)
    plot_end = min(n, a1 + pad + 1)
    x = np.arange(plot_start, plot_end)
    y = series[plot_start:plot_end]
    c = scores[plot_start:plot_end]
    lo, hi = float(np.nanmin(c)), float(np.nanmax(c))
    c_norm = (c - lo) / ((hi - lo) or 1.0)

    ax.axvspan(a0, a1, color="gray", alpha=0.2, zorder=0)
    ax.plot(x, y, color="gray", linewidth=0.5, alpha=0.35, zorder=1)
    sc = ax.scatter(
        x, y, c=c_norm, cmap=SCORE_CMAP, s=point_size, vmin=0, vmax=1,
        edgecolors="none", zorder=2,
    )
    ax.set_xlim(plot_start, plot_end - 1)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(False)
    return sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="clusters.csv (default: <figures_dir>/clusters.csv)",
    )
    parser.add_argument(
        "--forced",
        type=int,
        nargs="*",
        default=DEFAULT_FORCED,
        help=(
            "Paper titles (pre-filter positional) to force when using "
            "--from-clusters; mapped to UCR via title+1."
        ),
    )
    parser.add_argument(
        "--n-per-cluster",
        type=int,
        default=5,
        help="Examples per cluster when using --from-clusters.",
    )
    parser.add_argument(
        "--from-clusters",
        action="store_true",
        help="Select series from clusters.csv instead of the published Fig. 8 layout.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=1000,
        help="Timestamps of context kept on each side of the anomaly.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Scatter marker size (paper uses ~2).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Point-wise methods to aggregate (default: clustering list / all point_wise).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_paper_style()
    fig_dir = args.output_dir or cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = args.clusters or (cfg.figures_dir / "clusters.csv")

    methods = args.methods or cfg.point_methods_for_clustering
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    if not ucr_dir.exists():
        raise SystemExit(f"UCR data not found at {ucr_dir}")

    # Full table + tsad-style point-anomaly maps (positional 0 = UCR Dataset 1).
    anomaly_full = build_anomaly_df(ucr_dir)
    # Inclusive anomaly length to match binary label sums in tsad.
    anom_len = (
        anomaly_full["Anomaly End"].astype(int)
        - anomaly_full["Anomaly Start"].astype(int)
        + 1
    ).to_numpy()
    # anomaly_full is Dataset-sorted 1..250; positional i ↔ Dataset i+1.
    _, new_to_orig, dropped_pos = point_anomaly_index_maps(anom_len)
    anomaly_df = cfg.filter_sequence_anomalies(anomaly_full)
    kept_ucr = set(int(i) for i in anomaly_df.index)
    print(
        f"Point-anomaly filter: dropped {len(dropped_pos)} series "
        f"(positional {sorted(int(i) for i in dropped_pos)}); kept {len(kept_ucr)}."
    )

    # titles_by_cluster: paper titles for plot labels
    # ucr_by_cluster: UCR Dataset ids for loading series/scores
    if args.from_clusters:
        if not clusters_path.exists():
            raise SystemExit(f"clusters.csv not found: {clusters_path}")
        cluster_names = pd.read_csv(clusters_path, index_col=0).squeeze("columns")
        cluster_names.index = cluster_names.index.astype(int)
        rank_scores = None
        metrics_path = cfg.repo_root / "data" / "metrics" / "f1_pak.csv"
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path, index_col=0)
            mdf.index = mdf.index.astype(int)
            cols = [c for c in methods if c in mdf.columns]
            if cols:
                rank_scores = mdf[cols].median(axis=1)
        forced_ucr = [paper_title_to_ucr(t) for t in args.forced]
        ucr_by_cluster = {
            cluster: _pick_series(
                cluster_names,
                cluster,
                forced_ucr,
                args.n_per_cluster,
                rank_scores,
            )
            for cluster in CLUSTER_DISPLAY_ORDER
            if cluster in set(cluster_names.values)
        }
        titles_by_cluster = {
            c: [ucr_to_paper_title(u) for u in ids]
            for c, ids in ucr_by_cluster.items()
        }
    else:
        titles_by_cluster = {k: list(v) for k, v in PAPER_FIG8_TITLES.items()}
        ucr_by_cluster = {}
        print("Using published Fig. 8 titles; mapping TS T → UCR Dataset T+1.")
        for cluster, titles in titles_by_cluster.items():
            ucrs = [
                resolve_paper_title_to_ucr(t, kept_ucr, new_to_orig) for t in titles
            ]
            ucr_by_cluster[cluster] = ucrs
            for t, u in zip(titles, ucrs):
                via = "title+1" if u == paper_title_to_ucr(t) else "post-filter fallback"
                print(f"  {cluster}: TS {t} → UCR {u} ({via})")

    needed = sorted({sid for ids in ucr_by_cluster.values() for sid in ids})
    missing = [i for i in needed if i not in anomaly_df.index]
    if missing:
        raise SystemExit(f"Series not in filtered UCR table: {missing}")
    anomaly_df = anomaly_df.loc[needed]
    targets = build_binary_targets(anomaly_df)
    data_test, _ = load_ucr_series(ucr_dir, anomaly_df)

    print(f"Loading scores for {len(methods)} methods × {len(needed)} series…")
    df_scores = load_scores(
        cfg.results_dir / "Scores",
        methods,
        datasets=needed,
        schema=cfg.training_schema,
    )

    for cluster in CLUSTER_DISPLAY_ORDER:
        sample_ucr = ucr_by_cluster.get(cluster) or []
        sample_titles = titles_by_cluster.get(cluster) or [
            ucr_to_paper_title(u) for u in sample_ucr
        ]
        if not sample_ucr:
            print(f"Skip {cluster}: no series")
            continue
        print(f"{cluster}: titles={sample_titles} UCR={sample_ucr}")

        fig, axs = plt.subplots(len(sample_ucr), 1, figsize=(5.2, 1.55 * len(sample_ucr)))
        axs = np.atleast_1d(axs)
        last_sc = None
        for ax, sid, title_id in zip(axs, sample_ucr, sample_titles):
            arrays = []
            for m in methods:
                cell = df_scores.loc[sid, m] if m in df_scores.columns else None
                if cell is None or (isinstance(cell, float) and np.isnan(cell)):
                    continue
                arrays.append(np.asarray(cell, dtype=float).reshape(-1))
            if not arrays:
                ax.set_title(f"TS {title_id} (missing scores for UCR {sid})")
                continue
            agg = aggregate_scores(arrays)
            series = np.asarray(data_test.loc[sid], dtype=float).reshape(-1)
            lab = np.asarray(targets.loc[sid], dtype=int).reshape(-1)
            # align like main_paper
            n = min(len(series), len(agg), len(lab))
            series, agg, lab = series[-n:], agg[-n:], lab[-n:]
            nz = np.flatnonzero(lab)
            if len(nz) == 0:
                span = (0, n - 1)
            else:
                span = (int(nz[0]), int(nz[-1]))
            last_sc = _windowed_heatmap(
                ax,
                series,
                agg,
                anomaly_span=span,
                title=f"TS {title_id}",
                pad=args.context,
                point_size=args.point_size,
            )

        color = CLUSTER_COLORS.get(cluster, "#333")
        fig.suptitle(
            f"{CLUSTER_TITLE.get(cluster, cluster)}:",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
        if last_sc is not None:
            fig.colorbar(
                last_sc, ax=list(axs), fraction=0.025, pad=0.04, label="Anomaly score"
            )
        fig.subplots_adjust(hspace=0.45, top=0.92, right=0.88)
        short = cluster.replace("Point-", "").lower()
        out = fig_dir / f"fig8_heatmap_{short}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

    print("Fig. 8 heatmaps written to", fig_dir)


if __name__ == "__main__":
    main()

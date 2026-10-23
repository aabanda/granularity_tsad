"""Report for one series: curve + detector scores + per-metric table.

Index convention matches ``plot_fig8_cluster_heatmaps.py``: ``--series T`` is
the **paper title** ``TS T``, which loads UCR Dataset ``T + 1``.

Left column
  - top: time series (anomaly shaded)
  - below: anomaly scores for LSTMADalpha, FCVAE, SAND, MatrixProfile

Right column
  - performance table on that series: F1_pt, F1_seq, F1_PAK, wF1_PAK

Example
-------
    PYTHONPATH=. python scripts/plot_series_score_report.py --series 118
    PYTHONPATH=. python scripts/plot_series_score_report.py --ucr-id 119
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import (  # noqa: E402
    align_scores_targets,
    build_anomaly_df,
    build_binary_targets,
    load_scores,
)
from granularity_tsad.plots.style import set_paper_style  # noqa: E402

DEFAULT_METHODS = ["LSTMADalpha", "FCVAE", "SAND", "MatrixProfile"]
METHOD_COLORS = {
    "LSTMADalpha": "#1565c0",
    "FCVAE": "#7b1fa2",
    "SAND": "#ef6c00",
    "MatrixProfile": "#2e7d32",
}
METRIC_FILES = {
    "F1_pt": "f1_point.csv",
    "F1_seq": "f1_seq.csv",
    "F1_PAK": "f1_pak.csv",
    "wF1_PAK": "f1_wpak.csv",
}


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / ((hi - lo) or 1.0)


def _load_metrics(metrics_dir: Path, series_id: int, methods: list[str]) -> pd.DataFrame:
    """Rows = methods, columns = metric short names."""
    rows = {}
    for label, fname in METRIC_FILES.items():
        path = metrics_dir / fname
        if not path.exists():
            raise SystemExit(f"Missing metric table: {path}")
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(int)
        if series_id not in df.index:
            raise SystemExit(f"Series {series_id} not in {path}")
        for m in methods:
            if m not in df.columns:
                raise SystemExit(f"Method {m} not in {path}")
            rows.setdefault(m, {})[label] = float(df.loc[series_id, m])
    return pd.DataFrame(rows).T[list(METRIC_FILES)]


def plot_report(
    series: np.ndarray,
    scores: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    methods: list[str],
    *,
    paper_title: int | None,
    ucr_id: int,
    train_index: int,
    a0: int,
    a1: int,
    pad: int,
    density: str | None,
    out_path: Path,
) -> None:
    abs_a0, abs_a1 = a0, a1 + 1
    left = max(0, abs_a0 - pad)
    right = min(len(series), abs_a1 + pad)
    t = np.arange(left, right)

    n_score = len(methods)
    fig = plt.figure(figsize=(14.0, 2.2 + 1.35 * n_score))
    gs = GridSpec(
        1 + n_score,
        2,
        figure=fig,
        width_ratios=[2.4, 1.6],
        height_ratios=[1.45] + [1.0] * n_score,
        wspace=0.32,
        hspace=0.38,
    )

    # --- series ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, series[left:right], color="#263238", lw=0.9)
    ax0.axvspan(abs_a0, abs_a1, color="tomato", alpha=0.25, zorder=0)
    if paper_title is not None:
        title = f"TS {paper_title}  (UCR {ucr_id})"
    else:
        title = f"UCR {ucr_id}"
    if density:
        title += f"  [{density}]"
    title += f"  |  anomaly=[{a0}, {a1}]"
    ax0.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax0.set_ylabel("value", fontsize=10)
    ax0.tick_params(labelbottom=False)

    # --- scores ---
    score_axes = []
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[i + 1, 0], sharex=ax0)
        score_axes.append(ax)
        color = METHOD_COLORS.get(method, "#333333")
        if method not in scores:
            ax.text(0.5, 0.5, "missing score", transform=ax.transAxes, ha="center")
            ax.set_ylabel(method, fontsize=9, color=color)
            continue
        s = scores[method]
        full_score = np.full(len(series), np.nan, dtype=float)
        n = min(len(s), len(series) - train_index)
        full_score[train_index : train_index + n] = s[:n]
        window = full_score[left:right]
        valid = np.isfinite(window)
        if valid.any():
            y = np.full_like(window, np.nan)
            y[valid] = _minmax01(window[valid])
            ax.plot(t, y, color=color, lw=1.1)
        ax.axvspan(abs_a0, abs_a1, color="tomato", alpha=0.18, zorder=0)
        ax.set_ylabel(method, fontsize=9, color=color, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        if i < n_score - 1:
            ax.tick_params(labelbottom=False)
    score_axes[-1].set_xlabel("time index", fontsize=10)

    # --- per-metric ranking with values (2×2 on the right) ---
    gs_r = gs[:, 1].subgridspec(2, 2, wspace=0.35, hspace=0.45)
    metric_names = list(metrics.columns)
    short_name = {
        "LSTMADalpha": "LSTMADα",
        "MatrixProfile": "MatrixProfile",
        "FCVAE": "FCVAE",
        "SAND": "SAND",
    }
    for j, metric in enumerate(metric_names):
        ax_m = fig.add_subplot(gs_r[j // 2, j % 2])
        ax_m.set_axis_off()
        ax_m.set_title(metric, fontsize=11, fontweight="bold", color="#37474f", pad=2)
        order = metrics.loc[methods, metric].sort_values(ascending=False)
        lines = []
        colors = []
        weights = []
        for k, (method, val) in enumerate(order.items()):
            label = short_name.get(method, method)
            lines.append(f"{k + 1}. {label}   {val:.3f}")
            colors.append(METHOD_COLORS.get(method, "#333"))
            weights.append("bold" if k == 0 else "normal")
        # Draw top→bottom.
        y0 = 0.88
        for k, (line, color, weight) in enumerate(zip(lines, colors, weights)):
            ax_m.text(
                0.02,
                y0 - k * 0.22,
                line,
                transform=ax_m.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                color=color,
                fontweight=weight,
                family="sans-serif",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--series",
        type=int,
        default=118,
        help="Paper title TS T (same as Fig. 8); loads UCR Dataset T+1.",
    )
    parser.add_argument(
        "--ucr-id",
        type=int,
        default=None,
        help="Raw UCR Dataset id (skips paper title+1 mapping).",
    )
    parser.add_argument("--pad", type=int, default=1000, help="Context around anomaly (Fig. 8 default).")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "series_reports",
    )
    args = parser.parse_args()

    set_paper_style()
    cfg = load_config(args.config)
    methods = list(args.methods)

    # Fig. 8 convention: paper title TS T → UCR Dataset T+1.
    if args.ucr_id is not None:
        paper_title = None
        ucr_id = int(args.ucr_id)
    else:
        paper_title = int(args.series)
        ucr_id = paper_title + 1

    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    anomaly_df = cfg.filter_sequence_anomalies(build_anomaly_df(ucr_dir))
    if ucr_id not in anomaly_df.index:
        raise SystemExit(f"UCR {ucr_id} not in filtered UCR table")
    targets = build_binary_targets(anomaly_df)

    metrics_dir = cfg.data_dir / "metrics"
    metrics = _load_metrics(metrics_dir, ucr_id, methods)

    dens = None
    dens_path = metrics_dir / "with_density" / "f1_pak.csv"
    if dens_path.exists():
        ddf = pd.read_csv(dens_path, index_col=0)
        ddf.index = ddf.index.astype(int)
        if ucr_id in ddf.index and "density" in ddf.columns:
            dens = str(ddf.loc[ucr_id, "density"])

    print(
        f"Loading {'TS ' + str(paper_title) + ' → ' if paper_title is not None else ''}"
        f"UCR {ucr_id}"
    )
    df_scores = load_scores(
        cfg.results_dir / "Scores",
        methods,
        [ucr_id],
        schema=cfg.training_schema,
    )
    row = anomaly_df.loc[ucr_id]
    series = np.loadtxt(ucr_dir / row["File"]).astype(float).reshape(-1)
    train_index = int(row["Train Index"])
    a0, a1 = int(row["Anomaly Start"]), int(row["Anomaly End"])

    scores: dict[str, np.ndarray] = {}
    for m in methods:
        raw = df_scores.loc[ucr_id, m] if m in df_scores.columns else None
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            print(f"missing score: {m}")
            continue
        s, _ = align_scores_targets(np.asarray(raw, dtype=float), targets.loc[ucr_id], m)
        scores[m] = s

    print(metrics.round(3).to_string())
    stem = (
        f"report_ts_{paper_title}"
        if paper_title is not None
        else f"report_ucr_{ucr_id}"
    )
    plot_report(
        series,
        scores,
        metrics,
        methods,
        paper_title=paper_title,
        ucr_id=ucr_id,
        train_index=train_index,
        a0=a0,
        a1=a1,
        pad=args.pad,
        density=dens,
        out_path=args.output_dir / f"{stem}.png",
    )


if __name__ == "__main__":
    main()

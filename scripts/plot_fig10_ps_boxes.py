"""Per-series P×S range boxes (IQR), same layout as the median scatter.

Each dataset is a rectangle:
  - width  = IQR of point-wise method scores on that series
  - height = IQR of sequence-wise method scores on that series
Center mark = family medians (same as the scatter points).

Example
-------
    PYTHONPATH=. python scripts/plot_fig10_ps_boxes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]

from granularity_tsad.plots.style import (  # noqa: E402
    CLUSTER_COLORS,
    CLUSTER_DISPLAY_ORDER,
    set_paper_style,
)

METRIC_ORDER = ["F1_point", "F1_seq", "F1_PAK", "wF1_PAK"]
METRIC_LABELS = {
    "F1_point": r"$F1_{point}$",
    "F1_seq": r"$F1_{seq}$",
    "F1_PAK": r"$F1_{PAK}$",
    "wF1_PAK": r"$wF1_{PAK}$",
}
DEFAULT_CSV = REPO_ROOT / "figures" / "fig10_breakdown" / "fig10_scores_series_method.csv"
DEFAULT_OUT = REPO_ROOT / "figures" / "fig10_breakdown"


def _wpak_theoretical_max(k_values: list[int] | None = None) -> float:
    """Max ``wF1_PAK`` when F1(k)=1 for all k (same formula as ``wpak_auc``)."""
    from granularity_tsad.config import load_config
    from granularity_tsad.metrics.pak import DEFAULT_K_VALUES, wpak_auc

    if k_values is None:
        cfg_k = load_config().k_values
        k_values = list(cfg_k) if cfg_k else list(DEFAULT_K_VALUES)
    ones = [1.0] * len(k_values)
    return float(wpak_auc(ones, k_values=k_values))


def _normalize_wpak(df: pd.DataFrame, k_values: list[int] | None = None) -> pd.DataFrame:
    """Scale ``wF1_PAK`` scores by theoretical max so they lie in [0, 1]."""
    out = df.copy()
    mask = out["metric"].astype(str) == "wF1_PAK"
    if not mask.any():
        return out
    wmax = _wpak_theoretical_max(k_values)
    if wmax <= 0:
        raise ValueError(f"Invalid wF1 theoretical max: {wmax}")
    out.loc[mask, "score"] = out.loc[mask, "score"].astype(float) / wmax
    return out


def _series_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (series, metric): IQR and median of each family."""
    rows = []
    for (series_id, metric, cluster, density), g in df.groupby(
        ["series_id", "metric", "cluster", "density"], sort=False
    ):
        p = g.loc[g["family"] == "point_wise", "score"].astype(float)
        s = g.loc[g["family"] == "sequence_wise", "score"].astype(float)
        if p.empty or s.empty:
            continue
        rows.append(
            {
                "series_id": int(series_id),
                "metric": metric,
                "cluster": cluster,
                "density": density,
                "p_q25": float(p.quantile(0.25)),
                "p_q75": float(p.quantile(0.75)),
                "p_med": float(p.median()),
                "p_min": float(p.min()),
                "p_max": float(p.max()),
                "s_q25": float(s.quantile(0.25)),
                "s_q75": float(s.quantile(0.75)),
                "s_med": float(s.median()),
                "s_min": float(s.min()),
                "s_max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_ps_boxes(
    boxes: pd.DataFrame,
    out_dir: Path,
    *,
    spread: str = "iqr",
    alpha: float = 0.22,
    wpak_normalized: bool = False,
    outfile_stem: str | None = None,
) -> None:
    """Draw metric × density panels with one rectangle per series."""
    metrics = [m for m in METRIC_ORDER if m in set(boxes["metric"].astype(str))]
    clusters = [c for c in CLUSTER_DISPLAY_ORDER if c in set(boxes["cluster"].astype(str))]
    labels = dict(METRIC_LABELS)
    if wpak_normalized:
        labels["wF1_PAK"] = r"$\widetilde{wF1}_{PAK}$"

    if spread == "iqr":
        x0, x1, y0, y1 = "p_q25", "p_q75", "s_q25", "s_q75"
        spread_label = "IQR"
    elif spread == "minmax":
        x0, x1, y0, y1 = "p_min", "p_max", "s_min", "s_max"
        spread_label = "min–max"
    else:
        raise ValueError(f"Unknown spread: {spread!r}")

    fig, axs = plt.subplots(
        len(metrics),
        len(clusters),
        figsize=(3.4 * len(clusters), 3.2 * len(metrics)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            sub = boxes[(boxes["metric"] == metric) & (boxes["cluster"] == cluster)]
            color = CLUSTER_COLORS.get(cluster, "#555555")

            for _, row in sub.iterrows():
                left = float(row[x0])
                right = float(row[x1])
                bottom = float(row[y0])
                top = float(row[y1])
                # degenerate IQR (all methods equal) → thin visible mark
                width = max(right - left, 0.008)
                height = max(top - bottom, 0.008)
                if right - left < 0.008:
                    left = float(row["p_med"]) - width / 2
                if top - bottom < 0.008:
                    bottom = float(row["s_med"]) - height / 2
                ax.add_patch(
                    Rectangle(
                        (left, bottom),
                        width,
                        height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.6,
                        alpha=alpha,
                        zorder=2,
                    )
                )
                ax.plot(
                    row["p_med"],
                    row["s_med"],
                    marker=".",
                    markersize=3.5,
                    color=color,
                    alpha=0.9,
                    zorder=3,
                )

            ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1.0, zorder=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal", adjustable="box")
            n = len(sub)
            # who wins on medians (same summary as scatter)
            n_p = int((sub["p_med"] > sub["s_med"]).sum())
            n_s = int((sub["s_med"] > sub["p_med"]).sum())
            ax.text(
                0.04,
                0.96,
                f"n={n}\nP>S: {n_p}\nS>P: {n_s}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#333333",
            )
            if i == 0:
                ax.set_title(cluster, color=color)
            if j == 0:
                ax.set_ylabel(
                    f"{labels.get(metric, metric)}\nsequence-wise ({spread_label})"
                )
            if i == len(metrics) - 1:
                ax.set_xlabel(f"point-wise ({spread_label})")

    title = (
        f"Per-series score ranges: width = point-wise {spread_label}, "
        f"height = sequence-wise {spread_label}\n"
        "(dot = family median)"
    )
    if wpak_normalized:
        title += r"; $wF1$ scaled by theoretical max $\to[0,1]$"
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    tag = "iqr" if spread == "iqr" else "minmax"
    stem = outfile_stem or f"ps_boxes_{tag}_by_metric_density"
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--spread",
        choices=("iqr", "minmax"),
        default="iqr",
        help="Rectangle extent: IQR (default) or full min–max of methods.",
    )
    parser.add_argument("--alpha", type=float, default=0.22)
    parser.add_argument(
        "--normalize-wpak",
        action="store_true",
        help="Divide wF1_PAK by theoretical max (F1=1 ∀k) so it lies in [0, 1].",
    )
    args = parser.parse_args()

    set_paper_style()
    df = pd.read_csv(args.csv)
    if args.normalize_wpak:
        wmax = _wpak_theoretical_max()
        print(f"Normalizing wF1_PAK by theoretical max = {wmax:.6f}")
        df = _normalize_wpak(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boxes = _series_boxes(df)
    summary_name = (
        "ps_boxes_series_summary_wpak_norm.csv"
        if args.normalize_wpak
        else "ps_boxes_series_summary.csv"
    )
    boxes.to_csv(args.output_dir / summary_name, index=False)
    print(f"Wrote {args.output_dir / summary_name} ({len(boxes)} rows)")
    tag = "iqr" if args.spread == "iqr" else "minmax"
    stem = (
        f"ps_boxes_{tag}_wpak_norm_by_metric_density"
        if args.normalize_wpak
        else f"ps_boxes_{tag}_by_metric_density"
    )
    plot_ps_boxes(
        boxes,
        args.output_dir,
        spread=args.spread,
        alpha=args.alpha,
        wpak_normalized=args.normalize_wpak,
        outfile_stem=stem,
    )


if __name__ == "__main__":
    main()

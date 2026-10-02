"""Scatter P vs S (per-series family medians) faceted by metric × density.

Each point is one UCR series. Uses ``fig10_paired_medians_P_S.csv``.

Example
-------
    PYTHONPATH=. python scripts/plot_fig10_ps_scatter.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
DEFAULT_CSV = REPO_ROOT / "figures" / "fig10_breakdown" / "fig10_paired_medians_P_S.csv"
DEFAULT_OUT = REPO_ROOT / "figures" / "fig10_breakdown"


def plot_ps_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str))]
    clusters = [c for c in CLUSTER_DISPLAY_ORDER if c in set(df["cluster"].astype(str))]
    n_rows, n_cols = len(metrics), len(clusters)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 3.2 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            sub = df[(df["metric"] == metric) & (df["cluster"] == cluster)]
            color = CLUSTER_COLORS.get(cluster, "#555555")
            ax.scatter(
                sub["P_median"],
                sub["S_median"],
                s=18,
                alpha=0.65,
                color=color,
                edgecolors="none",
                zorder=3,
            )
            ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1.0, zorder=2)
            # count who wins
            n = len(sub)
            n_p = int((sub["P_median"] > sub["S_median"]).sum())
            n_s = int((sub["S_median"] > sub["P_median"]).sum())
            n_eq = n - n_p - n_s
            ax.text(
                0.04,
                0.96,
                f"n={n}\nP>S: {n_p}\nS>P: {n_s}"
                + (f"\n=: {n_eq}" if n_eq else ""),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#333333",
            )
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal", adjustable="box")
            if i == 0:
                ax.set_title(cluster, color=color)
            if j == 0:
                ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)}\nsequence-wise median")
            if i == n_rows - 1:
                ax.set_xlabel("point-wise median")

    fig.suptitle(
        "Per-series family medians: point-wise vs sequence-wise",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"ps_scatter_by_metric_density.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_delta_strip(df: pd.DataFrame, out_dir: Path) -> None:
    """Companion: Δ = P − S per series (same facet grid)."""
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str))]
    clusters = [c for c in CLUSTER_DISPLAY_ORDER if c in set(df["cluster"].astype(str))]
    n_rows, n_cols = len(metrics), len(clusters)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, metric in enumerate(metrics):
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            sub = df[(df["metric"] == metric) & (df["cluster"] == cluster)]
            color = CLUSTER_COLORS.get(cluster, "#555555")
            deltas = sub["delta_P_minus_S"].to_numpy(dtype=float)
            # jittered strip
            rng = np.random.default_rng(0)
            x = rng.normal(0.0, 0.04, size=len(deltas))
            ax.scatter(x, deltas, s=16, alpha=0.65, color=color, edgecolors="none", zorder=3)
            ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, zorder=2)
            if len(deltas):
                ax.axhline(
                    float(np.median(deltas)),
                    color=color,
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=4,
                )
            ax.set_xlim(-0.35, 0.35)
            ax.set_xticks([])
            if i == 0:
                ax.set_title(cluster, color=color)
            if j == 0:
                ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)}\n"
                              r"$\Delta$ = median(P) $-$ median(S)")
            if i == n_rows - 1:
                ax.set_xlabel("series (jittered)")

    fig.suptitle(
        r"Per-series paired difference $\Delta = P - S$ (line = median $\Delta$)",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"ps_delta_by_metric_density.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Skip the companion Δ strip plot.",
    )
    args = parser.parse_args()

    set_paper_style()
    df = pd.read_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_ps_scatter(df, args.output_dir)
    if not args.no_delta:
        plot_delta_strip(df, args.output_dir)


if __name__ == "__main__":
    main()

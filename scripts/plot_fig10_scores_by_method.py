"""Per-(series, method) scores by metric × density — no family aggregation.

Each point is one detector score on one UCR series (from
``fig10_scores_series_method.csv``). Methods are ordered point-wise then
sequence-wise so family differences are visible without taking a median.

Example
-------
    PYTHONPATH=. python scripts/plot_fig10_scores_by_method.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.plots.style import (  # noqa: E402
    CLUSTER_COLORS,
    CLUSTER_DISPLAY_ORDER,
    SCORE_CMAP,
    set_paper_style,
)

METRIC_ORDER = ["F1_point", "F1_seq", "F1_PAK", "wF1_PAK"]
METRIC_LABELS = {
    "F1_point": r"$F1_{point}$",
    "F1_seq": r"$F1_{seq}$",
    "F1_PAK": r"$F1_{PAK}$",
    "wF1_PAK": r"$wF1_{PAK}$",
}
FAMILY_COLORS = {"point_wise": "#1565c0", "sequence_wise": "#ef6c00"}
DEFAULT_CSV = REPO_ROOT / "figures" / "fig10_breakdown" / "fig10_scores_series_method.csv"
DEFAULT_OUT = REPO_ROOT / "figures" / "fig10_breakdown"


def _method_order(df: pd.DataFrame) -> list[str]:
    cfg = load_config()
    preferred = list(
        dict.fromkeys(cfg.point_methods_for_analysis + cfg.methods_sequence_wise)
    )
    present = list(df["method"].astype(str).unique())
    ordered = [m for m in preferred if m in present]
    ordered += sorted(set(present) - set(ordered))
    return ordered


def plot_scores_by_method(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str))]
    clusters = [c for c in CLUSTER_DISPLAY_ORDER if c in set(df["cluster"].astype(str))]
    methods = _method_order(df)
    method_to_x = {m: i for i, m in enumerate(methods)}
    family_of = (
        df.drop_duplicates("method").set_index("method")["family"].to_dict()
    )

    fig, axs = plt.subplots(
        len(metrics),
        len(clusters),
        figsize=(max(10.5, 0.55 * len(methods) * len(clusters) / 2), 3.0 * len(metrics)),
        sharey="row",
        squeeze=False,
    )
    rng = np.random.default_rng(0)

    for i, metric in enumerate(metrics):
        for j, cluster in enumerate(clusters):
            ax = axs[i, j]
            sub = df[(df["metric"] == metric) & (df["cluster"] == cluster)].copy()
            # light boxplot per method
            box_data, box_pos, box_colors = [], [], []
            for m in methods:
                vals = sub.loc[sub["method"] == m, "score"].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                box_data.append(vals)
                box_pos.append(method_to_x[m])
                fam = family_of.get(m, "point_wise")
                box_colors.append(FAMILY_COLORS.get(fam, "#555"))

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    positions=box_pos,
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="white", linewidth=1.1),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    boxprops=dict(linewidth=0.8, alpha=0.55),
                )
                for patch, color in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)

            # series points (jitter)
            for m in methods:
                vals = sub.loc[sub["method"] == m, "score"].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                fam = family_of.get(m, "point_wise")
                x = method_to_x[m] + rng.normal(0.0, 0.08, size=len(vals))
                ax.scatter(
                    x,
                    vals,
                    s=8,
                    alpha=0.35,
                    color=FAMILY_COLORS.get(fam, "#555"),
                    edgecolors="none",
                    zorder=3,
                )

            # family separator between last point-wise and first sequence-wise
            seq_start = next(
                (method_to_x[m] for m in methods if family_of.get(m) == "sequence_wise"),
                None,
            )
            if seq_start is not None and seq_start > 0:
                ax.axvline(seq_start - 0.5, color="#bbbbbb", linestyle="--", linewidth=0.9)

            ax.set_xlim(-0.7, len(methods) - 0.3)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(range(len(methods)))
            if i == len(metrics) - 1:
                ax.set_xticklabels(methods, rotation=75, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            if i == 0:
                ax.set_title(cluster, color=CLUSTER_COLORS.get(cluster, "#333"))

    # legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS["point_wise"],
            markersize=7,
            label="point-wise",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLORS["sequence_wise"],
            markersize=7,
            label="sequence-wise",
        ),
    ]
    axs[0, -1].legend(handles=handles, frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Per-series detector scores (no family aggregation)",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"scores_by_method_metric_density.{ext}"
        fig.savefig(path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_method_vs_series_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """One heatmap per metric: rows=series (grouped by cluster), cols=methods."""
    methods = _method_order(df)
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str))]

    for metric in metrics:
        sub = df[df["metric"] == metric].copy()
        # order series by cluster then id
        sub["cluster"] = pd.Categorical(sub["cluster"], CLUSTER_DISPLAY_ORDER, ordered=True)
        series_order = (
            sub[["series_id", "cluster"]]
            .drop_duplicates()
            .sort_values(["cluster", "series_id"])["series_id"]
            .tolist()
        )
        mat = (
            sub.pivot_table(index="series_id", columns="method", values="score", aggfunc="first")
            .reindex(index=series_order, columns=methods)
        )
        cluster_of = (
            sub.drop_duplicates("series_id").set_index("series_id")["cluster"].reindex(series_order)
        )

        fig_h = max(8.0, 0.035 * len(series_order))
        fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(methods)), fig_h))
        im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=75, ha="right", fontsize=8)
        # sparse y ticks: cluster boundaries
        ax.set_yticks([])
        y = 0
        for cluster in CLUSTER_DISPLAY_ORDER:
            n = int((cluster_of == cluster).sum())
            if n == 0:
                continue
            mid = y + n / 2 - 0.5
            ax.text(
                -0.8,
                mid,
                cluster.replace("Point-", ""),
                va="center",
                ha="right",
                fontsize=8,
                color=CLUSTER_COLORS.get(cluster, "#333"),
                clip_on=False,
            )
            if y > 0:
                ax.axhline(y - 0.5, color="white", linewidth=1.2)
            y += n
        # family separator
        fam = df.drop_duplicates("method").set_index("method")["family"].to_dict()
        seq_start = next((i for i, m in enumerate(methods) if fam.get(m) == "sequence_wise"), None)
        if seq_start:
            ax.axvline(seq_start - 0.5, color="white", linewidth=1.5)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — series × method")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        safe = metric.replace("/", "_")
        for ext in ("pdf", "png"):
            path = out_dir / f"heatmap_{safe}_series_method.{ext}"
            fig.savefig(path, dpi=120 if ext == "png" else None, bbox_inches="tight")
            print(f"Wrote {path}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip series×method heatmaps.",
    )
    args = parser.parse_args()

    set_paper_style()
    df = pd.read_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_scores_by_method(df, args.output_dir)
    if not args.no_heatmap:
        plot_method_vs_series_heatmap(df, args.output_dir)


if __name__ == "__main__":
    main()

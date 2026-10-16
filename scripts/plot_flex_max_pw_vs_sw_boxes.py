"""Boxplots of FlexMax mean scores: point-wise vs sequence-wise by density.

Reads ``data/metrics/with_density/flex_max_mean_by_density.csv`` (one row per
density cluster, one column per method) and draws three panels (dense / sparse /
normal). Each panel has two boxplots: the distribution of method means in that
cluster for point-wise vs sequence-wise detectors.

Example
-------
    PYTHONPATH=. python scripts/plot_flex_max_pw_vs_sw_boxes.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.plots.style import (  # noqa: E402
    CLUSTER_COLORS,
    CLUSTER_DISPLAY_ORDER,
    set_paper_style,
    style_axes,
)

DEFAULT_CSV = (
    REPO_ROOT / "data" / "metrics" / "with_density" / "flex_max_mean_by_density.csv"
)
DEFAULT_OUT = REPO_ROOT / "figures" / "flex_max_pw_vs_sw_by_density.pdf"

# CSV density key → paper cluster name (colors / panel order).
DENSITY_TO_CLUSTER = {
    "dense": "Point-dense",
    "sparse": "Point-sparse",
    "normal": "Point-normal",
}
FAMILY_COLORS = {"point_wise": "#1565c0", "sequence_wise": "#ef6c00"}
FAMILY_LABELS = {"point_wise": "Point-wise", "sequence_wise": "Sequence-wise"}


def _method_family(method: str, seq_methods: set[str]) -> str:
    return "sequence_wise" if method in seq_methods else "point_wise"


def melt_by_family(df: pd.DataFrame, seq_methods: set[str]) -> pd.DataFrame:
    """Long table: density, cluster, family, method, score."""
    method_cols = [c for c in df.columns if c != "density"]
    rows = []
    for _, row in df.iterrows():
        density = str(row["density"]).strip().lower()
        cluster = DENSITY_TO_CLUSTER.get(density, density)
        for m in method_cols:
            val = row[m]
            if pd.isna(val):
                continue
            fam = _method_family(m, seq_methods)
            rows.append(
                {
                    "density": density,
                    "cluster": cluster,
                    "family": fam,
                    "method": m,
                    "score": float(val),
                }
            )
    return pd.DataFrame(rows)


def plot_boxes(long_df: pd.DataFrame, out: Path) -> None:
    clusters = [c for c in CLUSTER_DISPLAY_ORDER if c in set(long_df["cluster"])]
    fig, axs = plt.subplots(
        1,
        len(clusters),
        figsize=(3.6 * len(clusters), 4.2),
        sharey=True,
        squeeze=False,
    )
    axs = axs[0]
    family_order = ["point_wise", "sequence_wise"]

    for ax, cluster in zip(axs, clusters):
        style_axes(ax)
        sub = long_df[long_df["cluster"] == cluster]
        data = [
            sub.loc[sub["family"] == fam, "score"].to_numpy(dtype=float)
            for fam in family_order
        ]
        positions = [0, 1]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "none",
                "markeredgecolor": "#555555",
            },
        )
        for patch, fam in zip(bp["boxes"], family_order):
            patch.set_facecolor(FAMILY_COLORS[fam])
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333333")
        for key in ("whiskers", "caps", "medians"):
            for line in bp[key]:
                line.set_color("#333333")
                line.set_linewidth(1.2)
        ax.set_xticks(positions)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in family_order], fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        color = CLUSTER_COLORS.get(cluster, "#333")
        panel = cluster.replace("Point-", "").capitalize()  # Dense / Sparse / Normal
        ax.set_title(panel, color=color, fontweight="bold", fontsize=14)
        n_p = int((sub["family"] == "point_wise").sum())
        n_s = int((sub["family"] == "sequence_wise").sum())
        ax.text(
            0.98,
            0.98,
            f"n={n_p}+{n_s}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#555555",
        )

    axs[0].set_ylabel("FlexMax mean score", fontsize=12)
    fig.suptitle(
        "FlexMax mean by density: point-wise vs sequence-wise methods",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "tight"}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    set_paper_style()
    cfg = load_config(args.config)
    seq_methods = set(cfg.methods_sequence_wise)

    df = pd.read_csv(args.csv)
    if "density" not in df.columns:
        raise SystemExit(f"Expected a 'density' column in {args.csv}")

    long_df = melt_by_family(df, seq_methods)
    # Sanity: report which columns went where.
    methods = sorted(long_df["method"].unique())
    point = [m for m in methods if m not in seq_methods]
    seq = [m for m in methods if m in seq_methods]
    print(f"Point-wise ({len(point)}): {point}")
    print(f"Sequence-wise ({len(seq)}): {seq}")

    plot_boxes(long_df, args.output)


if __name__ == "__main__":
    main()

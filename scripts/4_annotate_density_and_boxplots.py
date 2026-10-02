"""Annotate UCR metric CSVs with cluster density and regenerate boxplots.

Reads ``figures/clusters.csv`` (Point-dense / Point-sparse / Point-normal),
adds a ``density`` column (``dense`` / ``sparse`` / ``normal``) to each of
``f1_point.csv``, ``f1_seq.csv``, ``f1_pak.csv``, ``f1_wpak.csv``,
``flex_max.csv``, writes the annotated tables under ``data/metrics/with_density/``,
and regenerates the cluster-stratified boxplots (plus a FLEX_MAX mean table).

Example
-------
    python scripts/4_annotate_density_and_boxplots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.clustering import (  # noqa: E402
    CLUSTER_TO_DENSITY,
    DENSITY_TO_CLUSTER,
)
from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.plots import (  # noqa: E402
    plot_metric_comparison_by_density,
    set_paper_style,
)

METRIC_FILES = {
    "F1_point": "f1_point.csv",
    "F1_seq": "f1_seq.csv",
    "F1_PAK": "f1_pak.csv",
    "wF1_PAK": "f1_wpak.csv",
    "FLEX_MAX": "flex_max.csv",
}


def _load_clusters(path: Path) -> pd.Series:
    clusters = pd.read_csv(path, index_col=0).squeeze("columns")
    if not isinstance(clusters, pd.Series):
        clusters = clusters.iloc[:, 0]
    clusters.index = clusters.index.astype(int)
    return clusters.rename("cluster")


def _annotate(metric_df: pd.DataFrame, density: pd.Series) -> pd.DataFrame:
    out = metric_df.copy()
    out.index = out.index.astype(int)
    aligned = density.reindex(out.index)
    missing = int(aligned.isna().sum())
    if missing:
        raise ValueError(
            f"{missing} series in metric table have no cluster assignment"
        )
    # Keep metrics first, density last.
    out = out.drop(columns=["density"], errors="ignore")
    out["density"] = aligned.astype(str).values
    return out


def _metric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["density"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="Path to clusters.csv (default: <figures_dir>/clusters.csv)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Directory with f1_*.csv (default: data/metrics)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write annotated CSVs (default: data/metrics/with_density)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_paper_style()

    metrics_dir = args.metrics_dir or (REPO_ROOT / "data" / "metrics")
    out_dir = args.output_dir or (metrics_dir / "with_density")
    clusters_path = args.clusters or (cfg.figures_dir / "clusters.csv")
    if not clusters_path.exists():
        alt = REPO_ROOT / "figures" / "all_methods" / "clusters.csv"
        if alt.exists():
            clusters_path = alt
        else:
            raise SystemExit(
                f"clusters.csv not found at {clusters_path}. "
                "Run scripts/3_generate_figures.py first."
            )

    cluster_names = _load_clusters(clusters_path)
    density = cluster_names.map(CLUSTER_TO_DENSITY)
    if density.isna().any():
        unknown = sorted(set(cluster_names[density.isna()].astype(str)))
        raise SystemExit(f"Unknown cluster labels (no density map): {unknown}")

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated: dict[str, pd.DataFrame] = {}
    for metric_name, fname in METRIC_FILES.items():
        src = metrics_dir / fname
        if not src.exists():
            print(f"Skip missing {src}")
            continue
        raw = pd.read_csv(src, index_col=0)
        table = _annotate(raw, density)
        table.to_csv(out_dir / fname)
        annotated[metric_name] = table
        counts = table["density"].value_counts().to_dict()
        print(f"Wrote {out_dir / fname}  rows={len(table)}  density={counts}")

    if not annotated:
        raise SystemExit(f"No metric CSVs found in {metrics_dir}")

    # Boxplot panels use paper names; CSV stores short density labels.
    group = (
        annotated[next(iter(annotated))]["density"]
        .map(DENSITY_TO_CLUSTER)
        .rename("Anomaly cluster")
    )
    # Fig 9 uses F1_point / F1_seq / F1_PAK (not wF1_PAK / FLEX_MAX).
    boxplot_metrics = {
        name: _metric_only(df)
        for name, df in annotated.items()
        if name in {"F1_point", "F1_seq", "F1_PAK"}
    }
    algorithm_order = [
        m
        for m in cfg.point_methods_for_analysis
        if any(m in df.columns for df in boxplot_metrics.values())
    ]

    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    if boxplot_metrics:
        fig = plot_metric_comparison_by_density(
            boxplot_metrics,
            group,
            group_name="Anomaly cluster",
            algorithm_order=algorithm_order or None,
        )
        out_pdf = fig_dir / "metrics_by_cluster_from_density_csv.pdf"
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"Saved {out_pdf}")

    # BINARY FLEX_MAX: mean hit-rate by density × method (+ optional boxplot).
    if "FLEX_MAX" in annotated:
        flex = annotated["FLEX_MAX"]
        methods = [c for c in flex.columns if c != "density"]
        summary = (
            flex.groupby("density")[methods]
            .mean()
            .reindex(["dense", "sparse", "normal"])
        )
        summary_path = out_dir / "flex_max_mean_by_density.csv"
        summary.to_csv(summary_path)
        print(f"Wrote {summary_path}")
        print(summary.round(3).to_string())

        flex_box = {"FLEX_MAX": _metric_only(flex)}
        flex_order = [m for m in cfg.all_methods if m in methods] or methods
        fig_flex = plot_metric_comparison_by_density(
            flex_box,
            group,
            group_name="Anomaly cluster",
            algorithm_order=flex_order,
        )
        flex_pdf = fig_dir / "flex_max_by_cluster.pdf"
        fig_flex.savefig(flex_pdf)
        plt.close(fig_flex)
        print(f"Saved {flex_pdf}")

    print(
        "Compare visually with figures/metrics_by_cluster.pdf "
        "(same clustering + metrics → same boxes)."
    )


if __name__ == "__main__":
    main()

"""Generate the paper figures from precomputed metrics and scores.

Reproduces the granularity experiments of the paper:

* **Fig 6** dendrogram of the point-detectability profiles.
* **Fig 7** point-detectability profiles grouped by cluster.
* **Fig 8** score heatmaps for representative series per cluster
  (``scripts/plot_fig8_cluster_heatmaps.py``, uses saved scores).
* **Fig 9** per-metric boxplots of detector performance by anomaly cluster
  (point-dense / point-sparse / point-normal).
* **Fig 10** Bayesian point-wise vs sequence-wise comparison, per metric and
  per cluster (``F1_point``, ``F1_seq``, ``F1_PAK``).
* **Fig 12** the same Bayesian comparison under the stretched ``wF1_PAK``.

Figures are saved as editable vector graphics (PDF) under the configured
figures dir, alongside the cluster assignment and the Bayesian summary tables.

Example
-------
    python scripts/3_generate_figures.py --skip-profiles
    python scripts/plot_fig8_cluster_heatmaps.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.aggregation import (
    build_profiles,
    compute_pak_curve_frames,
)  # noqa: E402
from granularity_tsad.clustering import (  # noqa: E402
    CLUSTER_TO_DENSITY,
    cluster_profiles,
    name_clusters,
)
from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import (  # noqa: E402
    build_anomaly_df,
    build_binary_targets,
    load_scores,
)
from granularity_tsad.plots import (  # noqa: E402
    plot_dendrogram,
    plot_metric_comparison_by_density,
    plot_point_vs_seq_grid,
    plot_profiles_by_cluster,
    set_paper_style,
)

METRIC_FILES = {
    "F1_point": "f1_point.csv",
    "F1_seq": "f1_seq.csv",
    "F1_PAK": "f1_pak.csv",
}
WPAK_FILE = ("wF1_PAK", "f1_wpak.csv")


def _load_metric(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # Annotated CSVs may carry a density column; never treat it as a method.
    return df.drop(columns=["density"], errors="ignore")


def _write_metrics_with_density(
    metrics_dir: Path,
    cluster_names: pd.Series,
    fig_dir: Path,
) -> None:
    """Write f1_*.csv copies with a dense/sparse/normal density column."""
    density = cluster_names.map(CLUSTER_TO_DENSITY)
    if density.isna().any():
        print("Skipping density-annotated metrics (unknown cluster labels).")
        return
    out_dir = metrics_dir / "with_density"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in list(METRIC_FILES.values()) + [WPAK_FILE[1]]:
        src = metrics_dir / fname
        if not src.exists():
            continue
        table = pd.read_csv(src, index_col=0)
        table.index = table.index.astype(int)
        aligned = density.reindex(table.index.astype(int))
        table = table.drop(columns=["density"], errors="ignore")
        table["density"] = aligned.astype(str).values
        table.to_csv(out_dir / fname)
    # Also stash cluster → density next to figures for traceability.
    pd.DataFrame({"cluster": cluster_names, "density": density}).to_csv(
        fig_dir / "clusters_with_density.csv"
    )
    print(f"Wrote density-annotated metrics to {out_dir}")


def _bayesian_figures(
    metric_tables: dict[str, pd.DataFrame],
    point_methods: list[str],
    seq_methods: list[str],
    cluster_names: pd.Series,
    fig_dir: Path,
    tag: str,
) -> None:
    """Save one unified Bayesian grid (rows=metrics, cols=clusters) + CSV."""
    fig, summary = plot_point_vs_seq_grid(
        metric_tables, point_methods, seq_methods, cluster_names
    )
    fig.savefig(fig_dir / f"bayesian_{tag}.pdf")
    plt.close(fig)
    if not summary.empty:
        summary.to_csv(fig_dir / f"bayesian_{tag}.csv", index=False)
        print(f"Saved bayesian_{tag}.pdf / .csv ({len(summary)} cells)")
    else:
        print(f"No Bayesian cells for tag={tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="Reuse an existing clusters.csv and skip recomputing detectability "
        "profiles (Fig 6–7). Boxplots/Bayesian still run from saved metrics.",
    )
    parser.add_argument(
        "--skip-profiles",
        action="store_true",
        help="Like --clusters with the default <figures_dir>/clusters.csv.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_paper_style()
    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = cfg.repo_root / "data" / "metrics"
    if not metrics_dir.exists():
        metrics_dir = cfg.data_dir / "metrics"
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    methods = args.methods or cfg.all_methods
    point_methods_cluster = cfg.point_methods_for_clustering
    point_methods_analysis = cfg.point_methods_for_analysis
    seq_methods = cfg.methods_sequence_wise
    mode = "split" if cfg.split_point_wise else "all point_wise"
    print(
        f"Method mode: {mode}\n"
        f"  clustering ({len(point_methods_cluster)}): {point_methods_cluster}\n"
        f"  analysis   ({len(point_methods_analysis)}): {point_methods_analysis}\n"
        f"  sequence_wise_extended={cfg.sequence_wise_extended} "
        f"({len(seq_methods)}): {seq_methods}"
    )

    clusters_path = args.clusters
    if args.skip_profiles and clusters_path is None:
        clusters_path = fig_dir / "clusters.csv"

    cluster_names = pd.Series(dtype=object)
    if clusters_path is not None:
        if not Path(clusters_path).exists():
            print(f"clusters file not found: {clusters_path}")
            return
        cluster_names = pd.read_csv(clusters_path, index_col=0).squeeze("columns")
        cluster_names.index = cluster_names.index.astype(int)
        print(f"Loaded clusters from {clusters_path} (skipping profile recompute)")
        _write_metrics_with_density(metrics_dir, cluster_names, fig_dir)
    else:
        # --- Detectability profiles -> clustering (Fig 6 & 7) ---
        if not ucr_dir.exists():
            print(
                f"UCR data not found at {ucr_dir}. See the 'Data' section in README.md."
            )
            return
        anomaly_df = cfg.filter_sequence_anomalies(build_anomaly_df(ucr_dir))
        datasets = anomaly_df.index.tolist()
        targets = build_binary_targets(anomaly_df)
        df_scores = load_scores(
            cfg.results_dir / "Scores",
            point_methods_cluster,
            datasets,
            schema=cfg.training_schema,
        )
        frames = compute_pak_curve_frames(
            df_scores, targets, point_methods_cluster, cfg.k_values or None
        )
        profiles = build_profiles(
            frames["precision"], frames["recall"], point_methods_cluster
        )

        if not profiles.empty:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            plot_dendrogram(profiles, ax=ax)
            fig.savefig(fig_dir / "dendrogram.pdf")
            plt.close(fig)
            print("Saved dendrogram.pdf (Fig 6)")

            labels = cluster_profiles(profiles)
            names = name_clusters(labels, profiles)
            cluster_names = labels.map(names)
            cluster_names.to_csv(fig_dir / "clusters.csv")
            _write_metrics_with_density(metrics_dir, cluster_names, fig_dir)

            n_k = len(cfg.k_values) if cfg.k_values else (profiles.shape[1] // 2)
            fig = plot_profiles_by_cluster(profiles, cluster_names, n_k=n_k)
            fig.savefig(fig_dir / "profiles_by_cluster.pdf")
            plt.close(fig)
            print("Saved profiles_by_cluster.pdf (Fig 7)")

    # --- Load metric tables ---
    metric_tables = {
        name: _load_metric(metrics_dir / fname) for name, fname in METRIC_FILES.items()
    }
    metric_tables = {k: v for k, v in metric_tables.items() if v is not None}
    wpak_name, wpak_fname = WPAK_FILE
    wpak_table = _load_metric(metrics_dir / wpak_fname)

    if not metric_tables:
        print(
            "No metric tables found in",
            metrics_dir,
            "- run 2_compute_metrics.py first.",
        )
        return

    if cluster_names.empty:
        print(
            "No cluster assignment available (need detector scores); skipping "
            "cluster-stratified figures."
        )
        return

    # --- Boxplots of metrics by anomaly cluster (Fig 9) ---
    group = cluster_names.reindex(next(iter(metric_tables.values())).index)
    fig = plot_metric_comparison_by_density(
        metric_tables,
        group,
        group_name="Anomaly cluster",
        algorithm_order=point_methods_analysis,
    )
    fig.savefig(fig_dir / "metrics_by_cluster.pdf")
    plt.close(fig)
    print("Saved metrics_by_cluster.pdf (Fig 9)")

    # --- Bayesian point-wise vs sequence-wise by cluster (Fig 10) ---
    _bayesian_figures(
        metric_tables,
        point_methods_analysis,
        seq_methods,
        cluster_names,
        fig_dir,
        "pak",
    )

    # --- Same comparison under wF1_PAK (Fig 12) ---
    if wpak_table is not None:
        _bayesian_figures(
            {wpak_name: wpak_table},
            point_methods_analysis,
            seq_methods,
            cluster_names,
            fig_dir,
            "wpak",
        )

    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()

"""Generate the paper figures from precomputed metrics and scores.

Reproduces the granularity experiments of the paper:

* **Fig 6** dendrogram of the point-detectability profiles.
* **Fig 7** point-detectability profiles grouped by cluster.
* **Fig 9** per-metric boxplots of detector performance by anomaly cluster
  (point-dense / point-sparse / point-normal).
* **Fig 10** Bayesian point-wise vs sequence-wise comparison, per metric and
  per cluster (``F1_point``, ``F1_seq``, ``F1_PAK``).
* **Fig 12** the same Bayesian comparison under the stretched ``wF1_PAK``.

Figures are saved as editable vector graphics (PDF) under the configured
figures dir, alongside the cluster assignment and the Bayesian summary tables.

Example
-------
    python scripts/3_generate_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.aggregation import build_profiles, compute_pak_curve_frames  # noqa: E402
from granularity_tsad.clustering import cluster_profiles, name_clusters  # noqa: E402
from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import (  # noqa: E402
    build_anomaly_df,
    build_binary_targets,
    load_scores,
)
from granularity_tsad.plots import (  # noqa: E402
    plot_dendrogram,
    plot_metric_comparison_by_density,
    plot_profiles_by_cluster,
    point_vs_seq,
    point_vs_seq_by_cluster,
    set_paper_style,
)

METRIC_FILES = {
    "F1_point": "f1_point.csv",
    "F1_seq": "f1_seq.csv",
    "F1_PAK": "f1_pak.csv",
}
WPAK_FILE = ("wF1_PAK", "f1_wpak.csv")


def _load_metric(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, index_col=0) if path.exists() else None


def _bayesian_figures(
    metric_tables: dict[str, pd.DataFrame],
    point_methods: list[str],
    seq_methods: list[str],
    cluster_names: pd.Series,
    fig_dir: Path,
    tag: str,
) -> None:
    """Save per-(metric, cluster) simplex plots and a combined summary table."""
    summaries = []
    for metric_name, table in metric_tables.items():
        summary = point_vs_seq_by_cluster(
            table.dropna(how="all"), point_methods, seq_methods, cluster_names
        )
        if summary.empty:
            continue
        summary.insert(0, "Metric", metric_name)
        summaries.append(summary)

        aligned = cluster_names.reindex(table.index)
        rope = float(summary["rope"].iloc[0])
        for cluster in summary["Cluster"]:
            sub = table.loc[aligned.index[aligned == cluster]]
            probs, fig, _ = point_vs_seq(
                sub, point_methods, seq_methods, rope=rope, plot=True
            )
            if fig is None:
                continue
            safe = f"{metric_name}_{cluster}".replace(" ", "_").replace("/", "-")
            fig.savefig(fig_dir / f"bayesian_{safe}.pdf")
            plt.close(fig)

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        out.to_csv(fig_dir / f"bayesian_{tag}.csv", index=False)
        print(f"Saved bayesian_{tag}.csv ({len(out)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_paper_style()
    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = cfg.data_dir / "metrics"
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    if not ucr_dir.exists():
        print(f"UCR data not found at {ucr_dir}. See the 'Data' section in README.md.")
        return
    methods = args.methods or cfg.all_methods
    point_methods = cfg.methods_point_wise
    seq_methods = cfg.methods_sequence_wise

    # --- Detectability profiles -> clustering (Fig 6 & 7) ---
    anomaly_df = cfg.filter_sequence_anomalies(build_anomaly_df(ucr_dir))
    datasets = anomaly_df.index.tolist()
    targets = build_binary_targets(anomaly_df)
    df_scores = load_scores(
        cfg.results_dir / "Scores", point_methods, datasets, schema=cfg.training_schema
    )
    frames = compute_pak_curve_frames(df_scores, targets, point_methods, cfg.k_values or None)
    profiles = build_profiles(frames["precision"], frames["recall"], point_methods)

    cluster_names = pd.Series(dtype=object)
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
        print("No metric tables found in", metrics_dir, "- run 2_compute_metrics.py first.")
        return

    if cluster_names.empty:
        print("No cluster assignment available (need detector scores); skipping "
              "cluster-stratified figures.")
        return

    # --- Boxplots of metrics by anomaly cluster (Fig 9) ---
    group = cluster_names.reindex(next(iter(metric_tables.values())).index)
    fig = plot_metric_comparison_by_density(
        metric_tables, group, group_name="Anomaly cluster", algorithm_order=point_methods
    )
    fig.savefig(fig_dir / "metrics_by_cluster.pdf")
    plt.close(fig)
    print("Saved metrics_by_cluster.pdf (Fig 9)")

    # --- Bayesian point-wise vs sequence-wise by cluster (Fig 10) ---
    _bayesian_figures(metric_tables, point_methods, seq_methods, cluster_names, fig_dir, "pak")

    # --- Same comparison under wF1_PAK (Fig 12) ---
    if wpak_table is not None:
        _bayesian_figures(
            {wpak_name: wpak_table}, point_methods, seq_methods, cluster_names, fig_dir, "wpak"
        )

    print("Figures written to", fig_dir)


if __name__ == "__main__":
    main()

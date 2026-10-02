"""Regenerate figures/split with corrected cluster naming."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from granularity_tsad.aggregation import build_profiles, compute_pak_curve_frames
from granularity_tsad.clustering import cluster_profiles, name_clusters
from granularity_tsad.config import load_config
from granularity_tsad.data import build_anomaly_df, build_binary_targets, load_scores
from granularity_tsad.plots import (
    plot_dendrogram,
    plot_metric_comparison_by_density,
    plot_point_vs_seq_grid,
    plot_profiles_by_cluster,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]


def _load_metric(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0).drop(columns=["density"], errors="ignore")


def main() -> None:
    cfg = load_config()
    set_paper_style()

    ucr_dir = REPO / "data" / "raw" / "UCR_Anomaly_FullData"
    scores_root = REPO / "Results" / "Scores"
    metrics_dir = REPO / "data" / "metrics"
    fig_dir = REPO / "figures" / "split"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load((REPO / "config" / "paper.yaml").read_text())
    point_cluster = list(raw["methods"]["point_wise_for_clustering"])
    point_analysis = list(raw["methods"]["point_wise_for_analysis"])
    seq_methods = list(raw["methods"]["sequence_wise"])
    print("clustering", point_cluster)
    print("analysis", point_analysis)

    anomaly_df = cfg.filter_sequence_anomalies(build_anomaly_df(ucr_dir))
    datasets = anomaly_df.index.tolist()
    targets = build_binary_targets(anomaly_df)
    print(f"Computing profiles for {len(datasets)} series…", flush=True)
    df_scores = load_scores(
        scores_root, point_cluster, datasets, schema=cfg.training_schema
    )
    frames = compute_pak_curve_frames(
        df_scores, targets, point_cluster, cfg.k_values or None
    )
    profiles = build_profiles(frames["precision"], frames["recall"], point_cluster)
    print("profiles", profiles.shape, flush=True)

    labels = cluster_profiles(profiles)
    names = name_clusters(labels, profiles)
    cluster_names = labels.map(names)
    print("counts", cluster_names.value_counts().to_dict(), flush=True)
    print("id->name", names, flush=True)
    cluster_names.to_csv(fig_dir / "clusters.csv")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    plot_dendrogram(profiles, ax=ax)
    fig.savefig(fig_dir / "dendrogram.pdf")
    plt.close(fig)
    print("Saved dendrogram.pdf", flush=True)

    n_k = len(cfg.k_values) if cfg.k_values else profiles.shape[1] // 2
    fig = plot_profiles_by_cluster(profiles, cluster_names, n_k=n_k)
    fig.savefig(fig_dir / "profiles_by_cluster.pdf")
    plt.close(fig)
    print("Saved profiles_by_cluster.pdf", flush=True)

    mroot = metrics_dir / "split" if (metrics_dir / "split").exists() else metrics_dir
    metric_tables = {
        "F1_point": _load_metric(mroot / "f1_point.csv"),
        "F1_seq": _load_metric(mroot / "f1_seq.csv"),
        "F1_PAK": _load_metric(mroot / "f1_pak.csv"),
    }
    metric_tables = {k: v for k, v in metric_tables.items() if v is not None}
    point_analysis = [
        m
        for m in point_analysis
        if all(m in t.columns for t in metric_tables.values())
    ]
    print("analysis methods present", point_analysis, flush=True)

    group = cluster_names.reindex(next(iter(metric_tables.values())).index)
    fig = plot_metric_comparison_by_density(
        metric_tables,
        group,
        group_name="Anomaly cluster",
        algorithm_order=point_analysis,
    )
    fig.savefig(fig_dir / "metrics_by_cluster.pdf")
    plt.close(fig)
    print("Saved metrics_by_cluster.pdf", flush=True)

    fig, summary = plot_point_vs_seq_grid(
        metric_tables, point_analysis, seq_methods, cluster_names
    )
    fig.savefig(fig_dir / "bayesian_pak.pdf")
    plt.close(fig)
    summary.to_csv(fig_dir / "bayesian_pak.csv", index=False)
    print("Saved bayesian_pak.pdf", flush=True)

    wpak = _load_metric(mroot / "f1_wpak.csv")
    if wpak is not None:
        fig, summary = plot_point_vs_seq_grid(
            {"wF1_PAK": wpak}, point_analysis, seq_methods, cluster_names
        )
        fig.savefig(fig_dir / "bayesian_wpak.pdf")
        plt.close(fig)
        summary.to_csv(fig_dir / "bayesian_wpak.csv", index=False)
        print("Saved bayesian_wpak.pdf", flush=True)

    (fig_dir / "README.txt").write_text(
        "split_point_wise=true\n"
        f"clustering ({len(point_cluster)}): {point_cluster}\n"
        f"analysis ({len(point_analysis)}): {point_analysis}\n"
        "Cluster names: ascending mean PA-precision -> "
        "Point-normal < Point-sparse < Point-dense.\n"
        "Compare against figures/all_methods/ (split_point_wise=false).\n"
    )
    print("Done", flush=True)


if __name__ == "__main__":
    main()

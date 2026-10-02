"""ROPE sensitivity: p(better)/p(rope)/p(worse) vs MAD-scaled R.

Sweeps ``R = scale * MAD`` with ``scale in {0.25, 0.5, 1.0}`` (paper default
is 0.5) for each metric, keeping R fixed across density clusters within a
metric. Writes a CSV and a grid figure for the rebuttal / appendix.

Example
-------
    PYTHONPATH=. python scripts/plot_rope_sensitivity.py
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

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.plots import (  # noqa: E402
    plot_rope_sensitivity,
    rope_sensitivity_table,
    set_paper_style,
)

METRIC_FILES = {
    "F1_point": "f1_point.csv",
    "F1_seq": "f1_seq.csv",
    "F1_PAK": "f1_pak.csv",
    "wF1_PAK": "f1_wpak.csv",
}


def _load_metric(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    return df.drop(columns=["density"], errors="ignore")


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
        "--scales",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="MAD multipliers for R (default: 0.25 0.5 0.75 1.0).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_paper_style()
    fig_dir = args.output_dir or cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = cfg.repo_root / "data" / "metrics"
    if not metrics_dir.exists():
        metrics_dir = cfg.data_dir / "metrics"

    clusters_path = args.clusters or (cfg.figures_dir / "clusters.csv")
    if not clusters_path.exists():
        raise SystemExit(f"clusters.csv not found: {clusters_path}")
    cluster_names = pd.read_csv(clusters_path, index_col=0).squeeze("columns")
    if not isinstance(cluster_names, pd.Series):
        cluster_names = cluster_names.iloc[:, 0]
    cluster_names.index = cluster_names.index.astype(int)

    metric_tables = {
        name: df
        for name, fname in METRIC_FILES.items()
        if (df := _load_metric(metrics_dir / fname)) is not None
    }
    if not metric_tables:
        raise SystemExit(f"No metric CSVs in {metrics_dir}")

    point_methods = cfg.point_methods_for_analysis
    seq_methods = cfg.methods_sequence_wise
    print(
        f"ROPE sensitivity: scales={args.scales}\n"
        f"  point ({len(point_methods)}): {point_methods}\n"
        f"  seq   ({len(seq_methods)}): {seq_methods}\n"
        f"  metrics: {list(metric_tables)}"
    )

    summary = rope_sensitivity_table(
        metric_tables,
        point_methods,
        seq_methods,
        cluster_names,
        mad_scales=list(args.scales),
    )
    csv_path = fig_dir / "bayesian_rope_sensitivity.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(summary)} rows)")

    # Compact print at paper default scale 0.5
    base = summary[np.isclose(summary["mad_scale"], 0.5)]
    if not base.empty:
        print("\nAt paper default R = 0.5·MAD:")
        print(
            base[
                [
                    "Metric",
                    "Cluster",
                    "rope",
                    "P(point better)",
                    "P(rope)",
                    "P(sequence better)",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    fig = plot_rope_sensitivity(summary)
    pdf = fig_dir / "bayesian_rope_sensitivity.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()

"""Compute F1_point, F1_seq and F1_PAK for every (dataset, detector).

Loads detector scores (from ``Results/Scores`` or a cached pickle), rebuilds the
UCR targets, aligns them, and writes three metric tables to ``data/metrics/``.

Example
-------
    python scripts/2_compute_metrics.py --methods AE Donut LSTMADalpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import (  # noqa: E402
    align_scores_targets,
    build_anomaly_df,
    build_binary_targets,
    load_scores,
)
from granularity_tsad.metrics import f1_pak, f1_point, f1_seq, f1_wpak  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    if not ucr_dir.exists():
        print(f"UCR data not found at {ucr_dir}. See the 'Data' section in README.md.")
        return
    methods = args.methods or cfg.all_methods
    out_dir = args.output or (cfg.data_dir / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    anomaly_df = cfg.filter_sequence_anomalies(build_anomaly_df(ucr_dir))
    datasets = anomaly_df.index.tolist()
    targets = build_binary_targets(anomaly_df)
    df_scores = load_scores(
        cfg.results_dir / "Scores", methods, datasets, schema=cfg.training_schema
    )

    f1p = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    f1s = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    f1k = pd.DataFrame(index=datasets, columns=methods, dtype=float)
    wf1k = pd.DataFrame(index=datasets, columns=methods, dtype=float)

    for ds in datasets:
        for m in methods:
            scores = df_scores.loc[ds, m]
            if scores is None:
                continue
            s, t = align_scores_targets(scores, targets.loc[ds], m)
            if t.sum() == 0:
                continue
            f1p.loc[ds, m] = f1_point(s, t)
            f1s.loc[ds, m] = f1_seq(s, t)
            f1k.loc[ds, m] = f1_pak(s, t, k_values=cfg.k_values or None)
            wf1k.loc[ds, m] = f1_wpak(s, t, k_values=cfg.k_values or None)
        print(f"dataset {ds} done")

    f1p.to_csv(out_dir / "f1_point.csv")
    f1s.to_csv(out_dir / "f1_seq.csv")
    f1k.to_csv(out_dir / "f1_pak.csv")
    wf1k.to_csv(out_dir / "f1_wpak.csv")
    print("Saved metric tables to", out_dir)


if __name__ == "__main__":
    main()

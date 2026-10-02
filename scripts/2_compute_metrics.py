"""Compute F1_point, F1_seq, F1_PAK, wF1_PAK and BINARY FLEX_MAX.

Loads detector scores (from ``Results/Scores``), rebuilds the UCR targets,
aligns them, and writes metric tables to ``data/metrics/``.

Use ``--only flex_max`` to compute just BINARY FLEX_MAX without recomputing
the slower F1 / PAK tables.

Example
-------
    python scripts/2_compute_metrics.py --methods AE Donut LSTMADalpha
    python scripts/2_compute_metrics.py --only flex_max
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
from granularity_tsad.metrics import (  # noqa: E402
    binary_flex_max,
    f1_pak,
    f1_point,
    f1_seq,
    f1_wpak,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--only",
        choices=("all", "flex_max"),
        default="all",
        help="Compute all metrics (default) or only BINARY FLEX_MAX.",
    )
    parser.add_argument(
        "--flex-margin",
        type=int,
        default=100,
        help="FLEX_MAX window radius around the labelled anomaly (default: 100).",
    )
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

    do_f1 = args.only == "all"
    f1p = pd.DataFrame(index=datasets, columns=methods, dtype=float) if do_f1 else None
    f1s = pd.DataFrame(index=datasets, columns=methods, dtype=float) if do_f1 else None
    f1k = pd.DataFrame(index=datasets, columns=methods, dtype=float) if do_f1 else None
    wf1k = pd.DataFrame(index=datasets, columns=methods, dtype=float) if do_f1 else None
    flex = pd.DataFrame(index=datasets, columns=methods, dtype=float)

    for ds in datasets:
        for m in methods:
            scores = df_scores.loc[ds, m]
            if scores is None:
                continue
            s, t = align_scores_targets(scores, targets.loc[ds], m)
            if t.sum() == 0:
                continue
            flex.loc[ds, m] = binary_flex_max(s, t, margin=args.flex_margin)
            if do_f1:
                f1p.loc[ds, m] = f1_point(s, t)
                f1s.loc[ds, m] = f1_seq(s, t)
                f1k.loc[ds, m] = f1_pak(s, t, k_values=cfg.k_values or None)
                wf1k.loc[ds, m] = f1_wpak(s, t, k_values=cfg.k_values or None)
        print(f"dataset {ds} done")

    def _merge_save(path: Path, new: pd.DataFrame) -> None:
        """Update columns for freshly computed methods; keep other columns intact."""
        if path.exists():
            old = pd.read_csv(path, index_col=0)
            old.index = old.index.astype(int)
            new = new.copy()
            new.index = new.index.astype(int)
            for col in new.columns:
                old[col] = new[col]
            ordered = list(dict.fromkeys(list(old.columns) + list(new.columns)))
            old.reindex(columns=ordered).to_csv(path)
        else:
            new.to_csv(path)

    if do_f1:
        _merge_save(out_dir / "f1_point.csv", f1p)
        _merge_save(out_dir / "f1_seq.csv", f1s)
        _merge_save(out_dir / "f1_pak.csv", f1k)
        _merge_save(out_dir / "f1_wpak.csv", wf1k)
    _merge_save(out_dir / "flex_max.csv", flex)
    print("Saved metric tables to", out_dir)
    print(f"BINARY FLEX_MAX (margin={args.flex_margin}) → {out_dir / 'flex_max.csv'}")
    if cfg.sequence_wise_extended:
        print(
            "sequence_wise_extended=true → sequence-wise set:",
            cfg.methods_sequence_wise,
        )


if __name__ == "__main__":
    main()

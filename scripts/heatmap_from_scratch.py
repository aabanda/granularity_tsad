"""Compute an anomaly-score heatmap for a single UCR series from scratch.

Pipeline (paper requirement 3):

1. Run several point-wise EasyTSAD detectors on one UCR curve.
2. Aggregate their (min-max normalized) scores into a single profile.
3. Plot the series colored by the aggregated anomaly score.

Example
-------
    python scripts/heatmap_from_scratch.py --curve 91 \
        --methods AE Donut EncDecAD FCVAE LSTMADalpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.aggregation import aggregate_scores  # noqa: E402
from granularity_tsad.config import DEFAULT_POINTWISE_METHODS, load_config  # noqa: E402
from granularity_tsad.data import build_anomaly_df, load_ucr_series  # noqa: E402
from granularity_tsad.easytsad_runner import EasyTSADRunner  # noqa: E402
from granularity_tsad.plots import plot_score_heatmap, set_paper_style  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", required=True, help="UCR curve id, e.g. 91")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["AE", "Donut", "EncDecAD", "FCVAE", "LSTMADalpha"],
        help="Point-wise EasyTSAD detectors to aggregate.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-run", action="store_true", help="Skip running detectors; just read cached scores.")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the figure (PDF/SVG).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"

    runner = EasyTSADRunner(
        data_dir=cfg.data_dir,
        workspace=cfg.repo_root,
        training_schema=cfg.training_schema,
        preprocess=cfg.preprocess,
    )
    if not args.no_run:
        runner.run(args.methods, curves=[args.curve])

    score_arrays = runner.load_scores(args.methods, args.curve)
    aggregated = aggregate_scores(score_arrays)

    anomaly_df = build_anomaly_df(ucr_dir, datasets=[int(args.curve)])
    data_test, _ = load_ucr_series(ucr_dir, anomaly_df)
    row = anomaly_df.loc[int(args.curve)]
    train_idx = int(row["Train Index"])
    span = (
        int(row["Anomaly Start"]) - train_idx,
        int(row["Anomaly End"]) - train_idx,
    )
    series = data_test.loc[int(args.curve)]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(11, 3))
    plot_score_heatmap(
        series,
        aggregated,
        anomaly_span=span,
        ax=ax,
        title=f"Aggregated anomaly score — UCR {args.curve}",
    )
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output)
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

"""PAK precision / recall vs coverage ``k`` for nuevas (or any custom) series.

For **each** time series, write one PDF/PNG with two panels:

* left:  PAK **precision** vs ``k`` (%) for every method
* right: PAK **recall** vs ``k`` (%) for every method

This is the Point-Adjusted-K protocol (Kim et al., 2022): for each anomalous
segment, a detection counts toward recall only if at least ``k`` % of its
points are flagged. Sweeping ``k`` from 0 (any overlap) to 100 (full coverage)
gives the precision/recall %K curves used in the paper.

Examples
--------
    python scripts/plot_nuevas_pak_curves.py
    python scripts/plot_nuevas_pak_curves.py --methods AE SubLOF MatrixProfile
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import align_scores_targets  # noqa: E402
from granularity_tsad.metrics.pak import DEFAULT_K_VALUES, pak_curves  # noqa: E402
from granularity_tsad.plots import plot_pak_precision_recall, set_paper_style  # noqa: E402

DEFAULT_DATASET = "nuevas"
DEFAULT_OUT = REPO_ROOT / "figures" / "nuevas_pak_curves"


def _discover_methods(scores_root: Path, dataset: str, schema: str) -> list[str]:
    methods = []
    if not scores_root.is_dir():
        return methods
    for method_dir in sorted(scores_root.iterdir()):
        score_dir = method_dir / schema / dataset
        if score_dir.is_dir() and any(score_dir.glob("*.npy")):
            methods.append(method_dir.name)
    return methods


def _discover_curves(data_dir: Path, dataset: str) -> list[str]:
    root = data_dir / "UTS" / dataset
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _load_test_labels(data_dir: Path, dataset: str, curve: str) -> np.ndarray:
    path = data_dir / "UTS" / dataset / curve / "test_label.npy"
    return np.asarray(np.load(path), dtype=int).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--schema", default="naive")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--curves", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Coverage grid (default: paper logarithmic k-grid from config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_values = args.k_values or cfg.k_values or DEFAULT_K_VALUES
    scores_root = REPO_ROOT / "Results" / "Scores"
    methods = args.methods or _discover_methods(scores_root, args.dataset, args.schema)
    curves = args.curves or _discover_curves(args.data_dir, args.dataset)

    if not methods:
        print(f"No score files under {scores_root}/*/{args.schema}/{args.dataset}/")
        print("Run: python scripts/run_nuevas.py")
        return
    if not curves:
        print(f"No curves under {args.data_dir / 'UTS' / args.dataset}")
        return

    set_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    print(f"Methods: {methods}")
    print(f"Curves:  {curves}")
    print(f"k-grid:  {k_values}")
    print(f"Output:  {args.output_dir}")

    for curve in curves:
        targets_full = _load_test_labels(args.data_dir, args.dataset, curve)
        if targets_full.sum() == 0:
            print(f"  skip {curve}: no positive labels (cannot evaluate PAK)")
            continue

        method_curves: dict[str, dict[str, list[float]]] = {}
        for method in methods:
            score_path = (
                scores_root / method / args.schema / args.dataset / f"{curve}.npy"
            )
            if not score_path.exists():
                print(f"  skip missing: {method}/{curve}")
                continue
            scores = np.asarray(np.load(score_path), dtype=float).reshape(-1)
            s, t = align_scores_targets(scores, targets_full, method)
            if t.sum() == 0:
                print(f"  skip {method}/{curve}: no positives after alignment")
                continue
            curves_m = pak_curves(s, t, k_values=k_values)
            method_curves[method] = curves_m
            for i, k in enumerate(k_values):
                summary_rows.append(
                    {
                        "curve": curve,
                        "method": method,
                        "k": k,
                        "precision": curves_m["precision"][i],
                        "recall": curves_m["recall"][i],
                        "f1": curves_m["f1"][i],
                    }
                )

        if not method_curves:
            print(f"  no evaluable methods for {curve}")
            continue

        fig = plot_pak_precision_recall(
            method_curves,
            k_values=k_values,
            title=f"{curve} — PAK precision / recall vs k",
        )
        out_png = args.output_dir / f"{curve}_pak_pr.png"
        out_pdf = args.output_dir / f"{curve}_pak_pr.pdf"
        fig.savefig(out_png, dpi=150)
        fig.savefig(out_pdf)
        plt.close(fig)
        try:
            shown = out_png.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = out_png
        print(f"  wrote {shown} (+ pdf)")

    if summary_rows:
        csv_path = args.output_dir / "pak_precision_recall.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["curve", "method", "k", "precision", "recall", "f1"]
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote {csv_path.relative_to(REPO_ROOT)} ({len(summary_rows)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()

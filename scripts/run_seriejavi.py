"""Convert ``data/seriejavi/*.txt`` to EasyTSAD format and run detectors.

Your series already live in::

    data/seriejavi/<curve>_<train>_<anom_start>_<anom_end>.txt

This script:

1. Converts them to ``data/UTS/seriejavi/<curve_id>/``
2. Runs EasyTSAD detectors (all from ``config/paper.yaml``, or ``--methods``)
3. Writes scores to ``Results/Scores/<method>/naive/seriejavi/<curve_id>.npy``
4. Prints a short per-(method, series) score summary

Examples
--------
    # all methods, all 5 series
    python scripts/run_seriejavi.py

    # subset of methods (faster)
    python scripts/run_seriejavi.py --methods AE Donut EncDecAD SubLOF MatrixProfile

    # convert only
    python scripts/run_seriejavi.py --convert-only

    # summarize existing scores without re-running
    python scripts/run_seriejavi.py --summary-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.easytsad_runner import EasyTSADRunner  # noqa: E402
from granularity_tsad.synthetic_rho import (  # noqa: E402
    convert_synthetic_dir,
    list_synthetic_txt,
    parse_synthetic_filename,
)

DEFAULT_SOURCE = REPO_ROOT / "data" / "seriejavi"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DATASET_NAME = "seriejavi"


def _curve_meta(source_dir: Path) -> dict[str, dict]:
    meta = {}
    for path in list_synthetic_txt(source_dir):
        parsed = parse_synthetic_filename(path.name)
        if parsed:
            meta[parsed["curve_id"]] = parsed
    return meta


def summarize_scores(
    workspace: Path,
    dataset: str,
    schema: str,
    methods: list[str],
    curves: list[str],
    source_dir: Path,
    out_csv: Path | None = None,
) -> None:
    """Print mean/max scores inside vs outside the labeled anomaly span."""
    meta = _curve_meta(source_dir)
    rows = []
    header = (
        f"{'method':<16} {'curve':<16} {'len':>5} "
        f"{'mean_all':>9} {'mean_in':>9} {'mean_out':>9} {'max':>9} {'path'}"
    )
    print(header)
    print("-" * len(header))

    for method in methods:
        for curve in curves:
            path = (
                workspace
                / "Results"
                / "Scores"
                / method
                / schema
                / dataset
                / f"{curve}.npy"
            )
            if not path.exists():
                print(f"{method:<16} {curve:<16} MISSING  {path}")
                rows.append(
                    {
                        "method": method,
                        "curve": curve,
                        "status": "missing",
                        "path": str(path),
                    }
                )
                continue

            scores = np.asarray(np.load(path), dtype=float).reshape(-1)
            info = meta.get(curve)
            mean_in = mean_out = float("nan")
            if info is not None:
                # Scores align to the test region with a possible warm-up trim.
                # Prefer EasyTSAD test_label.npy end-aligned to the score length.
                test_path = (
                    REPO_ROOT / "data" / "UTS" / dataset / curve / "test_label.npy"
                )
                if test_path.exists():
                    labels = np.asarray(np.load(test_path), dtype=float).reshape(-1)
                    L = len(scores)
                    if len(labels) >= L:
                        labels = labels[len(labels) - L :]
                    else:
                        labels = np.pad(labels, (L - len(labels), 0))
                    inside = labels > 0.5
                    if inside.any():
                        mean_in = float(scores[inside].mean())
                    if (~inside).any():
                        mean_out = float(scores[~inside].mean())
                else:
                    train_index = info["train_index"]
                    a0 = info["anomaly_start"] - train_index
                    a1 = info["anomaly_end"] - train_index
                    L = len(scores)
                    lo = max(0, min(a0, L))
                    hi = max(0, min(a1, L))
                    if lo < hi:
                        mean_in = float(scores[lo:hi].mean())
                        outside = np.concatenate([scores[:lo], scores[hi:]])
                        if len(outside):
                            mean_out = float(outside.mean())

            row = {
                "method": method,
                "curve": curve,
                "status": "ok",
                "len": len(scores),
                "mean_all": float(scores.mean()),
                "mean_in": mean_in,
                "mean_out": mean_out,
                "max": float(scores.max()),
                "path": str(path),
            }
            rows.append(row)
            print(
                f"{method:<16} {curve:<16} {len(scores):5d} "
                f"{row['mean_all']:9.4f} {mean_in:9.4f} {mean_out:9.4f} "
                f"{row['max']:9.4f} {path}"
            )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "method",
            "curve",
            "status",
            "len",
            "mean_all",
            "mean_in",
            "mean_out",
            "max",
            "path",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote summary CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Folder with seriejavi .txt files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="EasyTSAD parent (contains UTS/).",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--curves", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-replace", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first method failure (default: continue).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "Results" / "seriejavi_score_summary.csv",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After running (or with --summary-only), also plot scores via plot_seriejavi_scores.py.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_dir = Path(args.source_dir)
    data_dir = Path(args.data_dir)

    if not source_dir.is_dir():
        print(f"Source folder not found: {source_dir}")
        return

    txt_files = list_synthetic_txt(source_dir)
    if not txt_files:
        print(f"No valid .txt series in {source_dir}")
        print("Expected: <curve_id>_<train>_<anom_start>_<anom_end>.txt")
        return

    print(f"Found {len(txt_files)} series in {source_dir}")
    for p in txt_files:
        print(f"  {p.name} -> {parse_synthetic_filename(p.name)}")

    curves = convert_synthetic_dir(source_dir, data_dir, dataset_name=args.dataset)
    if args.curves:
        wanted = set(args.curves)
        missing = wanted - set(curves)
        if missing:
            print(f"Requested curves missing: {sorted(missing)}")
            return
        curves = [c for c in curves if c in wanted]
    print(f"EasyTSAD folder: {data_dir / 'UTS' / args.dataset}")
    print(f"Curves: {curves}")

    methods = args.methods or cfg.methods_point_wise
    if not methods:
        methods = cfg.all_methods

    if args.convert_only:
        return

    if not args.summary_only:
        runner = EasyTSADRunner(
            data_dir=data_dir,
            workspace=cfg.repo_root,
            dataset=args.dataset,
            training_schema=cfg.training_schema,
            preprocess=cfg.preprocess,
        )
        print(f"\nRunning {len(methods)} methods on {len(curves)} curves...")
        print(f"Scores -> {cfg.repo_root / 'Results' / 'Scores'}/<method>/{cfg.training_schema}/{args.dataset}/")
        failed = runner.run(
            methods,
            curves=curves,
            replace=not args.no_replace,
            continue_on_error=not args.strict,
        )
        if failed:
            print(f"\nFinished with {len(failed)} failed method(s): {failed}")
        else:
            print("\nAll methods finished.")

    print("\n=== Score summary ===")
    summarize_scores(
        workspace=cfg.repo_root,
        dataset=args.dataset,
        schema=cfg.training_schema,
        methods=methods,
        curves=curves,
        source_dir=source_dir,
        out_csv=args.summary_csv,
    )

    if args.plot:
        import importlib.util

        print("\n=== Plots ===")
        plot_path = REPO_ROOT / "scripts" / "plot_seriejavi_scores.py"
        spec = importlib.util.spec_from_file_location("plot_seriejavi_scores", plot_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        argv = [
            "--dataset",
            args.dataset,
            "--source-dir",
            str(source_dir),
            "--data-dir",
            str(data_dir),
            "--schema",
            cfg.training_schema,
            "--curves",
            *curves,
            "--methods",
            *methods,
        ]
        old = sys.argv
        try:
            sys.argv = ["plot_seriejavi_scores.py", *argv]
            mod.main()
        finally:
            sys.argv = old


if __name__ == "__main__":
    main()

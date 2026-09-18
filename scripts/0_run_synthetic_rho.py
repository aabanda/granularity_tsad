"""Convert ``data/synthetic_rho/*.txt`` to EasyTSAD ``.npy`` and run detectors.

Source files live **outside** the UCR archive::

    data/synthetic_rho/<curve_id>_<train>_<anom_start>_<anom_end>.txt

They are converted to::

    data/UTS/synthetic_rho/<curve_id>/{train,test,train_label,test_label}.npy

Scores are written under::

    Results/Scores/<method>/naive/synthetic_rho/<curve_id>.npy

Examples
--------
    # convert placeholders + run all configured detectors on every synthetic curve
    python scripts/0_run_synthetic_rho.py

    # only convert (no detectors)
    python scripts/0_run_synthetic_rho.py --convert-only

    # one curve, few methods
    python scripts/0_run_synthetic_rho.py --curves javi --methods AE Donut
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.easytsad_runner import EasyTSADRunner  # noqa: E402
from granularity_tsad.synthetic_rho import (  # noqa: E402
    DATASET_NAME,
    convert_synthetic_dir,
    list_synthetic_txt,
    write_placeholder_txt,
)


def _ensure_placeholders(source_dir: Path) -> None:
    """Create a few replaceable example series if the folder is empty."""
    if list_synthetic_txt(source_dir):
        return
    print(f"No .txt series in {source_dir}; writing placeholders...")
    write_placeholder_txt(
        source_dir, "javi", 800, 1400, 1600, kind="sequence", seed=0
    )
    write_placeholder_txt(
        source_dir, "rho_dense", 1000, 1800, 2000, kind="dense", seed=1
    )
    write_placeholder_txt(
        source_dir, "rho_sparse", 1000, 1800, 2000, kind="sparse", seed=2
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Folder with synthetic .txt series (default: <repo>/data/synthetic_rho).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="EasyTSAD parent dir containing UTS/ (default: <repo>/data).",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help=f"EasyTSAD dataset name under UTS/ (default: {DATASET_NAME}).",
    )
    parser.add_argument("--curves", nargs="+", default=None, help="Subset of curve ids.")
    parser.add_argument("--methods", nargs="+", default=None, help="Override detectors.")
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert .txt → .npy; do not run detectors.",
    )
    parser.add_argument(
        "--no-placeholders",
        action="store_true",
        help="Do not auto-create example .txt files when the folder is empty.",
    )
    parser.add_argument("--no-replace", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Keep synthetic data self-contained in this repo (independent of UCR data_dir).
    data_dir = Path(args.data_dir) if args.data_dir else (REPO_ROOT / "data")
    source_dir = (
        Path(args.source_dir) if args.source_dir else (data_dir / "synthetic_rho")
    )
    source_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_placeholders:
        _ensure_placeholders(source_dir)

    txt_files = list_synthetic_txt(source_dir)
    if not txt_files:
        print(f"No valid synthetic .txt files in {source_dir}.")
        print(
            "Expected names like: <curve_id>_<train>_<anom_start>_<anom_end>.txt"
        )
        return

    print(f"Converting {len(txt_files)} series from {source_dir} ...")
    curves = convert_synthetic_dir(source_dir, data_dir, dataset_name=args.dataset)
    if args.curves:
        wanted = set(args.curves)
        missing = wanted - set(curves)
        if missing:
            print(f"Requested curves not found after conversion: {sorted(missing)}")
            return
        curves = [c for c in curves if c in wanted]
    print(f"EasyTSAD curves ready under {data_dir / 'UTS' / args.dataset}: {curves}")

    if args.convert_only:
        return

    methods = args.methods or cfg.all_methods
    runner = EasyTSADRunner(
        data_dir=data_dir,
        workspace=cfg.repo_root,
        dataset=args.dataset,
        training_schema=cfg.training_schema,
        preprocess=cfg.preprocess,
    )
    print(f"Running {len(methods)} methods on {len(curves)} synthetic curves...")
    failed = runner.run(
        methods, curves=curves, replace=not args.no_replace, continue_on_error=True
    )
    if failed:
        print(f"Finished with failures: {failed}")
    print("Done. Scores under:", cfg.repo_root / "Results" / "Scores")
    for curve in curves:
        print(f"  e.g. Results/Scores/<method>/{cfg.training_schema}/{args.dataset}/{curve}.npy")


if __name__ == "__main__":
    main()

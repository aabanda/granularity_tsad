"""Convert ``data/nuevas/*.txt`` to EasyTSAD format and run detectors.

Controlled-ρ series live in::

    data/nuevas/<curve>_<train>_<anom_start>_<anom_end>.txt
    data/nuevas/<curve>_pw.txt   # absolute point-anomaly indices (ρ)

This script:

1. Converts them to ``data/UTS/nuevas/<curve_id>/`` using ``*_pw.txt`` labels
2. Runs **all point-wise and sequence-wise** detectors from ``config/paper.yaml``
3. Writes scores to ``Results/Scores/<method>/naive/nuevas/<curve_id>.npy``
4. Optionally plots scores and PAK precision/recall %K curves

Examples
--------
    python scripts/run_nuevas.py
    python scripts/run_nuevas.py --plot
    python scripts/run_nuevas.py --methods AE SubLOF --convert-only
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.easytsad_runner import EasyTSADRunner  # noqa: E402
from granularity_tsad.synthetic_rho import (  # noqa: E402
    convert_synthetic_dir,
    list_synthetic_txt,
    parse_synthetic_filename,
)

DEFAULT_SOURCE = REPO_ROOT / "data" / "nuevas"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DATASET_NAME = "nuevas"


def _load_seriejavi_helpers():
    """Load summarize_scores from run_seriejavi without treating scripts/ as a package."""
    path = REPO_ROOT / "scripts" / "run_seriejavi.py"
    spec = importlib.util.spec_from_file_location("run_seriejavi", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run_plot_script(script_name: str, argv: list[str]) -> None:
    plot_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), plot_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    old = sys.argv
    try:
        sys.argv = [script_name, *argv]
        mod.main()
    finally:
        sys.argv = old


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--curves", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-replace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--pw-labels",
        action="store_true",
        help="Label only *_pw.txt point indices (default: full filename segment).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "Results" / "nuevas_score_summary.csv",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After scores are ready, plot score panels + PAK precision/recall curves.",
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

    use_pw = args.pw_labels
    curves = convert_synthetic_dir(
        source_dir,
        data_dir,
        dataset_name=args.dataset,
        use_pw_labels=use_pw,
        anomaly_end_inclusive=True,  # filename end is inclusive (matches pw max index)
    )
    if args.curves:
        wanted = set(args.curves)
        missing = wanted - set(curves)
        if missing:
            print(f"Requested curves missing: {sorted(missing)}")
            return
        curves = [c for c in curves if c in wanted]
    print(f"EasyTSAD folder: {data_dir / 'UTS' / args.dataset}")
    print(f"Labels: {'*_pw.txt point indices' if use_pw else 'filename segment (inclusive)'}")
    print(f"Curves: {curves}")

    # Point-wise ∪ sequence-wise from paper config (not the analysis-only extras).
    if args.methods:
        methods = args.methods
    else:
        seen: list[str] = []
        for name in cfg.methods_point_wise + cfg.methods_sequence_wise:
            if name not in seen:
                seen.append(name)
        methods = seen
    print(f"Methods ({len(methods)}): {methods}")

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
        print(
            f"Scores -> {cfg.repo_root / 'Results' / 'Scores'}/"
            f"<method>/{cfg.training_schema}/{args.dataset}/"
        )
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
    summarize_scores = _load_seriejavi_helpers().summarize_scores
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
        print("\n=== Combined score + PAK plots ===")
        _run_plot_script(
            "plot_nuevas_scores_with_pak.py",
            [
                "--dataset",
                args.dataset,
                "--source-dir",
                str(source_dir),
                "--data-dir",
                str(data_dir),
                "--schema",
                cfg.training_schema,
                "--output-dir",
                str(REPO_ROOT / "figures" / "nuevas_scores"),
                "--curves",
                *curves,
                "--methods",
                *methods,
            ],
        )


if __name__ == "__main__":
    main()

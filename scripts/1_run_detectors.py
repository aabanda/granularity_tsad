"""Run the EasyTSAD detector benchmark over the UCR archive.

Wraps the EasyTSAD controller to run all configured detectors and persist their
anomaly scores under ``Results/Scores/``. Scores are later consumed by
``2_compute_metrics.py``.

Example
-------
    python scripts/1_run_detectors.py --curves 1 2 3
    python scripts/1_run_detectors.py            # all curves
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import build_anomaly_df  # noqa: E402
from granularity_tsad.easytsad_runner import EasyTSADRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None, help="Override the configured detectors.")
    parser.add_argument("--curves", nargs="+", default=None, help="UCR curve ids (default: all available).")
    parser.add_argument("--no-replace", action="store_true", help="Do not overwrite existing results.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    if not ucr_dir.exists():
        print(f"UCR data not found at {ucr_dir}. See the 'Data' section in README.md.")
        return
    methods = args.methods or cfg.all_methods

    if args.curves:
        curves = list(args.curves)
    else:
        curves = [str(d) for d in build_anomaly_df(ucr_dir).index.tolist()]

    runner = EasyTSADRunner(
        data_dir=cfg.data_dir,
        workspace=cfg.repo_root,
        training_schema=cfg.training_schema,
        preprocess=cfg.preprocess,
    )
    print(f"Running {len(methods)} methods on {len(curves)} curves...")
    runner.run(methods, curves=curves, replace=not args.no_replace)
    print("Done. Scores under:", cfg.repo_root / "Results" / "Scores")


if __name__ == "__main__":
    main()

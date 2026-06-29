"""Loading and alignment of UCR Anomaly Archive data and detector scores.

The UCR Anomaly Archive encodes the train split and the anomalous segment in
each filename, e.g.::

    135_UCR_Anomaly_InternalBleeding16_1200_4187_4197.txt
        |                              |    |    |
        dataset id                     |    |    anomaly end
                                       |    anomaly start
                                       train/test split index

This module rebuilds the per-dataset metadata, the binary ground-truth targets
for the test region, loads detector scores, and aligns scores to targets to
absorb the leading offset introduced by windowing/forecasting detectors.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Detectors whose scores are shifted by a fixed amount beyond the generic
# "trim leading targets to match score length" rule (ported from the paper).
SPECIAL_ALIGN = {
    "AR": {"trim_scores_head_to": 16},      # keep only the last len(targets)-16 scores
    "MyAlgo_ESN": {"scores_skip": 20, "targets_skip": 21},
}


def parse_ucr_filename(filename: str) -> dict | None:
    """Extract dataset id, train index and anomaly span from a UCR filename."""
    numbers = re.findall(r"\d+", filename)
    if len(numbers) < 4:
        return None
    return {
        "dataset": int(numbers[0]),
        "train_index": int(numbers[-3]),
        "anomaly_start": int(numbers[-2]),
        "anomaly_end": int(numbers[-1]),
    }


def build_anomaly_df(ucr_dir: Path, datasets: list[int] | None = None) -> pd.DataFrame:
    """Build the per-dataset anomaly metadata table from UCR filenames."""
    ucr_dir = Path(ucr_dir)
    rows = []
    for file in sorted(ucr_dir.iterdir()):
        meta = parse_ucr_filename(file.name)
        if meta is None:
            continue
        if datasets is not None and meta["dataset"] not in datasets:
            continue
        total_length = len(np.loadtxt(file))
        rows.append(
            {
                "Dataset": meta["dataset"],
                "Train Index": meta["train_index"],
                "Total Length": total_length,
                "Anomaly Length": meta["anomaly_end"] - meta["anomaly_start"],
                "Anomaly Start": meta["anomaly_start"],
                "Anomaly End": meta["anomaly_end"],
                "File": file.name,
            }
        )
    df = pd.DataFrame(rows).set_index("Dataset").sort_index()
    df.index = df.index.astype(int)
    return df


def load_ucr_series(
    ucr_dir: Path, anomaly_df: pd.DataFrame, normalize: bool = True
) -> tuple[pd.Series, pd.Series]:
    """Load train/test splits for every dataset, min-max normalized on train."""
    ucr_dir = Path(ucr_dir)
    data_test, data_train, index = [], [], []
    for dataset, row in anomaly_df.iterrows():
        ts = np.loadtxt(ucr_dir / row["File"])
        train_idx = int(row["Train Index"])
        train = ts[:train_idx]
        test = ts[train_idx - 1:]
        if normalize:
            lo, hi = float(np.min(train)), float(np.max(train))
            denom = (hi - lo) or 1.0
            train = (train - lo) / denom
            test = (test - lo) / denom
        data_train.append(train)
        data_test.append(test)
        index.append(int(dataset))
    return (
        pd.Series(data_test, index=index).sort_index(),
        pd.Series(data_train, index=index).sort_index(),
    )


def build_binary_targets(anomaly_df: pd.DataFrame) -> pd.Series:
    """Binary anomaly labels for the test region of every dataset."""
    targets = {}
    for dataset, row in anomaly_df.iterrows():
        arr = np.zeros(int(row["Total Length"]), dtype=int)
        arr[int(row["Anomaly Start"]): int(row["Anomaly End"]) + 1] = 1
        targets[int(dataset)] = arr[int(row["Train Index"]):]
    return pd.Series(targets).sort_index()


def load_scores(
    scores_dir: Path,
    methods: list[str],
    datasets: list[int],
    schema: str = "naive",
    dataset_name: str = "UCR",
) -> pd.DataFrame:
    """Load per-(dataset, method) score arrays from an EasyTSAD score tree.

    Expected layout: ``scores_dir/<method>/<schema>/<dataset_name>/<id>.npy``.
    """
    scores_dir = Path(scores_dir)
    data = {}
    for method in methods:
        col = []
        for ds in datasets:
            path = scores_dir / method / schema / dataset_name / f"{ds}.npy"
            col.append(np.load(path) if path.exists() else None)
        data[method] = col
    return pd.DataFrame(data, index=datasets).sort_index()


def align_scores_targets(
    scores: np.ndarray, targets: np.ndarray, method: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Align a detector's scores to the binary targets.

    Generic rule: trim the leading ``len(targets) - len(scores)`` targets (the
    warm-up window consumed by windowed/forecasting detectors), then pad or
    truncate scores so both sequences share the same length. ``method``-specific
    pre-trims from :data:`SPECIAL_ALIGN` are applied first.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)

    rule = SPECIAL_ALIGN.get(method, {})
    if "trim_scores_head_to" in rule:
        scores = scores[len(targets) - rule["trim_scores_head_to"]:]
    if "scores_skip" in rule:
        scores = scores[rule["scores_skip"]:]
    if "targets_skip" in rule:
        targets = targets[rule["targets_skip"]:]

    if len(scores) < len(targets):
        # Generic warm-up trim: drop the leading targets the detector never saw.
        targets = targets[len(targets) - len(scores):]

    if len(scores) < len(targets):
        scores = np.concatenate([np.zeros(len(targets) - len(scores)), scores])
    elif len(scores) > len(targets):
        scores = scores[: len(targets)]
    return scores, targets

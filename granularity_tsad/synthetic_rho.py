"""Synthetic ρ series: plain ``.txt`` sources converted to EasyTSAD layout.

Drop files under ``data/synthetic_rho/`` (outside the UCR archive). Each
filename encodes the train split and anomaly span::

    <curve_id>_<train_index>_<anomaly_start>_<anomaly_end>.txt

Examples::

    javi_800_1400_1600.txt
    rho_dense_1000_1800_2000.txt

Content is one float per line (full univariate series). Conversion writes
EasyTSAD's ``train.npy`` / ``test.npy`` / ``*_label.npy`` under::

    {data_dir}/UTS/synthetic_rho/<curve_id>/
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

DATASET_NAME = "synthetic_rho"
DEFAULT_SOURCE_DIRNAME = "synthetic_rho"


def parse_synthetic_filename(filename: str) -> dict | None:
    """Parse ``<curve_id>_<train>_<anom_start>_<anom_end>.txt``.

    ``curve_id`` may contain underscores (e.g. ``rho_dense``). The last three
    underscore-separated tokens must be integers.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        train_index = int(parts[-3])
        anomaly_start = int(parts[-2])
        anomaly_end = int(parts[-1])
    except ValueError:
        return None
    curve_id = "_".join(parts[:-3])
    # Allow dots so ρ tags like ``seriejavi_0.05`` are valid curve ids.
    if not curve_id or not re.fullmatch(r"[A-Za-z0-9_.\-]+", curve_id):
        return None
    if not (0 < train_index < anomaly_start < anomaly_end):
        return None
    return {
        "curve_id": curve_id,
        "train_index": train_index,
        "anomaly_start": anomaly_start,
        "anomaly_end": anomaly_end,
    }


def list_synthetic_txt(source_dir: Path) -> list[Path]:
    """Return sorted ``.txt`` series files with a valid synthetic filename."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    files = []
    for path in sorted(source_dir.glob("*.txt")):
        if parse_synthetic_filename(path.name) is not None:
            files.append(path)
    return files


def find_pw_label_file(txt_path: Path, curve_id: str) -> Path | None:
    """Locate ``{curve_id}_pw.txt`` next to the series or under ``labels/``."""
    txt_path = Path(txt_path)
    candidates = [
        txt_path.with_name(f"{curve_id}_pw.txt"),
        txt_path.parent / "labels" / f"{curve_id}_pw.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_pw_indices(pw_path: Path) -> np.ndarray:
    """Load absolute point-anomaly indices from a ``*_pw.txt`` file."""
    pw_path = Path(pw_path)
    if pw_path.stat().st_size == 0:
        return np.asarray([], dtype=int)
    idxs = np.loadtxt(pw_path, dtype=int).reshape(-1)
    return np.asarray(idxs, dtype=int)


def convert_txt_to_easytsad(
    txt_path: Path,
    export_dir: Path,
    *,
    anomaly_end_inclusive: bool = False,
    use_pw_labels: bool = False,
) -> str:
    """Convert one synthetic ``.txt`` into an EasyTSAD curve folder.

    Parameters
    ----------
    txt_path
        Source file (one float per line).
    export_dir
        Parent of curve folders, typically ``{data_dir}/UTS/synthetic_rho``.
    anomaly_end_inclusive
        If True, label ``[start, end]`` inclusive (UCR paper convention in
        ``data.py``). EasyTSAD's original UCR converter used an exclusive end;
        default here is exclusive ``[start, end)`` to match EasyTSAD.
    use_pw_labels
        If True and a sibling ``{curve_id}_pw.txt`` exists, label only those
        absolute indices (controlled-ρ point anomalies). An empty ``_pw`` file
        yields an all-negative label (ρ = 0).

    Returns
    -------
    curve_id
        Folder / curve name used by EasyTSAD.
    """
    meta = parse_synthetic_filename(txt_path.name)
    if meta is None:
        raise ValueError(f"Invalid synthetic filename: {txt_path.name}")

    series = np.loadtxt(txt_path).astype(float).reshape(-1)
    if series.ndim != 1 or len(series) == 0:
        raise ValueError(f"Expected a non-empty 1-D series in {txt_path}")

    train_index = meta["train_index"]
    a0, a1 = meta["anomaly_start"], meta["anomaly_end"]
    if train_index >= len(series) or a1 > len(series):
        raise ValueError(
            f"{txt_path.name}: indices out of range for length {len(series)} "
            f"(train={train_index}, anomaly=[{a0}, {a1}))"
        )

    label = np.zeros(len(series), dtype=float)
    pw_path = find_pw_label_file(txt_path, meta["curve_id"]) if use_pw_labels else None
    if pw_path is not None:
        idxs = load_pw_indices(pw_path)
        if len(idxs):
            if idxs.min() < 0 or idxs.max() >= len(series):
                raise ValueError(
                    f"{pw_path.name}: pw indices out of range for length {len(series)}"
                )
            label[idxs] = 1.0
        label_source = pw_path.name
    elif anomaly_end_inclusive:
        label[a0 : a1 + 1] = 1.0
        label_source = "segment_inclusive"
    else:
        label[a0:a1] = 1.0
        label_source = "segment_exclusive"

    train = series[:train_index]
    test = series[train_index:]
    train_label = label[:train_index]
    test_label = label[train_index:]

    if train_label.sum() > 0:
        raise ValueError(
            f"{txt_path.name}: training region must be anomaly-free "
            f"(found {int(train_label.sum())} labeled points before index {train_index})"
        )

    curve_dir = Path(export_dir) / meta["curve_id"]
    curve_dir.mkdir(parents=True, exist_ok=True)
    np.save(curve_dir / "train.npy", train)
    np.save(curve_dir / "test.npy", test)
    np.save(curve_dir / "train_label.npy", train_label)
    np.save(curve_dir / "test_label.npy", test_label)

    test_ano = float(test_label.sum()) / max(len(test_label), 1)
    total_ano = float(label.sum()) / max(len(label), 1)
    seg_len = (a1 - a0 + 1) if anomaly_end_inclusive or pw_path is not None else (a1 - a0)
    info = {
        "intervals": 1,
        "training set anomaly ratio": 0.0,
        "testset anomaly ratio": round(test_ano, 5),
        "total anomaly ratio": round(total_ano, 5),
        "source": txt_path.name,
        "label_source": label_source,
        "n_anomaly_points": int(label.sum()),
        "segment_rho": round(float(label.sum()) / max(seg_len, 1), 5),
        "len data": int(len(series)),
        "len train": int(len(train)),
        "len test": int(len(test)),
    }
    with open(curve_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    return meta["curve_id"]


def convert_synthetic_dir(
    source_dir: Path,
    data_dir: Path,
    dataset_name: str = DATASET_NAME,
    *,
    use_pw_labels: bool = False,
    anomaly_end_inclusive: bool = False,
) -> list[str]:
    """Convert every valid ``.txt`` under ``source_dir`` to EasyTSAD layout.

    Writes to ``{data_dir}/UTS/{dataset_name}/<curve_id>/``.
    """
    export_dir = Path(data_dir) / "UTS" / dataset_name
    export_dir.mkdir(parents=True, exist_ok=True)
    curves = []
    for txt_path in list_synthetic_txt(source_dir):
        curves.append(
            convert_txt_to_easytsad(
                txt_path,
                export_dir,
                use_pw_labels=use_pw_labels,
                anomaly_end_inclusive=anomaly_end_inclusive,
            )
        )
    return curves


def generate_placeholder_series(
    length: int,
    train_index: int,
    anomaly_start: int,
    anomaly_end: int,
    *,
    kind: str = "dense",
    seed: int = 0,
) -> np.ndarray:
    """Build a simple sine series with a sequence anomaly (for placeholders)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)
    series = np.sin(2 * np.pi * t / 40.0) + 0.05 * rng.normal(size=length)

    # Collective anomaly: amplitude + frequency change inside the labeled span.
    span = slice(anomaly_start, anomaly_end)
    local = np.arange(anomaly_end - anomaly_start, dtype=float)
    series[span] = 1.6 * np.sin(2 * np.pi * local / 18.0) + 0.3

    if kind == "dense":
        # Many point spikes inside the region (high ρ).
        n_spikes = max(8, (anomaly_end - anomaly_start) // 12)
        idxs = rng.choice(np.arange(anomaly_start, anomaly_end), size=n_spikes, replace=False)
        series[idxs] += rng.choice([-1.0, 1.0], size=n_spikes) * rng.uniform(1.5, 2.5, size=n_spikes)
    elif kind == "sparse":
        # Few point spikes (low ρ).
        n_spikes = 2
        idxs = rng.choice(np.arange(anomaly_start, anomaly_end), size=n_spikes, replace=False)
        series[idxs] += rng.choice([-1.0, 1.0], size=n_spikes) * 2.2
    elif kind == "sequence":
        pass  # collective change only
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Sanity: keep train clean.
    assert train_index <= anomaly_start
    return series


def write_placeholder_txt(
    source_dir: Path,
    curve_id: str,
    train_index: int,
    anomaly_start: int,
    anomaly_end: int,
    *,
    length: int | None = None,
    kind: str = "sequence",
    seed: int = 0,
) -> Path:
    """Write one placeholder ``.txt`` the user can later replace."""
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    if length is None:
        length = anomaly_end + max(200, train_index // 4)
    series = generate_placeholder_series(
        length, train_index, anomaly_start, anomaly_end, kind=kind, seed=seed
    )
    path = source_dir / f"{curve_id}_{train_index}_{anomaly_start}_{anomaly_end}.txt"
    np.savetxt(path, series, fmt="%.8e")
    return path

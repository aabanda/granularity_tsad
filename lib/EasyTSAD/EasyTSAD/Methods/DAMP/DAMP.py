"""DAMP (Discord Aware Matrix Profile) as an EasyTSAD sequence-wise detector.

Lu et al., KDD 2022: left-discords (nearest neighbour only in the past).
On UCR we score the **test** region against the **train(+valid)** history via
stumpy AB-join — the same left-discord objective as DAMP, in a form that scales
to the archive's long series (up to ~1e6 points).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...DataFactory import TSData
from .. import BaseMethod


def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _left_discord_scores(
    history: np.ndarray,
    query: np.ndarray,
    m: int,
    *,
    max_query_len: int = 12000,
) -> np.ndarray:
    """Nearest-neighbour z-normalized distance of each query window into history."""
    import stumpy

    history = _as_1d(history)
    query = _as_1d(query)
    m = int(m)
    n_q = len(query)
    if len(history) < m or n_q < m:
        return np.zeros(max(n_q, 1), dtype=float)

    # Downsample extremely long UCR series so STOMP stays tractable.
    factor = 1
    if n_q > max_query_len:
        factor = int(np.ceil(n_q / max_query_len))
        history_ds = history[::factor]
        query_ds = query[::factor]
        m_ds = max(4, m // factor)
        if len(history_ds) < m_ds or len(query_ds) < m_ds:
            factor = 1
            history_ds, query_ds, m_ds = history, query, m
    else:
        history_ds, query_ds, m_ds = history, query, m

    mp = stumpy.stump(query_ds, m_ds, history_ds, ignore_trivial=False)
    amp = np.asarray(mp[:, 0], dtype=float)
    # Score per downsampled timestamp, then upsample.
    scores_ds = np.empty(len(query_ds), dtype=float)
    pad = len(query_ds) - len(amp)
    if pad > 0:
        scores_ds[:pad] = amp[0] if len(amp) else 0.0
        scores_ds[pad:] = amp
    else:
        scores_ds[:] = amp[: len(query_ds)]

    if factor == 1:
        return scores_ds
    # Repeat each score ``factor`` times and trim/pad to original length.
    out = np.repeat(scores_ds, factor)[:n_q]
    if len(out) < n_q:
        out = np.concatenate([out, np.full(n_q - len(out), out[-1])])
    return out


class DAMP(BaseMethod):
    """EasyTSAD wrapper: sequence-wise left-discord scores on the test region."""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__()
        params = params or {}
        self.window = int(params.get("window", 100))
        self.__anomaly_score: np.ndarray | None = None
        self.score_train = None

    def train_valid_phase(self, tsTrain: TSData) -> None:
        train = _as_1d(tsTrain.train)
        if len(train) >= 2 * self.window:
            half = len(train) // 2
            self.score_train = _left_discord_scores(train[:half], train[half:], self.window)
        else:
            self.score_train = np.zeros(max(len(train), 1), dtype=float)

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]) -> None:
        pass

    def test_phase(self, tsData: TSData) -> None:
        train = _as_1d(tsData.train)
        valid = (
            _as_1d(tsData.valid)
            if getattr(tsData, "valid", None) is not None and len(np.asarray(tsData.valid))
            else np.array([], dtype=float)
        )
        test = _as_1d(tsData.test)
        history = np.concatenate([train, valid]) if valid.size else train
        # Ensure history is long enough for the window.
        if len(history) < self.window:
            history = np.concatenate([history, test[: self.window]])
        self.__anomaly_score = _left_discord_scores(history, test, self.window)

    def anomaly_score(self) -> np.ndarray:
        assert self.__anomaly_score is not None
        return self.__anomaly_score

"""Sequence-wise (range-based) evaluation metric (paper Section 2.4).

Sequence-wise performance is measured with the range-based F-score of
Tatbul et al. (NeurIPS 2018), as implemented by the ``prts`` library
(``ts_fscore``). The metric is computed by sweeping detection thresholds over
the score range and keeping the best (maximum) range-based F1, using the
``prts`` default parameters.
"""

from __future__ import annotations

import numpy as np


def _ts_fscore():
    """Lazily import prts so the package imports without it installed."""
    try:
        from prts import ts_fscore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "prts is required for the range-based sequence-wise metric. "
            "Install it with `pip install prts`."
        ) from exc
    return ts_fscore


def f1_seq(
    scores: np.ndarray,
    targets: np.ndarray,
    n_thresholds: int = 50,
) -> float:
    """Best range-based (sequence-wise) F1 over thresholds.

    Reproduces the paper computation: for each threshold in
    ``linspace(min, max, n_thresholds)`` (endpoints excluded) binarize the
    scores and evaluate ``prts.ts_fscore`` with the package default parameters,
    returning the maximum F1.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)

    if targets.sum() == 0 or np.ptp(scores) == 0:
        return float("nan")

    ts_fscore = _ts_fscore()
    best = 0.0
    for threshold in np.linspace(scores.min(), scores.max(), n_thresholds)[1:-1]:
        pred = (scores >= threshold).astype(int)
        if pred.sum() == 0:
            continue
        score = ts_fscore(targets, pred)  # prts default parameters
        if score > best:
            best = score
    return float(best)

"""BINARY FLEX_MAX: does the score peak fall near the labelled anomaly?

A detector scores ``1`` on a series when (at least one) timestamp attaining the
global maximum anomaly score lies inside the flex window

    ``[anomaly_start - margin, anomaly_end + margin]``

(clipped to the series), and ``0`` otherwise. Indices are in the same
coordinate system as the aligned ``scores`` / ``targets`` arrays.
"""

from __future__ import annotations

import numpy as np


def binary_flex_max(
    scores: np.ndarray,
    targets: np.ndarray,
    margin: int = 100,
) -> float:
    """Binary FLEX_MAX in ``{0, 1}``.

    Parameters
    ----------
    scores : per-timestamp anomaly scores (aligned to ``targets``).
    targets : binary ground-truth labels (same length as ``scores``).
    margin : flex radius in timestamps around the labelled anomaly span.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=int).reshape(-1)
    n = min(len(scores), len(targets))
    if n == 0:
        return 0.0
    scores, targets = scores[:n], targets[:n]

    nz = np.flatnonzero(targets)
    if len(nz) == 0 or not np.isfinite(scores).any():
        return 0.0

    a0, a1 = int(nz[0]), int(nz[-1])
    lo = max(0, a0 - int(margin))
    hi = min(n - 1, a1 + int(margin))

    # Treat NaNs as -inf so they never win the max.
    safe = np.where(np.isfinite(scores), scores, -np.inf)
    peak = float(np.max(safe))
    if not np.isfinite(peak):
        return 0.0
    peak_locs = np.flatnonzero(safe == peak)
    hit = np.any((peak_locs >= lo) & (peak_locs <= hi))
    return 1.0 if hit else 0.0

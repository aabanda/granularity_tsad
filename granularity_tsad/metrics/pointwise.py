"""Point-wise evaluation metric (paper Section 2.4).

Point-wise F1 scores each timestamp independently, with no point adjustment.
It is the strictest granularity: a detector must flag the exact anomalous
timestamps to be rewarded.
"""

from __future__ import annotations

import numpy as np

from .pak import _tadpak_evaluate


def f1_point(
    scores: np.ndarray,
    targets: np.ndarray,
    interval: int = 100,
) -> float:
    """Best point-wise F1 over thresholds (no point adjustment).

    Parameters
    ----------
    scores : per-timestamp anomaly scores.
    targets : binary ground-truth labels.
    interval : number of thresholds explored when searching the best F1.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    ev = _tadpak_evaluate().evaluate(scores, targets, pa=False, interval=interval)
    # tadpak exposes point-wise keys without the "_w_pa" suffix.
    for key in ("best_f1", "best_f1_wo_pa"):
        if key in ev:
            return float(ev[key])
    raise KeyError(f"Could not find a point-wise F1 key in tadpak output: {list(ev)}")

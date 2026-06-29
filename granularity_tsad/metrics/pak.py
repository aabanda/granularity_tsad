"""PAK / WPAK metrics (paper Sections 2.3 and 3.4).

The Point-Adjusted-K (PAK) protocol [Kim et al., 2022] generalizes point
adjustment: for each anomalous segment, recall counts only if at least ``k`` %
of its points are detected. Sweeping ``k`` from 0 to 100 yields a curve that
interpolates between sequence-wise (``k=0``, any overlap) and strict
point-coverage (``k=100``) behaviour. The area under this F1 curve is the
scalar ``F1_PAK`` metric used throughout the paper.
"""

from __future__ import annotations

import numpy as np


def _tadpak_evaluate():
    """Lazily import tadpak so the package imports without it installed."""
    try:
        from tadpak import evaluate
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tadpak is required for PAK metrics. Install it with `pip install tadpak`."
        ) from exc
    return evaluate


# Logarithmic k-grid used in the paper (Section 3).
DEFAULT_K_VALUES: list[int] = [0, 1, 2, 3, 5, 6, 8, 11, 15, 19, 25, 33, 44, 58, 76, 100]


def pak_curves(
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: list[int] | None = None,
    interval: int = 100,
) -> dict[str, list[float]]:
    """Return PAK precision/recall/F1 curves over the ``k`` grid.

    Parameters
    ----------
    scores : per-timestamp anomaly scores.
    targets : binary ground-truth labels (same length as ``scores``).
    k_values : coverage thresholds in percent; defaults to :data:`DEFAULT_K_VALUES`.
    interval : number of thresholds explored by tadpak when searching the best F1.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=int)
    evaluate = _tadpak_evaluate()

    f1, precision, recall = [], [], []
    for k in k_values:
        ev = evaluate.evaluate(scores, targets, pa=True, interval=interval, k=k)
        f1.append(ev["best_f1_w_pa"])
        precision.append(ev["best_precision_w_pa"])
        recall.append(ev["best_recall_w_pa"])
    return {"k_values": list(k_values), "f1": f1, "precision": precision, "recall": recall}


def pak_f1_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: list[int] | None = None,
) -> list[float]:
    """Convenience wrapper returning only the PAK F1 curve."""
    return pak_curves(scores, targets, k_values=k_values)["f1"]


def pak_auc(f1_curve: list[float], normalize: bool = True) -> float:
    """Area under the PAK F1 curve (the scalar ``F1_PAK`` metric)."""
    f1_curve = np.asarray(f1_curve, dtype=float)
    auc = float(np.trapz(f1_curve, dx=1))
    if normalize and len(f1_curve):
        auc /= len(f1_curve)
    return auc


def f1_pak(
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: list[int] | None = None,
    normalize: bool = True,
) -> float:
    """Compute the scalar ``F1_PAK`` directly from scores and targets."""
    curve = pak_f1_curve(scores, targets, k_values=k_values)
    return pak_auc(curve, normalize=normalize)


def _wpak_weights(k_values: np.ndarray) -> np.ndarray:
    """Decreasing weights ``w(k) = 1/k`` with ``w(0) = 1`` (no division by zero).

    ``k = 0`` (any-overlap detection) receives the maximum weight, matching the
    "assign decreasing weight to higher k" rationale of Section 3.4.
    """
    k_values = np.asarray(k_values, dtype=float)
    return np.where(k_values > 0, 1.0 / np.where(k_values == 0, 1.0, k_values), 1.0)


def wpak_auc(
    f1_curve: list[float],
    k_values: list[int] | None = None,
) -> float:
    """Weighted PAK-AUC with ``w(k) = 1/k`` (stretched WPAK, Section 3.4).

    Down-weights the high-coverage region so that detecting *some* anomalous
    points matters more than achieving full segment coverage. ``k = 0`` (when
    present in the grid) is given weight 1 to avoid division by zero.
    """
    f1_curve = np.asarray(f1_curve, dtype=float)
    if k_values is None:
        k_values = list(range(1, len(f1_curve) + 1))
    weights = _wpak_weights(np.asarray(k_values, dtype=float))
    weighted = f1_curve * weights
    return float(np.trapz(weighted, dx=1)) / len(f1_curve)


def f1_wpak(
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: list[int] | None = None,
) -> float:
    """Compute the scalar weighted ``wF1_PAK`` directly from scores and targets."""
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    curve = pak_f1_curve(scores, targets, k_values=k_values)
    return wpak_auc(curve, k_values=k_values)

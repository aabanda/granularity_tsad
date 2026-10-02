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


def _tadpak_pak():
    """Lazily import tadpak's point-adjust-k transform."""
    try:
        from tadpak.pak import pak
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tadpak is required for PAK metrics. Install it with `pip install tadpak`."
        ) from exc
    return pak


def _tadpak_evaluate():
    """Lazily import tadpak's evaluate helper (used by point-wise F1)."""
    try:
        from tadpak import evaluate
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tadpak is required for PAK metrics. Install it with `pip install tadpak`."
        ) from exc
    return evaluate


# Logarithmic k-grid used in the paper (Section 3).
DEFAULT_K_VALUES: list[int] = [0, 1, 2, 3, 5, 6, 8, 11, 15, 19, 25, 33, 44, 58, 76, 100]


def _prf1(pred: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    """Precision / recall / F1 from binary arrays (zero-safe)."""
    pred = pred.astype(bool, copy=False)
    targets = targets.astype(bool, copy=False)
    tp = int(np.logical_and(pred, targets).sum())
    fp = int(np.logical_and(pred, ~targets).sum())
    fn = int(np.logical_and(~pred, targets).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def pak_curves(
    scores: np.ndarray,
    targets: np.ndarray,
    k_values: list[int] | None = None,
    interval: int = 100,
    *,
    threshold_mode: str = "per_k",
) -> dict[str, list[float]]:
    """Return PAK precision/recall/F1 curves over the ``k`` grid.

    Parameters
    ----------
    scores : per-timestamp anomaly scores.
    targets : binary ground-truth labels (same length as ``scores``).
    k_values : coverage thresholds in percent; defaults to :data:`DEFAULT_K_VALUES`.
    interval : number of score thresholds explored when searching the best F1.
    threshold_mode :
        ``"per_k"`` (default)
            For each ``k``, re-optimize the score threshold (protocol behind the
            scalar ``F1_PAK`` AUC / tadpak ``best_f1_w_pa``).
        ``"fixed_k0"``
            Choose a **single** threshold that maximizes F1 at ``k=0`` (PA with
            any-overlap), then evaluate P/R/F1 at that same threshold for every
            ``k``. This matches the illustrative PAK curves in the paper figures
            (score density → how fast F1/recall drop as ``k`` grows).
        ``"fixed_nopa"``
            Choose a **single** threshold that maximizes point-wise F1 **without**
            point adjustment, then evaluate PAK P/R/F1 at that threshold for every
            ``k``. Avoids the k=0 pathology where an ultra-high τ* (one peak tip)
            gets perfect F1 via PA but collapses as soon as ``k>0``.
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    if threshold_mode not in {"per_k", "fixed_k0", "fixed_nopa"}:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")

    scores = np.asarray(scores, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=int).reshape(-1)
    if len(scores) != len(targets):
        raise ValueError("scores and targets must have the same length")
    if targets.sum() == 0:
        zeros = [0.0] * len(k_values)
        return {
            "k_values": list(k_values),
            "f1": list(zeros),
            "precision": list(zeros),
            "recall": list(zeros),
            "threshold": None,
            "threshold_mode": threshold_mode,
        }

    pak = _tadpak_pak()
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        ths = np.asarray([lo if np.isfinite(lo) else 0.0], dtype=float)
    else:
        ths = np.linspace(lo, hi, int(interval))

    def _best_at_k(k: int) -> tuple[float, float, float, float]:
        best = (-1.0, 0.0, 0.0, float(ths[0]))  # f1, P, R, thr
        for th in ths:
            pred = pak(scores, targets, float(th), int(k))
            precision, recall, f1 = _prf1(pred, targets)
            if f1 > best[0]:
                best = (f1, precision, recall, float(th))
        return best

    if threshold_mode == "fixed_k0":
        _f1, _p, _r, thr_star = _best_at_k(0)
        f1_out, precision_out, recall_out = [], [], []
        for k in k_values:
            pred = pak(scores, targets, thr_star, int(k))
            precision, recall, f1 = _prf1(pred, targets)
            f1_out.append(float(f1))
            precision_out.append(float(precision))
            recall_out.append(float(recall))
        return {
            "k_values": list(k_values),
            "f1": f1_out,
            "precision": precision_out,
            "recall": recall_out,
            "threshold": thr_star,
            "threshold_mode": threshold_mode,
        }

    if threshold_mode == "fixed_nopa":
        # Best point-wise F1 threshold (no PA), then freeze it across k.
        thr_star = None
        best_f1 = -1.0
        for th in ths:
            pred = scores >= float(th)
            _p, _r, f1 = _prf1(pred, targets)
            if f1 > best_f1:
                best_f1 = f1
                thr_star = float(th)
        assert thr_star is not None
        f1_out, precision_out, recall_out = [], [], []
        for k in k_values:
            pred = pak(scores, targets, thr_star, int(k))
            precision, recall, f1 = _prf1(pred, targets)
            f1_out.append(float(f1))
            precision_out.append(float(precision))
            recall_out.append(float(recall))
        return {
            "k_values": list(k_values),
            "f1": f1_out,
            "precision": precision_out,
            "recall": recall_out,
            "threshold": thr_star,
            "threshold_mode": threshold_mode,
        }

    # per_k
    f1_out, precision_out, recall_out = [], [], []
    for k in k_values:
        f1, precision, recall, _thr = _best_at_k(int(k))
        f1_out.append(float(f1 if f1 >= 0 else 0.0))
        precision_out.append(float(precision))
        recall_out.append(float(recall))
    return {
        "k_values": list(k_values),
        "f1": f1_out,
        "precision": precision_out,
        "recall": recall_out,
        "threshold": None,
        "threshold_mode": threshold_mode,
    }


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

"""Visualize seriejavi (or other custom) anomaly scores.

For **each** time series, write one PNG with:

1. **Top**: the raw series cropped to
   ``[anomaly_start - pad, anomaly_end + pad]`` (default pad=100),
   with the labeled anomaly shaded.
2. **Below**: one subplot per method showing that method's anomaly score
   **min-max normalized to [0, 1]** on the plotted window. Each panel
   annotates ``density = mean(normalized score[anomaly])``.

Examples
--------
    python scripts/plot_seriejavi_scores.py
    python scripts/plot_seriejavi_scores.py --pad 100 --methods AE Donut
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.plots import set_paper_style  # noqa: E402
from granularity_tsad.synthetic_rho import (  # noqa: E402
    list_synthetic_txt,
    parse_synthetic_filename,
)

DEFAULT_DATASET = "seriejavi"
DEFAULT_SOURCE = REPO_ROOT / "data" / "seriejavi"
DEFAULT_OUT = REPO_ROOT / "figures" / "seriejavi_scores"


def _discover_methods(scores_root: Path, dataset: str, schema: str) -> list[str]:
    methods = []
    if not scores_root.is_dir():
        return methods
    for method_dir in sorted(scores_root.iterdir()):
        score_dir = method_dir / schema / dataset
        if score_dir.is_dir() and any(score_dir.glob("*.npy")):
            methods.append(method_dir.name)
    return methods


def _discover_curves(source_dir: Path) -> list[str]:
    curves = []
    for path in list_synthetic_txt(source_dir):
        meta = parse_synthetic_filename(path.name)
        if meta:
            curves.append(meta["curve_id"])
    return curves


def _minmax01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize finite values to [0, 1] (NaNs preserved)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return out
    lo, hi = float(np.nanmin(x[valid])), float(np.nanmax(x[valid]))
    denom = (hi - lo) if hi > lo else 1.0
    out[valid] = (x[valid] - lo) / denom
    return out


def _anomaly_score_density(scores01: np.ndarray, a0: int, a1: int) -> float:
    """Mean of [0, 1]-normalized scores on the labeled anomaly span."""
    region = np.asarray(scores01[a0:a1], dtype=float).reshape(-1)
    if not np.isfinite(region).any():
        return float("nan")
    return float(np.nanmean(region))


def _load_full_series(
    data_dir: Path, dataset: str, curve: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (full_series, full_labels, train_index)."""
    curve_dir = data_dir / "UTS" / dataset / curve
    train = np.asarray(np.load(curve_dir / "train.npy"), dtype=float).reshape(-1)
    test = np.asarray(np.load(curve_dir / "test.npy"), dtype=float).reshape(-1)
    train_label = np.asarray(np.load(curve_dir / "train_label.npy"), dtype=float).reshape(-1)
    test_label = np.asarray(np.load(curve_dir / "test_label.npy"), dtype=float).reshape(-1)
    series = np.concatenate([train, test])
    labels = np.concatenate([train_label, test_label])
    return series, labels, len(train)


def _score_on_full_timeline(
    scores: np.ndarray, full_len: int
) -> np.ndarray:
    """Place detector scores on the full series axis (end-aligned, NaN pad)."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    out = np.full(full_len, np.nan, dtype=float)
    n = min(full_len, len(scores))
    out[full_len - n :] = scores[len(scores) - n :]
    return out


def plot_one_curve(
    series: np.ndarray,
    labels: np.ndarray,
    method_scores: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    pad: int = 100,
    window: tuple[int, int] | None = None,
) -> None:
    """One PNG: series window on top, [0,1]-normalized score subplots below.

    ``window`` is ``(anomaly_start, anomaly_end_exclusive)`` used for shading
    when labels are sparse or empty (controlled-ρ ``*_pw`` labels).
    """
    anom = np.where(labels > 0.5)[0]
    if len(anom) > 0:
        a0, a1 = int(anom[0]), int(anom[-1]) + 1
    elif window is not None:
        a0, a1 = int(window[0]), int(window[1])
    else:
        raise ValueError(f"{title}: no anomaly labels found")
    w0 = max(0, a0 - pad)
    w1 = min(len(series), a1 + pad)

    series_w = series[w0:w1]
    x = np.arange(w0, w1)  # absolute indices
    methods = list(method_scores.keys())

    n_panels = 1 + len(methods)
    fig, axs = plt.subplots(
        n_panels,
        1,
        figsize=(11, 1.8 + 1.05 * len(methods)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6] + [1.0] * len(methods)},
    )
    axs = np.atleast_1d(axs)

    ax0 = axs[0]
    ax0.axvspan(a0, a1, color="gray", alpha=0.28, zorder=0, label="anomaly")
    # Mark individual labeled points when they are sparse inside the window.
    if len(anom) > 0 and len(anom) < (a1 - a0):
        ax0.scatter(
            anom,
            series[anom],
            s=14,
            c="#c62828",
            zorder=3,
            label="pw labels",
        )
    ax0.plot(x, series_w, color="#222222", linewidth=1.0)
    ax0.set_ylabel("Value")
    ax0.set_title(
        f"{title}  |  window [{w0}, {w1})  (anomaly [{a0}, {a1}), pad=±{pad})"
    )
    ax0.set_xlim(w0, w1 - 1)

    for ax, method in zip(axs[1:], methods):
        sc_full = method_scores[method]
        # Min-max on the plotted window, then density = mean on anomaly labels
        # (fallback: mean on the shaded segment when labels are empty).
        sc01_full = np.full_like(sc_full, np.nan, dtype=float)
        sc_w = sc_full[w0:w1]
        sc01_w = _minmax01(sc_w)
        sc01_full[w0:w1] = sc01_w
        if len(anom) > 0:
            density = float(np.nanmean(sc01_full[anom]))
        else:
            density = _anomaly_score_density(sc01_full, a0, a1)
        ax.axvspan(a0, a1, color="gray", alpha=0.28, zorder=0)
        ax.plot(x, sc01_w, color="#b71c1c", linewidth=1.0)
        ax.set_ylabel(method, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.text(
            0.99,
            0.92,
            f"density={density:.4g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    axs[-1].set_xlabel("Index (full series)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--schema", default="naive")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--curves", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pad",
        type=int,
        default=100,
        help="Points before/after the anomaly region to include (default: 100).",
    )
    args = parser.parse_args()

    scores_root = REPO_ROOT / "Results" / "Scores"
    methods = args.methods or _discover_methods(scores_root, args.dataset, args.schema)
    curves = args.curves or _discover_curves(args.source_dir)

    if not methods:
        print(f"No score files under {scores_root}/*/{args.schema}/{args.dataset}/")
        print("Run: python scripts/run_seriejavi.py")
        return
    if not curves:
        print(f"No curves found in {args.source_dir}")
        return

    set_paper_style()
    print(f"Methods: {methods}")
    print(f"Curves:  {curves}")
    print(f"Pad:     ±{args.pad}")
    print(f"Output:  {args.output_dir}")

    # Remove older plot styles so the folder only has the new layout.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Map curve_id -> filename segment for empty / sparse label fallbacks.
    curve_window: dict[str, tuple[int, int]] = {}
    for path in list_synthetic_txt(args.source_dir):
        meta = parse_synthetic_filename(path.name)
        if meta:
            # Exclusive end matches convert_txt_to_easytsad default; +1 if
            # the last labeled index equals anomaly_end (inclusive pw files).
            curve_window[meta["curve_id"]] = (
                meta["anomaly_start"],
                meta["anomaly_end"] + 1,
            )

    for curve in curves:
        series, labels, _train_index = _load_full_series(
            args.data_dir, args.dataset, curve
        )
        method_scores: dict[str, np.ndarray] = {}
        for method in methods:
            path = scores_root / method / args.schema / args.dataset / f"{curve}.npy"
            if not path.exists():
                print(f"  skip missing: {method}/{curve}")
                continue
            raw = np.load(path)
            method_scores[method] = _score_on_full_timeline(raw, len(series))

        if not method_scores:
            print(f"  no scores for {curve}")
            continue

        out = args.output_dir / f"{curve}.png"
        plot_one_curve(
            series,
            labels,
            method_scores,
            title=curve,
            out_path=out,
            pad=args.pad,
            window=curve_window.get(curve),
        )
        try:
            shown = out.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"  wrote {shown}")

    print("Done.")


if __name__ == "__main__":
    main()

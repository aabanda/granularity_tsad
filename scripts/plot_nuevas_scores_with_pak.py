"""Combined score + PAK %K plots for nuevas (one figure per series).

For each time series:

1. **Top** (full width): raw series window with anomaly shaded / pw points.
2. **Each method row**:
   - **Left**: anomaly score min-max normalized to [0, 1] (same as before).
   - **Right**: that method's PAK **precision** and **recall** vs coverage ``k``.

Examples
--------
    python scripts/plot_nuevas_scores_with_pak.py
    python scripts/plot_nuevas_scores_with_pak.py --methods AE SubLOF MatrixProfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import align_scores_targets  # noqa: E402
from granularity_tsad.metrics.pak import DEFAULT_K_VALUES, pak_curves  # noqa: E402
from granularity_tsad.plots import set_paper_style  # noqa: E402
from granularity_tsad.synthetic_rho import (  # noqa: E402
    find_pw_label_file,
    list_synthetic_txt,
    load_pw_indices,
    parse_synthetic_filename,
)

# Reuse helpers from the score-only plotter.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "plot_seriejavi_scores", REPO_ROOT / "scripts" / "plot_seriejavi_scores.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

_discover_methods = _mod._discover_methods
_discover_curves = _mod._discover_curves
_minmax01 = _mod._minmax01
_anomaly_score_density = _mod._anomaly_score_density
_load_full_series = _mod._load_full_series
_score_on_full_timeline = _mod._score_on_full_timeline

DEFAULT_DATASET = "nuevas"
DEFAULT_SOURCE = REPO_ROOT / "data" / "nuevas"
DEFAULT_OUT = REPO_ROOT / "figures" / "nuevas_scores"


def plot_one_curve_with_pak(
    series: np.ndarray,
    labels: np.ndarray,
    method_scores_full: dict[str, np.ndarray],
    method_pak: dict[str, dict[str, list[float]]],
    title: str,
    out_path: Path,
    pad: int = 100,
    window: tuple[int, int] | None = None,
    k_values: list[int] | None = None,
    pw_indices: np.ndarray | None = None,
) -> None:
    """Series on top; each method row = score | PAK precision/recall.

    ``labels`` should be the **segment** ground truth used for PAK (one contiguous
    anomalous window). Optional ``pw_indices`` are over-plotted as red dots to
    show the controlled-ρ point anomalies without changing the PAK target.
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
    x = np.arange(w0, w1)
    methods = list(method_scores_full.keys())
    k_values = list(k_values or DEFAULT_K_VALUES)
    pw = np.asarray(pw_indices if pw_indices is not None else [], dtype=int).reshape(-1)
    pw = pw[(pw >= 0) & (pw < len(series))]

    n_methods = len(methods)
    fig = plt.figure(figsize=(14, 1.7 + 1.15 * n_methods))
    gs = GridSpec(
        n_methods + 1,
        2,
        figure=fig,
        height_ratios=[1.7] + [1.0] * n_methods,
        width_ratios=[2.4, 1.0],
        hspace=0.35,
        wspace=0.28,
    )

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axvspan(a0, a1, color="gray", alpha=0.28, zorder=0)
    if len(pw):
        ax0.scatter(pw, series[pw], s=14, c="#c62828", zorder=3, label="pw (ρ)")
    ax0.plot(x, series_w, color="#222222", linewidth=1.0)
    ax0.set_ylabel("Value")
    seg_len = max(a1 - a0, 1)
    ax0.set_title(
        f"{title}  |  window [{w0}, {w1})  "
        f"segment [{a0}, {a1}) len={seg_len}"
        + (f"  pw={len(pw)} (ρ≈{len(pw)/seg_len:.2f})" if len(pw) else "")
        + f"  pad=±{pad}"
    )
    ax0.set_xlim(w0, w1 - 1)
    ax0.set_xlabel("")

    score_axes = []
    pak_axes = []
    for i, method in enumerate(methods):
        ax_s = fig.add_subplot(gs[i + 1, 0], sharex=ax0 if i == 0 else score_axes[0])
        ax_p = fig.add_subplot(gs[i + 1, 1])
        score_axes.append(ax_s)
        pak_axes.append(ax_p)

        sc_full = method_scores_full[method]
        sc01_full = np.full_like(sc_full, np.nan, dtype=float)
        sc_w = sc_full[w0:w1]
        sc01_w = _minmax01(sc_w)
        sc01_full[w0:w1] = sc01_w
        # Density over the full shaded segment (matches PAK target).
        density = _anomaly_score_density(sc01_full, a0, a1)

        ax_s.axvspan(a0, a1, color="gray", alpha=0.28, zorder=0)
        ax_s.plot(x, sc01_w, color="#b71c1c", linewidth=1.0)
        ax_s.set_ylabel(method, fontsize=8)
        ax_s.set_ylim(-0.05, 1.05)
        ax_s.set_yticks([0.0, 0.5, 1.0])
        ax_s.text(
            0.99,
            0.92,
            f"density={density:.4g}",
            transform=ax_s.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
            },
        )

        curves = method_pak.get(method)
        if curves is None:
            ax_p.set_axis_off()
            ax_p.text(0.5, 0.5, "no PAK\n(no labels)", ha="center", va="center", fontsize=8)
        else:
            precision = np.asarray(curves["precision"], dtype=float)
            recall = np.asarray(curves["recall"], dtype=float)
            f1 = np.asarray(curves["f1"], dtype=float)
            auc = float(np.trapz(f1) / max(len(f1), 1))
            ax_p.plot(
                k_values,
                precision,
                marker="o",
                markersize=2.2,
                linewidth=1.1,
                color="#1565c0",
                label="P",
            )
            ax_p.plot(
                k_values,
                recall,
                marker="s",
                markersize=2.2,
                linewidth=1.1,
                color="#2e7d32",
                label="R",
            )
            ax_p.set_ylim(-0.05, 1.05)
            ax_p.set_yticks([0.0, 0.5, 1.0])
            ax_p.set_xlim(0, 100)
            ax_p.legend(frameon=False, fontsize=6, loc="lower left", ncols=2)
            thr = curves.get("threshold")
            thr_txt = f"τ*={thr:.3g}\n" if thr is not None else ""
            ax_p.text(
                0.98,
                0.92,
                f"{thr_txt}F1 AUC={auc:.3f}",
                transform=ax_p.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                },
            )
        if i == 0:
            ax_p.set_title("PAK P/R vs k (single τ*, best F1 w/o PA)", fontsize=8)
        if i < n_methods - 1:
            ax_s.tick_params(axis="x", labelbottom=False)
            ax_p.tick_params(axis="x", labelbottom=False)
        else:
            ax_s.set_xlabel("Index (full series)")
            ax_p.set_xlabel(r"$k$ (%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--schema", default="naive")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--curves", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pad", type=int, default=100)
    parser.add_argument("--k-values", type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_values = args.k_values or cfg.k_values or DEFAULT_K_VALUES
    scores_root = REPO_ROOT / "Results" / "Scores"
    available = set(
        args.methods or _discover_methods(scores_root, args.dataset, args.schema)
    )
    # Point-wise first (config order), then sequence-wise at the end.
    methods = [m for m in cfg.methods_point_wise if m in available]
    methods += [m for m in cfg.methods_sequence_wise if m in available and m not in methods]
    # Any leftover discovered methods (not in config) go after point-wise, before seq.
    leftovers = [m for m in sorted(available) if m not in methods]
    if leftovers:
        # Keep leftovers just before sequence-wise block.
        seq = [m for m in methods if m in cfg.methods_sequence_wise]
        point = [m for m in methods if m not in cfg.methods_sequence_wise]
        methods = point + leftovers + seq
    curves = args.curves or _discover_curves(args.source_dir)

    if not methods:
        print(f"No scores under {scores_root}/*/{args.schema}/{args.dataset}/")
        return
    if not curves:
        print(f"No curves in {args.source_dir}")
        return

    set_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    curve_window: dict[str, tuple[int, int]] = {}
    curve_src: dict[str, Path] = {}
    for path in list_synthetic_txt(args.source_dir):
        meta = parse_synthetic_filename(path.name)
        if meta:
            curve_window[meta["curve_id"]] = (
                meta["anomaly_start"],
                meta["anomaly_end"] + 1,
            )
            curve_src[meta["curve_id"]] = path

    print(f"Methods: {methods}")
    print(f"Curves:  {curves}")
    print(f"k-grid:  {k_values}")
    print(f"Output:  {args.output_dir}")
    print("PAK target: full anomaly segment; threshold: single τ* = best F1 without PA")

    for curve in curves:
        series, labels, train_index = _load_full_series(
            args.data_dir, args.dataset, curve
        )
        # Build / force segment GT for PAK so coverage-k is meaningful.
        win = curve_window.get(curve)
        if win is not None:
            seg_labels = np.zeros(len(series), dtype=float)
            seg_labels[win[0] : win[1]] = 1.0
            labels = seg_labels
        test_labels = labels[train_index:].astype(int)

        pw = np.asarray([], dtype=int)
        src = curve_src.get(curve)
        if src is not None:
            pw_path = find_pw_label_file(src, curve)
            if pw_path is not None:
                pw = load_pw_indices(pw_path)

        method_scores_full: dict[str, np.ndarray] = {}
        method_pak: dict[str, dict[str, list[float]]] = {}
        for method in methods:
            path = scores_root / method / args.schema / args.dataset / f"{curve}.npy"
            if not path.exists():
                print(f"  skip missing: {method}/{curve}")
                continue
            raw = np.asarray(np.load(path), dtype=float).reshape(-1)
            method_scores_full[method] = _score_on_full_timeline(raw, len(series))
            if test_labels.sum() > 0:
                s, t = align_scores_targets(raw, test_labels, method)
                if t.sum() > 0:
                    method_pak[method] = pak_curves(
                        s,
                        t,
                        k_values=k_values,
                        interval=100,
                        threshold_mode="fixed_nopa",
                    )

        if not method_scores_full:
            print(f"  no scores for {curve}")
            continue

        out = args.output_dir / f"{curve}.png"
        plot_one_curve_with_pak(
            series,
            labels,
            method_scores_full,
            method_pak,
            title=curve,
            out_path=out,
            pad=args.pad,
            window=curve_window.get(curve),
            k_values=k_values,
            pw_indices=pw,
        )
        try:
            shown = out.resolve().relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        n_pak = len(method_pak)
        print(f"  wrote {shown} (+ pdf)  (PAK for {n_pak}/{len(method_scores_full)} methods)")

    print("Done.")


if __name__ == "__main__":
    main()

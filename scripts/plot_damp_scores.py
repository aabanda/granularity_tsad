"""Plot anomaly scores vs UCR series (sanity check for DAMP / families).

For each selected series:
  - top: time series with anomaly shaded
  - below: 3 point-wise + 3 sequence-wise scores (min-max on the window)

Example
-------
    python scripts/plot_damp_scores.py
    python scripts/plot_damp_scores.py --curves 141 14 110 --pad 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import (  # noqa: E402
    align_scores_targets,
    build_anomaly_df,
    build_binary_targets,
    load_scores,
)
from granularity_tsad.plots import set_paper_style  # noqa: E402

DEFAULT_POINT = ["AE", "Donut", "TimesNet"]
DEFAULT_SEQ = ["SubLOF", "MatrixProfile", "DAMP"]
POINT_COLOR = "#1565c0"
SEQ_COLOR = "#ef6c00"


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    denom = (hi - lo) if hi > lo else 1.0
    return (x - lo) / denom


def _default_curves(pak: pd.DataFrame) -> list[int]:
    """High / mid / lower DAMP F1_PAK within each density group."""
    curves: list[int] = []
    dens_col = "density" if "density" in pak.columns else None
    if dens_col is None:
        s = pak["DAMP"].sort_values(ascending=False)
        return [int(s.index[i]) for i in (0, 1, len(s) // 2, -3, -2, -1)]
    for d in ("dense", "sparse", "normal"):
        sub = pak[pak[dens_col] == d].sort_values("DAMP", ascending=False)
        if sub.empty:
            continue
        idxs = [0, len(sub) // 3, 2 * len(sub) // 3]
        for i in idxs:
            curves.append(int(sub.index[min(i, len(sub) - 1)]))
    seen: set[int] = set()
    return [c for c in curves if not (c in seen or seen.add(c))]


def _plot_one(
    series: np.ndarray,
    scores: dict[str, np.ndarray],
    method_order: list[str],
    family_of: dict[str, str],
    *,
    dataset_id: int,
    train_index: int,
    a0: int,
    a1: int,
    pad: int,
    density: str | None,
    f1_pak_damp: float | None,
    out_path: Path,
) -> None:
    abs_a0, abs_a1 = a0, a1 + 1
    left = max(0, abs_a0 - pad)
    right = min(len(series), abs_a1 + pad)

    methods = [m for m in method_order if m in scores]
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(11, 2.0 + 1.35 * len(methods)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5] + [1.0] * len(methods)},
    )
    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    t = np.arange(left, right)
    ax0.plot(t, series[left:right], color="#263238", lw=0.9)
    ax0.axvspan(abs_a0, abs_a1, color="tomato", alpha=0.25, zorder=0, label="anomaly")
    title = f"UCR {dataset_id}"
    if density:
        title += f"  [{density}]"
    title += f"  train_idx={train_index}, anomaly=[{a0},{a1}]"
    if f1_pak_damp is not None and np.isfinite(f1_pak_damp):
        title += f"  |  DAMP F1_PAK={f1_pak_damp:.3f}"
    ax0.set_title(title, fontsize=11)
    ax0.set_ylabel("value")
    ax0.legend(loc="upper right", frameon=False, fontsize=8)

    for ax, method in zip(axes[1:], methods):
        s = scores[method]
        full_score = np.full(len(series), np.nan, dtype=float)
        n = min(len(s), len(series) - train_index)
        full_score[train_index : train_index + n] = s[:n]
        window = full_score[left:right]
        valid = np.isfinite(window)
        fam = family_of.get(method, "other")
        color = POINT_COLOR if fam == "point" else SEQ_COLOR
        if valid.any():
            window01 = np.full_like(window, np.nan)
            window01[valid] = _minmax01(window[valid])
            ax.plot(t, window01, color=color, lw=1.0)
            ta0 = max(0, a0 - train_index)
            ta1 = max(ta0, a1 - train_index + 1)
            dens = (
                float(np.nanmean(_minmax01(s)[ta0:ta1])) if ta1 > ta0 else float("nan")
            )
            ax.set_title(
                f"{method}  ({fam}-wise)  |  dens_anom={dens:.3f}",
                fontsize=9,
                color=color,
            )
        else:
            ax.text(0.5, 0.5, "no score in window", transform=ax.transAxes, ha="center")
            ax.set_title(f"{method}  ({fam}-wise)", fontsize=9, color=color)
        ax.axvspan(abs_a0, abs_a1, color="tomato", alpha=0.18, zorder=0)
        ax.set_ylabel("score")
        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel("time index (full series)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--curves", nargs="+", type=int, default=None)
    parser.add_argument("--pad", type=int, default=250)
    parser.add_argument("--point-methods", nargs="+", default=DEFAULT_POINT)
    parser.add_argument("--seq-methods", nargs="+", default=DEFAULT_SEQ)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "damp_scores",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_paper_style()
    ucr_dir = cfg.data_dir / "raw" / "UCR_Anomaly_FullData"
    anomaly_df = cfg.filter_sequence_anomalies(build_anomaly_df(ucr_dir))
    targets = build_binary_targets(anomaly_df)

    pak_path = cfg.data_dir / "metrics" / "with_density" / "f1_pak.csv"
    if not pak_path.exists():
        pak_path = cfg.data_dir / "metrics" / "f1_pak.csv"
    pak = pd.read_csv(pak_path, index_col=0)
    pak.index = pak.index.astype(int)

    curves = args.curves or _default_curves(pak)
    point_methods = list(args.point_methods)
    seq_methods = list(args.seq_methods)
    methods = point_methods + seq_methods
    family_of = {m: "point" for m in point_methods}
    family_of.update({m: "sequence" for m in seq_methods})

    df_scores = load_scores(
        cfg.results_dir / "Scores",
        methods,
        curves,
        schema=cfg.training_schema,
    )

    dens_map = (
        pak["density"].astype(str).to_dict() if "density" in pak.columns else {}
    )

    for ds in curves:
        if ds not in anomaly_df.index:
            print(f"skip {ds}: not in filtered anomaly table")
            continue
        row = anomaly_df.loc[ds]
        series = np.loadtxt(ucr_dir / row["File"]).astype(float).reshape(-1)
        train_index = int(row["Train Index"])
        a0, a1 = int(row["Anomaly Start"]), int(row["Anomaly End"])

        scores: dict[str, np.ndarray] = {}
        for m in methods:
            raw = df_scores.loc[ds, m] if m in df_scores.columns else None
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                print(f"  {ds}/{m}: missing score")
                continue
            s, _t = align_scores_targets(
                np.asarray(raw, dtype=float), targets.loc[ds], m
            )
            scores[m] = s
        if not scores:
            continue

        f1 = (
            float(pak.loc[ds, "DAMP"])
            if "DAMP" in pak.columns and ds in pak.index
            else None
        )
        dens = dens_map.get(ds)
        _plot_one(
            series,
            scores,
            methods,
            family_of,
            dataset_id=int(ds),
            train_index=train_index,
            a0=a0,
            a1=a1,
            pad=args.pad,
            density=dens,
            f1_pak_damp=f1,
            out_path=args.output_dir / f"scores_ucr_{ds}.png",
        )


if __name__ == "__main__":
    main()

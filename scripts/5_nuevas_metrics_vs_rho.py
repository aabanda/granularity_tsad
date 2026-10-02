"""Minimal nuevas experiment figures: metrics vs ρ and ranking flip.

Computes ``F1_point``, ``F1_seq``, ``F1_PAK`` for every (ρ, method) on
``data/UTS/nuevas`` (segment GT from the filename), then writes:

1. ``figures/nuevas_experiment/metrics_vs_rho.csv``
2. ``metrics_vs_rho.pdf/.png`` — one panel per metric; selected point-wise
   and sequence-wise detectors vs ρ
3. ``family_dist_vs_rho.pdf/.png`` — full score distribution (boxplots) of
   point-wise vs sequence-wise methods at each ρ
4. ``ranking_flip.pdf/.png`` — which family wins (Δ = point − sequence median)

Example
-------
    python scripts/5_nuevas_metrics_vs_rho.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from granularity_tsad.config import load_config  # noqa: E402
from granularity_tsad.data import align_scores_targets  # noqa: E402
from granularity_tsad.metrics import f1_pak, f1_point, f1_seq  # noqa: E402
from granularity_tsad.plots import set_paper_style  # noqa: E402
from granularity_tsad.synthetic_rho import (  # noqa: E402
    list_synthetic_txt,
    parse_synthetic_filename,
)

DEFAULT_OUT = REPO_ROOT / "figures" / "nuevas_experiment"
# Representative methods for the line plot (paper-style claim).
HIGHLIGHT_POINT = ["AE", "LSTMADalpha"]
HIGHLIGHT_SEQ = ["SubLOF", "MatrixProfile"]


def _parse_rho(curve_id: str) -> float | None:
    m = re.search(r"(\d+\.\d+)$", curve_id)
    if not m:
        m = re.search(r"(\d+\.\d+)", curve_id)
    return float(m.group(1)) if m else None


def _segment_test_labels(
    data_dir: Path, dataset: str, curve: str, train_index: int, a0: int, a1: int
) -> np.ndarray:
    """Binary labels on the test region for the filename segment [a0, a1] inclusive."""
    test = np.load(data_dir / "UTS" / dataset / curve / "test.npy")
    full_len = train_index + len(np.asarray(test).reshape(-1))
    lab = np.zeros(full_len, dtype=int)
    lab[a0 : a1 + 1] = 1
    return lab[train_index:]


def compute_table(
    cfg,
    data_dir: Path,
    source_dir: Path,
    dataset: str,
    methods: list[str],
) -> pd.DataFrame:
    scores_root = cfg.repo_root / "Results" / "Scores"
    schema = cfg.training_schema
    rows = []
    metas = {}
    for path in list_synthetic_txt(source_dir):
        meta = parse_synthetic_filename(path.name)
        if meta:
            metas[meta["curve_id"]] = meta

    curves = sorted(metas.keys(), key=lambda c: _parse_rho(c) or -1)
    point_set = set(cfg.methods_point_wise)
    seq_set = set(cfg.methods_sequence_wise)

    for curve in curves:
        meta = metas[curve]
        rho = _parse_rho(curve)
        targets = _segment_test_labels(
            data_dir,
            dataset,
            curve,
            meta["train_index"],
            meta["anomaly_start"],
            meta["anomaly_end"],
        )
        for method in methods:
            sp = scores_root / method / schema / dataset / f"{curve}.npy"
            if not sp.exists():
                print(f"  skip missing {method}/{curve}")
                continue
            raw = np.asarray(np.load(sp), dtype=float).reshape(-1)
            s, t = align_scores_targets(raw, targets, method)
            if t.sum() == 0:
                continue
            fam = (
                "point_wise"
                if method in point_set
                else ("sequence_wise" if method in seq_set else "other")
            )
            row = {
                "curve": curve,
                "rho": rho,
                "method": method,
                "family": fam,
                "f1_point": float(f1_point(s, t)),
                "f1_seq": float(f1_seq(s, t)),
                "f1_pak": float(f1_pak(s, t, k_values=cfg.k_values or None)),
            }
            rows.append(row)
            print(
                f"  {curve} {method}: point={row['f1_point']:.3f} "
                f"seq={row['f1_seq']:.3f} pak={row['f1_pak']:.3f}"
            )
    return pd.DataFrame(rows)


def plot_metrics_vs_rho(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [("f1_point", r"$F1_{point}$"), ("f1_seq", r"$F1_{seq}$"), ("f1_pak", r"$F1_{PAK}$")]
    highlight = HIGHLIGHT_POINT + HIGHLIGHT_SEQ
    colors = {
        "AE": "#1565c0",
        "LSTMADalpha": "#6a1b9a",
        "SubLOF": "#2e7d32",
        "MatrixProfile": "#ef6c00",
    }
    styles = {
        "AE": "-",
        "LSTMADalpha": "--",
        "SubLOF": "-",
        "MatrixProfile": "--",
    }

    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True)
    for ax, (col, ylab) in zip(axs, metrics):
        for method in highlight:
            sub = df[df["method"] == method].sort_values("rho")
            if sub.empty:
                continue
            ax.plot(
                sub["rho"],
                sub[col],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=colors.get(method, "#333"),
                linestyle=styles.get(method, "-"),
                label=method,
            )
        ax.set_xlabel(r"point density $\rho$")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(ylab)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_vs_rho.pdf")
    fig.savefig(out_dir / "metrics_vs_rho.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'metrics_vs_rho.pdf'}")


def plot_family_and_flip(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [("f1_point", r"$F1_{point}$"), ("f1_seq", r"$F1_{seq}$"), ("f1_pak", r"$F1_{PAK}$")]
    sub = df[df["family"].isin(["point_wise", "sequence_wise"])].copy()
    rhos = sorted(sub["rho"].unique())
    fam_med = (
        sub.groupby(["rho", "family"], as_index=False)[["f1_point", "f1_seq", "f1_pak"]]
        .median()
    )

    # Full distribution per family × ρ (grouped boxplots)
    colors = {"point_wise": "#1565c0", "sequence_wise": "#ef6c00"}
    labels = {"point_wise": "point-wise", "sequence_wise": "sequence-wise"}
    width = 0.35
    offset = {"point_wise": -width / 2, "sequence_wise": width / 2}

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), sharex=True)
    for ax, (col, ylab) in zip(axs, metrics):
        for i, rho in enumerate(rhos):
            for family in ("point_wise", "sequence_wise"):
                vals = sub.loc[(sub["rho"] == rho) & (sub["family"] == family), col].dropna()
                if vals.empty:
                    continue
                bp = ax.boxplot(
                    [vals.to_numpy()],
                    positions=[i + offset[family]],
                    widths=width * 0.9,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(
                        marker="o",
                        markersize=3,
                        markerfacecolor=colors[family],
                        markeredgecolor=colors[family],
                        alpha=0.6,
                    ),
                    medianprops=dict(color="white", linewidth=1.2),
                    whiskerprops=dict(color=colors[family]),
                    capprops=dict(color=colors[family]),
                    boxprops=dict(
                        facecolor=colors[family],
                        edgecolor=colors[family],
                        alpha=0.75,
                    ),
                )
                # keep a single legend entry per family
                if i == 0:
                    bp["boxes"][0].set_label(labels[family])
        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels([f"{r:g}" for r in rhos], rotation=45, ha="right")
        ax.set_xlabel(r"point density $\rho$")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(ylab)
        ax.set_xlim(-0.7, len(rhos) - 0.3)
    axs[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for name in ("family_dist_vs_rho", "family_median_vs_rho"):
        fig.savefig(out_dir / f"{name}.pdf")
        fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'family_dist_vs_rho.pdf'}")

    # Ranking flip: Δ = point median − sequence median (>0 → point wins)
    pivot = fam_med.pivot(index="rho", columns="family")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for col, ylab, color in [
        ("f1_point", r"$\Delta F1_{point}$", "#1565c0"),
        ("f1_seq", r"$\Delta F1_{seq}$", "#2e7d32"),
        ("f1_pak", r"$\Delta F1_{PAK}$", "#6a1b9a"),
    ]:
        if ("point_wise" not in pivot[col].columns) or (
            "sequence_wise" not in pivot[col].columns
        ):
            continue
        delta = pivot[col]["point_wise"] - pivot[col]["sequence_wise"]
        ax.plot(
            delta.index,
            delta.values,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=color,
            label=ylab,
        )
    ax.axhline(0.0, color="#666", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"point density $\rho$")
    ax.set_ylabel(r"median(point) $-$ median(sequence)")
    ax.set_title("Ranking flip: who wins as density grows?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ranking_flip.pdf")
    fig.savefig(out_dir / "ranking_flip.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'ranking_flip.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", default="nuevas")
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data" / "nuevas")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Reuse an existing metrics CSV instead of recomputing.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    if args.csv and Path(args.csv).exists():
        df = pd.read_csv(args.csv)
        print(f"Loaded {args.csv} ({len(df)} rows)")
    else:
        methods = args.methods or (
            list(dict.fromkeys(cfg.methods_point_wise + cfg.methods_sequence_wise))
        )
        print(f"Computing metrics for {len(methods)} methods…")
        df = compute_table(cfg, args.data_dir, args.source_dir, args.dataset, methods)
        csv_path = out_dir / "metrics_vs_rho.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    if df.empty:
        print("No metrics computed.")
        return

    plot_metrics_vs_rho(df, out_dir)
    plot_family_and_flip(df, out_dir)
    print("Done. Score+PAK per-ρ panels already in figures/nuevas_scores/")


if __name__ == "__main__":
    main()

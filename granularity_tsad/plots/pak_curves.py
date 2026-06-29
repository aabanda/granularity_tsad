"""PAK F1 curves and the WPAK weighting (paper Figures 9 & 11)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..metrics.pak import DEFAULT_K_VALUES
from .style import style_axes


def plot_pak_curves(
    curves: dict[str, list[float]],
    k_values: list[int] | None = None,
    ylabel: str = "F1",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot one or more PAK F1 curves (``{label: f1_curve}``) vs. coverage k."""
    k_values = k_values or DEFAULT_K_VALUES
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    style_axes(ax)
    for label, curve in curves.items():
        ax.plot(k_values, curve, marker="o", markersize=4, label=label)
    ax.set_xlabel("Coverage threshold k (%)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    if len(curves) > 1:
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    return ax


def plot_wpak_weighting(
    f1_curve: list[float],
    k_values: list[int] | None = None,
    title: str = "PAK vs. WPAK weighting",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Show a PAK F1 curve alongside its ``w(k)=1/k`` weighted (WPAK) version."""
    k_values = k_values or DEFAULT_K_VALUES
    f1 = np.asarray(f1_curve, dtype=float)
    ks = np.asarray(k_values, dtype=float)
    weights = np.where(ks > 0, 1.0 / np.where(ks == 0, 1, ks), 1.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    style_axes(ax)
    ax.plot(k_values, f1, marker="o", markersize=4, color="#1565c0", label="PAK F1")
    ax.plot(
        k_values,
        f1 * weights,
        marker="s",
        markersize=4,
        color="#c62828",
        label="WPAK (w=1/k)",
    )
    ax.set_xlabel("Coverage threshold k (%)")
    ax.set_ylabel("F1")
    ax.legend(frameon=False)
    ax.set_title(title)
    return ax

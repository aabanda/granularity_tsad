"""Shared plotting style for publication-quality, editable vector figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Cluster palette reused across figures.
CLUSTER_COLORS = {
    "Point-sparse": "#2196F3",
    "Point-normal": "#4CAF50",
    "Point-dense": "#FF9800",
}

SCORE_CMAP = "YlOrRd"  # anomaly-score colormap used in the heatmaps


def set_paper_style() -> None:
    """Configure matplotlib for clean, editable vector output (PDF/SVG).

    Keeps text as real text (not paths) so figures remain editable and crisp,
    uses a white background, a visible axes box and no grid.
    """
    mpl.rcParams.update(
        {
            # Editable text in vector formats.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            # White background, boxed axes, no grid.
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def style_axes(ax: plt.Axes) -> plt.Axes:
    """Apply the boxed, grid-free look to a single axes."""
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    return ax

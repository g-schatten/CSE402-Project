"""Shared matplotlib style so report figures are visibly the same project
as the web dashboard (PLAN.md section 8): the same compartment/solver
colors as web/src/design/tokens.css, vector output, Type 1 fonts, no
rasterized text.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

COMPARTMENT_COLOR = {"S": "#2f7fd6", "E": "#8a5fc9", "I": "#d43f4e", "R": "#22a06b", "V": "#c98a1f"}
SOLVER_COLOR = {
    "euler": "#d43f4e",
    "heun": "#c98a1f",
    "rk4": "#22a06b",
    "rk45": "#2f7fd6",
    "backward_euler": "#8a5fc9",
    "trapezoidal": "#1f9aa0",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333f52",
            "axes.labelcolor": "#131a24",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "text.color": "#131a24",
            "xtick.color": "#4d5b70",
            "ytick.color": "#4d5b70",
            "font.family": "sans-serif",
            "font.size": 10,
            "grid.color": "#dde3ec",
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,  # Type 1 / embeddable, not rasterized
            "ps.fonttype": 42,
        }
    )

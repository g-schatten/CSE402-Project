"""Common result type and dense-output interpolation for every solver.

`integrate()` is the single entry point every experiment calls; it dispatches
to the requested solver and returns a SolveResult with a uniform shape
regardless of which method produced it, so downstream code (sensitivity,
estimation, diagnostics) never has to special-case a particular solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sirlab.models.base import Model


@dataclass
class SolveResult:
    t: NDArray[np.float64]
    y: NDArray[np.float64]  # shape (n_steps+1, n_state)
    n_fev: int
    n_steps: int
    n_rejected: int = 0
    h_history: NDArray[np.float64] | None = None
    diverged: bool = False
    negativity_events: int = 0

    def at(self, t_query: NDArray[np.float64]) -> NDArray[np.float64]:
        """Dense-output evaluation at arbitrary times via piecewise cubic
        Hermite interpolation using the stored trajectory's local slopes
        (finite-difference estimated from the stored grid -- exact for the
        polynomial solvers' own stages would require storing stage data,
        which dense_output() in solvers/rk.py does when requested). This
        default is used when no per-solver dense output is available.
        """
        return _hermite_interp(self.t, self.y, np.asarray(t_query, dtype=float))


def _hermite_interp(t_grid: NDArray[np.float64], y_grid: NDArray[np.float64], t_query: NDArray[np.float64]) -> NDArray[np.float64]:
    n_state = y_grid.shape[1]
    # finite-difference slopes at grid nodes (central where possible)
    slopes = np.gradient(y_grid, t_grid, axis=0)
    out = np.empty((len(t_query), n_state))
    idx = np.clip(np.searchsorted(t_grid, t_query, side="right") - 1, 0, len(t_grid) - 2)
    for j, i in enumerate(idx):
        h = t_grid[i + 1] - t_grid[i]
        s = (t_query[j] - t_grid[i]) / h if h > 0 else 0.0
        y0, y1 = y_grid[i], y_grid[i + 1]
        m0, m1 = slopes[i] * h, slopes[i + 1] * h
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        out[j] = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
    return out


def uniform_grid(t_span: tuple[float, float], h: float) -> NDArray[np.float64]:
    t0, tf = t_span
    n = int(round((tf - t0) / h))
    return t0 + h * np.arange(n + 1)

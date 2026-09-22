"""Grid-based integration of sampled arrays (composite Simpson / trapezoid),
used to turn a solver's own dense trajectory into incidence observations:
incidence(t_k) = integral of beta*S*I/N over [t_{k-1}, t_k] (PLAN.md 2.5).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def trapezoid(y: NDArray[np.float64], t: NDArray[np.float64]) -> float:
    """Trapezoidal integral of y sampled at (possibly non-uniform) times t."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def simpson_nonuniform(y: NDArray[np.float64], t: NDArray[np.float64]) -> float:
    """Composite Simpson's rule on a non-uniform grid.

    Falls back to the trapezoidal rule on any 2-point interval; uses the
    standard 3-point parabola formula for unequal spacing otherwise.
    """
    n = len(t) - 1
    if n < 2:
        return trapezoid(y, t)
    total = 0.0
    i = 0
    while i < n - 1:
        h0 = t[i + 1] - t[i]
        h1 = t[i + 2] - t[i + 1]
        y0, y1, y2 = y[i], y[i + 1], y[i + 2]
        if h0 <= 0 or h1 <= 0:
            total += trapezoid(y[i : i + 2], t[i : i + 2])
            i += 1
            continue
        total += (h0 + h1) / 6.0 * (
            (2.0 - h1 / h0) * y0 + (h0 + h1) ** 2 / (h0 * h1) * y1 + (2.0 - h0 / h1) * y2
        )
        i += 2
    if i < n:  # one leftover interval when n is odd
        total += trapezoid(y[i:], t[i:])
    return float(total)

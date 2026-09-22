"""Unified dispatch: `integrate(model, y0, theta, t_span, h, solver=...)` is
the single entry point every experiment and every estimator calls, so the
rest of the codebase never needs to know which concrete solver ran.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.models.base import Model
from sirlab.solvers.base import SolveResult
from sirlab.solvers.explicit import integrate_euler, integrate_heun, integrate_rk4
from sirlab.solvers.implicit import integrate_backward_euler, integrate_trapezoidal
from sirlab.solvers.rk45 import integrate_rk45

SOLVER_ORDER = {"euler": 1, "heun": 2, "rk4": 4, "backward_euler": 1, "trapezoidal": 2, "rk45": 5}
SOLVER_FEV_PER_STEP = {"euler": 1, "heun": 2, "rk4": 4, "backward_euler": None, "trapezoidal": None, "rk45": 6}

FIXED_STEP_SOLVERS = {
    "euler": integrate_euler,
    "heun": integrate_heun,
    "rk4": integrate_rk4,
    "backward_euler": integrate_backward_euler,
    "trapezoidal": integrate_trapezoidal,
}


def integrate(
    model: Model,
    y0: NDArray[np.float64],
    theta: NDArray[np.float64],
    t_span: tuple[float, float],
    *,
    solver: str = "rk4",
    h: float | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> SolveResult:
    """Dispatch to the requested solver. Fixed-step solvers require `h`;
    'rk45' is adaptive and uses (rtol, atol) instead."""
    if solver == "rk45":
        return integrate_rk45(model, np.asarray(y0, dtype=float), np.asarray(theta, dtype=float), t_span, rtol=rtol, atol=atol)
    if solver not in FIXED_STEP_SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {sorted(FIXED_STEP_SOLVERS) + ['rk45']}")
    if h is None:
        raise ValueError(f"solver {solver!r} is fixed-step and requires h")
    return FIXED_STEP_SOLVERS[solver](model, np.asarray(y0, dtype=float), np.asarray(theta, dtype=float), t_span, h)


__all__ = [
    "integrate",
    "SolveResult",
    "SOLVER_ORDER",
    "SOLVER_FEV_PER_STEP",
    "FIXED_STEP_SOLVERS",
]

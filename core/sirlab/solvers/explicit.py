"""Fixed-step explicit solvers: Forward Euler, Heun (explicit trapezoid /
RK2), and classical RK4 (PLAN.md section 2.3). Each takes and returns the
same shapes so they are interchangeable everywhere else in the codebase.

The stage-combination formulas are written in *exactly* the textbook form
given in PLAN.md 2.3 (not algebraically simplified), because the
TypeScript twin (web/src/numerics) mirrors this literally so the two
implementations parity-test to ~1e-12.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.models.base import Model
from sirlab.solvers.base import SolveResult, uniform_grid


def integrate_euler(model: Model, y0: NDArray[np.float64], theta: NDArray[np.float64], t_span: tuple[float, float], h: float) -> SolveResult:
    t = uniform_grid(t_span, h)
    y = np.empty((len(t), model.n_state))
    y[0] = y0
    n_fev = 0
    negativity_events = 0
    diverged = False
    for k in range(len(t) - 1):
        yk = y[k]
        f = model.rhs(t[k], yk, theta)
        n_fev += 1
        y[k + 1] = yk + h * f
        if np.any(y[k + 1] < 0):
            negativity_events += 1
        if not np.all(np.isfinite(y[k + 1])) or np.any(np.abs(y[k + 1]) > 1e8 * max(1.0, np.max(np.abs(y0)))):
            diverged = True
            y[k + 2 :] = np.nan
            break
    return SolveResult(t=t, y=y, n_fev=n_fev, n_steps=len(t) - 1, negativity_events=negativity_events, diverged=diverged)


def integrate_heun(model: Model, y0: NDArray[np.float64], theta: NDArray[np.float64], t_span: tuple[float, float], h: float) -> SolveResult:
    t = uniform_grid(t_span, h)
    y = np.empty((len(t), model.n_state))
    y[0] = y0
    n_fev = 0
    negativity_events = 0
    diverged = False
    for k in range(len(t) - 1):
        yk = y[k]
        f0 = model.rhs(t[k], yk, theta)
        y_pred = yk + h * f0
        f1 = model.rhs(t[k + 1], y_pred, theta)
        n_fev += 2
        y[k + 1] = yk + (h / 2.0) * (f0 + f1)
        if np.any(y[k + 1] < 0):
            negativity_events += 1
        if not np.all(np.isfinite(y[k + 1])) or np.any(np.abs(y[k + 1]) > 1e8 * max(1.0, np.max(np.abs(y0)))):
            diverged = True
            y[k + 2 :] = np.nan
            break
    return SolveResult(t=t, y=y, n_fev=n_fev, n_steps=len(t) - 1, negativity_events=negativity_events, diverged=diverged)


def integrate_rk4(model: Model, y0: NDArray[np.float64], theta: NDArray[np.float64], t_span: tuple[float, float], h: float) -> SolveResult:
    t = uniform_grid(t_span, h)
    y = np.empty((len(t), model.n_state))
    y[0] = y0
    n_fev = 0
    negativity_events = 0
    diverged = False
    for k in range(len(t) - 1):
        yk = y[k]
        tk = t[k]
        k1 = model.rhs(tk, yk, theta)
        k2 = model.rhs(tk + h / 2.0, yk + (h / 2.0) * k1, theta)
        k3 = model.rhs(tk + h / 2.0, yk + (h / 2.0) * k2, theta)
        k4 = model.rhs(tk + h, yk + h * k3, theta)
        n_fev += 4
        y[k + 1] = yk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if np.any(y[k + 1] < 0):
            negativity_events += 1
        if not np.all(np.isfinite(y[k + 1])) or np.any(np.abs(y[k + 1]) > 1e8 * max(1.0, np.max(np.abs(y0)))):
            diverged = True
            y[k + 2 :] = np.nan
            break
    return SolveResult(t=t, y=y, n_fev=n_fev, n_steps=len(t) - 1, negativity_events=negativity_events, diverged=diverged)

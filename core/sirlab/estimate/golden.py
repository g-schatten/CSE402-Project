"""Golden-section line search wrapped in coordinate descent (PLAN.md section
2.6, item 2 -- the textbook Week 9-10 one-dimensional optimizer, demonstrated
directly on the same cost landscape the grid search visualizes).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import FitResult, ForwardModel

_PHI = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618


def golden_section_1d(f, lo: float, hi: float, *, tol: float = 1e-6, max_iter: int = 200) -> tuple[float, float, int]:
    """Minimize a unimodal scalar function f on [lo, hi]. Returns (x*, f(x*), n_fev)."""
    a, b = lo, hi
    c = b - _PHI * (b - a)
    d = a + _PHI * (b - a)
    fc, fd = f(c), f(d)
    n_fev = 2
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _PHI * (b - a)
            fd = f(d)
        n_fev += 1
    x_star = (a + b) / 2.0
    return x_star, f(x_star), n_fev + 1


def coordinate_descent_golden(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta0: NDArray[np.float64],
    bounds: list[tuple[float, float]],
    *,
    weights: NDArray[np.float64] | None = None,
    n_sweeps: int = 15,
    tol: float = 1e-8,
) -> FitResult:
    theta = np.array(theta0, dtype=float)
    path = [theta.copy()]
    n_fev = 0
    prev_cost = fwd.cost(theta, obs, weights)
    n_fev += 1
    converged = False
    for sweep in range(n_sweeps):
        for j in range(len(theta)):
            lo, hi = bounds[j]

            def f1d(v, j=j):
                t = theta.copy()
                t[j] = v
                return fwd.cost(t, obs, weights)

            x_star, f_star, nf = golden_section_1d(f1d, lo, hi, tol=1e-6)
            theta[j] = x_star
            n_fev += nf
            path.append(theta.copy())
        cost = fwd.cost(theta, obs, weights)
        n_fev += 1
        if abs(prev_cost - cost) < tol * max(1.0, prev_cost):
            converged = True
            prev_cost = cost
            break
        prev_cost = cost
    return FitResult(
        theta_hat=theta,
        cost=prev_cost,
        n_iter=len(path) - 1,
        n_fev=n_fev,
        converged=converged,
        path=np.array(path),
        message="coordinate descent with golden-section line search",
    )

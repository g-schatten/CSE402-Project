"""Nelder-Mead simplex method (derivative-free baseline, PLAN.md 2.6 item 3).
The simplex vertices at every iteration are recorded so the Fitting Studio
page can animate the "crawling simplex" alongside Gauss-Newton's straight
jumps on the same landscape.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import FitResult, ForwardModel


def nelder_mead(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta0: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None = None,
    step: float = 0.1,
    max_iter: int = 400,
    tol: float = 1e-10,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> FitResult:
    n = len(theta0)
    theta0 = np.array(theta0, dtype=float)

    def cost(t):
        return fwd.cost(t, obs, weights)

    simplex = [theta0.copy()]
    for i in range(n):
        v = theta0.copy()
        v[i] += step * max(abs(theta0[i]), 1.0)
        simplex.append(v)
    simplex = np.array(simplex)
    f_vals = np.array([cost(v) for v in simplex])
    n_fev = n + 1
    path = [simplex[np.argmin(f_vals)].copy()]

    for it in range(max_iter):
        order = np.argsort(f_vals)
        simplex, f_vals = simplex[order], f_vals[order]
        path.append(simplex[0].copy())

        if np.std(f_vals) < tol * max(1.0, abs(f_vals[0])):
            break

        centroid = simplex[:-1].mean(axis=0)
        worst, f_worst = simplex[-1], f_vals[-1]

        # reflection
        x_r = centroid + alpha * (centroid - worst)
        f_r = cost(x_r)
        n_fev += 1
        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1], f_vals[-1] = x_r, f_r
            continue

        if f_r < f_vals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = cost(x_e)
            n_fev += 1
            if f_e < f_r:
                simplex[-1], f_vals[-1] = x_e, f_e
            else:
                simplex[-1], f_vals[-1] = x_r, f_r
            continue

        # contraction
        x_c = centroid + rho * (worst - centroid)
        f_c = cost(x_c)
        n_fev += 1
        if f_c < f_worst:
            simplex[-1], f_vals[-1] = x_c, f_c
            continue

        # shrink
        best = simplex[0]
        for i in range(1, len(simplex)):
            simplex[i] = best + sigma * (simplex[i] - best)
            f_vals[i] = cost(simplex[i])
            n_fev += 1

    order = np.argsort(f_vals)
    theta_hat = simplex[order[0]]
    return FitResult(
        theta_hat=theta_hat,
        cost=float(f_vals[order[0]]),
        n_iter=it + 1,
        n_fev=n_fev,
        converged=np.std(f_vals) < tol * max(1.0, abs(f_vals[order[0]])),
        path=np.array(path),
        message="Nelder-Mead simplex",
    )

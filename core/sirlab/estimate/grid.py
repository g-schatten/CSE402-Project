"""Grid search over (theta_1, theta_2) -- also the primary source of the cost
landscape J(beta, gamma) used throughout the UI and by the profile-likelihood
/ UQ code (PLAN.md 2.6, item 1). Only supports 2-parameter models by design
(it is a diagnostic/visualization tool, not the production estimator).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import FitResult, ForwardModel


def grid_search(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta1_range: tuple[float, float],
    theta2_range: tuple[float, float],
    *,
    n1: int = 60,
    n2: int = 60,
    weights: NDArray[np.float64] | None = None,
) -> tuple[FitResult, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Returns (FitResult at the grid minimum, theta1_grid, theta2_grid, cost_surface)
    with cost_surface shape (n2, n1) so it can be plotted directly with
    imshow/contour (rows = theta2, cols = theta1)."""
    t1 = np.linspace(*theta1_range, n1)
    t2 = np.linspace(*theta2_range, n2)
    surface = np.full((n2, n1), np.nan)
    n_fev = 0
    best_cost = np.inf
    best_theta = None
    for i2, v2 in enumerate(t2):
        for i1, v1 in enumerate(t1):
            theta = np.array([v1, v2])
            c = fwd.cost(theta, obs, weights)
            surface[i2, i1] = c
            n_fev += 1
            if c < best_cost:
                best_cost = c
                best_theta = theta
    result = FitResult(
        theta_hat=best_theta,
        cost=best_cost,
        n_iter=n1 * n2,
        n_fev=n_fev,
        converged=True,
        path=np.array([best_theta]),
        message="grid search minimum (resolution-limited)",
    )
    return result, t1, t2, surface

"""Residual bootstrap (PLAN.md section 2.7, route 2): resample residuals
B times, add them back onto the fitted curve, refit, collect the cloud of
theta_hat*. Captures the estimator's actual sampling distribution shape
(not just a Gaussian approximation), at the cost of B refits.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt


def residual_bootstrap(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta_hat: NDArray[np.float64],
    *,
    n_boot: int = 1000,
    weights: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    rng = rng or np.random.default_rng()
    fitted = fwd.predict(theta_hat)
    residuals = obs - fitted
    n = len(residuals)
    thetas = np.empty((n_boot, len(theta_hat)))
    for b in range(n_boot):
        resampled = residuals[rng.integers(0, n, size=n)]
        obs_star = fitted + resampled
        fit = levenberg_marquardt(fwd, obs_star, theta_hat, weights=weights, max_iter=40)
        thetas[b] = fit.theta_hat
    return thetas

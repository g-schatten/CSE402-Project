"""Monte Carlo over fresh noise realizations (PLAN.md section 2.7, route 3).
Unlike the bootstrap (which resamples the residuals of ONE fit), this
generates M entirely new synthetic datasets from the known ground truth and
noise model, refitting each. Because it knows the ground truth, it is the
only UQ route in this project that measures true *bias*, not just sampling
spread -- this is what experiments E07/E08/E13 use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.observe import NoiseModel


@dataclass
class MonteCarloResult:
    thetas: NDArray[np.float64]  # (n_trials, n_param)
    converged: NDArray[np.bool_]
    theta_true: NDArray[np.float64]

    @property
    def bias(self) -> NDArray[np.float64]:
        ok = self.thetas[self.converged]
        return np.mean(ok, axis=0) - self.theta_true if len(ok) else np.full_like(self.theta_true, np.nan)

    @property
    def variance(self) -> NDArray[np.float64]:
        ok = self.thetas[self.converged]
        return np.var(ok, axis=0) if len(ok) else np.full_like(self.theta_true, np.nan)

    @property
    def rmse(self) -> NDArray[np.float64]:
        ok = self.thetas[self.converged]
        return np.sqrt(np.mean((ok - self.theta_true) ** 2, axis=0)) if len(ok) else np.full_like(self.theta_true, np.nan)


def monte_carlo_recovery(
    fwd: ForwardModel,
    theta_true: NDArray[np.float64],
    noise_model: NoiseModel,
    theta0: NDArray[np.float64],
    *,
    n_trials: int = 500,
    weights_fn=None,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    rng = rng or np.random.default_rng()
    clean = fwd.predict(theta_true)
    thetas = np.empty((n_trials, len(theta_true)))
    converged = np.zeros(n_trials, dtype=bool)
    for m in range(n_trials):
        obs = noise_model.sample(clean, rng)
        weights = weights_fn(obs) if weights_fn is not None else None
        fit = levenberg_marquardt(fwd, obs, theta0, weights=weights, max_iter=60)
        thetas[m] = fit.theta_hat
        converged[m] = fit.converged and np.all(np.isfinite(fit.theta_hat))
    return MonteCarloResult(thetas=thetas, converged=converged, theta_true=theta_true)

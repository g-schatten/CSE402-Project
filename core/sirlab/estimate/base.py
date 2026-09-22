"""Shared types for the estimation module (PLAN.md section 2.6, section 4.5).

Every optimizer below solves the same problem: given a model, an observation
operator, a noisy dataset, and a solver+step-size to run the forward model,
recover theta_hat minimizing the (possibly weighted) least-squares cost

    J(theta) = sum_k w_k * (obs_k - h(y(t_k; theta)))^2                (2.7)

`path` (every iterate) is mandatory on FitResult -- the web dashboard's
Fitting Studio page animates the optimizer descending the cost landscape.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class FitResult:
    theta_hat: NDArray[np.float64]
    cost: float
    n_iter: int
    n_fev: int
    converged: bool
    path: NDArray[np.float64]  # shape (n_iter+1, n_param), every iterate incl. the start
    jacobian: NDArray[np.float64] | None = None
    cov: NDArray[np.float64] | None = None
    message: str = ""


class ForwardModel:
    """Wraps a model + solver configuration + observation operator into a
    single callable theta -> predicted observations, so every optimizer can
    be written against one interface regardless of what's underneath.
    """

    def __init__(self, model, y0, t_obs, obs_fn, *, solver="rk4", h=None, rtol=1e-9, atol=1e-11, n_total=None):
        self.model = model
        self.y0 = np.asarray(y0, dtype=float)
        self.t_obs = np.asarray(t_obs, dtype=float)
        self.obs_fn = obs_fn  # (model, result, theta, t_obs, **kwargs) -> NDArray
        self.solver = solver
        self.h = h
        self.rtol = rtol
        self.atol = atol
        self.n_total = n_total

    def predict(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        from sirlab.solvers import integrate

        t_span = (0.0, float(self.t_obs[-1]))
        result = integrate(
            self.model, self.y0, theta, t_span, solver=self.solver, h=self.h, rtol=self.rtol, atol=self.atol
        )
        return self.obs_fn(result, self.t_obs)

    def residuals(self, theta: NDArray[np.float64], obs: NDArray[np.float64], weights: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        pred = self.predict(theta)
        r = pred - obs
        if weights is not None:
            r = r * np.sqrt(weights)
        return r

    def cost(self, theta: NDArray[np.float64], obs: NDArray[np.float64], weights: NDArray[np.float64] | None = None) -> float:
        r = self.residuals(theta, obs, weights)
        return float(0.5 * np.sum(r**2))

    def jacobian(self, theta: NDArray[np.float64], weights: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """d(residual)/d(theta) via the forward sensitivity equations
        (PLAN.md 2.4), NOT finite differences."""
        from sirlab.sensitivity import integrate_with_sensitivities, unpack

        t_span = (0.0, float(self.t_obs[-1]))
        aug_result = integrate_with_sensitivities(
            self.model, self.y0, theta, t_span, solver=self.solver, h=self.h, rtol=self.rtol, atol=self.atol
        )
        _, sens = unpack(aug_result, self.model.n_state, self.model.n_param)
        # interpolate sensitivities onto t_obs the same way the observation does:
        # build a lightweight SolveResult-like wrapper reusing dense-output Hermite interp
        from sirlab.solvers.base import SolveResult

        n_state, n_param = self.model.n_state, self.model.n_param
        jac_obs = np.empty((len(self.t_obs), n_param))
        for j in range(n_param):
            fake = SolveResult(t=aug_result.t, y=sens[:, :, j], n_fev=0, n_steps=0)
            sens_j_at_obs = fake.at(self.t_obs)  # (n_t_obs, n_state)
            jac_obs[:, j] = self._project_state_jacobian_to_observation(sens_j_at_obs)
        if weights is not None:
            jac_obs = jac_obs * np.sqrt(weights)[:, None]
        return jac_obs

    def _project_state_jacobian_to_observation(self, sens_at_obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Default: prevalence observation on state index 1 (I). Overridden
        via subclassing or monkey-patched `obs_state_index` attribute for
        other observation operators."""
        idx = getattr(self, "obs_state_index", 1)
        return sens_at_obs[:, idx]

"""Observation operators and noise models (PLAN.md section 2.5).

Two observation types:
  - prevalence: obs_k = I(t_k)
  - incidence:  obs_k = integral_{t_{k-1}}^{t_k} beta S I / N dt, times a
    reporting fraction rho in (0, 1], computed by composite Simpson on the
    solver's own grid (decoupled from the observation times via dense
    output, so h and the sampling interval Delta-t are genuinely independent
    experimental factors, as RQ3 requires).

Four noise models: additive Gaussian, proportional Gaussian, Poisson, and
negative binomial (overdispersed Poisson). Under Poisson noise the noise
level is NOT an independent knob -- it is set by the epidemic size, which
is documented as a deliberate finding in PLAN.md 2.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sirlab.quadrature.simpson import simpson_nonuniform
from sirlab.solvers.base import SolveResult


def prevalence(result: SolveResult, i_index: int, t_obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """I(t_obs) via dense-output interpolation of the solver trajectory."""
    y_at = result.at(t_obs)
    return y_at[:, i_index]


def incidence(
    model,
    result: SolveResult,
    theta: NDArray[np.float64],
    t_obs: NDArray[np.float64],
    *,
    s_index: int = 0,
    i_index: int = 1,
    n_total: float,
    reporting_fraction: float = 1.0,
) -> NDArray[np.float64]:
    """Integral of beta*S*I/N over each [t_obs[k-1], t_obs[k]] window, on the
    solver's own grid (PLAN.md 2.5), scaled by the reporting fraction. The
    first observation window is [t_obs[0]-dt, t_obs[0]] using the same
    spacing as the first interior interval if t_obs[0] > result.t[0].
    """
    beta = theta[0]
    t_grid = result.t
    y_grid = result.y
    s_all = y_grid[:, s_index]
    i_all = y_grid[:, i_index]
    flux_grid = beta * s_all * i_all / n_total  # instantaneous incidence rate on the solver grid

    out = np.empty(len(t_obs))
    prev_t = t_obs[0] - (t_obs[1] - t_obs[0]) if len(t_obs) > 1 else t_obs[0]
    for k, tk in enumerate(t_obs):
        t_lo = prev_t if k == 0 else t_obs[k - 1]
        t_hi = tk
        mask = (t_grid >= t_lo) & (t_grid <= t_hi)
        sub_t = np.concatenate(([t_lo], t_grid[mask], [t_hi]))
        sub_t = np.unique(sub_t)
        sub_flux = np.interp(sub_t, t_grid, flux_grid)
        out[k] = simpson_nonuniform(sub_flux, sub_t) if len(sub_t) >= 3 else np.trapezoid(sub_flux, sub_t)
        prev_t = tk
    return reporting_fraction * out


@dataclass
class NoiseModel:
    kind: str  # "additive_gaussian" | "proportional_gaussian" | "poisson" | "negbinom"
    sigma: float = 0.0  # for gaussian models: absolute (additive) or relative (proportional) std
    dispersion: float = 10.0  # for negbinom: higher = closer to Poisson

    def sample(self, clean: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        clean = np.asarray(clean, dtype=float)
        if self.kind == "additive_gaussian":
            return clean + rng.normal(scale=self.sigma, size=clean.shape)
        if self.kind == "proportional_gaussian":
            return clean * (1.0 + rng.normal(scale=self.sigma, size=clean.shape))
        if self.kind == "poisson":
            lam = np.clip(clean, 0.0, None)
            return rng.poisson(lam=lam).astype(float)
        if self.kind == "negbinom":
            mean = np.clip(clean, 1e-9, None)
            r = self.dispersion
            p = r / (r + mean)
            return rng.negative_binomial(r, p).astype(float)
        raise ValueError(f"unknown noise kind {self.kind!r}")

    def weight(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Weights for the weighted least-squares cost (2.7): 1 for
        homoscedastic (additive Gaussian), 1/max(obs,1) for count-like data
        (Poisson-ish approximate variance stabilization)."""
        if self.kind == "additive_gaussian":
            return np.ones_like(obs)
        return 1.0 / np.clip(obs, 1.0, None)

"""E05: Cost landscape (PLAN.md section 5).

J(beta, gamma) surfaces at several noise levels, plus the valley axis
(principal axis of curvature at the minimum via eigendecomposition of the
Hessian approximation J^T J, by our power method with Hotelling deflation)
and kappa(J^T J) -- the concrete identifiability picture that E06's
optimizers are then raced across.
"""
from __future__ import annotations

import numpy as np

from experiments.common import build_prevalence_forward_model, quick_or_full
from sirlab.linalg.eigen import symmetric_eigen
from sirlab.linalg.lu import cond_estimate
from sirlab.models import SIR
from sirlab.observe import NoiseModel


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    theta_true = np.array([config["theta_true"]["beta"], config["theta_true"]["gamma"]])
    y0 = np.array([n_total - i0, i0, 0.0])
    model = SIR(n_total)
    n_obs = quick_or_full(config, "n_obs", profile=profile)
    t_obs = np.linspace(1.0, config["t_obs_final"], n_obs)
    fwd = build_prevalence_forward_model(model, y0, t_obs, solver=config["solver"], h=config["h"])

    clean = fwd.predict(theta_true)
    n1 = quick_or_full(config, "grid_n", profile=profile)
    n2 = n1
    b_grid = np.linspace(*config["beta_range"], n1)
    g_grid = np.linspace(*config["gamma_range"], n2)

    rng = np.random.default_rng(config.get("seed", 0))
    surfaces = {}
    for sigma_frac in config["noise_levels"]:
        sigma = sigma_frac * np.max(clean)
        noise = NoiseModel(kind="additive_gaussian", sigma=sigma) if sigma > 0 else None
        obs = noise.sample(clean, rng) if noise else clean
        surf = np.empty((n2, n1))
        for i2, g in enumerate(g_grid):
            for i1, b in enumerate(b_grid):
                surf[i2, i1] = fwd.cost(np.array([b, g]), obs)
        jac = fwd.jacobian(theta_true)
        jtj = jac.T @ jac
        eigvals, eigvecs = symmetric_eigen(jtj)
        surfaces[str(sigma_frac)] = {
            "surface": surf.tolist(),
            "valley_axis": eigvecs[:, np.argmin(eigvals)].tolist(),
            "curvature_eigvals": eigvals.tolist(),
            "cond_jtj": float(cond_estimate(jtj + 1e-12 * np.eye(2))),
            "cond_j": float(np.sqrt(cond_estimate(jtj + 1e-12 * np.eye(2)))),
        }

    return {
        "experiment": "E05",
        "beta_grid": b_grid.tolist(),
        "gamma_grid": g_grid.tolist(),
        "theta_true": theta_true.tolist(),
        "surfaces": surfaces,
    }

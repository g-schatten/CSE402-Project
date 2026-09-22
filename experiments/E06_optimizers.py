"""E06: Optimizer shoot-out (PLAN.md section 5).

Five optimizers x many random starts x two parameterizations (linear vs
log-space), on one fixed noisy dataset -- iterations, f-evals, success rate,
and the full convergence path (for the Fitting Studio's side-by-side
animation of Nelder-Mead's crawling simplex vs Gauss-Newton's long jumps).
"""
from __future__ import annotations

import numpy as np

from experiments.common import build_prevalence_forward_model, quick_or_full
from sirlab.estimate import coordinate_descent_golden, gauss_newton, levenberg_marquardt, nelder_mead
from sirlab.models import SIR
from sirlab.observe import NoiseModel

_SUCCESS_ATOL = 0.02  # relative tolerance on theta to call a run "successful"


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    theta_true = np.array([config["theta_true"]["beta"], config["theta_true"]["gamma"]])
    y0 = np.array([n_total - i0, i0, 0.0])
    model = SIR(n_total)
    t_obs = np.linspace(1.0, config["t_obs_final"], config["n_obs"])
    fwd = build_prevalence_forward_model(model, y0, t_obs, solver=config["solver"], h=config["h"])

    rng = np.random.default_rng(config.get("seed", 0))
    clean = fwd.predict(theta_true)
    sigma = config["noise_sigma_frac"] * np.max(clean)
    noise = NoiseModel(kind="additive_gaussian", sigma=sigma)
    obs = noise.sample(clean, rng)

    n_starts = quick_or_full(config, "n_starts", profile=profile)
    bounds = [tuple(config["bounds"]["beta"]), tuple(config["bounds"]["gamma"])]
    starts = np.column_stack(
        [rng.uniform(*bounds[0], size=n_starts), rng.uniform(*bounds[1], size=n_starts)]
    )

    optimizers = {
        "grid_refined_gauss_newton": lambda t0: gauss_newton(fwd, obs, t0, log_space=False),
        "gauss_newton_log": lambda t0: gauss_newton(fwd, obs, t0, log_space=True),
        "levenberg_marquardt": lambda t0: levenberg_marquardt(fwd, obs, t0, log_space=True),
        "nelder_mead": lambda t0: nelder_mead(fwd, obs, t0, max_iter=300),
        "coordinate_descent_golden": lambda t0: coordinate_descent_golden(fwd, obs, t0, bounds=bounds, n_sweeps=10),
    }

    results = {}
    example_paths = {}
    for name, opt in optimizers.items():
        n_iters, n_fevs, successes = [], [], []
        for k, t0 in enumerate(starts):
            try:
                fit = opt(t0)
            except Exception:
                successes.append(False)
                continue
            n_iters.append(fit.n_iter)
            n_fevs.append(fit.n_fev)
            ok = fit.converged and np.allclose(fit.theta_hat, theta_true, rtol=_SUCCESS_ATOL, atol=1e-3)
            successes.append(bool(ok))
            if k == 0:
                example_paths[name] = fit.path.tolist()
        results[name] = {
            "mean_iterations": float(np.mean(n_iters)) if n_iters else None,
            "mean_fevals": float(np.mean(n_fevs)) if n_fevs else None,
            "success_rate": float(np.mean(successes)),
            "n_starts": n_starts,
        }

    return {
        "experiment": "E06",
        "theta_true": theta_true.tolist(),
        "starts": starts.tolist(),
        "results": results,
        "example_paths": example_paths,
    }

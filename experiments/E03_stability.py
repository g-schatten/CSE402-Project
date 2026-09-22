"""E03: Stability frontier (PLAN.md section 5).

For each (solver, beta, gamma) on a grid, find the largest h in the sweep
that stays "stable" (no negativity events, no divergence, and bounded
relative to the reference trajectory) -- the empirical stability boundary.
Also records max(h * |eigenvalue|) of the local Jacobian along the
trajectory, linking the empirical boundary to the textbook linear-stability
picture.
"""
from __future__ import annotations

import numpy as np

from experiments.common import quick_or_full
from sirlab.diagnostics import stability_metric
from sirlab.models import SIR
from sirlab.solvers import integrate


def _is_stable(res, n_total: float) -> bool:
    if res.diverged or res.negativity_events > 0:
        return False
    if not np.all(np.isfinite(res.y)):
        return False
    if np.any(res.y < -1e-6 * n_total) or np.any(res.y > 1.5 * n_total):
        return False
    return True


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    t_final = float(config["t_final"])
    n_grid = quick_or_full(config, "grid_n", profile=profile)
    betas = np.linspace(*config["beta_range"], n_grid)
    gammas = np.linspace(*config["gamma_range"], n_grid)
    hs = np.array(quick_or_full(config, "step_sizes", profile=profile))
    model = SIR(n_total)
    y0 = np.array([n_total - i0, i0, 0.0])

    out = {}
    for solver in config["solvers"]:
        first_failure_h = np.full((n_grid, n_grid), np.nan)
        max_h_lambda = np.full((n_grid, n_grid), np.nan)
        for ib, beta in enumerate(betas):
            for ig, gamma in enumerate(gammas):
                theta = np.array([beta, gamma])
                stable_hs = []
                for h in hs:
                    res = integrate(model, y0, theta, (0.0, t_final), solver=solver, h=h)
                    if _is_stable(res, n_total):
                        stable_hs.append(h)
                        jac = model.jacobian_y(0.0, res.y[len(res.y) // 3], theta)
                        max_h_lambda[ib, ig] = max(max_h_lambda[ib, ig] if not np.isnan(max_h_lambda[ib, ig]) else 0, stability_metric(h, jac))
                    else:
                        break
                first_failure_h[ib, ig] = stable_hs[-1] if stable_hs else 0.0
        out[solver] = {
            "first_failure_h": first_failure_h.tolist(),
            "max_stable_h_times_lambda": max_h_lambda.tolist(),
        }

    return {
        "experiment": "E03",
        "n_total": n_total,
        "beta_grid": betas.tolist(),
        "gamma_grid": gammas.tolist(),
        "step_sizes_tested": hs.tolist(),
        "solvers": out,
    }

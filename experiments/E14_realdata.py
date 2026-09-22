"""E14: Real data -- the Eyam 1666 plague outbreak (PLAN.md section 5,
listed as an optional/time-permitting extension: "a demonstration, not a
proof"). Fits (beta, gamma) to the historical susceptible-count series
using the same estimation machinery as the synthetic experiments, with
both RK4 and Euler as the internal solver, so any solver-choice sensitivity
shows up on real data too. See data/raw/provenance.md for the data-source
caveat that should be resolved before this experiment's numbers are cited
in the final report.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.models import SIR
from sirlab.observe import prevalence
from sirlab.uq import confidence_ellipse, correlation

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _load_csv(filename: str) -> tuple[np.ndarray, np.ndarray]:
    days, s_vals = [], []
    with open(DATA_DIR / filename) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            days.append(float(row["day"]))
            s_vals.append(float(row["susceptible"]))
    return np.array(days), np.array(s_vals)


def _obs_fn_s(result, t_obs):
    return prevalence(result, i_index=0, t_obs=t_obs)


def run(config: dict) -> dict:
    days, s_obs = _load_csv(config["data_file"])
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    y0 = np.array([n_total - i0, i0, 0.0])
    theta0 = np.array([config["theta0"]["beta"], config["theta0"]["gamma"]])

    t_obs = days.copy()
    t_obs[0] = max(t_obs[0], 1e-6)  # avoid a zero-length integration span

    results = {}
    for solver in config["solvers"]:
        model = SIR(n_total)
        fwd = ForwardModel(model, y0, t_obs, _obs_fn_s, solver=solver, h=config["h"])
        fwd.obs_state_index = 0
        fit = levenberg_marquardt(fwd, s_obs, theta0, max_iter=100)
        pred = fwd.predict(fit.theta_hat)
        residuals = (s_obs - pred).tolist()
        r0 = float(fit.theta_hat[0] / fit.theta_hat[1])

        entry = {
            "theta_hat": fit.theta_hat.tolist(),
            "r0": r0,
            "converged": bool(fit.converged),
            "cost": fit.cost,
            "predicted_s": pred.tolist(),
            "residuals": residuals,
        }
        if fit.cov is not None:
            ellipse, chi2_val = confidence_ellipse(fit.cov, fit.theta_hat)
            entry["cov"] = fit.cov.tolist()
            entry["ellipse"] = ellipse.tolist()
            entry["rho"] = correlation(fit.cov)
        results[solver] = entry

    return {
        "experiment": "E14",
        "data_source": "data/raw/eyam_1666.csv (see data/raw/provenance.md)",
        "n_total": n_total,
        "i0": i0,
        "days": days.tolist(),
        "s_observed": s_obs.tolist(),
        "results": results,
    }

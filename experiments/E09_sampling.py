"""E09: Sampling & window (PLAN.md section 5).

Sweeps observation spacing dt and the fraction of the epidemic actually
observed ("window_frac" of the way from t=0 to ~6x the peak time, i.e. does
the dataset end before, at, or well after the peak). RMSE vs #observations,
and -- the qualitative headline of this experiment -- an identifiability
collapse when the window is truncated before the peak (visible as RMSE
blowing up and, in E11's profile-likelihood view, a flat profile).
"""
from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.common import quick_or_full
from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.models import SIR
from sirlab.observe import prevalence
from sirlab.reference import SIRReference


def _obs_fn(result, t_obs):
    return prevalence(result, i_index=1, t_obs=t_obs)


def _one_trial(args):
    beta, gamma, n_total, i0, dt, window_frac, sigma_frac, solver, h, seed = args
    s0 = n_total - i0
    ref = SIRReference(beta=beta, gamma=gamma, n_total=n_total, s0=s0, i0=i0)
    t_peak = ref.peak_time()
    t_final = window_frac * 6.0 * t_peak
    t_obs = np.arange(dt, max(t_final, dt * 2), dt)
    clean_i = ref.trajectory(t_obs)[:, 1]
    rng = np.random.default_rng(seed)
    sigma = sigma_frac * np.max(clean_i)
    obs = clean_i + rng.normal(scale=sigma, size=clean_i.shape) if sigma > 0 else clean_i.copy()

    model = SIR(n_total)
    y0 = np.array([s0, i0, 0.0])
    fwd = ForwardModel(model, y0, t_obs, _obs_fn, solver=solver, h=h)
    fwd.obs_state_index = 1
    theta0 = np.array([beta, gamma]) * rng.uniform(0.6, 1.6, size=2)
    try:
        fit = levenberg_marquardt(fwd, obs, theta0, max_iter=60)
        ok = fit.converged and np.all(fit.theta_hat > 0) and np.all(fit.theta_hat < 5)
        return fit.theta_hat, bool(ok), len(t_obs)
    except Exception:
        return np.array([np.nan, np.nan]), False, len(t_obs)


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total, i0 = float(config["n_total"]), float(config["i0"])
    beta, gamma = config["theta_true"]["beta"], config["theta_true"]["gamma"]
    dts = quick_or_full(config, "dts", profile=profile)
    window_fracs = quick_or_full(config, "window_fracs", profile=profile)
    n_rep = quick_or_full(config, "n_replicates", profile=profile)
    seed = int(config.get("seed", 0))
    n_jobs = int(config.get("n_jobs", 4))

    cells = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for dt in dts:
            for wf in window_fracs:
                jobs = []
                for r in range(n_rep):
                    key = f"{dt}|{wf}|{r}"
                    s = (seed * 10_000_000 + zlib.crc32(key.encode())) % (2**31 - 1)
                    jobs.append((beta, gamma, n_total, i0, dt, wf, config["sigma_frac"], config["solver"], config["h"], s))
                thetas, oks, n_obs_list = [], [], []
                for theta_hat, ok, n_obs in ex.map(_one_trial, jobs, chunksize=2):
                    thetas.append(theta_hat)
                    oks.append(ok)
                    n_obs_list.append(n_obs)
                thetas = np.array(thetas)
                oks = np.array(oks)
                good = thetas[oks]
                theta_true = np.array([beta, gamma])
                rmse = np.sqrt(np.mean((good - theta_true) ** 2, axis=0)) if len(good) else np.array([np.nan, np.nan])
                cells.append(
                    {
                        "dt": dt,
                        "window_frac": wf,
                        "n_obs": n_obs_list[0],
                        "n_converged": int(oks.sum()),
                        "n_total": n_rep,
                        "rmse": rmse.tolist(),
                    }
                )

    return {"experiment": "E09", "theta_true": [beta, gamma], "cells": cells}

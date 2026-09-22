"""E08: Noise degradation (PLAN.md section 5).

RMSE(beta_hat), RMSE(gamma_hat) vs sigma, with bootstrap confidence
intervals on the RMSE itself (resampling the M replicate estimates, not the
original residuals -- a second, independent use of the bootstrap idea from
uq/bootstrap.py at one level up: quantifying uncertainty in the noise-vs-
accuracy curve, not in a single fit).
"""
from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.common import quick_or_full
from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.models import SIR
from sirlab.observe import NoiseModel, prevalence
from sirlab.reference import SIRReference


def _obs_fn(result, t_obs):
    return prevalence(result, i_index=1, t_obs=t_obs)


def _one_trial(args):
    beta, gamma, n_total, i0, t_obs_final, dt, sigma_frac, solver, h, seed = args
    s0 = n_total - i0
    ref = SIRReference(beta=beta, gamma=gamma, n_total=n_total, s0=s0, i0=i0)
    t_obs = np.arange(dt, t_obs_final + 1e-9, dt)
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
        ok = fit.converged and np.all(fit.theta_hat > 0)
        return fit.theta_hat, bool(ok)
    except Exception:
        return np.array([np.nan, np.nan]), False


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total, i0 = float(config["n_total"]), float(config["i0"])
    beta, gamma = config["theta_true"]["beta"], config["theta_true"]["gamma"]
    sigma_fracs = quick_or_full(config, "sigma_fracs", profile=profile)
    n_rep = quick_or_full(config, "n_replicates", profile=profile)
    seed = int(config.get("seed", 0))
    n_jobs = int(config.get("n_jobs", 4))

    curve = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for sigma_frac in sigma_fracs:
            jobs = []
            for r in range(n_rep):
                key = f"{sigma_frac}|{r}"
                s = (seed * 10_000_000 + zlib.crc32(key.encode())) % (2**31 - 1)
                jobs.append((beta, gamma, n_total, i0, config["t_obs_final"], config["dt"], sigma_frac, config["solver"], config["h"], s))
            thetas, oks = [], []
            for theta_hat, ok in ex.map(_one_trial, jobs, chunksize=2):
                thetas.append(theta_hat)
                oks.append(ok)
            thetas = np.array(thetas)
            oks = np.array(oks)
            good = thetas[oks]
            theta_true = np.array([beta, gamma])
            rmse = np.sqrt(np.mean((good - theta_true) ** 2, axis=0)) if len(good) else np.array([np.nan, np.nan])

            # bootstrap CI on the RMSE itself, resampling the M trial estimates
            boot_rmses = []
            rng = np.random.default_rng(seed + 1)
            for _ in range(300):
                idx = rng.integers(0, len(good), size=len(good)) if len(good) else []
                if len(idx) == 0:
                    continue
                boot_rmses.append(np.sqrt(np.mean((good[idx] - theta_true) ** 2, axis=0)))
            boot_rmses = np.array(boot_rmses) if boot_rmses else np.full((1, 2), np.nan)
            ci_lo = np.percentile(boot_rmses, 2.5, axis=0)
            ci_hi = np.percentile(boot_rmses, 97.5, axis=0)

            curve.append(
                {
                    "sigma_frac": sigma_frac,
                    "n_converged": int(oks.sum()),
                    "n_total": n_rep,
                    "rmse": rmse.tolist(),
                    "rmse_ci_lo": ci_lo.tolist(),
                    "rmse_ci_hi": ci_hi.tolist(),
                }
            )

    return {"experiment": "E08", "theta_true": [beta, gamma], "curve": curve}

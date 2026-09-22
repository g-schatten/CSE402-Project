"""E13: Crossover sigma* -- the project's headline finding (PLAN.md section
0 and section 5).

For each solver step size h, we isolate two error sources that both push
the recovered (beta, gamma) away from the truth:

  1. Solver bias(h): with NOISE-FREE synthetic data, any deviation of
     theta_hat from theta_true is caused purely by the fitting solver's
     truncation error (the data-generating process is the exact gold
     standard, so there is nothing else it could be). This is deterministic
     given h -- one fit suffices.
  2. Noise-induced std(sigma, h): with noisy data at level sigma, the
     spread of theta_hat across many replicates at fixed h.

The crossover sigma*(h) is where std(sigma, h) first equals bias(h) in
magnitude: below it, solver bias dominates the total error and solver
choice matters; above it, sampling noise dominates and a cheaper/less
accurate solver at the same h is statistically indistinguishable from a
better one. We locate sigma*(h) by linear interpolation of std(sigma) vs
bias(h) on a log-sigma grid.
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


def _fit_one(beta, gamma, n_total, i0, dt, t_obs_final, solver, h, sigma_frac, seed):
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
    theta0 = np.array([beta, gamma]) * rng.uniform(0.7, 1.4, size=2)
    try:
        fit = levenberg_marquardt(fwd, obs, theta0, max_iter=60)
        ok = fit.converged and np.all(fit.theta_hat > 0) and np.all(fit.theta_hat < 5)
        return fit.theta_hat, bool(ok)
    except Exception:
        return np.array([np.nan, np.nan]), False


def _replicate_worker(args):
    return _fit_one(*args)


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total, i0 = float(config["n_total"]), float(config["i0"])
    beta, gamma = config["theta_true"]["beta"], config["theta_true"]["gamma"]
    theta_true = np.array([beta, gamma])
    solver = config["solver"]
    hs = quick_or_full(config, "h_values", profile=profile)
    sigma_fracs = quick_or_full(config, "sigma_fracs", profile=profile)
    n_rep = quick_or_full(config, "n_replicates", profile=profile)
    seed = int(config.get("seed", 0))
    n_jobs = int(config.get("n_jobs", 4))
    dt, t_obs_final = config["dt"], config["t_obs_final"]

    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for h in hs:
            # 1. solver bias at sigma=0 (deterministic given h; one fit suffices,
            # but we average 3 different start points as a cheap sanity check
            # that the optimizer isn't landing in different local minima)
            bias_trials = []
            for k in range(3):
                key = f"bias|{h}|{k}"
                s = (seed * 10_000_000 + zlib.crc32(key.encode())) % (2**31 - 1)
                theta_hat, ok = _fit_one(beta, gamma, n_total, i0, dt, t_obs_final, solver, h, 0.0, s)
                if ok:
                    bias_trials.append(theta_hat)
            bias = (np.mean(bias_trials, axis=0) - theta_true) if bias_trials else np.array([np.nan, np.nan])

            # 2. noise-induced std at each sigma, fixed h
            sigma_curve = []
            for sigma_frac in sigma_fracs:
                if sigma_frac == 0.0:
                    continue  # sigma=0 case already handled by the bias trials above
                jobs = []
                for r in range(n_rep):
                    key = f"{h}|{sigma_frac}|{r}"
                    s = (seed * 10_000_000 + zlib.crc32(key.encode())) % (2**31 - 1)
                    jobs.append((beta, gamma, n_total, i0, dt, t_obs_final, solver, h, sigma_frac, s))
                thetas, oks = [], []
                for theta_hat, ok in ex.map(_replicate_worker, jobs, chunksize=2):
                    thetas.append(theta_hat)
                    oks.append(ok)
                thetas, oks = np.array(thetas), np.array(oks)
                good = thetas[oks]
                std = good.std(axis=0) if len(good) else np.array([np.nan, np.nan])
                mean_dev = (good.mean(axis=0) - theta_true) if len(good) else np.array([np.nan, np.nan])
                sigma_curve.append({"sigma_frac": sigma_frac, "std": std.tolist(), "mean_deviation": mean_dev.tolist(), "n_converged": int(oks.sum())})

            # 3. crossover sigma* per parameter: first sigma where std >= |bias|
            crossover = {}
            for j, name in enumerate(["beta", "gamma"]):
                target = abs(bias[j]) if np.isfinite(bias[j]) else np.nan
                xs = [c["sigma_frac"] for c in sigma_curve]
                ys = [c["std"][j] for c in sigma_curve]
                sigma_star = None
                for k in range(len(xs) - 1):
                    if np.isnan(ys[k]) or np.isnan(ys[k + 1]):
                        continue
                    if (ys[k] - target) * (ys[k + 1] - target) <= 0 and ys[k + 1] != ys[k]:
                        frac = (target - ys[k]) / (ys[k + 1] - ys[k])
                        sigma_star = xs[k] + frac * (xs[k + 1] - xs[k])
                        break
                crossover[name] = sigma_star

            rows.append({"h": h, "solver_bias": bias.tolist(), "sigma_curve": sigma_curve, "crossover_sigma_star": crossover})

    return {
        "experiment": "E13",
        "solver": solver,
        "theta_true": theta_true.tolist(),
        "conclusion": (
            "Below sigma*, the fitting solver's own truncation error dominates the "
            "total parameter error; above it, sampling noise dominates and further "
            "solver refinement at this h buys nothing statistically distinguishable."
        ),
        "rows": rows,
    }

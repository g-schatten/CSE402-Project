"""E07: Solver-in-the-loop recovery -- the project's centrepiece (PLAN.md
section 0 and section 5).

Synthetic observations are generated from the *exact* semi-analytic gold
standard (reference.py), sampled at spacing dt and corrupted with additive
Gaussian noise at level sigma. We then fit (beta, gamma) using Levenberg-
Marquardt whose internal forward model is deliberately run with a specific
(solver, h) -- so any bias in the recovered parameters is attributable
purely to that solver's truncation error, isolated from the data-generating
process. Repeated over many noise replicates per (solver, h, sigma, dt,
theta_true) cell gives bias, variance, and RMSE -- the numbers behind the
project's headline crossover-sigma* claim (E13 consumes this same
machinery at finer resolution).

Runs the factorial in parallel across CPU cores (PLAN.md section 5: "a
multiprocessing pool" -- ProcessPoolExecutor here).
"""
from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from experiments.common import quick_or_full
from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.models import SIR
from sirlab.observe import NoiseModel, prevalence
from sirlab.reference import SIRReference


@dataclass(frozen=True)
class Cell:
    solver: str
    h: float
    sigma_frac: float
    dt: float
    beta_true: float
    gamma_true: float


def _make_obs_fn(i_index: int = 1):
    def obs_fn(result, t_obs):
        return prevalence(result, i_index=i_index, t_obs=t_obs)

    return obs_fn


def _run_replicate(args) -> tuple:
    cell, n_total, i0, t_obs_final, replicate_seed = args
    beta, gamma = cell.beta_true, cell.gamma_true
    theta_true = np.array([beta, gamma])
    s0 = n_total - i0

    ref = SIRReference(beta=beta, gamma=gamma, n_total=n_total, s0=s0, i0=i0)
    t_obs = np.arange(cell.dt, t_obs_final + 1e-9, cell.dt)
    true_traj = ref.trajectory(t_obs)
    clean_i = true_traj[:, 1]

    rng = np.random.default_rng(replicate_seed)
    sigma = cell.sigma_frac * np.max(clean_i)
    noise = NoiseModel(kind="additive_gaussian", sigma=sigma) if sigma > 0 else None
    obs = noise.sample(clean_i, rng) if noise else clean_i.copy()

    model = SIR(n_total)
    y0 = np.array([s0, i0, 0.0])
    fwd = ForwardModel(model, y0, t_obs, _make_obs_fn(), solver=cell.solver, h=cell.h)
    fwd.obs_state_index = 1

    theta0 = theta_true * rng.uniform(0.6, 1.6, size=2)
    try:
        fit = levenberg_marquardt(fwd, obs, theta0, max_iter=60)
        theta_hat = fit.theta_hat
        converged = bool(fit.converged) and np.all(np.isfinite(theta_hat)) and np.all(theta_hat > 0)
    except Exception:
        theta_hat = np.array([np.nan, np.nan])
        converged = False

    return cell, theta_hat, converged


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    t_obs_final = float(config["t_obs_final"])
    seed = int(config.get("seed", 0))
    n_jobs = int(config.get("n_jobs", 4))

    solvers = list(config["solver_step_sizes"].keys())
    theta_grid = quick_or_full(config, "theta_grid", profile=profile)
    sigma_fracs = quick_or_full(config, "sigma_fracs", profile=profile)
    dts = quick_or_full(config, "sample_dts", profile=profile)
    n_replicates = quick_or_full(config, "n_replicates", profile=profile)

    cells: list[Cell] = []
    for solver in solvers:
        hs = config["solver_step_sizes"][solver][profile]
        for h in hs:
            for sigma_frac in sigma_fracs:
                for dt in dts:
                    for theta in theta_grid:
                        cells.append(Cell(solver, h, sigma_frac, dt, theta["beta"], theta["gamma"]))

    jobs = []
    for cell in cells:
        for r in range(n_replicates):
            # Deterministic (not Python's randomized str-hash()) seed derivation
            # so a rerun with an unchanged config reproduces every trial exactly,
            # per PLAN.md section 5's reproducibility contract.
            key = f"{cell.solver}|{cell.h}|{cell.sigma_frac}|{cell.dt}|{cell.beta_true}|{cell.gamma_true}|{r}"
            replicate_seed = (seed * 10_000_000 + zlib.crc32(key.encode())) % (2**31 - 1)
            jobs.append((cell, n_total, i0, t_obs_final, replicate_seed))

    results: dict[Cell, list] = {c: [] for c in cells}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for cell, theta_hat, converged in ex.map(_run_replicate, jobs, chunksize=4):
            results[cell].append((theta_hat.tolist(), converged))

    cells_out = []
    for cell, trials in results.items():
        thetas = np.array([t for t, ok in trials if ok])
        n_ok = len(thetas)
        theta_true = np.array([cell.beta_true, cell.gamma_true])
        if n_ok > 0:
            bias = (thetas.mean(axis=0) - theta_true).tolist()
            variance = thetas.var(axis=0).tolist()
            rmse = np.sqrt(np.mean((thetas - theta_true) ** 2, axis=0)).tolist()
        else:
            bias = variance = rmse = [None, None]
        cells_out.append(
            {
                "solver": cell.solver,
                "h": cell.h,
                "sigma_frac": cell.sigma_frac,
                "dt": cell.dt,
                "beta_true": cell.beta_true,
                "gamma_true": cell.gamma_true,
                "n_converged": n_ok,
                "n_total_trials": len(trials),
                "bias": bias,
                "variance": variance,
                "rmse": rmse,
            }
        )

    return {
        "experiment": "E07",
        "profile": profile,
        "n_cells": len(cells),
        "n_replicates": n_replicates,
        "cells": cells_out,
    }

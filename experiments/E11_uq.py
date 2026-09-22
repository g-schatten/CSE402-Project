"""E11: Uncertainty quantification (PLAN.md section 5 / section 2.7).

Runs all four UQ routes on one fitted dataset (asymptotic, bootstrap, Monte
Carlo, MCMC with Gelman-Rubin/ESS) so they can be overlaid on one figure,
plus two profile-likelihood curves: one on the full observation window
(identifiable, sharp minimum) and one on a truncated pre-peak window
(the non-identifiability signature: a flat profile).
"""
from __future__ import annotations

import numpy as np

from experiments.common import build_prevalence_forward_model
from sirlab.models import SIR
from sirlab.observe import NoiseModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.reference import SIRReference
from sirlab.uq import (
    confidence_ellipse,
    correlation,
    effective_sample_size,
    gelman_rubin,
    monte_carlo_recovery,
    profile_likelihood_1d,
    residual_bootstrap,
    run_mcmc,
)


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total, i0 = float(config["n_total"]), float(config["i0"])
    beta, gamma = config["theta_true"]["beta"], config["theta_true"]["gamma"]
    theta_true = np.array([beta, gamma])
    model = SIR(n_total)
    y0 = np.array([n_total - i0, i0, 0.0])
    t_obs = np.arange(config["dt"], config["t_obs_final"] + 1e-9, config["dt"])
    fwd = build_prevalence_forward_model(model, y0, t_obs, solver=config["solver"], h=config["h"])

    rng = np.random.default_rng(config.get("seed", 0))
    clean = fwd.predict(theta_true)
    sigma = config["sigma_frac"] * np.max(clean)
    noise = NoiseModel(kind="additive_gaussian", sigma=sigma)
    obs = noise.sample(clean, rng)

    fit = levenberg_marquardt(fwd, obs, theta_true * 1.2)

    # 1. asymptotic
    ellipse, chi2_val = confidence_ellipse(fit.cov, fit.theta_hat)
    rho = correlation(fit.cov)

    # 2. bootstrap
    n_boot = config["n_bootstrap"][profile] if isinstance(config["n_bootstrap"], dict) else config["n_bootstrap"]
    boot = residual_bootstrap(fwd, obs, fit.theta_hat, n_boot=n_boot, rng=rng)

    # 3. Monte Carlo (measures true bias, knows ground truth)
    n_mc = config["n_montecarlo"][profile] if isinstance(config["n_montecarlo"], dict) else config["n_montecarlo"]
    mc = monte_carlo_recovery(fwd, theta_true, noise, theta_true * 1.1, n_trials=n_mc, rng=rng)

    # 4. MCMC
    mc_cfg = config["mcmc"]
    n_samples = mc_cfg["n_samples"][profile]
    burn_in = mc_cfg["burn_in"][profile]
    mcmc = run_mcmc(
        fwd, obs, fit.theta_hat, n_chains=mc_cfg["n_chains"], n_samples=n_samples, burn_in=burn_in,
        prop_scale=np.array(mc_cfg["prop_scale"]), rng=rng,
    )
    rhat = gelman_rubin(mcmc.chains)
    pooled = mcmc.pooled()
    ess_beta = effective_sample_size(pooled[:, 0])
    ess_gamma = effective_sample_size(pooled[:, 1])

    # Profile likelihoods: full window vs truncated pre-peak window
    n_grid = config["profile_grid_n"][profile] if isinstance(config["profile_grid_n"], dict) else config["profile_grid_n"]
    beta_grid = np.linspace(0.5 * beta, 2.0 * beta, n_grid)
    profile_full = profile_likelihood_1d(fwd, obs, 0, beta_grid, (0.3 * gamma, 3.0 * gamma))

    ref = SIRReference(beta=beta, gamma=gamma, n_total=n_total, s0=y0[0], i0=y0[1])
    t_peak = ref.peak_time()
    t_trunc = config["truncated_window_frac"] * t_peak
    t_obs_trunc = t_obs[t_obs <= t_trunc]
    fwd_trunc = build_prevalence_forward_model(model, y0, t_obs_trunc, solver=config["solver"], h=config["h"])
    obs_trunc = obs[: len(t_obs_trunc)]
    profile_trunc = profile_likelihood_1d(fwd_trunc, obs_trunc, 0, beta_grid, (0.3 * gamma, 3.0 * gamma))

    return {
        "experiment": "E11",
        "theta_true": theta_true.tolist(),
        "point_estimate": fit.theta_hat.tolist(),
        "asymptotic": {"ellipse": ellipse.tolist(), "chi2": chi2_val, "rho": rho, "cov": fit.cov.tolist()},
        "bootstrap": {"thetas": boot.tolist(), "mean": boot.mean(0).tolist(), "std": boot.std(0).tolist()},
        "montecarlo": {"bias": mc.bias.tolist(), "variance": mc.variance.tolist(), "rmse": mc.rmse.tolist(), "n_converged": int(mc.converged.sum())},
        "mcmc": {
            "pooled_samples": pooled[:: max(1, len(pooled) // 2000)].tolist(),  # thin for the JSON artifact
            "rhat": rhat.tolist(),
            "ess_beta": ess_beta,
            "ess_gamma": ess_gamma,
            "acceptance_rate": mcmc.acceptance_rate.tolist(),
            "posterior_mean": pooled.mean(0).tolist(),
        },
        "profile_likelihood": {
            "beta_grid": beta_grid.tolist(),
            "cost_full_window": profile_full.tolist(),
            "cost_truncated_window": profile_trunc.tolist(),
            "t_peak": t_peak,
            "t_truncated": t_trunc,
        },
    }

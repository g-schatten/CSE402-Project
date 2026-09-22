"""UQ acceptance tests (PLAN.md section 4.6 / section 2.7). Trial counts are
kept small here for CI speed; production experiments (E11, E13) use the
--full profile's larger budgets.
"""
from __future__ import annotations

import numpy as np
import pytest

from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.models import SIR
from sirlab.observe import NoiseModel, prevalence
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

N = 1000.0
THETA_TRUE = np.array([0.3, 0.1])
Y0 = np.array([999.0, 1.0, 0.0])
T_OBS = np.linspace(1.0, 60.0, 25)


def _obs_fn(result, t_obs):
    return prevalence(result, i_index=1, t_obs=t_obs)


@pytest.fixture
def fwd_and_obs():
    model = SIR(N)
    fwd = ForwardModel(model, Y0, T_OBS, _obs_fn, solver="rk4", h=0.25)
    fwd.obs_state_index = 1
    rng = np.random.default_rng(7)
    clean = fwd.predict(THETA_TRUE)
    noise = NoiseModel(kind="additive_gaussian", sigma=0.05 * np.max(clean))
    obs = noise.sample(clean, rng)
    return fwd, obs, noise, rng


def test_chi2_quantile_2df_known_value():
    from sirlab.uq._chi2 import chi2_quantile_2df

    assert chi2_quantile_2df(0.95) == pytest.approx(5.991464547, abs=1e-6)


def test_confidence_ellipse_and_correlation(fwd_and_obs):
    fwd, obs, noise, rng = fwd_and_obs
    fit = levenberg_marquardt(fwd, obs, THETA_TRUE * 1.3)
    assert fit.cov is not None
    ellipse, chi2_val = confidence_ellipse(fit.cov, fit.theta_hat)
    assert ellipse.shape == (200, 2)
    assert chi2_val == pytest.approx(5.991464547, abs=1e-6)
    rho = correlation(fit.cov)
    assert -1.0 < rho < 1.0


def test_bootstrap_recovers_truth_on_average(fwd_and_obs):
    fwd, obs, noise, rng = fwd_and_obs
    fit = levenberg_marquardt(fwd, obs, THETA_TRUE * 1.3)
    boot = residual_bootstrap(fwd, obs, fit.theta_hat, n_boot=40, rng=rng)
    assert np.allclose(boot.mean(axis=0), THETA_TRUE, atol=0.02)
    assert np.all(boot.std(axis=0) > 0)


def test_monte_carlo_measures_small_bias(fwd_and_obs):
    fwd, obs, noise, rng = fwd_and_obs
    mc = monte_carlo_recovery(fwd, THETA_TRUE, noise, THETA_TRUE * 1.1, n_trials=40, rng=rng)
    assert np.all(np.abs(mc.bias) < 0.02)
    assert np.all(mc.rmse > 0)


def test_mcmc_recovers_posterior_and_mixes(fwd_and_obs):
    fwd, obs, noise, rng = fwd_and_obs
    fit = levenberg_marquardt(fwd, obs, THETA_TRUE * 1.2)
    mcmc = run_mcmc(
        fwd, obs, fit.theta_hat, n_chains=3, n_samples=200, burn_in=200, prop_scale=np.array([0.015, 0.015]), rng=rng
    )
    assert np.all(mcmc.acceptance_rate > 0.1)
    rhat = gelman_rubin(mcmc.chains)
    assert np.all(rhat < 1.15)  # loose bound for a fast test; production runs target <1.01
    posterior_mean = mcmc.pooled().mean(axis=0)
    assert np.allclose(posterior_mean, THETA_TRUE, atol=0.03)
    ess = effective_sample_size(mcmc.pooled()[:, 0])
    assert ess > 1


def test_profile_likelihood_has_minimum_near_truth(fwd_and_obs):
    fwd, obs, noise, rng = fwd_and_obs
    fixed_vals = np.linspace(0.2, 0.4, 11)
    costs = profile_likelihood_1d(fwd, obs, 0, fixed_vals, (0.02, 0.3))
    assert fixed_vals[np.argmin(costs)] == pytest.approx(0.3, abs=0.03)

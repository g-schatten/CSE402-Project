"""Observation operators and noise models (PLAN.md section 2.5)."""
from __future__ import annotations

import numpy as np

from sirlab.models import SIR
from sirlab.observe import NoiseModel, incidence, prevalence
from sirlab.solvers import integrate

N = 1000.0


def _solve():
    model = SIR(N)
    y0 = np.array([N - 1, 1.0, 0.0])
    theta = np.array([0.3, 0.1])
    return model, integrate(model, y0, theta, (0.0, 60.0), solver="rk4", h=0.05), theta


def test_prevalence_matches_interpolated_i():
    model, res, theta = _solve()
    t_obs = np.array([10.0, 20.0, 30.0])
    obs = prevalence(res, i_index=1, t_obs=t_obs)
    # interpolated value should lie close to the nearest grid samples (h=0.05, so very close)
    for t, o in zip(t_obs, obs):
        idx = np.argmin(np.abs(res.t - t))
        assert abs(o - res.y[idx, 1]) < 1e-2


def test_incidence_is_nonnegative_and_reasonable():
    model, res, theta = _solve()
    t_obs = np.arange(1.0, 60.0, 1.0)
    inc = incidence(model, res, theta, t_obs, n_total=N)
    assert np.all(inc >= -1e-8)
    # Exact identity: integral_0^T (beta S I / N) dt = S(0) - S(T) (from dS/dt itself),
    # NOT total recovered R(T) -- that would double-count the lag between infection
    # and recovery (people counted in cumulative incidence who are still in I at T).
    s_at_window_end = np.interp(t_obs[-1], res.t, res.y[:, 0])
    expected = res.y[0, 0] - s_at_window_end
    assert abs(np.sum(inc) - expected) < 0.03 * expected


def test_incidence_reporting_fraction_scales_linearly():
    model, res, theta = _solve()
    t_obs = np.arange(1.0, 60.0, 1.0)
    full = incidence(model, res, theta, t_obs, n_total=N, reporting_fraction=1.0)
    half = incidence(model, res, theta, t_obs, n_total=N, reporting_fraction=0.5)
    assert np.allclose(half, 0.5 * full)


def test_noise_models_sample_with_expected_scale(rng):
    clean = np.full(2000, 100.0)
    additive = NoiseModel(kind="additive_gaussian", sigma=5.0)
    samples = additive.sample(clean, rng)
    assert abs(np.std(samples) - 5.0) < 0.5
    assert abs(np.mean(samples) - 100.0) < 1.0

    proportional = NoiseModel(kind="proportional_gaussian", sigma=0.1)
    samples_p = proportional.sample(clean, rng)
    assert abs(np.std(samples_p) - 10.0) < 1.5

    poisson = NoiseModel(kind="poisson")
    samples_pois = poisson.sample(clean, rng)
    # Poisson variance = mean, so std ~ sqrt(100) = 10
    assert abs(np.std(samples_pois) - 10.0) < 2.0

    negbinom = NoiseModel(kind="negbinom", dispersion=10.0)
    samples_nb = negbinom.sample(clean, rng)
    # negative binomial variance = mean + mean^2/dispersion > Poisson variance
    assert np.var(samples_nb) > np.var(samples_pois) * 0.5


def test_weight_function_matches_noise_kind():
    obs = np.array([0.0, 1.0, 100.0])
    additive = NoiseModel(kind="additive_gaussian", sigma=1.0)
    assert np.allclose(additive.weight(obs), 1.0)
    poisson = NoiseModel(kind="poisson")
    w = poisson.weight(obs)
    assert w[2] < w[1]  # larger counts get smaller weight (variance-stabilizing)

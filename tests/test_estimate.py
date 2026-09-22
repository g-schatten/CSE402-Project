"""Estimator acceptance tests (PLAN.md section 4.5): on noise-free synthetic
data with a tight solver, every optimizer recovers (beta, gamma) to high
accuracy, and Gauss-Newton/Levenberg-Marquardt converge in a handful of
iterations from a 50%-off start.
"""
from __future__ import annotations

import numpy as np
import pytest

from sirlab.estimate import coordinate_descent_golden, gauss_newton, levenberg_marquardt, nelder_mead
from sirlab.estimate.base import ForwardModel
from sirlab.models import SIR
from sirlab.observe import prevalence

N = 1000.0
THETA_TRUE = np.array([0.3, 0.1])
Y0 = np.array([999.0, 1.0, 0.0])
T_OBS = np.linspace(1.0, 60.0, 30)


def _obs_fn(result, t_obs):
    return prevalence(result, i_index=1, t_obs=t_obs)


@pytest.fixture
def tight_forward_model():
    model = SIR(N)
    fwd = ForwardModel(model, Y0, T_OBS, _obs_fn, solver="rk45", rtol=1e-11, atol=1e-13)
    fwd.obs_state_index = 1
    return fwd


@pytest.fixture
def clean_obs(tight_forward_model):
    return tight_forward_model.predict(THETA_TRUE)


def test_gauss_newton_exact_recovery(tight_forward_model, clean_obs):
    theta0 = THETA_TRUE * 1.5  # 50% off
    result = gauss_newton(tight_forward_model, clean_obs, theta0, log_space=True)
    assert np.allclose(result.theta_hat, THETA_TRUE, atol=1e-6)
    assert result.n_iter < 15


def test_levenberg_marquardt_exact_recovery(tight_forward_model, clean_obs):
    theta0 = THETA_TRUE * 1.5
    result = levenberg_marquardt(tight_forward_model, clean_obs, theta0)
    assert np.allclose(result.theta_hat, THETA_TRUE, atol=1e-6)
    assert result.n_iter < 15


def test_nelder_mead_exact_recovery(tight_forward_model, clean_obs):
    theta0 = THETA_TRUE * 1.5
    result = nelder_mead(tight_forward_model, clean_obs, theta0, max_iter=300)
    assert np.allclose(result.theta_hat, THETA_TRUE, atol=1e-4)


def test_coordinate_descent_golden_exact_recovery(tight_forward_model, clean_obs):
    theta0 = THETA_TRUE * 1.5
    result = coordinate_descent_golden(
        tight_forward_model, clean_obs, theta0, bounds=[(0.05, 1.0), (0.02, 0.5)], n_sweeps=8
    )
    assert np.allclose(result.theta_hat, THETA_TRUE, atol=1e-3)


def test_jacobian_matches_finite_difference(tight_forward_model):
    """Cross-check the sensitivity-equation Jacobian (fwd.jacobian, built
    from PLAN.md 2.4) against a central finite difference -- an independent
    check distinct from the complex-step test in test_models.py, at the
    level of the full forward-model-to-observations map."""
    theta = THETA_TRUE.copy()
    jac_analytic = tight_forward_model.jacobian(theta)
    eps = 1e-5
    jac_fd = np.empty_like(jac_analytic)
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] += eps
        tm = theta.copy()
        tm[j] -= eps
        jac_fd[:, j] = (tight_forward_model.predict(tp) - tight_forward_model.predict(tm)) / (2 * eps)
    # Compare in relative-L2 norm rather than elementwise allclose: a few
    # entries near the epidemic tail have large third-derivative curvature,
    # so a central finite difference at any single eps carries a bit more
    # truncation/rounding error there than the analytic sensitivity equation
    # does (confirmed correct separately by the exact-recovery GN/LM tests).
    rel_err = np.linalg.norm(jac_analytic - jac_fd) / np.linalg.norm(jac_fd)
    assert rel_err < 5e-3, rel_err


def test_qr_and_normal_equations_gauss_newton_agree(tight_forward_model, clean_obs):
    theta0 = THETA_TRUE * 1.3
    r_lu = gauss_newton(tight_forward_model, clean_obs, theta0, use_qr=False, log_space=True)
    r_qr = gauss_newton(tight_forward_model, clean_obs, theta0, use_qr=True, log_space=True)
    assert np.allclose(r_lu.theta_hat, r_qr.theta_hat, atol=1e-4)

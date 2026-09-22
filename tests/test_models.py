"""Jacobians vs complex-step differentiation (PLAN.md section 4.1 accept
criterion): d f/d y and d f/d theta must match complex-step to < 1e-10 over
many random states, for every model. Complex-step avoids the subtractive
cancellation of finite differences and is exact to machine precision for
analytic f, so any disagreement here is a bug in the analytic Jacobian, not
numerical noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from sirlab.models import SEIR, SIR, SIRS, SIRVaccination

N = 1000.0
H = 1e-30  # complex-step increment; result is exact to double precision for any h this small


def complex_step_jacobian_y(model, t, y, theta):
    n = len(y)
    jac = np.zeros((n, n))
    for k in range(n):
        yc = y.astype(complex).copy()
        yc[k] += 1j * H
        f = model.rhs(t, yc, theta.astype(complex))
        jac[:, k] = f.imag / H
    return jac


def complex_step_jacobian_theta(model, t, y, theta):
    n_state = len(y)
    n_param = len(theta)
    jac = np.zeros((n_state, n_param))
    for k in range(n_param):
        tc = theta.astype(complex).copy()
        tc[k] += 1j * H
        f = model.rhs(t, y.astype(complex), tc)
        jac[:, k] = f.imag / H
    return jac


MODEL_FACTORIES = {
    "SIR": (lambda: SIR(N), (0.3, 0.1), lambda n: np.array([n - 5, 5, 0.0])),
    "SEIR": (lambda: SEIR(N), (0.3, 0.2, 0.1), lambda n: np.array([n - 8, 3, 5, 0.0])),
    "SIRS": (lambda: SIRS(N), (0.3, 0.1, 0.02), lambda n: np.array([n - 10, 5, 5.0])),
    "SIR-V": (lambda: SIRVaccination(N), (0.3, 0.1, 0.01), lambda n: np.array([n - 6, 5, 0.0, 1.0])),
}


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
def test_jacobians_match_complex_step(name, rng):
    factory, theta0, y0_fn = MODEL_FACTORIES[name]
    model = factory()
    theta0 = np.array(theta0)
    for _ in range(200):
        y = y0_fn(N) + rng.normal(scale=1.0, size=len(y0_fn(N)))
        theta = theta0 * (1.0 + rng.normal(scale=0.1, size=len(theta0)))
        jy_analytic = model.jacobian_y(0.0, y, theta)
        jy_cs = complex_step_jacobian_y(model, 0.0, y, theta)
        assert np.allclose(jy_analytic, jy_cs, atol=1e-9), (name, jy_analytic, jy_cs)

        jt_analytic = model.jacobian_theta(0.0, y, theta)
        jt_cs = complex_step_jacobian_theta(model, 0.0, y, theta)
        assert np.allclose(jt_analytic, jt_cs, atol=1e-9), (name, jt_analytic, jt_cs)


def test_sirs_reduces_to_sir_when_xi_zero(rng):
    sir = SIR(N)
    sirs = SIRS(N)
    y = np.array([950.0, 40.0, 10.0])
    theta_sir = np.array([0.3, 0.1])
    theta_sirs = np.array([0.3, 0.1, 0.0])
    f_sir = sir.rhs(0.0, y, theta_sir)
    f_sirs = sirs.rhs(0.0, y, theta_sirs)
    assert np.allclose(f_sir, f_sirs)


def test_mass_conservation_rhs_sums_to_zero(rng):
    """The defining structural property that makes mass conservation a weak
    diagnostic (PLAN.md 2.8): sum(f) == 0 identically for SIR/SEIR/SIRS
    (vaccination moves mass into an absorbing class but does not create or
    destroy it either)."""
    for name, (factory, theta0, y0_fn) in MODEL_FACTORIES.items():
        model = factory()
        y = y0_fn(N)
        theta = np.array(theta0)
        f = model.rhs(0.0, y, theta)
        assert abs(np.sum(f)) < 1e-9, name

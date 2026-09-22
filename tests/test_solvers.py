"""Observed convergence order verification (PLAN.md section 4.2 accept
criterion and section 7 "order verification" -- the canonical
numerical-analysis test): each solver's error against the SIR gold standard
must shrink at its theoretical rate as h -> 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from sirlab.diagnostics import observed_order
from sirlab.models import SIR
from sirlab.reference import SIRReference
from sirlab.solvers import integrate

N = 1000.0
BETA, GAMMA = 0.3, 0.1
Y0 = np.array([999.0, 1.0, 0.0])
THETA = np.array([BETA, GAMMA])
T_FINAL = 40.0  # comfortably before the tail singularity, well past the peak (~38.4)


@pytest.mark.parametrize(
    "solver,expected_order,tolerance",
    [
        ("euler", 1.0, 0.15),
        ("heun", 2.0, 0.15),
        ("rk4", 4.0, 0.25),
        ("backward_euler", 1.0, 0.15),
        ("trapezoidal", 2.0, 0.2),
    ],
)
def test_observed_convergence_order(solver, expected_order, tolerance):
    model = SIR(N)
    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=Y0[0], i0=Y0[1])
    true_final = ref.trajectory(np.array([T_FINAL]))[0]

    hs = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])
    errs = np.empty_like(hs)
    for k, h in enumerate(hs):
        res = integrate(model, Y0, THETA, (0.0, T_FINAL), solver=solver, h=h)
        errs[k] = np.linalg.norm(res.y[-1] - true_final)

    p_hat = observed_order(errs, hs)
    assert abs(p_hat - expected_order) < tolerance, (solver, p_hat, errs)


def test_rk45_hits_tight_tolerance():
    model = SIR(N)
    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=Y0[0], i0=Y0[1])
    true_final = ref.trajectory(np.array([T_FINAL]))[0]
    res = integrate(model, Y0, THETA, (0.0, T_FINAL), solver="rk45", rtol=1e-10, atol=1e-12)
    err = np.linalg.norm(res.y[-1] - true_final)
    assert err < 1e-4


def test_stability_function_on_scalar_decay():
    """y' = lambda y, lambda<0: each explicit method's stability boundary
    matches its textbook stability function R(z), z = h*lambda."""
    lam = -1.0

    class Scalar:
        n_state = 1

        def rhs(self, t, y, theta):
            return lam * y

    model = Scalar()
    y0 = np.array([1.0])

    def r_euler(z):
        return 1 + z

    def r_heun(z):
        return 1 + z + z**2 / 2

    def r_rk4(z):
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24

    for h, r_fn, name in [(0.5, r_euler, "euler"), (0.5, r_heun, "heun"), (0.5, r_rk4, "rk4")]:
        res = integrate(model, y0, np.array([]), (0.0, h), solver=name, h=h)
        z = h * lam
        assert res.y[-1, 0] == pytest.approx(r_fn(z), abs=1e-10)


def test_euler_instability_and_negativity_at_large_h():
    """Demonstrates the RQ1 stability-boundary claim: Forward Euler at a
    large step size on SIR produces negativity events (and/or blow-up),
    while RK4 at the same h remains well-behaved for this regime."""
    model = SIR(N)
    res_euler = integrate(model, Y0, THETA, (0.0, 80.0), solver="euler", h=10.0)
    res_rk4 = integrate(model, Y0, THETA, (0.0, 80.0), solver="rk4", h=10.0)
    assert res_euler.negativity_events > 0 or res_euler.diverged
    assert res_rk4.negativity_events == 0
    assert not res_rk4.diverged


def test_mass_conservation_is_uninformative_but_phase_invariant_is_not():
    """The deliberate PLAN.md 2.8 finding: explicit RK conserves S+I+R to
    roundoff at ANY h (because the RHS sums to zero identically), even when
    the solution is catastrophically wrong, whereas the phase invariant
    (2.1) drifts at the solver's convergence order and IS discriminating.
    """
    from sirlab.diagnostics import mass_error, phase_invariant_drift

    model = SIR(N)
    res_bad = integrate(model, Y0, THETA, (0.0, 60.0), solver="euler", h=5.0)
    res_good = integrate(model, Y0, THETA, (0.0, 60.0), solver="rk4", h=0.01)

    mass_err_bad = mass_error(res_bad.y, N)
    assert np.max(mass_err_bad) < 1e-8  # "conserved" despite being wrong

    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=Y0[0], i0=Y0[1])
    true_final = ref.trajectory(np.array([60.0]))[0]
    bad_err = np.linalg.norm(res_bad.y[-1] - true_final)
    good_err = np.linalg.norm(res_good.y[-1] - true_final)
    assert bad_err > 50 * good_err  # euler-at-h=5 is badly wrong vs rk4-at-h=0.01

    drift_bad = phase_invariant_drift(res_bad.y[:, 0], res_bad.y[:, 2], BETA, GAMMA, N)
    drift_good = phase_invariant_drift(res_good.y[:, 0], res_good.y[:, 2], BETA, GAMMA, N)
    assert np.max(drift_bad) > 100 * np.max(drift_good)  # phase invariant DOES discriminate

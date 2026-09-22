"""Property-based tests (PLAN.md section 7 "Property-based" layer): for any
valid (beta, gamma, y0), the RK4 (near-exact) trajectory should keep S
non-increasing, R non-decreasing, and every compartment within [0, N].
These are structural invariants of the continuous SIR system that any
sufficiently accurate solver must respect away from pathological step
sizes; we use RK4 at a modest step size, which is accurate enough for these
qualitative (not quantitative) properties to hold.
"""
from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sirlab.models import SIR
from sirlab.solvers import integrate


@settings(max_examples=60, deadline=None)
@given(
    beta=st.floats(min_value=0.05, max_value=1.5),
    gamma=st.floats(min_value=0.02, max_value=0.8),
    n_total=st.floats(min_value=100.0, max_value=10_000.0),
    i0_frac=st.floats(min_value=1e-4, max_value=0.2),
)
def test_sir_structural_invariants(beta, gamma, n_total, i0_frac):
    i0 = i0_frac * n_total
    y0 = np.array([n_total - i0, i0, 0.0])
    model = SIR(n_total)
    res = integrate(model, y0, np.array([beta, gamma]), (0.0, 60.0), solver="rk4", h=0.05)

    s, i, r = res.y[:, 0], res.y[:, 1], res.y[:, 2]
    tol = 1e-6 * n_total

    # S is non-increasing (monotone depletion of susceptibles)
    assert np.all(np.diff(s) <= tol)
    # R is non-decreasing (recovery is one-way)
    assert np.all(np.diff(r) >= -tol)
    # every compartment stays within [0, N] (up to solver/roundoff slack)
    assert np.all(s >= -tol) and np.all(s <= n_total + tol)
    assert np.all(i >= -tol) and np.all(i <= n_total + tol)
    assert np.all(r >= -tol) and np.all(r <= n_total + tol)
    # mass conservation holds throughout (structural, not an accuracy claim)
    assert np.allclose(s + i + r, n_total, atol=tol)


@settings(max_examples=40, deadline=None)
@given(
    beta=st.floats(min_value=0.05, max_value=1.5),
    gamma=st.floats(min_value=0.02, max_value=0.8),
)
def test_r0_consistent_with_epidemic_growth_direction(beta, gamma):
    """If R0 = beta/gamma > 1, I(t) must rise above I0 at some point shortly
    after t=0 (the epidemic takes off); if R0 < 1, I(t) must not exceed I0
    by more than solver/roundoff slack (it dies out immediately)."""
    n_total = 1000.0
    i0 = 1.0
    y0 = np.array([n_total - i0, i0, 0.0])
    model = SIR(n_total)
    res = integrate(model, y0, np.array([beta, gamma]), (0.0, 5.0 / gamma), solver="rk4", h=0.01)
    i_max = np.max(res.y[:, 1])
    r0 = beta / gamma
    if r0 > 1.01:
        assert i_max > i0 * 1.001
    elif r0 < 0.99:
        assert i_max <= i0 * 1.001 + 1e-6

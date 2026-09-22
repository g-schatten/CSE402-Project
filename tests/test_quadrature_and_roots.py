"""Adaptive quadrature and root-finding primitives (PLAN.md section 4).

Includes a regression test for the tolerance-halving bug found and fixed
during development: naive per-panel tol-halving drives the error budget
below floating-point roundoff long before max_depth is reached, causing
silent under-convergence. See core/sirlab/quadrature/adaptive.py's
docstring and web/src/numerics/reference.ts's parallel fix.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sirlab.quadrature.adaptive import adaptive_simpson, composite_simpson
from sirlab.quadrature.simpson import simpson_nonuniform, trapezoid
from sirlab.roots.newton import bisection, newton, newton_safe


def test_adaptive_simpson_matches_known_integral():
    # integral of sin(x) from 0 to pi is exactly 2
    val = adaptive_simpson(math.sin, 0.0, math.pi, tol=1e-12)
    assert abs(val - 2.0) < 1e-9


def test_adaptive_simpson_does_not_silently_underconverge_on_tight_tolerance():
    """Regression test: a wide, tight-tolerance integral used to force
    per-panel tol-halving below the roundoff floor before max_depth was
    reached, causing panels to be force-accepted without truly converging
    (observed error ~1e-2 on an integral of magnitude ~38 before the fix)."""

    def integrand(r: float) -> float:
        beta, gamma, n, s0 = 0.3, 0.1, 1000.0, 999.0
        k = beta / (gamma * n)
        s = s0 * math.exp(-k * r)
        i = max(n - r - s, 1e-300)
        return 1.0 / (gamma * i)

    r_at_peak = 365.87059611150863
    val = adaptive_simpson(integrand, 0.0, r_at_peak, tol=1e-13)
    # ground truth from a fine (2M-point) trapezoid check performed during
    # development; both this and the TS twin must agree with it.
    assert abs(val - 38.35530772768317) < 1e-6


def test_adaptive_simpson_handles_near_singular_tail_without_hanging():
    """Near the SIR final-size asymptote the integrand diverges; the
    quadrature must degrade gracefully (via max_evals) rather than hang."""

    def integrand(r: float) -> float:
        beta, gamma, n, s0 = 0.3, 0.1, 1000.0, 999.0
        k = beta / (gamma * n)
        s = s0 * math.exp(-k * r)
        i = max(n - r - s, 1e-300)
        return 1.0 / (gamma * i)

    r_inf = 500.4999995
    val = adaptive_simpson(integrand, 0.0, r_inf * (1 - 1e-9), tol=1e-13, max_evals=5000)
    assert np.isfinite(val)
    assert val > 0


def test_composite_simpson_matches_known_integral():
    val = composite_simpson(lambda x: x**2, 0.0, 3.0, n=100)
    assert abs(val - 9.0) < 1e-6


def test_simpson_nonuniform_and_trapezoid_agree_on_linear_function():
    t = np.array([0.0, 0.5, 1.3, 2.0, 3.5, 4.0])
    y = 2.0 * t + 1.0  # exact for both rules
    assert abs(trapezoid(y, t) - (2.0 * t[-1] ** 2 / 2 + t[-1])) < 1e-9
    assert abs(simpson_nonuniform(y, t) - (2.0 * t[-1] ** 2 / 2 + t[-1])) < 1e-9


def test_newton_converges_on_simple_root():
    root = newton(lambda x: x**2 - 4, lambda x: 2 * x, x0=3.0)
    assert abs(root - 2.0) < 1e-10


def test_bisection_converges_and_validates_bracket():
    root = bisection(lambda x: x**3 - 2, 0.0, 2.0)
    assert abs(root - 2 ** (1 / 3)) < 1e-10
    with pytest.raises(ValueError):
        bisection(lambda x: x**2 + 1, -1.0, 1.0)  # no sign change


def test_newton_safe_matches_plain_newton_on_well_behaved_function():
    root_safe = newton_safe(lambda x: x**2 - 4, lambda x: 2 * x, 0.0, 5.0)
    assert abs(root_safe - 2.0) < 1e-10


def test_newton_safe_handles_a_case_where_plain_newton_would_overshoot():
    """f(x) = x^3 - x - 2 has a root near 1.5214; a poor initial Newton
    step from a flat-derivative region can overshoot badly. newton_safe
    must still converge inside the given bracket."""
    f = lambda x: x**3 - x - 2
    fp = lambda x: 3 * x**2 - 1
    root = newton_safe(f, fp, 0.0, 3.0)
    assert abs(f(root)) < 1e-9

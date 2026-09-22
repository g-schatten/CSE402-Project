"""Hand-written LU and QR vs SciPy/NumPy (PLAN.md section 4.3 accept
criterion). SciPy/NumPy appear only in this test file, never inside
core/sirlab itself (see test_no_scipy_in_core.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from sirlab.linalg.lu import cond_estimate, det, lu_factor, lu_solve, solve
from sirlab.linalg.qr import householder_qr, qr_solve


def test_lu_solve_matches_scipy(rng):
    for n in (2, 5, 10, 20):
        a = rng.normal(size=(n, n)) + n * np.eye(n)  # diagonally dominant -> well conditioned
        b = rng.normal(size=n)
        x_ours = solve(a, b)
        x_scipy = sla.solve(a, b)
        assert np.allclose(x_ours, x_scipy, atol=1e-9), n


def test_lu_solve_matches_scipy_multi_rhs(rng):
    n = 8
    a = rng.normal(size=(n, n)) + n * np.eye(n)
    b = rng.normal(size=(n, 3))
    x_ours = solve(a, b)
    x_scipy = sla.solve(a, b)
    assert np.allclose(x_ours, x_scipy, atol=1e-9)


def test_lu_det_matches_numpy(rng):
    for n in (3, 6, 10):
        a = rng.normal(size=(n, n)) + n * np.eye(n)
        assert abs(det(a) - np.linalg.det(a)) < 1e-6 * max(1.0, abs(np.linalg.det(a)))


def test_lu_raises_on_singular():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank-deficient
    with pytest.raises(np.linalg.LinAlgError):
        lu_factor(a)


def test_cond_estimate_matches_numpy(rng):
    n = 6
    a = rng.normal(size=(n, n)) + n * np.eye(n)
    ours = cond_estimate(a)
    npc = np.linalg.cond(a, 1)
    assert abs(ours - npc) / npc < 1e-6


def test_qr_reconstructs_and_is_orthonormal(rng):
    m, n = 12, 4
    a = rng.normal(size=(m, n))
    q, r = householder_qr(a)
    assert np.allclose(q @ r, a, atol=1e-10)
    assert np.allclose(q.T @ q, np.eye(n), atol=1e-10)
    assert np.allclose(r, np.triu(r))


def test_qr_solve_matches_numpy_lstsq(rng):
    m, n = 15, 5
    a = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    x_ours = qr_solve(a, b)
    x_np = np.linalg.lstsq(a, b, rcond=None)[0]
    assert np.allclose(x_ours, x_np, atol=1e-8)


def test_condition_number_squares_under_normal_equations(rng):
    """The concrete numerical-analysis point behind preferring QR over
    J^T J for the estimator (PLAN.md 2.6): kappa(J^T J) = kappa(J)^2."""
    m, n = 20, 4
    a = rng.normal(size=(m, n))
    kappa_j = np.linalg.cond(a)
    kappa_jtj = np.linalg.cond(a.T @ a)
    assert kappa_jtj == pytest.approx(kappa_j**2, rel=1e-6)

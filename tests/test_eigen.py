"""Hand-written power method, inverse power method, and Hotelling deflation
(linalg/eigen.py) against the worked examples of the CSE 401 lecture
"Eigenvalue Decomposition: The Power Method", and against NumPy's
eigensolvers -- which appear only in this test file, never inside
core/sirlab (see test_hygiene.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from sirlab.linalg.eigen import (
    cond2,
    dominant_pair,
    hotelling_deflate,
    inverse_power_method,
    power_method,
    spectral_radius,
    symmetric_eigen,
)
from sirlab.models import SIR

LECTURE_2X2 = np.array([[2.0, 1.0], [1.0, 2.0]])
LECTURE_3X3 = np.array([[3.556, -1.778, 0.0], [-1.778, 3.556, -1.778], [0.0, -1.778, 3.556]])
LECTURE_4X4 = np.array([[8.0, 2.0, 0.0, 0.0], [2.0, 8.0, 0.0, 0.0], [0.0, 0.0, 3.0, 1.0], [0.0, 0.0, 1.0, 3.0]])


def test_power_method_reproduces_lecture_2x2_table():
    # slide 27: x0 = [1, 0]
    res = power_method(LECTURE_2X2, np.array([1.0, 0.0]))
    assert res.factors[:4] == pytest.approx([2.000, 2.500, 2.800, 2.929], abs=5e-4)
    assert res.converged
    assert res.eigenvalue == pytest.approx(3.0, rel=1e-10)
    assert res.eigenvector == pytest.approx([1.0, 1.0], abs=1e-10)


def test_power_method_reproduces_lecture_3x3_table():
    # slide 30, including the sign change of the factor at iteration 3
    res = power_method(LECTURE_3X3)
    table = [1.778, 3.556, -7.112, 6.223, 6.096, 6.07483, 6.07122, 6.07060, 6.07049, 6.07048]
    assert res.factors[:10] == pytest.approx(table, abs=5e-6)
    assert res.eigenvalue == pytest.approx(3.556 + 1.778 * np.sqrt(2.0), rel=1e-11)
    assert res.eigenvector == pytest.approx([-1.0 / np.sqrt(2.0), 1.0, -1.0 / np.sqrt(2.0)], abs=1e-10)


def test_inverse_power_method_reproduces_lecture_4x4_table():
    # slides 36-37: x0 = [1, 0, 1, 0]; the factors approach 1/lambda_min = 0.5
    res = inverse_power_method(LECTURE_4X4, np.array([1.0, 0.0, 1.0, 0.0]))
    table = [0.37500, 0.41667, 0.45000, 0.47222, 0.48529, 0.49242]
    assert res.factors[:6] == pytest.approx(table, abs=5e-6)
    assert res.eigenvalue == pytest.approx(2.0, rel=1e-10)
    assert res.eigenvector == pytest.approx([0.0, 0.0, 1.0, -1.0], abs=1e-9)


def test_inverse_power_method_raises_on_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        inverse_power_method(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_hotelling_deflation_matches_lecture_example():
    # slide 42: A_2 = A - 3 v_hat v_hat^T with v = [1, 1]
    a2 = hotelling_deflate(LECTURE_2X2, 3.0, np.array([1.0, 1.0]))
    assert a2 == pytest.approx(np.array([[0.5, -0.5], [-0.5, 0.5]]))


def test_symmetric_eigen_on_lecture_matrices():
    vals, vecs = symmetric_eigen(LECTURE_4X4)
    assert vals == pytest.approx([2.0, 4.0, 6.0, 10.0], abs=1e-10)
    assert np.allclose(LECTURE_4X4 @ vecs, vecs * vals, atol=1e-9)
    # the all-ones start vector is the 2x2's first eigenvector, so the second
    # stage must not start from it after deflation
    vals, _ = symmetric_eigen(LECTURE_2X2)
    assert vals == pytest.approx([1.0, 3.0], abs=1e-12)


def test_symmetric_eigen_matches_numpy_eigh(rng):
    for _ in range(200):
        n = int(rng.integers(2, 7))
        m = rng.normal(size=(n, n))
        s = m + m.T
        vals, vecs = symmetric_eigen(s)
        ref = np.linalg.eigvalsh(s)
        scale = np.max(np.abs(ref))
        assert np.allclose(vals, ref, atol=1e-10 * scale)
        assert np.allclose(vecs.T @ vecs, np.eye(n), atol=1e-8)
        assert np.allclose(s @ vecs, vecs * vals, atol=1e-7 * scale)


def test_symmetric_eigen_handles_equal_and_opposite_eigenvalues(rng):
    # +2 and -2 share the largest magnitude, so the plain power method never settles
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    s = q @ np.diag([2.0, -2.0, 0.5]) @ q.T
    assert not power_method(s).converged
    vals, vecs = symmetric_eigen(s)
    assert vals == pytest.approx([-2.0, 0.5, 2.0], abs=1e-10)
    assert np.allclose(s @ vecs, vecs * vals, atol=1e-9)


def test_symmetric_eigen_rejects_nonsymmetric_matrix():
    with pytest.raises(ValueError):
        symmetric_eigen(np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_spectral_radius_of_complex_conjugate_pair():
    # eigenvalues 0.3 +- 0.4i, modulus 0.5
    a = np.array([[0.3, -0.4], [0.4, 0.3]])
    assert not power_method(a).converged
    assert spectral_radius(a) == pytest.approx(0.5, rel=1e-12)
    pair = dominant_pair(a, np.array([1.0, 0.0]))
    assert sorted(lam.imag for lam in pair) == pytest.approx([-0.4, 0.4], abs=1e-12)
    assert all(lam.real == pytest.approx(0.3, abs=1e-12) for lam in pair)


def test_spectral_radius_of_sir_jacobian_at_the_epidemic_peak():
    # At the peak beta S/N = gamma, so the nonzero eigenvalues are
    # (-a +- sqrt(a^2 - 4 a gamma)) / 2 with a = beta I/N: complex when
    # a < 4 gamma, with |lambda|^2 = a * gamma.
    n_total, beta, gamma = 1000.0, 0.3, 0.1
    s = n_total * gamma / beta
    i = 300.0
    jac = SIR(n_total).jacobian_y(0.0, np.array([s, i, n_total - s - i]), np.array([beta, gamma]))
    a = beta * i / n_total
    assert a < 4.0 * gamma
    assert not power_method(jac).converged
    assert spectral_radius(jac) == pytest.approx(np.sqrt(a * gamma), rel=1e-12)


def test_spectral_radius_matches_numpy_on_sir_jacobians(rng):
    n_total = 1000.0
    model = SIR(n_total)
    for _ in range(300):
        theta = np.array([rng.uniform(0.05, 1.5), rng.uniform(0.05, 1.0)])
        s = rng.uniform(0.0, n_total)
        i = rng.uniform(0.0, n_total - s)
        jac = model.jacobian_y(0.0, np.array([s, i, n_total - s - i]), theta)
        ref = np.max(np.abs(np.linalg.eigvals(jac)))
        assert spectral_radius(jac) == pytest.approx(ref, rel=1e-7)


def test_spectral_radius_matches_numpy_eigvals(rng):
    for _ in range(300):
        n = int(rng.integers(2, 7))
        a = rng.normal(size=(n, n))
        mods = np.sort(np.abs(np.linalg.eigvals(a)))[::-1]
        if n > 2 and mods[2] > 0.99 * mods[0]:
            continue  # three eigenvalues of nearly equal modulus: see the next test
        assert spectral_radius(a) == pytest.approx(mods[0], rel=1e-8)


def test_spectral_radius_raises_when_three_eigenvalues_share_the_modulus(rng):
    # a rotation (eigenvalues e^{+-i t}) plus eigenvalue 1: the iterates never
    # settle into a 2-D invariant subspace, and we say so rather than guess
    t = 1.0
    block = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    with pytest.raises(np.linalg.LinAlgError):
        spectral_radius(q @ block @ q.T)


def test_cond2_matches_numpy(rng):
    for n in (2, 3, 5):
        a = rng.normal(size=(30, n)) * np.logspace(0, 3, n)
        assert cond2(a) == pytest.approx(np.linalg.cond(a), rel=1e-9)

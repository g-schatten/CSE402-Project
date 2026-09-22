"""Hand-written Householder QR, used to solve the least-squares normal
equations a second, better-conditioned way (PLAN.md section 2.6): we form
J = Q R for the Jacobian of residuals and solve R delta = -Q^T r directly,
avoiding the condition-number-squaring of J^T J. Cross-checked against
numpy.linalg.qr in tests/test_linalg.py.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def householder_qr(a: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reduced QR via Householder reflections. a is (m, n), m >= n.

    Returns (Q, R) with Q (m, n) orthonormal columns and R (n, n) upper
    triangular, such that A = Q R.
    """
    a = np.array(a, dtype=float, copy=True)
    m, n = a.shape
    if m < n:
        raise ValueError("householder_qr requires m >= n (tall or square matrix)")
    q_full = np.eye(m)
    r = a.copy()

    for k in range(n):
        x = r[k:, k].copy()
        alpha = -np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else -np.linalg.norm(x)
        if alpha == 0.0:
            continue
        v = x.copy()
        v[0] -= alpha
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-300:
            continue
        v /= v_norm
        # apply H_k = I - 2 v v^T to the trailing submatrix of R
        r[k:, k:] -= 2.0 * np.outer(v, v @ r[k:, k:])
        # accumulate Q = H_1 H_2 ... H_n  (apply the same reflection to Q_full's columns)
        q_full[:, k:] -= 2.0 * np.outer(q_full[:, k:] @ v, v)

    q = q_full[:, :n]
    r = np.triu(r[:n, :])
    return q, r


def qr_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Least-squares solve of A x ~= b via Householder QR (A tall, full column rank)."""
    q, r = householder_qr(a)
    qtb = q.T @ b
    n = r.shape[0]
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (qtb[i] - r[i, i + 1 :] @ x[i + 1 :]) / r[i, i]
    return x

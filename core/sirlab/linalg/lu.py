"""Hand-written LU decomposition with partial pivoting (Doolittle form).

Used by: backward Euler / trapezoidal Newton solves (solvers/implicit.py) and
the Gauss-Newton normal equations (estimate/gaussnewton.py). Cross-checked
against scipy.linalg.lu_factor / lu_solve in tests/test_linalg.py -- SciPy
never appears here, only in the test that verifies this code.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lu_factor(a: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """In-place-style Doolittle LU with partial pivoting.

    Returns (lu, piv) where `lu` packs L (unit lower, diagonal implicit) and
    U (upper) into one matrix, and `piv` records row swaps in the compact
    LAPACK convention: piv[k] is the row that row k was swapped with at
    step k (piv[k] >= k).
    """
    a = np.array(a, dtype=float, copy=True)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("lu_factor requires a square matrix")
    piv = np.arange(n)

    for k in range(n):
        # partial pivoting: largest magnitude in column k, at or below row k
        p = k + int(np.argmax(np.abs(a[k:, k])))
        if a[p, k] == 0.0:
            raise np.linalg.LinAlgError(f"singular matrix (zero pivot at column {k})")
        if p != k:
            a[[k, p], :] = a[[p, k], :]
            piv[k] = p
        else:
            piv[k] = k
        for i in range(k + 1, n):
            factor = a[i, k] / a[k, k]
            a[i, k] = factor  # store multiplier in the (now unused) lower slot
            a[i, k + 1 :] -= factor * a[k, k + 1 :]
    return a, piv


def _apply_pivots(b: NDArray[np.float64], piv: NDArray[np.int_]) -> NDArray[np.float64]:
    b = b.copy()
    for k, p in enumerate(piv):
        if p != k:
            b[[k, p]] = b[[p, k]]
    return b


def lu_solve(lu: NDArray[np.float64], piv: NDArray[np.int_], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve A x = b given the (lu, piv) factors from lu_factor."""
    n = lu.shape[0]
    b = np.asarray(b, dtype=float)
    single = b.ndim == 1
    rhs = b.reshape(n, 1) if single else b
    x = np.empty_like(rhs)
    for col in range(rhs.shape[1]):
        v = _apply_pivots(rhs[:, col], piv)
        # forward substitution, L unit-lower-triangular
        y = np.empty(n)
        for i in range(n):
            y[i] = v[i] - lu[i, :i] @ y[:i]
        # back substitution, U upper-triangular
        xcol = np.empty(n)
        for i in range(n - 1, -1, -1):
            xcol[i] = (y[i] - lu[i, i + 1 :] @ xcol[i + 1 :]) / lu[i, i]
        x[:, col] = xcol
    return x[:, 0] if single else x


def solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    lu, piv = lu_factor(a)
    return lu_solve(lu, piv, b)


def det(a: NDArray[np.float64]) -> float:
    lu, piv = lu_factor(a)
    n_swaps = int(np.sum(np.arange(len(piv)) != piv))
    sign = -1.0 if n_swaps % 2 else 1.0
    return float(sign * np.prod(np.diag(lu)))


def cond_estimate(a: NDArray[np.float64]) -> float:
    """1-norm condition number estimate: ||A||_1 * ||A^-1||_1, via explicit
    inverse (fine for the small matrices -- Jacobians of a 2-4 state model --
    this library ever factors)."""
    n = a.shape[0]
    lu, piv = lu_factor(a)
    inv = lu_solve(lu, piv, np.eye(n))
    norm_a = np.max(np.sum(np.abs(a), axis=0))
    norm_inv = np.max(np.sum(np.abs(inv), axis=0))
    return float(norm_a * norm_inv)

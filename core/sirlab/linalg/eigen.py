"""Hand-written eigenvalue routines from the CSE 401 lecture "Eigenvalue
Decomposition: The Power Method": the power method, the inverse power method,
and Hotelling deflation.

Used by: the stability metric h * rho(J_y f) (diagnostics.py, E03), the
confidence-ellipse axes (uq/asymptotic.py, E11/E14), the cost-valley
direction and curvatures (E05), and the 2-norm condition number of the
residual Jacobian (E10). Cross-checked against numpy.linalg.eig / eigh / cond
in tests/test_eigen.py -- NumPy's eigensolvers never appear here, only in the
test that verifies this code.

The lecture's power method needs a single dominant eigenvalue,
|lambda_1| > |lambda_2|. The SIR Jacobian breaks that assumption. Its
eigenvalues are 0 and those of the 2x2 block [[-a, -b], [a, b - gamma]]
(a = beta I/N, b = beta S/N); at the epidemic peak b = gamma, so they are
(-a +- sqrt(a^2 - 4 a gamma)) / 2, a complex-conjugate pair of equal modulus
whenever a < 4 gamma. The normalizing factor then oscillates instead of
converging. Whenever the power method stalls like this, the iterates still
settle into the 2-D invariant subspace of the two largest eigenvalues, so we
project onto that subspace and solve the 2x2 problem there (_project_2d).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sirlab.linalg.lu import lu_factor, lu_solve

Operator = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# Two entries whose magnitudes agree to this relative margin count as tied for
# largest (see _iterate). Far below any real gap between entries, so it never
# changes which entry the lecture's rule picks when one is clearly largest.
_TIE_RTOL = 1e-10


@dataclass
class PowerResult:
    eigenvalue: float
    eigenvector: NDArray[np.float64]  # scaled so its largest-magnitude entry is 1
    n_iter: int
    converged: bool
    factors: NDArray[np.float64]  # normalizing factor at every iteration (the lecture's table column)


def _square(a) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("eigenvalue routines require a square matrix")
    return a


def _iterate(apply: Operator, x0: NDArray[np.float64], *, tol: float, atol: float, max_iter: int) -> PowerResult:
    """The loop shared by the power and inverse power methods (lecture slide
    25): y = apply(x); the largest-magnitude entry of y is the normalizing
    factor and the eigenvalue estimate; x = y / factor.

    Stopping: the lecture stops once the factor stops changing,
    |f_k - f_{k-1}| <= tol * |f_k| (its |eps_a| test; atol adds an absolute
    floor for eigenvalues at roundoff level). We also require the normalized
    vector to have stopped changing, max_i |x_k - x_{k-1}| <= tol. The factor
    alone can stall early: for the lecture's 4x4 matrix (slide 31) started
    from all-ones, the largest entry equals 10 from the first step while x is
    still far from the eigenvector, and deflating that x (as symmetric_eigen
    does) would corrupt every later eigenvalue.

    Ties: while the previous iteration's entry is still (to _TIE_RTOL) the
    largest in magnitude, we keep dividing by it. Without this, an
    eigenvector with two entries of equal magnitude and opposite sign, such as
    [1, -1], lets roundoff pick a different entry each iteration and flip the
    factor's sign, so the estimate never settles.
    """
    x = np.array(x0, dtype=float)
    if not np.any(x):
        raise ValueError("the starting vector must be nonzero")
    factors: list[float] = []
    m_prev = None
    for k in range(1, max_iter + 1):
        y = apply(x)
        mag = np.abs(y)
        m = int(np.argmax(mag))
        if m_prev is not None and mag[m_prev] >= (1.0 - _TIE_RTOL) * mag[m]:
            m = m_prev
        factor = float(y[m])
        if factor == 0.0:
            # apply(x) = 0 = 0 * x, so x is an eigenvector with eigenvalue 0
            factors.append(0.0)
            return PowerResult(0.0, x / x[np.argmax(np.abs(x))], k, True, np.array(factors))
        x_new = y / factor
        factors.append(factor)
        if k > 1 and abs(factor - factors[-2]) <= tol * abs(factor) + atol and np.max(np.abs(x_new - x)) <= tol:
            return PowerResult(factor, x_new, k, True, np.array(factors))
        x = x_new
        m_prev = m
    return PowerResult(factors[-1], x, max_iter, False, np.array(factors))


def power_method(
    a: NDArray[np.float64],
    x0: NDArray[np.float64] | None = None,
    *,
    tol: float = 1e-12,
    atol: float = 0.0,
    max_iter: int = 1000,
) -> PowerResult:
    """Dominant (largest-magnitude) eigenpair of a square matrix (lecture
    slides 23-26). x0 defaults to the all-ones vector, as in the lecture's 3x3
    example; it must have a nonzero component along the dominant eigenvector.

    `converged` is False when the iteration never settles. For a real matrix
    that means its two largest eigenvalues have (nearly) equal modulus: a
    complex-conjugate pair, a pair +lambda / -lambda, or two real eigenvalues
    so close that max_iter iterations are not enough.
    """
    a = _square(a)
    x0 = np.ones(a.shape[0]) if x0 is None else x0
    return _iterate(lambda x: a @ x, x0, tol=tol, atol=atol, max_iter=max_iter)


def inverse_power_method(
    a: NDArray[np.float64],
    x0: NDArray[np.float64] | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> PowerResult:
    """Smallest-magnitude eigenpair (lecture slides 34-37): the power method
    on A^-1, whose dominant eigenvalue is 1/lambda_min. Rather than forming
    A^-1, each iteration solves A y = x with our LU factorization, factored
    once and reused (slide 35).

    `factors` holds the normalizing factors, i.e. estimates of 1/lambda_min
    as in the lecture's table; `eigenvalue` is lambda_min = 1/factor. Raises
    LinAlgError if A is singular (lambda_min = 0).
    """
    a = _square(a)
    lu, piv = lu_factor(a)
    x0 = np.ones(a.shape[0]) if x0 is None else x0
    res = _iterate(lambda x: lu_solve(lu, piv, x), x0, tol=tol, atol=0.0, max_iter=max_iter)
    res.eigenvalue = 1.0 / res.eigenvalue  # A^-1 x = 0 is impossible for x != 0, so the factor is nonzero
    return res


def hotelling_deflate(a: NDArray[np.float64], eigenvalue: float, eigenvector: NDArray[np.float64]) -> NDArray[np.float64]:
    """A - lambda v_hat v_hat^T (lecture slides 41-44). For symmetric A this
    replaces lambda by 0 and leaves every other eigenpair unchanged, because
    the eigenvectors of a symmetric matrix are orthogonal."""
    v = np.asarray(eigenvector, dtype=float)
    v = v / np.sqrt(v @ v)
    return np.asarray(a, dtype=float) - eigenvalue * np.outer(v, v)


def _project_2d(
    apply: Operator, x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Project the operator onto span{x, apply(x)}: orthonormalize the two
    vectors (Gram-Schmidt) into the columns of Q and return (Q, H, invariant)
    with H = Q^T apply(Q). When x is a stalled power-method iterate, that span
    is the 2-D invariant subspace of the two largest eigenvalues, and H's
    eigenvalues are exactly those two. `invariant` checks this: it is True
    when apply(Q) = Q H to 1e-10, i.e. the components along every other
    eigenvector have died out. If apply(x) is parallel to x, x is already an
    eigenvector: Q has one column and H is 1x1."""
    q1 = np.asarray(x, dtype=float)
    q1 = q1 / np.sqrt(q1 @ q1)
    aq1 = apply(q1)
    w = aq1.copy()
    for _ in range(2):  # Gram-Schmidt twice, for orthogonality to roundoff
        w = w - (q1 @ w) * q1
    w_norm = np.sqrt(w @ w)
    if w_norm <= 1e-14 * max(np.sqrt(aq1 @ aq1), 1e-300):
        return q1[:, None], np.array([[q1 @ aq1]]), True
    q = np.column_stack([q1, w / w_norm])
    aq = np.column_stack([aq1, apply(q[:, 1])])
    h = q.T @ aq
    invariant = bool(np.max(np.abs(aq - q @ h)) <= 1e-10 * max(np.max(np.abs(aq)), 1e-300))
    return q, h, invariant


def _eig_2x2(h: NDArray[np.float64]) -> tuple[complex, complex]:
    """Roots of det(H - lambda I) = lambda^2 - tr(H) lambda + det(H) = 0
    (lecture slide 7), larger modulus first."""
    if h.shape == (1, 1):
        return complex(h[0, 0]), complex(h[0, 0])
    trace = h[0, 0] + h[1, 1]
    det = h[0, 0] * h[1, 1] - h[0, 1] * h[1, 0]
    root = np.sqrt(complex(trace * trace - 4.0 * det))
    pair = sorted([(trace + root) / 2.0, (trace - root) / 2.0], key=abs, reverse=True)
    return pair[0], pair[1]


def _dominant_eigenvalue(apply: Operator, n: int, *, tol: float, max_iter: int, max_rounds: int = 20) -> complex:
    """Largest-magnitude eigenvalue of an operator: the power method, and if
    it stalls, the larger root of the 2-D projection (_project_2d). If the
    projected subspace is not yet invariant (a third eigenvalue of similar
    modulus is still decaying), keep iterating from where we stopped, for up
    to max_rounds rounds of max_iter iterations."""
    x = np.ones(n)
    for _ in range(max_rounds):
        res = _iterate(apply, x, tol=tol, atol=0.0, max_iter=max_iter)
        if res.converged:
            return complex(res.eigenvalue)
        _, h, invariant = _project_2d(apply, res.eigenvector)
        if invariant:
            return _eig_2x2(h)[0]
        x = res.eigenvector
    raise np.linalg.LinAlgError("power method did not settle into a 2-D invariant subspace")


def dominant_pair(a: NDArray[np.float64], x: NDArray[np.float64]) -> tuple[complex, complex]:
    """The two largest-magnitude eigenvalues of a real matrix, given a
    power-method iterate x whose components outside their 2-D invariant
    subspace have died out -- for example a complex-conjugate pair, where
    |lambda| = sqrt(det H)."""
    a = _square(a)
    return _eig_2x2(_project_2d(lambda v: a @ v, x)[1])


def spectral_radius(a: NDArray[np.float64], *, tol: float = 1e-12, max_iter: int = 500) -> float:
    """rho(A) = max_i |lambda_i| of a real, possibly non-symmetric matrix.
    If the power method converges the dominant eigenvalue is real and
    rho = |factor|; if it stalls (for the SIR Jacobian near the peak, a
    complex-conjugate pair), rho comes from the 2-D projection."""
    a = _square(a)
    return float(abs(_dominant_eigenvalue(lambda v: a @ v, a.shape[0], tol=tol, max_iter=max_iter)))


def _start_vector(found: NDArray[np.float64]) -> NDArray[np.float64]:
    """All-ones vector, or failing that a standard basis vector, with the
    unit eigenvectors already found (columns of `found`) projected out."""
    n = found.shape[0]
    for cand in [np.ones(n), *np.eye(n)]:
        x = cand.copy()
        for _ in range(2):  # Gram-Schmidt twice, for orthogonality to roundoff
            x = x - found @ (found.T @ x)
        if np.sqrt(x @ x) > 1e-8 * np.sqrt(cand @ cand):
            return x
    raise np.linalg.LinAlgError("found eigenvectors already span the space")


def _symmetric_2x2(h: NDArray[np.float64]) -> list[tuple[float, NDArray[np.float64]]]:
    """Eigenpairs of a symmetric 2x2 matrix [[p, r], [r, s]] in closed form:
    lambda = (p + s)/2 +- sqrt(((p - s)/2)^2 + r^2), with eigenvector
    [r, lambda - p] or [lambda - s, r], whichever is longer."""
    p, s, r = h[0, 0], h[1, 1], 0.5 * (h[0, 1] + h[1, 0])
    mid, rad = 0.5 * (p + s), float(np.hypot(0.5 * (p - s), r))
    pairs = []
    for lam in (mid + rad, mid - rad):
        u1, u2 = np.array([r, lam - p]), np.array([lam - s, r])
        u = u1 if u1 @ u1 >= u2 @ u2 else u2
        if not np.any(u):  # h is a multiple of I: any orthonormal pair works
            u = np.array([1.0, 0.0]) if not pairs else np.array([0.0, 1.0])
        pairs.append((float(lam), u / np.sqrt(u @ u)))
    return pairs


def symmetric_eigen(
    a: NDArray[np.float64],
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """All eigenpairs of a symmetric matrix: find the dominant pair by the
    power method, remove it by Hotelling deflation, repeat (lecture slides
    38-44). Returns (eigenvalues in ascending order, unit eigenvectors as
    columns) -- the convention of numpy.linalg.eigh.

    Each stage starts from the all-ones vector with the eigenvectors already
    found projected out. Without that, the lecture's own 2x2 example fails:
    for A = [[2, 1], [1, 2]] the all-ones vector is the first eigenvector, so
    A_2 @ ones = 0 after deflation and the second stage would report a
    spurious zero eigenvalue.

    A stage stalls when the two largest remaining eigenvalues have (nearly)
    equal magnitude, e.g. lambda and -lambda. It then takes eigenpairs from
    the 2-D projection instead, keeping only those with residual
    |A v - lambda v| <= 1e-10 * max|A|: if the iterate had almost settled on
    one of the two eigenvectors, the other direction is poorly determined, and
    deflating it would contaminate every later stage; the next stage finds it
    cleanly instead. LinAlgError is raised if no eigenpair passes.

    Deflating with a slightly inexact eigenvector leaves a small component of
    it in the deflated matrix, which leaks into later eigenvectors. Since the
    eigenvectors of a symmetric matrix are orthogonal, each new one is
    Gram-Schmidt orthogonalized against those already found, which removes
    exactly that leak.
    """
    a = _square(a)
    scale = float(np.max(np.abs(a)))
    if np.max(np.abs(a - a.T)) > 1e-10 * max(scale, 1e-300):
        raise ValueError("symmetric_eigen requires a symmetric matrix")
    a = 0.5 * (a + a.T)
    b = a.copy()
    n = a.shape[0]
    vals = np.empty(n)
    vecs = np.empty((n, n))
    i = 0
    while i < n:
        res = power_method(b, _start_vector(vecs[:, :i]), tol=tol, atol=1e-14 * scale, max_iter=max_iter)
        if res.converged:
            pairs = [(res.eigenvalue, res.eigenvector / np.sqrt(res.eigenvector @ res.eigenvector))]
        else:
            q, h, _ = _project_2d(lambda v: b @ v, res.eigenvector)
            if q.shape[1] == 1:
                candidates = [(float(h[0, 0]), q[:, 0])]
            else:
                candidates = [(lam, q @ u) for lam, u in _symmetric_2x2(h)]
            pairs = [
                (lam, v) for lam, v in candidates if np.max(np.abs(a @ v - lam * v)) <= 1e-10 * max(scale, 1e-300)
            ][: n - i]
            if not pairs:
                raise np.linalg.LinAlgError(f"power method failed to resolve eigenpair {i + 1} of {n}")
        for lam, v in pairs:
            for _ in range(2):  # Gram-Schmidt twice, for orthogonality to roundoff
                v = v - vecs[:, :i] @ (vecs[:, :i].T @ v)
            v = v / np.sqrt(v @ v)
            vals[i] = lam
            vecs[:, i] = v
            b = hotelling_deflate(b, lam, v)
            i += 1
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def cond2(a: NDArray[np.float64], *, tol: float = 1e-13, max_iter: int = 10_000) -> float:
    """2-norm condition number sigma_max / sigma_min of a full-column-rank
    matrix. The squared singular values of A are the eigenvalues of the
    symmetric matrix A^T A, so kappa_2(A) = sqrt(lambda_max / lambda_min),
    with lambda_max from the power method and lambda_min from the inverse
    power method (LU factored once). Forming A^T A squares the condition
    number, which costs about 1e-16 * kappa(A)^2 relative accuracy in
    lambda_min -- negligible at the kappa(A) of a few thousand this project's
    Jacobians reach."""
    a = np.asarray(a, dtype=float)
    ata = a.T @ a
    n = ata.shape[0]
    lu, piv = lu_factor(ata)
    lam_max = _dominant_eigenvalue(lambda v: ata @ v, n, tol=tol, max_iter=max_iter).real
    inv_lam_min = _dominant_eigenvalue(lambda v: lu_solve(lu, piv, v), n, tol=tol, max_iter=max_iter).real
    return float(np.sqrt(lam_max * inv_lam_min))

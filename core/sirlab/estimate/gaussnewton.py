"""Gauss-Newton (PLAN.md 2.6 item 4). Solves the normal equations
(J^T J) delta = -J^T r via our own LU factorization, using J from the
forward sensitivity equations (2.4) -- never finite differences. Also
exposes a QR variant that solves min||J delta + r|| directly (better
conditioned: kappa(J) instead of kappa(J)^2, tested in test_linalg.py).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import FitResult, ForwardModel
from sirlab.linalg.lu import lu_factor, lu_solve
from sirlab.linalg.qr import qr_solve


def gauss_newton(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta0: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None = None,
    max_iter: int = 50,
    tol: float = 1e-10,
    use_qr: bool = False,
    log_space: bool = False,
) -> FitResult:
    """If log_space, optimizes u = log(theta) internally (keeps parameters
    positive and better-conditions the landscape near typical epidemic
    (beta, gamma) valleys, per PLAN.md 2.6); theta_hat is still returned in
    the original (linear) parameterization.
    """
    theta = np.array(theta0, dtype=float)
    u = np.log(theta) if log_space else theta.copy()
    path = [theta.copy()]
    n_fev = 0

    def to_theta(u_):
        return np.exp(u_) if log_space else u_

    cost_prev = fwd.cost(to_theta(u), obs, weights)
    n_fev += 1
    converged = False
    jac = None
    it = 0
    for it in range(max_iter):
        theta_cur = to_theta(u)
        r = fwd.residuals(theta_cur, obs, weights)
        jac_theta = fwd.jacobian(theta_cur, weights)  # d(residual)/d(theta), shape (n_obs, n_param)
        n_fev += 1
        jac = jac_theta * theta_cur[None, :] if log_space else jac_theta  # chain rule d/du = theta * d/dtheta

        jtj = jac.T @ jac
        jtr = jac.T @ r
        try:
            if use_qr:
                delta = qr_solve(jac, -r)
            else:
                lu, piv = lu_factor(jtj + 1e-12 * np.eye(len(u)))
                delta = lu_solve(lu, piv, -jtr)
        except np.linalg.LinAlgError:
            break

        # simple backtracking line search to guarantee descent
        step = 1.0
        new_cost = None
        for _ in range(30):
            u_new = u + step * delta
            c = fwd.cost(to_theta(u_new), obs, weights)
            n_fev += 1
            if np.isfinite(c) and c < cost_prev:
                new_cost = c
                break
            step *= 0.5
        if new_cost is None:
            converged = True  # no descent direction found; treat as converged (local min)
            break

        u = u_new
        theta = to_theta(u)
        path.append(theta.copy())
        if abs(cost_prev - new_cost) < tol * max(1.0, cost_prev):
            cost_prev = new_cost
            converged = True
            break
        cost_prev = new_cost

    theta_hat = to_theta(u)
    cov = None
    if jac is not None:
        try:
            n_obs, n_param = jac.shape
            dof = max(n_obs - n_param, 1)
            sigma2 = 2.0 * cost_prev / dof
            jtj = jac.T @ jac
            lu, piv = lu_factor(jtj + 1e-12 * np.eye(n_param))
            jtj_inv = lu_solve(lu, piv, np.eye(n_param))
            cov_u = sigma2 * jtj_inv
            if log_space:
                # delta method: Cov(theta) = diag(theta) Cov(u) diag(theta)
                d = np.diag(theta_hat)
                cov = d @ cov_u @ d
            else:
                cov = cov_u
        except np.linalg.LinAlgError:
            cov = None

    return FitResult(
        theta_hat=theta_hat,
        cost=cost_prev,
        n_iter=it + 1,
        n_fev=n_fev,
        converged=converged,
        path=np.array(path),
        jacobian=jac,
        cov=cov,
        message=f"Gauss-Newton ({'QR' if use_qr else 'normal equations (LU)'}{', log-space' if log_space else ''})",
    )

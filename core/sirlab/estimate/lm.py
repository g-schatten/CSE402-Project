"""Levenberg-Marquardt (PLAN.md 2.6 item 5) -- the production estimator.
(J^T J + lambda * diag(J^T J)) delta = -J^T r, solved with our own LU, with
the classical multiplicative lambda adaptation (increase on a rejected
step, decrease on an accepted one).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import FitResult, ForwardModel
from sirlab.linalg.lu import lu_factor, lu_solve


def levenberg_marquardt(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta0: NDArray[np.float64],
    *,
    weights: NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    lambda0: float = 1e-2,
    log_space: bool = True,
) -> FitResult:
    theta = np.array(theta0, dtype=float)
    u = np.log(theta) if log_space else theta.copy()
    lam = lambda0
    path = [theta.copy()]
    n_fev = 0

    def to_theta(u_):
        return np.exp(u_) if log_space else u_

    cost_cur = fwd.cost(to_theta(u), obs, weights)
    n_fev += 1
    converged = False
    jac = None
    it = 0
    for it in range(max_iter):
        theta_cur = to_theta(u)
        r = fwd.residuals(theta_cur, obs, weights)
        jac_theta = fwd.jacobian(theta_cur, weights)
        n_fev += 1
        jac = jac_theta * theta_cur[None, :] if log_space else jac_theta

        jtj = jac.T @ jac
        jtr = jac.T @ r
        diag = np.diag(np.diag(jtj))

        accepted = False
        for _ in range(30):
            try:
                lu, piv = lu_factor(jtj + lam * diag + 1e-14 * np.eye(len(u)))
                delta = lu_solve(lu, piv, -jtr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            u_try = u + delta
            c_try = fwd.cost(to_theta(u_try), obs, weights)
            n_fev += 1
            if np.isfinite(c_try) and c_try < cost_cur:
                accepted = True
                break
            lam *= 10.0
            if lam > 1e12:
                break

        if not accepted:
            converged = True
            break

        lam = max(lam / 10.0, 1e-12)
        u = u_try
        theta = to_theta(u)
        path.append(theta.copy())
        if abs(cost_cur - c_try) < tol * max(1.0, cost_cur):
            cost_cur = c_try
            converged = True
            break
        cost_cur = c_try

    theta_hat = to_theta(u)
    cov = None
    if jac is not None:
        try:
            n_obs, n_param = jac.shape
            dof = max(n_obs - n_param, 1)
            sigma2 = 2.0 * cost_cur / dof
            jtj = jac.T @ jac
            lu, piv = lu_factor(jtj + 1e-12 * np.eye(n_param))
            jtj_inv = lu_solve(lu, piv, np.eye(n_param))
            cov_u = sigma2 * jtj_inv
            if log_space:
                d = np.diag(theta_hat)
                cov = d @ cov_u @ d
            else:
                cov = cov_u
        except np.linalg.LinAlgError:
            cov = None

    return FitResult(
        theta_hat=theta_hat,
        cost=cost_cur,
        n_iter=it + 1,
        n_fev=n_fev,
        converged=converged,
        path=np.array(path),
        jacobian=jac,
        cov=cov,
        message=f"Levenberg-Marquardt{' (log-space)' if log_space else ''}",
    )

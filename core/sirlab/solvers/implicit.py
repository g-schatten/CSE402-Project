"""A-stable implicit solvers: backward Euler and the trapezoidal rule
(Crank-Nicolson), both fixed-step. Each implicit stage is a nonlinear
system solved by our own Newton's method, with the linear system at every
Newton iteration solved by our own LU factorization (PLAN.md section 2.3;
this is the concrete tie to the Week 5-6 LU-decomposition syllabus topic).

Backward Euler:  y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})
    G(y) = y - y_n - h f(t_{n+1}, y) = 0
    Newton: (I - h J_f(y)) delta = -G(y)

Trapezoidal:     y_{n+1} = y_n + h/2 [f(t_n,y_n) + f(t_{n+1}, y_{n+1})]
    G(y) = y - y_n - h/2 [f(t_n,y_n) + f(t_{n+1}, y)] = 0
    Newton: (I - h/2 J_f(y)) delta = -G(y)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.linalg.lu import lu_factor, lu_solve
from sirlab.models.base import Model
from sirlab.solvers.base import SolveResult, uniform_grid


def _newton_solve_stage(
    model: Model,
    t_next: float,
    theta: NDArray[np.float64],
    guess: NDArray[np.float64],
    residual_fn,
    jacobian_fn,
    *,
    tol: float = 1e-11,
    max_iter: int = 50,
) -> tuple[NDArray[np.float64], int, bool]:
    y = guess.copy()
    n_fev = 0
    converged = False
    for _ in range(max_iter):
        g = residual_fn(y)
        n_fev += 1
        if np.linalg.norm(g, ord=np.inf) < tol:
            converged = True
            break
        jac = jacobian_fn(y)
        try:
            lu, piv = lu_factor(jac)
            delta = lu_solve(lu, piv, -g)
        except np.linalg.LinAlgError:
            break
        y = y + delta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            g = residual_fn(y)
            n_fev += 1
            converged = np.linalg.norm(g, ord=np.inf) < tol * 10
            break
    return y, n_fev, converged


def integrate_backward_euler(
    model: Model, y0: NDArray[np.float64], theta: NDArray[np.float64], t_span: tuple[float, float], h: float
) -> SolveResult:
    t = uniform_grid(t_span, h)
    n = model.n_state
    y = np.empty((len(t), n))
    y[0] = y0
    n_fev = 0
    negativity_events = 0
    diverged = False
    eye = np.eye(n)
    for k in range(len(t) - 1):
        yk = y[k]
        t_next = t[k + 1]

        def residual(yg, yk=yk, t_next=t_next):
            return yg - yk - h * model.rhs(t_next, yg, theta)

        def jac(yg, t_next=t_next):
            return eye - h * model.jacobian_y(t_next, yg, theta)

        guess = yk + h * model.rhs(t[k], yk, theta)  # explicit-Euler predictor
        y_next, nf, ok = _newton_solve_stage(model, t_next, theta, guess, residual, jac)
        n_fev += nf + 1  # +1 for the predictor rhs evaluation
        y[k + 1] = y_next
        if not ok:
            diverged = True
            y[k + 2 :] = np.nan
            break
        if np.any(y_next < 0):
            negativity_events += 1
        if not np.all(np.isfinite(y_next)):
            diverged = True
            y[k + 2 :] = np.nan
            break
    return SolveResult(t=t, y=y, n_fev=n_fev, n_steps=len(t) - 1, negativity_events=negativity_events, diverged=diverged)


def integrate_trapezoidal(
    model: Model, y0: NDArray[np.float64], theta: NDArray[np.float64], t_span: tuple[float, float], h: float
) -> SolveResult:
    t = uniform_grid(t_span, h)
    n = model.n_state
    y = np.empty((len(t), n))
    y[0] = y0
    n_fev = 0
    negativity_events = 0
    diverged = False
    eye = np.eye(n)
    for k in range(len(t) - 1):
        yk = y[k]
        t_next = t[k + 1]
        f_k = model.rhs(t[k], yk, theta)
        n_fev += 1

        def residual(yg, yk=yk, t_next=t_next, f_k=f_k):
            return yg - yk - (h / 2.0) * (f_k + model.rhs(t_next, yg, theta))

        def jac(yg, t_next=t_next):
            return eye - (h / 2.0) * model.jacobian_y(t_next, yg, theta)

        guess = yk + h * f_k  # explicit-Euler predictor
        y_next, nf, ok = _newton_solve_stage(model, t_next, theta, guess, residual, jac)
        n_fev += nf
        y[k + 1] = y_next
        if not ok:
            diverged = True
            y[k + 2 :] = np.nan
            break
        if np.any(y_next < 0):
            negativity_events += 1
        if not np.all(np.isfinite(y_next)):
            diverged = True
            y[k + 2 :] = np.nan
            break
    return SolveResult(t=t, y=y, n_fev=n_fev, n_steps=len(t) - 1, negativity_events=negativity_events, diverged=diverged)

"""Adaptive Dormand-Prince RK45 (DOPRI5 / "ode45"), hand-written, with a
standard PI step-size controller and dense (4th-order) output. Used as a
high-accuracy reference-quality solver in the experiments (distinct from
the SIR-specific semi-analytic gold standard in reference.py, which only
exists for the plain SIR model -- RK45 is what we use for SEIR/SIRS/
vaccination "truth" trajectories where no closed form exists).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sirlab.models.base import Model
from sirlab.solvers.base import SolveResult

# Dormand-Prince Butcher tableau
_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])  # 5th order (=stage 7, FSAL)
_B4 = np.array(
    [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)  # 4th order embedded


def integrate_rk45(
    model: Model,
    y0: NDArray[np.float64],
    theta: NDArray[np.float64],
    t_span: tuple[float, float],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    h0: float | None = None,
    h_max: float | None = None,
    max_steps: int = 200_000,
) -> SolveResult:
    t0, tf = t_span
    direction = 1.0 if tf >= t0 else -1.0
    y = y0.astype(float).copy()
    t = t0
    n_state = model.n_state

    if h0 is None:
        f0 = model.rhs(t0, y, theta)
        scale = atol + np.abs(y) * rtol
        d0 = np.linalg.norm(y / scale)
        d1 = np.linalg.norm(f0 / scale)
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        h0 = min(h0, abs(tf - t0))
    h = direction * abs(h0)
    if h_max is not None:
        h = direction * min(abs(h), h_max)

    ts = [t0]
    ys = [y.copy()]
    n_fev = 0
    n_steps = 0
    n_rejected = 0
    h_hist = []
    order = 5

    safety = 0.9
    min_factor, max_factor = 0.2, 5.0

    f_prev = model.rhs(t, y, theta)
    n_fev += 1

    while (direction > 0 and t < tf - 1e-14) or (direction < 0 and t > tf + 1e-14):
        if (direction > 0 and t + h > tf) or (direction < 0 and t + h < tf):
            h = tf - t
        if n_steps + n_rejected > max_steps:
            break

        k = [f_prev]
        for i in range(1, 7):
            yi = y.copy()
            for j, aij in enumerate(_A[i]):
                yi = yi + h * aij * k[j]
            ki = model.rhs(t + _C[i] * h, yi, theta)
            k.append(ki)
        n_fev += 6
        k_arr = np.array(k)  # (7, n_state)

        y5 = y + h * (_B5 @ k_arr)
        y4 = y + h * (_B4 @ k_arr)
        err = y5 - y4
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        if err_norm <= 1.0 or abs(h) < 1e-14:
            t = t + h
            y = y5
            f_prev = k[6]  # FSAL: k7 evaluated at the accepted (t+h, y5)
            ts.append(t)
            ys.append(y.copy())
            h_hist.append(h)
            n_steps += 1
            factor = max_factor if err_norm == 0 else min(max_factor, max(min_factor, safety * err_norm ** (-1 / (order + 1))))
        else:
            factor = max(min_factor, safety * err_norm ** (-1 / (order + 1)))
            n_rejected += 1
        h = h * factor
        if h_max is not None:
            h = direction * min(abs(h), h_max)

    diverged = n_steps + n_rejected > max_steps or not np.all(np.isfinite(y))
    return SolveResult(
        t=np.array(ts),
        y=np.array(ys),
        n_fev=n_fev,
        n_steps=n_steps,
        n_rejected=n_rejected,
        h_history=np.array(h_hist) if h_hist else None,
        diverged=diverged,
    )

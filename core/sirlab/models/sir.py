"""The classical SIR model (PLAN.md section 2.1).

    dS/dt = -beta S I / N
    dI/dt = +beta S I / N - gamma I
    dR/dt = +gamma I

theta = (beta, gamma). State order is (S, I, R).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.models.base import Model


class SIR(Model):
    name = "SIR"
    state_names = ("S", "I", "R")
    param_names = ("beta", "gamma")

    def __init__(self, n_total: float):
        self.n_total = float(n_total)

    def rhs(self, t, y, theta):
        S, I, R = y
        beta, gamma = theta
        N = self.n_total
        infection = beta * S * I / N
        recovery = gamma * I
        return np.array([-infection, infection - recovery, recovery], dtype=y.dtype)

    def jacobian_y(self, t, y, theta):
        S, I, R = y
        beta, gamma = theta
        N = self.n_total
        return np.array(
            [
                [-beta * I / N, -beta * S / N, 0.0],
                [beta * I / N, beta * S / N - gamma, 0.0],
                [0.0, gamma, 0.0],
            ],
            dtype=float,
        )

    def jacobian_theta(self, t, y, theta):
        S, I, R = y
        N = self.n_total
        # columns: d/d(beta), d/d(gamma)
        return np.array(
            [
                [-S * I / N, 0.0],
                [S * I / N, -I],
                [0.0, I],
            ],
            dtype=float,
        )

    def r0(self, theta):
        beta, gamma = theta
        return float(beta / gamma)

    def initial_state(self, *, n, i0, e0=0.0, r0=0.0, v0=0.0):
        s0 = n - i0 - r0
        return np.array([s0, i0, r0], dtype=float)

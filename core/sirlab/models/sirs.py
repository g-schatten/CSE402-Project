"""SIRS model: SIR plus waning immunity, allowing endemic re-circulation.

    dS/dt = -beta S I / N + xi R
    dI/dt = +beta S I / N - gamma I
    dR/dt = +gamma I - xi R

theta = (beta, gamma, xi). State order is (S, I, R). xi = 0 reduces exactly
to SIR, which is asserted in tests/test_models.py.
"""
from __future__ import annotations

import numpy as np

from sirlab.models.base import Model


class SIRS(Model):
    name = "SIRS"
    state_names = ("S", "I", "R")
    param_names = ("beta", "gamma", "xi")

    def __init__(self, n_total: float):
        self.n_total = float(n_total)

    def rhs(self, t, y, theta):
        S, I, R = y
        beta, gamma, xi = theta
        N = self.n_total
        infection = beta * S * I / N
        recovery = gamma * I
        waning = xi * R
        return np.array([-infection + waning, infection - recovery, recovery - waning], dtype=y.dtype)

    def jacobian_y(self, t, y, theta):
        S, I, R = y
        beta, gamma, xi = theta
        N = self.n_total
        return np.array(
            [
                [-beta * I / N, -beta * S / N, xi],
                [beta * I / N, beta * S / N - gamma, 0.0],
                [0.0, gamma, -xi],
            ],
            dtype=float,
        )

    def jacobian_theta(self, t, y, theta):
        S, I, R = y
        return np.array(
            [
                [-S * I / self.n_total, 0.0, R],
                [S * I / self.n_total, -I, 0.0],
                [0.0, I, -R],
            ],
            dtype=float,
        )

    def r0(self, theta):
        beta, gamma, xi = theta
        return float(beta / gamma)

    def initial_state(self, *, n, i0, e0=0.0, r0=0.0, v0=0.0):
        s0 = n - i0 - r0
        return np.array([s0, i0, r0], dtype=float)

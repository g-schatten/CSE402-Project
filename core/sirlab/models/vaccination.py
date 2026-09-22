"""SIR model with ongoing vaccination of susceptibles into an absorbing V class.

    dS/dt = -beta S I / N - nu S
    dI/dt = +beta S I / N - gamma I
    dR/dt = +gamma I
    dV/dt = +nu S

theta = (beta, gamma, nu). State order is (S, I, R, V). V counts toward N.
"""
from __future__ import annotations

import numpy as np

from sirlab.models.base import Model


class SIRVaccination(Model):
    name = "SIR-V"
    state_names = ("S", "I", "R", "V")
    param_names = ("beta", "gamma", "nu")

    def __init__(self, n_total: float):
        self.n_total = float(n_total)

    def rhs(self, t, y, theta):
        S, I, R, V = y
        beta, gamma, nu = theta
        N = self.n_total
        infection = beta * S * I / N
        recovery = gamma * I
        vaccination = nu * S
        return np.array([-infection - vaccination, infection - recovery, recovery, vaccination], dtype=y.dtype)

    def jacobian_y(self, t, y, theta):
        S, I, R, V = y
        beta, gamma, nu = theta
        N = self.n_total
        return np.array(
            [
                [-beta * I / N - nu, -beta * S / N, 0.0, 0.0],
                [beta * I / N, beta * S / N - gamma, 0.0, 0.0],
                [0.0, gamma, 0.0, 0.0],
                [nu, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def jacobian_theta(self, t, y, theta):
        S, I, R, V = y
        N = self.n_total
        return np.array(
            [
                [-S * I / N, 0.0, -S],
                [S * I / N, -I, 0.0],
                [0.0, I, 0.0],
                [0.0, 0.0, S],
            ],
            dtype=float,
        )

    def r0(self, theta):
        beta, gamma, nu = theta
        return float(beta / gamma)

    def initial_state(self, *, n, i0, e0=0.0, r0=0.0, v0=0.0):
        s0 = n - i0 - r0 - v0
        return np.array([s0, i0, r0, v0], dtype=float)

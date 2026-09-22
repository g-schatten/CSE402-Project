"""SEIR model: SIR plus an exposed (latent, non-infectious) compartment.

    dS/dt = -beta S I / N
    dE/dt = +beta S I / N - sigma_inc E
    dI/dt = +sigma_inc E - gamma I
    dR/dt = +gamma I

theta = (beta, sigma_inc, gamma). State order is (S, E, I, R).
R0 = beta / gamma (the latent stage delays but does not change R0).
"""
from __future__ import annotations

import numpy as np

from sirlab.models.base import Model


class SEIR(Model):
    name = "SEIR"
    state_names = ("S", "E", "I", "R")
    param_names = ("beta", "sigma_inc", "gamma")

    def __init__(self, n_total: float):
        self.n_total = float(n_total)

    def rhs(self, t, y, theta):
        S, E, I, R = y
        beta, sigma_inc, gamma = theta
        N = self.n_total
        infection = beta * S * I / N
        progression = sigma_inc * E
        recovery = gamma * I
        return np.array(
            [-infection, infection - progression, progression - recovery, recovery],
            dtype=y.dtype,
        )

    def jacobian_y(self, t, y, theta):
        S, E, I, R = y
        beta, sigma_inc, gamma = theta
        N = self.n_total
        return np.array(
            [
                [-beta * I / N, 0.0, -beta * S / N, 0.0],
                [beta * I / N, -sigma_inc, beta * S / N, 0.0],
                [0.0, sigma_inc, -gamma, 0.0],
                [0.0, 0.0, gamma, 0.0],
            ],
            dtype=float,
        )

    def jacobian_theta(self, t, y, theta):
        S, E, I, R = y
        N = self.n_total
        return np.array(
            [
                [-S * I / N, 0.0, 0.0],
                [S * I / N, -E, 0.0],
                [0.0, E, -I],
                [0.0, 0.0, I],
            ],
            dtype=float,
        )

    def r0(self, theta):
        beta, sigma_inc, gamma = theta
        return float(beta / gamma)

    def initial_state(self, *, n, i0, e0=0.0, r0=0.0, v0=0.0):
        s0 = n - e0 - i0 - r0
        return np.array([s0, e0, i0, r0], dtype=float)

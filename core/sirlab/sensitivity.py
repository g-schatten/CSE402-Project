"""Forward sensitivity equations (PLAN.md section 2.4).

Estimation needs d y / d theta. We do NOT use finite differences for this:
we augment the state with the sensitivities s_j = dy/dtheta_j and integrate
them with the *same solver* used for the trajectory, so solver error enters
the Jacobian exactly the way it enters the state -- which is precisely what
RQ2 asks about.

    d(s_j)/dt = J_y f(y) @ s_j + df/dtheta_j,     s_j(0) = 0

The augmented state is [y (n_state) ; s_1 ; s_2 ; ... ; s_p], each s_j of
length n_state, stacked so the augmented system has n_state * (1 + n_param)
components. It exposes a `.rhs` method and `.n_state` attribute so it can be
handed directly to any solver in sirlab.solvers without modification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sirlab.models.base import Model
from sirlab.solvers import integrate


@dataclass
class _AugmentedSystem:
    model: Model
    base_n_state: int
    n_param: int

    @property
    def n_state(self) -> int:
        """Total augmented dimension: the solvers key their array allocation
        off this, not off the base model's state count."""
        return self.base_n_state * (1 + self.n_param)

    def rhs(self, t: float, y_aug: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
        n, p = self.base_n_state, self.n_param
        y = y_aug[:n]
        f = self.model.rhs(t, y, theta)
        jy = self.model.jacobian_y(t, y, theta)
        jtheta = self.model.jacobian_theta(t, y, theta)
        out = np.empty_like(y_aug)
        out[:n] = f
        for j in range(p):
            s_j = y_aug[n + j * n : n + (j + 1) * n]
            out[n + j * n : n + (j + 1) * n] = jy @ s_j + jtheta[:, j]
        return out


def integrate_with_sensitivities(
    model: Model,
    y0: NDArray[np.float64],
    theta: NDArray[np.float64],
    t_span: tuple[float, float],
    *,
    solver: str = "rk4",
    h: float | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Returns (SolveResult on the augmented system). Extract the plain
    trajectory with `.y[:, :n_state]` and the sensitivity of state i to
    parameter j at time index k with `.y[k, n_state + j*n_state + i]`, or
    use `unpack()` below for a friendlier shape.
    """
    n_state = model.n_state
    n_param = model.n_param
    aug = _AugmentedSystem(model=model, base_n_state=n_state, n_param=n_param)
    y0_aug = np.concatenate([y0, np.zeros(n_state * n_param)])
    return integrate(aug, y0_aug, theta, t_span, solver=solver, h=h, rtol=rtol, atol=atol)


def unpack(result, n_state: int, n_param: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split an augmented-system SolveResult into (y, sens) where y has
    shape (n_t, n_state) and sens has shape (n_t, n_state, n_param)."""
    y = result.y[:, :n_state]
    n_t = result.y.shape[0]
    sens = np.empty((n_t, n_state, n_param))
    for j in range(n_param):
        sens[:, :, j] = result.y[:, n_state + j * n_state : n_state + (j + 1) * n_state]
    return y, sens

"""Common interface every compartmental model implements.

Every model is an autonomous ODE system dy/dt = f(t, y; theta) on a fixed
total population N = sum(y). Models must supply analytic Jacobians
(d f / d y and d f / d theta) -- these feed the sensitivity equations
(sensitivity.py) and the Gauss-Newton / Levenberg-Marquardt estimators.
Jacobians are unit-tested against complex-step differentiation in
tests/test_models.py; finite differences are deliberately not used anywhere
in this codebase (see PLAN.md section 2.4).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


class Model(ABC):
    """Abstract base for a compartmental epidemic model.

    Subclasses fix `state_names` and `param_names` (order matters -- it
    defines the vector layout used everywhere else) and implement `rhs`,
    `jacobian_y`, and `jacobian_theta`.
    """

    name: str
    state_names: tuple[str, ...]
    param_names: tuple[str, ...]

    @property
    def n_state(self) -> int:
        return len(self.state_names)

    @property
    def n_param(self) -> int:
        return len(self.param_names)

    @abstractmethod
    def rhs(self, t: float, y: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """dy/dt at (t, y) given parameters theta. Shape (n_state,) in, out."""

    @abstractmethod
    def jacobian_y(self, t: float, y: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """d f / d y, shape (n_state, n_state)."""

    @abstractmethod
    def jacobian_theta(self, t: float, y: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """d f / d theta, shape (n_state, n_param)."""

    def rhs_complex(self, t: float, y: NDArray[np.complexfloating], theta: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
        """Complex-safe re-evaluation of rhs, used only by the complex-step
        differentiation tests. Default implementation just calls rhs(); models
        whose rhs uses numpy ufuncs that support complex dtypes (add, mul, exp)
        need no override, which is the case for every model in this package.
        """
        return self.rhs(t, y, theta)

    def r0(self, theta: NDArray[np.float64]) -> float:
        """Basic reproduction number, where defined. Overridden per model."""
        raise NotImplementedError

    def initial_state(self, *, n: float, i0: float, e0: float = 0.0, r0: float = 0.0, v0: float = 0.0) -> NDArray[np.float64]:
        raise NotImplementedError


@dataclass
class ModelSpec:
    """A resolved (model, theta, y0, N) bundle used to configure experiments."""

    model: Model
    theta: NDArray[np.float64]
    y0: NDArray[np.float64]
    n_total: float
    extra: dict = field(default_factory=dict)

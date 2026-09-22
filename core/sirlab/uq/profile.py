"""Profile likelihood (PLAN.md section 2.7): fix one parameter, re-optimize
the other, plot cost vs the fixed value. A flat profile is the visual
signature of non-identifiability -- this is what makes the truncated
observation-window experiment (E09) legible.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import ForwardModel
from sirlab.estimate.golden import golden_section_1d


def profile_likelihood_1d(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    fixed_index: int,
    fixed_values: NDArray[np.float64],
    other_bounds: tuple[float, float],
    *,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """For each value v in fixed_values, fix theta[fixed_index]=v and
    minimize cost over the remaining parameter via golden-section search.
    Only supports the 2-parameter case (as used throughout this project).
    Returns the array of minimized costs, same length as fixed_values."""
    other_index = 1 - fixed_index
    costs = np.empty(len(fixed_values))
    for i, v in enumerate(fixed_values):

        def f1d(other_val, v=v):
            theta = np.empty(2)
            theta[fixed_index] = v
            theta[other_index] = other_val
            return fwd.cost(theta, obs, weights)

        _, cost_star, _ = golden_section_1d(f1d, *other_bounds, tol=1e-7)
        costs[i] = cost_star
    return costs

"""Chi-squared quantile for 2 degrees of freedom, in closed form (no SciPy
needed): the chi-squared(2) CDF is F(x) = 1 - exp(-x/2), so the p-quantile
is x_p = -2 ln(1-p). Used only for the 2-parameter (beta, gamma) confidence
ellipse in uq/asymptotic.py.
"""
from __future__ import annotations

import numpy as np


def chi2_quantile_2df(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return float(-2.0 * np.log(1.0 - p))

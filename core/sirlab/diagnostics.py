"""Error norms, observed convergence order, and stability/invariant
diagnostics (PLAN.md section 2.8).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.linalg.eigen import spectral_radius


def error_norms(y_approx: NDArray[np.float64], y_true: NDArray[np.float64]) -> dict[str, float]:
    """Global error at the final time, max-over-time (L-infinity), and
    time-L2 norm, computed per-compartment-summed. y_approx, y_true: (n_t, n_state)."""
    diff = y_approx - y_true
    return {
        "global_final": float(np.linalg.norm(diff[-1])),
        "linf": float(np.max(np.linalg.norm(diff, axis=1))),
        "l2_time": float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))),
    }


def observed_order(errors: NDArray[np.float64], hs: NDArray[np.float64]) -> float:
    """Fit p in error ~ C h^p via a log-log least-squares slope (our own
    normal-equations fit, not np.polyfit, to keep the estimate traceable to
    this project's own least-squares machinery)."""
    mask = (errors > 0) & np.isfinite(errors)
    if mask.sum() < 2:
        return float("nan")
    x = np.log(hs[mask])
    y = np.log(errors[mask])
    xm, ym = x.mean(), y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = np.sum((x - xm) ** 2)
    return float(num / den) if den > 0 else float("nan")


def richardson_order(y_h: NDArray[np.float64], y_h2: NDArray[np.float64], y_h4: NDArray[np.float64]) -> float:
    """Observed order from three successively halved step sizes with no
    reference solution: p = log2( ||y_h - y_h2|| / ||y_h2 - y_h4|| )."""
    num = np.linalg.norm(y_h - y_h2)
    den = np.linalg.norm(y_h2 - y_h4)
    if den <= 0 or num <= 0:
        return float("nan")
    return float(np.log2(num / den))


def mass_error(y: NDArray[np.float64], n_total: float) -> NDArray[np.float64]:
    """|sum(compartments) - N| at every time point. Deliberately included as
    a diagnostic even though PLAN.md 2.8 shows it has ~zero discriminating
    power for SIR (every explicit RK method conserves it to roundoff at any
    h, because the RHS sums to zero identically) -- the point is to
    *demonstrate* that emptiness, not to rely on it."""
    return np.abs(np.sum(y, axis=1) - n_total)


def phase_invariant_drift(s: NDArray[np.float64], r: NDArray[np.float64], beta: float, gamma: float, n_total: float) -> NDArray[np.float64]:
    """|Q(t) - Q(0)| using the SIR phase invariant (reference.py eq. 2.1).
    Unlike mass conservation, this drifts at the solver's convergence order
    and is a genuine solver-agnostic accuracy proxy."""
    k = beta / (gamma * n_total)
    q = np.log(np.clip(s, 1e-300, None)) + k * r
    return np.abs(q - q[0])


def stability_metric(h: float, jac_y: NDArray[np.float64]) -> float:
    """h * rho(J), rho the largest eigenvalue modulus of the local Jacobian --
    used to test a trajectory point against a solver's stability region. rho
    comes from our power method (linalg/eigen.py), which also handles the
    complex-conjugate eigenvalue pair the SIR Jacobian has near the peak."""
    return float(h * spectral_radius(jac_y))


def work_precision_fevals(n_steps: int, fevals_per_step: int) -> int:
    return n_steps * fevals_per_step

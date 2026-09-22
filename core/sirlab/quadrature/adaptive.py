"""Adaptive Simpson quadrature (hand-written), used by reference.py to evaluate
the SIR gold-standard integral (PLAN.md eq. 2.2) to near machine precision.
"""
from __future__ import annotations

from collections.abc import Callable


def _simpson(f: Callable[[float], float], a: float, b: float, fa: float, fm: float, fb: float) -> float:
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb)


def adaptive_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-13,
    max_depth: int = 40,
    max_evals: int = 200_000,
) -> float:
    """Integrate f on [a, b] to (approximately) absolute tolerance `tol`.

    Iterative (worklist-based, not recursive) adaptive Simpson's rule, so a
    pathological integrand -- e.g. the near-singular tail of the SIR
    quadrature (2.2) close to R_infinity, where I -> 0 and the integrand
    blows up -- cannot cause unbounded recursion or exponential blow-up. Once
    `max_evals` sub-interval evaluations are spent, remaining panels are
    accepted at their current estimate rather than subdivided further: the
    result degrades gracefully near a singularity instead of hanging.

    A second safeguard fixes a subtler failure mode: naive tol-halving per
    split drives the per-panel error budget to tol / 2**depth, which
    underflows below floating-point roundoff (~1e-16 relative) long before
    max_depth is reached whenever the caller asks for a very tight absolute
    tol (this project's callers do: reference.py wants ~1e-13). Once that
    happens the stopping criterion can never be satisfied on the noise
    floor, so panels get force-terminated at max_depth *without having
    converged*, silently contributing O(panel count x noise) accumulated
    error -- exactly the kind of bug this quadrature exists to avoid
    introducing elsewhere. We therefore also accept a panel once its error
    estimate is already at the roundoff floor for its own magnitude,
    regardless of how small the halved tol_ budget has become.
    """
    if a == b:
        return 0.0
    fa, fb = f(a), f(b)
    m0 = 0.5 * (a + b)
    fm0 = f(m0)
    whole0 = _simpson(f, a, b, fa, fm0, fb)

    total = 0.0
    evals = 3
    # stack of panels: (a, b, fa, fm, fb, whole_estimate, tol_budget, depth)
    stack = [(a, b, fa, fm0, fb, whole0, tol, max_depth)]
    while stack:
        a_, b_, fa_, fm_, fb_, whole, tol_, depth = stack.pop()
        m = 0.5 * (a_ + b_)
        lm = 0.5 * (a_ + m)
        rm = 0.5 * (m + b_)
        flm = f(lm)
        frm = f(rm)
        evals += 2
        left = _simpson(f, a_, m, fa_, flm, fm_)
        right = _simpson(f, m, b_, fm_, frm, fb_)
        refined = left + right
        err_est = abs(refined - whole)
        roundoff_floor = 1e-14 * max(abs(refined), abs(whole), 1.0)
        if depth <= 0 or evals >= max_evals or err_est <= max(15.0 * tol_, roundoff_floor):
            total += refined + (refined - whole) / 15.0
            continue
        stack.append((a_, m, fa_, flm, fm_, left, tol_ / 2.0, depth - 1))
        stack.append((m, b_, fm_, frm, fb_, right, tol_ / 2.0, depth - 1))
    return total


def composite_simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Fixed-grid composite Simpson's rule with n subintervals (n must be even).
    Used for incidence integration on a solver's own uniform grid (PLAN.md
    section 2.5), where the grid, not accuracy-adaptivity, is the point.
    """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += (4.0 if i % 2 else 2.0) * f(x)
    return total * h / 3.0

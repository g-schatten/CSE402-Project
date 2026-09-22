"""Scalar Newton's method and bisection, hand-written.

Used by reference.py to invert t(R) (Newton, with a bisection fallback for
safety near the boundary) and to solve the final-size equation (2.4).
"""
from __future__ import annotations

from collections.abc import Callable


def newton(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> float:
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = fprime(x)
        if dfx == 0.0:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return x_new
        x = x_new
    return x


def bisection(f: Callable[[float], float], lo: float, hi: float, *, tol: float = 1e-13, max_iter: int = 200) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("bisection requires a sign change on [lo, hi]")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def newton_safe(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-13,
    max_iter: int = 100,
) -> float:
    """Newton's method safeguarded by bisection (the standard "rtsafe" scheme):
    a Newton step is taken only when it stays inside the current bracket and
    at least halves the bracket width; otherwise we bisect. This guarantees
    monotone bracket shrinkage every iteration (unlike a naive safeguard that
    can fall back to the *same* bisection point twice and falsely report
    convergence). Used for the mildly nonlinear equations this library solves
    (final size, t(R) inversion)."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("newton_safe requires a sign change on [lo, hi]")

    # Orient so that f(lo) < 0 < f(hi); track lo/hi purely as the bracket,
    # regardless of which endpoint originally held which sign.
    if flo > 0:
        lo, hi = hi, lo

    x = 0.5 * (lo + hi)
    dx_old = abs(hi - lo)
    dx = dx_old
    fx = f(x)
    dfx = fprime(x)

    for _ in range(max_iter):
        newton_out_of_bracket = ((x - hi) * dfx - fx) * ((x - lo) * dfx - fx) > 0
        newton_too_slow = abs(2.0 * fx) > abs(dx_old * dfx)
        if dfx == 0.0 or newton_out_of_bracket or newton_too_slow:
            dx_old = dx
            dx = 0.5 * (hi - lo)
            x_new = lo + dx
            if x_new == lo:
                return x_new
        else:
            dx_old = dx
            dx = fx / dfx
            x_new = x - dx
            if x_new == x:
                return x_new

        if abs(dx) < tol:
            return x_new

        x = x_new
        fx = f(x)
        dfx = fprime(x)
        if fx < 0:
            lo = x
        else:
            hi = x

    return x

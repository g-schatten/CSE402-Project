"""The semi-analytic SIR gold standard (PLAN.md section 2.2).

SIR admits an exact reduction that needs no ODE solver at all:

    dS/dR = -(beta / (gamma N)) S                       (phase relation)
    =>  S(t) = S0 * exp( -(beta/(gamma N)) * (R(t) - R0_init) )     (2.1)
    =>  Q(t) := ln S(t) + (beta/(gamma N)) * R(t)  is constant in t

Substituting S(R) into dR/dt = gamma*I = gamma*(N - R - S) gives a *scalar*
separable ODE in R alone, so

    t(R) = integral_0^R  dr / [ gamma * (N - r - S0 * exp(-(beta/(gammaN)) r)) ]   (2.2)

The integrand is smooth and strictly positive on [0, R_infinity), but grows
without bound as R -> R_infinity (I -> 0 there, logarithmically in time --
full recovery is only reached asymptotically). We therefore build the
inverse map t -> R in two stages:

  1. A monotone table of (rho, t) pairs is built ONCE per parameter set by
     summing many small, fast, near-machine-precision sub-integrals over a
     grid clustered at both ends (Chebyshev-like spacing) -- dense where the
     integrand varies fastest (rho near 0) and dense approaching the
     asymptote (rho near the gap, clipped well short of it).
  2. A query t is located in the table (binary search) for a *narrow*
     bracket, refined by a single short quadrature-based safeguarded-Newton
     solve local to that bracket (few iterations, tiny sub-integral each).

This is both accurate (each table cell and each refinement step individually
hits ~1e-13-relative quadrature tolerance) and fast (no query ever
integrates over more than one table cell from scratch). Every RQ1 error
measurement in this project is made against this reference, never against
"RK4 at a very small h" (which would be circular). See
tests/test_reference.py for the self-consistency gate that must pass before
any other experiment is trusted.

Also implemented here: the closed-form peak height (2.3) and final size
(2.4), and the phase-invariant Q(t) (2.1) used as a solver-agnostic accuracy
diagnostic in place of the (numerically useless, see PLAN.md 2.8) mass
conservation check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sirlab.quadrature.adaptive import adaptive_simpson
from sirlab.roots.newton import newton_safe

_TABLE_N = 2000  # table cells for the cached (rho, t) monotone map


@dataclass
class SIRReference:
    beta: float
    gamma: float
    n_total: float
    s0: float
    i0: float
    r0_init: float = 0.0

    _table_rho: NDArray[np.float64] | None = field(default=None, init=False, repr=False, compare=False)
    _table_t: NDArray[np.float64] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def r0_number(self) -> float:
        """Basic reproduction number beta/gamma (not to be confused with
        r0_init, the initial value of the R compartment)."""
        return self.beta / self.gamma

    @property
    def r_infinity(self) -> float:
        """Total ever-recovered as t -> infinity, i.e. R(inf). This is the
        final size S_infinity translated into R units: R_inf = N - S_inf."""
        return self.n_total - self.s_infinity

    # ---- eq. (2.1): phase relation ----------------------------------------
    def s_of_r(self, r: float | NDArray) -> float | NDArray:
        k = self.beta / (self.gamma * self.n_total)
        return self.s0 * np.exp(-k * (np.asarray(r) - self.r0_init))

    def phase_invariant(self, s: float, r: float) -> float:
        """Q(t) = ln S + (beta/(gamma N)) R, constant along any true
        trajectory. Drift of Q from its t=0 value is our solver-agnostic
        accuracy diagnostic (PLAN.md 2.8)."""
        k = self.beta / (self.gamma * self.n_total)
        return float(np.log(s) + k * r)

    # ---- eq. (2.2): t(R) via quadrature ------------------------------------
    def _dt_drho(self, rho: float) -> float:
        """dt/drho at offset rho = r - r0_init (rho >= 0)."""
        s = self.s_of_r(self.r0_init + rho)
        i = self.n_total - (self.r0_init + rho) - s
        i = max(float(i), 1e-300)
        return 1.0 / (self.gamma * i)

    def _dt_dr(self, r: float) -> float:
        return self._dt_drho(r - self.r0_init)

    def _ensure_table(self) -> None:
        if self._table_t is not None:
            return
        gap = self.r_infinity - self.r0_init
        # Clip well short of the asymptote (integrand -> infinity there).
        rho_max = gap * (1.0 - 1e-9)
        n = _TABLE_N
        # Chebyshev-like clustering at both ends of [0, rho_max].
        u = np.linspace(0.0, np.pi, n + 1)
        rho_grid = rho_max * (1.0 - np.cos(u)) / 2.0
        t_grid = np.empty(n + 1)
        t_grid[0] = 0.0
        for k in range(n):
            t_grid[k + 1] = t_grid[k] + adaptive_simpson(
                self._dt_drho, rho_grid[k], rho_grid[k + 1], tol=1e-13, max_depth=40, max_evals=2000
            )
        self._table_rho = rho_grid
        self._table_t = t_grid

    def t_of_r(self, r: float, *, tol: float = 1e-13) -> float:
        """Direct (uncached) quadrature evaluation of t(R). Used for
        one-off queries and by tests; r_of_t uses the cached table + local
        refinement for speed."""
        rho = r - self.r0_init
        if rho <= 0:
            return 0.0
        return adaptive_simpson(self._dt_drho, 0.0, rho, tol=tol)

    def r_of_t(self, t: float, *, tol: float = 1e-13) -> float:
        """Invert t(R) = t for R (dR/dt = gamma*I > 0, so t(R) is strictly
        increasing and the inversion is well posed). See module docstring
        for the table + local-refinement strategy."""
        if t <= 0:
            return self.r0_init
        self._ensure_table()
        rho_grid, t_grid = self._table_rho, self._table_t
        assert rho_grid is not None and t_grid is not None

        if t >= t_grid[-1]:
            # Beyond the table's clipped range: R is already
            # indistinguishable from R_infinity at this resolution.
            return self.r0_init + rho_grid[-1]

        idx = int(np.searchsorted(t_grid, t, side="right"))
        idx = max(1, min(idx, len(t_grid) - 1))
        rho_lo, rho_hi = rho_grid[idx - 1], rho_grid[idx]
        t_lo, t_hi = t_grid[idx - 1], t_grid[idx]

        # f(rho) = t_lo + integral_{rho_lo}^{rho} dt/drho' - t, restricted to
        # this one narrow table cell -- cheap, and rtsafe converges in a
        # handful of iterations from a bracket this tight.
        def f(rho: float) -> float:
            return t_lo + adaptive_simpson(self._dt_drho, rho_lo, rho, tol=tol, max_evals=2000) - t

        def fprime(rho: float) -> float:
            return self._dt_drho(rho)

        rho_root = newton_safe(f, fprime, rho_lo, rho_hi, tol=tol * max(1.0, t))
        return self.r0_init + rho_root

    def trajectory(self, t_eval: NDArray[np.float64], *, tol: float = 1e-13) -> NDArray[np.float64]:
        """Return the reference (S, I, R) trajectory at the given times,
        shape (len(t_eval), 3)."""
        out = np.empty((len(t_eval), 3))
        for idx, t in enumerate(t_eval):
            r = self.r_of_t(float(t), tol=tol)
            s = float(self.s_of_r(r))
            i = self.n_total - s - r
            out[idx] = (s, i, r)
        return out

    # ---- eq. (2.3): closed-form epidemic peak -----------------------------
    @property
    def peak_i(self) -> float:
        r0n = self.r0_number
        n, s0, i0 = self.n_total, self.s0, self.i0
        s_at_peak = n / r0n
        return i0 + s0 - s_at_peak + s_at_peak * np.log(n / (r0n * s0))

    @property
    def peak_s(self) -> float:
        return self.n_total / self.r0_number

    def peak_time(self, *, tol: float = 1e-13) -> float:
        """No closed form for the peak *time* -- it comes from the same
        quadrature machinery: peak occurs when S(t) = N/R0, i.e. at
        R(t_peak) = R such that S_of_r(R) = N/R0."""
        r0n = self.r0_number
        s_target = self.n_total / r0n
        k = self.beta / (self.gamma * self.n_total)
        r_at_peak = self.r0_init - np.log(s_target / self.s0) / k
        return self.t_of_r(float(r_at_peak), tol=tol)

    # ---- eq. (2.4): final size equation, solved by safeguarded Newton -----
    @property
    def s_infinity(self) -> float:
        r0n = self.r0_number
        n = self.n_total

        def g(s: float) -> float:
            return np.log(self.s0 / s) - r0n * (n - s) / n

        def gprime(s: float) -> float:
            return -1.0 / s + r0n / n

        lo = 1e-9 * n
        hi = self.s0
        return newton_safe(g, gprime, lo, hi, tol=1e-13)

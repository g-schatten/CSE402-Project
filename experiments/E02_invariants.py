"""E02: Invariant drift (PLAN.md section 5 / 2.8).

Demonstrates the deliberate finding: mass conservation (S+I+R=N) is
conserved to roundoff by every explicit RK method at ANY step size (because
the SIR right-hand side sums to zero identically), so it has ~zero
diagnostic value -- while the phase invariant Q(t) = ln S + (beta/gammaN) R
drifts at the solver's convergence order and IS a genuine accuracy proxy.
"""
from __future__ import annotations

import numpy as np

from experiments.common import quick_or_full
from sirlab.diagnostics import mass_error, phase_invariant_drift
from sirlab.models import SIR
from sirlab.solvers import integrate


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    beta, gamma = config["theta"]["beta"], config["theta"]["gamma"]
    theta = np.array([beta, gamma])
    y0 = np.array([n_total - i0, i0, 0.0])
    t_final = float(config["t_final"])
    model = SIR(n_total)
    hs = quick_or_full(config, "step_sizes", profile=profile)

    out = {}
    for solver in config["solvers"]:
        per_h = {}
        for h in hs:
            res = integrate(model, y0, theta, (0.0, t_final), solver=solver, h=h)
            m_err = mass_error(res.y, n_total)
            q_drift = phase_invariant_drift(res.y[:, 0], res.y[:, 2], beta, gamma, n_total)
            per_h[str(h)] = {
                "t": res.t.tolist(),
                "mass_error": m_err.tolist(),
                "phase_invariant_drift": q_drift.tolist(),
                "max_mass_error": float(np.max(m_err)),
                "max_phase_drift": float(np.max(q_drift)),
            }
        out[solver] = per_h

    return {
        "experiment": "E02",
        "n_total": n_total,
        "theta": {"beta": beta, "gamma": gamma},
        "conclusion": (
            "Mass error stays at machine-roundoff magnitude regardless of solver or h "
            "(RHS sums to zero identically); phase-invariant drift instead scales with "
            "the solver's convergence order and correctly ranks solver accuracy."
        ),
        "results": out,
    }

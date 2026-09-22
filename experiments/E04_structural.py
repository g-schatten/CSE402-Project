"""E04: Structural accuracy (PLAN.md section 5).

Peak height, peak time, and final-size errors vs the closed-form quantities
in reference.py (eq. 2.3 / 2.4), as a function of solver and h -- these are
"the numbers a policymaker actually reads off an epidemic curve", distinct
from the abstract state-vector error norms in E01.
"""
from __future__ import annotations

import numpy as np

from experiments.common import quick_or_full, sir_reference_from_config
from sirlab.models import SIR
from sirlab.solvers import integrate


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    theta = np.array([config["theta"]["beta"], config["theta"]["gamma"]])
    y0 = np.array([n_total - i0, i0, 0.0])
    model = SIR(n_total)
    ref = sir_reference_from_config(config)
    true_peak_i, true_peak_t, true_s_inf = ref.peak_i, ref.peak_time(), ref.s_infinity
    t_final_long = true_peak_t * 6  # comfortably past the peak for final-size convergence
    hs = quick_or_full(config, "step_sizes", profile=profile)

    out = {}
    for solver in config["solvers"]:
        peak_i_err, peak_t_err, final_size_err = [], [], []
        for h in hs:
            res = integrate(model, y0, theta, (0.0, t_final_long), solver=solver, h=h)
            idx_peak = int(np.argmax(res.y[:, 1]))
            peak_i_err.append(float(abs(res.y[idx_peak, 1] - true_peak_i)))
            peak_t_err.append(float(abs(res.t[idx_peak] - true_peak_t)))
            final_size_err.append(float(abs(res.y[-1, 0] - true_s_inf)))
        out[solver] = {
            "h": list(hs),
            "peak_i_error": peak_i_err,
            "peak_t_error": peak_t_err,
            "final_size_error": final_size_err,
        }

    return {
        "experiment": "E04",
        "true_peak_i": true_peak_i,
        "true_peak_t": true_peak_t,
        "true_s_infinity": true_s_inf,
        "t_final_long": t_final_long,
        "solvers": out,
    }

"""E01: Convergence & order (PLAN.md section 5).

For each (regime, solver, h): integrate SIR and compare the final state
against the semi-analytic gold standard (reference.py). Fits the observed
convergence order via log-log least squares and reports work-precision
(error vs cumulative f-evaluations) alongside error vs h.
"""
from __future__ import annotations

import numpy as np

from experiments.common import quick_or_full, sir_reference_from_config
from sirlab.diagnostics import observed_order
from sirlab.models import SIR
from sirlab.solvers import SOLVER_FEV_PER_STEP, integrate


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    i0 = float(config["i0"])
    t_final = float(config["t_final"])
    solvers = config["solvers"]
    hs = np.array(quick_or_full(config, "step_sizes", profile=profile), dtype=float)

    regimes_out = []
    for regime in config["regimes"]:
        theta = np.array([regime["theta"]["beta"], regime["theta"]["gamma"]])
        y0 = np.array([n_total - i0, i0, 0.0])
        model = SIR(n_total)
        ref = sir_reference_from_config({"n_total": n_total, "i0": i0, "theta": regime["theta"]})
        true_final = ref.trajectory(np.array([t_final]))[0]
        true_peak_i = ref.peak_i
        true_peak_t = ref.peak_time()
        true_s_inf = ref.s_infinity

        solver_results = {}
        for solver in solvers:
            errs, fevs, peak_err_i, peak_err_t = [], [], [], []
            for h in hs:
                res = integrate(model, y0, theta, (0.0, t_final), solver=solver, h=h)
                err = float(np.linalg.norm(res.y[-1] - true_final))
                errs.append(err)
                fevs.append(res.n_fev)
                i_peak_idx = int(np.argmax(res.y[:, 1]))
                peak_err_i.append(float(abs(res.y[i_peak_idx, 1] - true_peak_i)))
                peak_err_t.append(float(abs(res.t[i_peak_idx] - true_peak_t)))
            errs = np.array(errs)
            p_hat = observed_order(errs, hs)
            solver_results[solver] = {
                "h": hs.tolist(),
                "error_final_state": errs.tolist(),
                "n_fev": fevs,
                "observed_order": p_hat,
                "theoretical_order": {"euler": 1, "heun": 2, "rk4": 4, "backward_euler": 1, "trapezoidal": 2}[solver],
                "peak_i_error": peak_err_i,
                "peak_t_error": peak_err_t,
            }

        regimes_out.append(
            {
                "name": regime["name"],
                "theta": regime["theta"],
                "true_final_state": true_final.tolist(),
                "true_peak_i": true_peak_i,
                "true_peak_t": true_peak_t,
                "true_s_infinity": true_s_inf,
                "solvers": solver_results,
            }
        )

    return {"experiment": "E01", "n_total": n_total, "i0": i0, "t_final": t_final, "regimes": regimes_out}

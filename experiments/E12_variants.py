"""E12: Model variants (PLAN.md section 5).

Repeats the E01-style convergence-order measurement on SEIR / SIRS /
SIR-Vaccination, none of which has a closed-form solution -- so the "truth"
here is a tight adaptive RK45 solve (rtol=atol~1e-12), not the SIR-only
semi-analytic gold standard. SIR itself is included as the baseline row
(using the true gold standard) so the table is directly comparable.
"""
from __future__ import annotations

import numpy as np

from experiments.common import quick_or_full
from sirlab.diagnostics import observed_order
from sirlab.models import REGISTRY
from sirlab.reference import SIRReference
from sirlab.solvers import integrate


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    t_final = float(config["t_final"])
    hs = np.array(quick_or_full(config, "step_sizes", profile=profile))
    solvers = config["solvers"]

    variants_out = []
    for variant in config["variants"]:
        model_name = variant["model"]
        model = REGISTRY[model_name](n_total)
        y0 = model.initial_state(
            n=n_total, i0=variant.get("i0", 1.0), e0=variant.get("e0", 0.0), r0=variant.get("r0", 0.0), v0=variant.get("v0", 0.0)
        )
        theta = np.array([variant["theta"][name] for name in model.param_names])

        if model_name == "SIR":
            ref = SIRReference(beta=theta[0], gamma=theta[1], n_total=n_total, s0=y0[0], i0=y0[1])
            true_final = ref.trajectory(np.array([t_final]))[0]
        else:
            tight = integrate(model, y0, theta, (0.0, t_final), solver="rk45", rtol=1e-12, atol=1e-14)
            true_final = tight.y[-1]

        solver_results = {}
        for solver in solvers:
            errs = []
            for h in hs:
                res = integrate(model, y0, theta, (0.0, t_final), solver=solver, h=h)
                errs.append(float(np.linalg.norm(res.y[-1] - true_final)))
            errs = np.array(errs)
            solver_results[solver] = {
                "h": hs.tolist(),
                "error_final_state": errs.tolist(),
                "observed_order": observed_order(errs, hs),
            }

        variants_out.append(
            {
                "model": model_name,
                "state_names": list(model.state_names),
                "param_names": list(model.param_names),
                "theta": variant["theta"],
                "r0": model.r0(theta),
                "solvers": solver_results,
            }
        )

    return {"experiment": "E12", "t_final": t_final, "variants": variants_out}

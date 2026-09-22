"""Generates web/tests/fixtures/*.json: full-precision Python trajectories
for a grid of (model, theta, y0, h, T, solver) inputs, consumed by
web/tests/parity.test.ts to prove the TypeScript numerics twin agrees with
the Python core to ~1e-12 (solvers) / ~1e-9 (reference gold standard;
looser only because the two adaptive-quadrature implementations use
independent (but equivalent) recursion strategies) -- PLAN.md section 6.4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sirlab.models import SEIR, SIR, SIRS, SIRVaccination
from sirlab.reference import SIRReference
from sirlab.solvers import integrate

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "tests" / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 1000.0

CASES = [
    {"model": "SIR", "theta": [0.3, 0.1], "y0": [999.0, 1.0, 0.0], "h": 0.5, "t_final": 40.0, "solver": "euler"},
    {"model": "SIR", "theta": [0.3, 0.1], "y0": [999.0, 1.0, 0.0], "h": 0.25, "t_final": 40.0, "solver": "heun"},
    {"model": "SIR", "theta": [0.3, 0.1], "y0": [999.0, 1.0, 0.0], "h": 0.5, "t_final": 40.0, "solver": "rk4"},
    {"model": "SIR", "theta": [0.6, 0.15], "y0": [995.0, 5.0, 0.0], "h": 0.1, "t_final": 30.0, "solver": "rk4"},
    {"model": "SEIR", "theta": [0.3, 0.2, 0.1], "y0": [995.0, 2.0, 3.0, 0.0], "h": 0.25, "t_final": 40.0, "solver": "rk4"},
    {"model": "SIRS", "theta": [0.3, 0.1, 0.02], "y0": [990.0, 5.0, 5.0], "h": 0.25, "t_final": 60.0, "solver": "rk4"},
    {"model": "SIR-V", "theta": [0.3, 0.1, 0.01], "y0": [994.0, 5.0, 0.0, 1.0], "h": 0.25, "t_final": 40.0, "solver": "rk4"},
]

MODEL_CLASSES = {"SIR": SIR, "SEIR": SEIR, "SIRS": SIRS, "SIR-V": SIRVaccination}


def main():
    fixtures = []
    for case in CASES:
        model = MODEL_CLASSES[case["model"]](N)
        y0 = np.array(case["y0"])
        theta = np.array(case["theta"])
        res = integrate(model, y0, theta, (0.0, case["t_final"]), solver=case["solver"], h=case["h"])
        fixtures.append(
            {
                **case,
                "n_total": N,
                "t": res.t.tolist(),
                "y": res.y.tolist(),
            }
        )

    ref = SIRReference(beta=0.3, gamma=0.1, n_total=N, s0=999.0, i0=1.0)
    t_eval = np.linspace(0.0, 60.0, 13)
    ref_traj = ref.trajectory(t_eval)
    reference_fixture = {
        "beta": 0.3,
        "gamma": 0.1,
        "n_total": N,
        "s0": 999.0,
        "i0": 1.0,
        "t_eval": t_eval.tolist(),
        "trajectory": ref_traj.tolist(),
        "peak_i": ref.peak_i,
        "peak_time": ref.peak_time(),
        "s_infinity": ref.s_infinity,
    }

    (OUT_DIR / "solver_cases.json").write_text(json.dumps(fixtures, indent=2))
    (OUT_DIR / "reference_case.json").write_text(json.dumps(reference_fixture, indent=2))
    print(f"wrote {len(fixtures)} solver fixtures + 1 reference fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()

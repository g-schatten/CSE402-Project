"""Gold-standard self-consistency (PLAN.md section 4.4 accept criterion).
This test gates everything else: RQ1's error measurements are only
meaningful if the reference trajectory is itself correct.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate as si

from sirlab.reference import SIRReference

PARAM_SETS = [
    dict(beta=0.3, gamma=0.1, n_total=1000.0, s0=999.0, i0=1.0),
    dict(beta=1.5, gamma=0.5, n_total=10_000.0, s0=9_990.0, i0=10.0),
    dict(beta=0.15, gamma=0.1, n_total=5_000.0, s0=4_995.0, i0=5.0),  # R0 close to 1
]


@pytest.mark.parametrize("params", PARAM_SETS)
def test_phase_invariant_self_consistency(params):
    ref = SIRReference(**params)
    t_eval = np.linspace(0.0, 80.0, 21)
    traj = ref.trajectory(t_eval)
    q0 = ref.phase_invariant(traj[0, 0], traj[0, 2])
    for row in traj:
        q = ref.phase_invariant(row[0], row[2])
        assert abs(q - q0) < 1e-9


@pytest.mark.parametrize("params", PARAM_SETS)
def test_reference_agrees_with_scipy_dop853(params):
    beta, gamma, n = params["beta"], params["gamma"], params["n_total"]

    def f(t, y):
        s, i, r = y
        inf = beta * s * i / n
        return [-inf, inf - gamma * i, gamma * i]

    ref = SIRReference(**params)
    t_eval = np.linspace(0.0, 80.0, 25)
    traj = ref.trajectory(t_eval)
    sol = si.solve_ivp(f, [0.0, 80.0], [params["s0"], params["i0"], 0.0], t_eval=t_eval, rtol=1e-13, atol=1e-14, method="DOP853")
    diff = np.max(np.abs(sol.y.T - traj))
    assert diff < 1e-5, diff  # both sides are within roundoff of the true solution


@pytest.mark.parametrize("params", PARAM_SETS)
def test_peak_matches_scipy(params):
    beta, gamma, n = params["beta"], params["gamma"], params["n_total"]

    def f(t, y):
        s, i, r = y
        inf = beta * s * i / n
        return [-inf, inf - gamma * i, gamma * i]

    ref = SIRReference(**params)
    t_peak = ref.peak_time()
    sol = si.solve_ivp(f, [0.0, t_peak * 2], [params["s0"], params["i0"], 0.0], max_step=t_peak / 2000, rtol=1e-12, atol=1e-13)
    numeric_peak = np.max(sol.y[1])
    assert abs(numeric_peak - ref.peak_i) < 1e-4 * ref.peak_i


@pytest.mark.parametrize("params", PARAM_SETS)
def test_final_size_matches_long_run_scipy(params):
    beta, gamma, n = params["beta"], params["gamma"], params["n_total"]

    def f(t, y):
        s, i, r = y
        inf = beta * s * i / n
        return [-inf, inf - gamma * i, gamma * i]

    ref = SIRReference(**params)
    sol = si.solve_ivp(f, [0.0, 5000.0], [params["s0"], params["i0"], 0.0], rtol=1e-12, atol=1e-13)
    s_final = sol.y[0, -1]
    assert abs(s_final - ref.s_infinity) < 1e-3 * ref.s_infinity


def test_mass_conserved_along_reference():
    ref = SIRReference(**PARAM_SETS[0])
    t_eval = np.linspace(0, 100, 11)
    traj = ref.trajectory(t_eval)
    n = PARAM_SETS[0]["n_total"]
    assert np.allclose(np.sum(traj, axis=1), n, atol=1e-6)

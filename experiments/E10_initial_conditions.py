"""E10: Initial conditions (PLAN.md section 5).

Compares two estimation scenarios: (a) I0 known exactly (standard 2-param
(beta, gamma) fit) vs (b) I0 unknown and estimated as a 3rd parameter
alongside (beta, gamma). Reports the extra variance incurred by not knowing
I0, and the 2-norm condition number of the 3-parameter J vs the 2-parameter
one (sirlab.linalg.eigen.cond2: power and inverse power method on J^T J).

Note on method: for the 3-parameter case, S0 = N - I0 depends on the
parameter being estimated, so the clean forward-sensitivity-equation
machinery in sirlab.sensitivity (which assumes a theta-independent initial
condition) does not directly apply without extending it. Rather than
special-case sirlab.estimate's analytic-Jacobian contract for one
experiment, we fit the 3-parameter case with the derivative-free
Nelder-Mead optimizer (already exact-recovery tested in test_estimate.py)
and compute the condition-number diagnostic from a local central finite
difference here in the experiment script -- an explicitly scoped exception
to the "no finite differences" rule, which applies to sirlab/estimate's
production estimators, not to a one-off analysis diagnostic.
"""
from __future__ import annotations

import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.common import quick_or_full
from sirlab.estimate.base import ForwardModel
from sirlab.estimate.lm import levenberg_marquardt
from sirlab.estimate.neldermead import nelder_mead
from sirlab.linalg.eigen import cond2
from sirlab.models import SIR
from sirlab.observe import prevalence
from sirlab.reference import SIRReference


def _obs_fn(result, t_obs):
    return prevalence(result, i_index=1, t_obs=t_obs)


class _I0ForwardModel(ForwardModel):
    """theta = (beta, gamma, i0); builds y0 = (N - i0, i0, 0) per-evaluation."""

    def predict(self, theta):
        from sirlab.solvers import integrate

        beta, gamma, i0 = theta
        y0 = np.array([self.n_total - i0, i0, 0.0])
        t_span = (0.0, float(self.t_obs[-1]))
        result = integrate(self.model, y0, np.array([beta, gamma]), t_span, solver=self.solver, h=self.h)
        return self.obs_fn(result, self.t_obs)


def _numeric_jacobian(fwd, theta, eps=1e-5):
    jac = np.empty((len(fwd.t_obs), len(theta)))
    for j in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += eps * max(abs(theta[j]), 1.0)
        tm[j] -= eps * max(abs(theta[j]), 1.0)
        jac[:, j] = (fwd.predict(tp) - fwd.predict(tm)) / (tp[j] - tm[j])
    return jac


def _one_trial(args):
    beta, gamma, i0_true, n_total, dt, sigma_frac, solver, h, t_obs_final, seed = args
    s0 = n_total - i0_true
    ref = SIRReference(beta=beta, gamma=gamma, n_total=n_total, s0=s0, i0=i0_true)
    t_obs = np.arange(dt, t_obs_final + 1e-9, dt)
    clean_i = ref.trajectory(t_obs)[:, 1]
    rng = np.random.default_rng(seed)
    sigma = sigma_frac * np.max(clean_i)
    obs = clean_i + rng.normal(scale=sigma, size=clean_i.shape)

    model = SIR(n_total)
    y0_known = np.array([s0, i0_true, 0.0])

    # (a) I0 known
    fwd_known = ForwardModel(model, y0_known, t_obs, _obs_fn, solver=solver, h=h)
    fwd_known.obs_state_index = 1
    theta0 = np.array([beta, gamma]) * rng.uniform(0.7, 1.4, size=2)
    fit_known = levenberg_marquardt(fwd_known, obs, theta0, max_iter=60)

    # (b) I0 unknown, estimated jointly
    fwd_unknown = _I0ForwardModel(model, y0_known, t_obs, _obs_fn, solver=solver, h=h)
    fwd_unknown.n_total = n_total
    theta0_3 = np.array([theta0[0], theta0[1], i0_true * rng.uniform(0.5, 2.0)])
    fit_unknown = nelder_mead(fwd_unknown, obs, theta0_3, max_iter=400)

    return (
        fit_known.theta_hat,
        bool(fit_known.converged and np.all(fit_known.theta_hat > 0)),
        fit_unknown.theta_hat,
        bool(fit_unknown.converged),
    )


def run(config: dict) -> dict:
    profile = config.get("_profile", "quick")
    n_total = float(config["n_total"])
    beta, gamma = config["theta_true"]["beta"], config["theta_true"]["gamma"]
    i0_values = quick_or_full(config, "i0_values", profile=profile)
    n_rep = quick_or_full(config, "n_replicates", profile=profile)
    seed = int(config.get("seed", 0))
    n_jobs = int(config.get("n_jobs", 4))

    cells = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i0 in i0_values:
            jobs = []
            for r in range(n_rep):
                key = f"{i0}|{r}"
                s = (seed * 10_000_000 + zlib.crc32(key.encode())) % (2**31 - 1)
                jobs.append((beta, gamma, i0, n_total, config["dt"], config["sigma_frac"], config["solver"], config["h"], config["t_obs_final"], s))

            known_thetas, known_oks, unknown_thetas, unknown_oks = [], [], [], []
            for th_k, ok_k, th_u, ok_u in ex.map(_one_trial, jobs, chunksize=2):
                known_thetas.append(th_k)
                known_oks.append(ok_k)
                unknown_thetas.append(th_u[:2])
                unknown_oks.append(ok_u)

            known_thetas, unknown_thetas = np.array(known_thetas), np.array(unknown_thetas)
            known_oks, unknown_oks = np.array(known_oks), np.array(unknown_oks)
            theta_true = np.array([beta, gamma])

            var_known = np.var(known_thetas[known_oks], axis=0).tolist() if known_oks.any() else [None, None]
            var_unknown = np.var(unknown_thetas[unknown_oks], axis=0).tolist() if unknown_oks.any() else [None, None]

            # condition-number diagnostic at the true parameter values
            s0 = n_total - i0
            t_obs = np.arange(config["dt"], config["t_obs_final"] + 1e-9, config["dt"])
            model = SIR(n_total)
            y0_known = np.array([s0, i0, 0.0])
            fwd_known = ForwardModel(model, y0_known, t_obs, _obs_fn, solver=config["solver"], h=config["h"])
            fwd_known.obs_state_index = 1
            jac2 = fwd_known.jacobian(theta_true)
            fwd_unknown = _I0ForwardModel(model, y0_known, t_obs, _obs_fn, solver=config["solver"], h=config["h"])
            fwd_unknown.n_total = n_total
            jac3 = _numeric_jacobian(fwd_unknown, np.array([beta, gamma, i0]))

            cells.append(
                {
                    "i0": i0,
                    "variance_i0_known": var_known,
                    "variance_i0_unknown": var_unknown,
                    "cond_2param": cond2(jac2),
                    "cond_3param": cond2(jac3),
                    "n_converged_known": int(known_oks.sum()),
                    "n_converged_unknown": int(unknown_oks.sum()),
                    "n_total": n_rep,
                }
            )

    return {"experiment": "E10", "theta_true": [beta, gamma], "cells": cells}

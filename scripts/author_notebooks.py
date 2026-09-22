"""Authors the four RQ notebooks (PLAN.md section 6 / phase 8). These
narrate results that experiments/ already computed -- they load
results/*.json and plot; they never recompute the underlying sweeps.
"""
from __future__ import annotations

from pathlib import Path

from make_notebook import code, md, write_notebook

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(exist_ok=True)

COMMON_SETUP = """\
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("..").resolve() / "scripts"))
from plot_style import apply_style, SOLVER_COLOR, COMPARTMENT_COLOR
apply_style()

RESULTS = Path("..") / "results"

def load(exp_id):
    return json.loads((RESULTS / f"{exp_id}.json").read_text())
"""

# ---------------------------------------------------------------- RQ1 ----
rq1 = [
    md(
        "# RQ1 -- Solver comparison\n\n"
        "*How do Forward Euler, Heun, and RK4 differ in accuracy, observed convergence order, "
        "and numerical stability when solving the SIR system?*\n\n"
        "This notebook narrates the results already computed by `experiments/E01_convergence.py`, "
        "`E02_invariants.py`, `E03_stability.py`, and `E04_structural.py`. Run "
        "`make experiments` (or `make experiment ID=E01` etc.) before executing these cells."
    ),
    code(COMMON_SETUP),
    md(
        "## Convergence order\n\n"
        "Each solver's error against the semi-analytic gold standard (`sirlab.reference`), "
        "as a function of step size h. The fitted log-log slope is the *observed* convergence order "
        "p&#770; -- it should match the theoretical order (1, 2, 4, 1, 2 for Euler / Heun / RK4 / "
        "backward Euler / trapezoidal)."
    ),
    code(
        """\
d = load("E01")
regime = d["regimes"][0]
fig, ax = plt.subplots(figsize=(6, 4.5))
for solver, res in regime["solvers"].items():
    ax.loglog(res["h"], res["error_final_state"], "o-", color=SOLVER_COLOR.get(solver, "gray"),
               label=f"{solver} (p̂={res['observed_order']:.2f}, theory={res['theoretical_order']})")
ax.set_xlabel("step size h"); ax.set_ylabel(f"error at t={d['t_final']}")
ax.set_title(f"Convergence -- {regime['name']}")
ax.legend()
plt.show()
"""
    ),
    md(
        "## The mass-conservation-is-uninformative finding\n\n"
        "Because the SIR right-hand side sums to zero identically, every explicit Runge-Kutta "
        "method conserves S+I+R to roundoff **at any step size** -- including a solve that is "
        "otherwise catastrophically wrong. The phase invariant Q(t) = ln S + (beta/gamma N) R "
        "drifts at the solver's true convergence order instead, and is the diagnostic that "
        "actually discriminates solver quality."
    ),
    code(
        """\
d2 = load("E02")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for solver, per_h in d2["results"].items():
    for h_str, entry in per_h.items():
        axes[0].semilogy(entry["t"], np.maximum(entry["mass_error"], 1e-18), color=SOLVER_COLOR.get(solver, "gray"), alpha=0.6)
        axes[1].semilogy(entry["t"], np.maximum(entry["phase_invariant_drift"], 1e-18), color=SOLVER_COLOR.get(solver, "gray"), alpha=0.6)
axes[0].set_title("Mass error (uninformative)"); axes[1].set_title("Phase-invariant drift (discriminating)")
plt.show()
"""
    ),
    md("## Stability frontier and structural accuracy\n\nSee `E03` (empirical stability boundary in (β, γ, h)) and `E04` (peak height / peak time / final-size errors) for the remaining RQ1 evidence."),
    code(
        """\
d3 = load("E03")
fig, axes = plt.subplots(1, len(d3["solvers"]), figsize=(4 * len(d3["solvers"]), 4))
for ax, (solver, res) in zip(np.atleast_1d(axes), d3["solvers"].items()):
    im = ax.pcolormesh(d3["beta_grid"], d3["gamma_grid"], np.array(res["first_failure_h"]).T, shading="auto")
    ax.set_title(f"{solver}: largest stable h"); ax.set_xlabel("β"); ax.set_ylabel("γ")
    fig.colorbar(im, ax=ax)
plt.show()
"""
    ),
]

# ---------------------------------------------------------------- RQ2 ----
rq2 = [
    md(
        "# RQ2 -- Parameter recovery\n\n"
        "*How does the choice of solver propagate into the accuracy of parameters (β, γ) recovered "
        "via least-squares estimation on synthetic data?*\n\n"
        "Narrates `experiments/E05_landscape.py`, `E06_optimizers.py`, and `E07_solver_recovery.py`."
    ),
    code(COMMON_SETUP),
    md("## The cost landscape J(β, γ)\n\nBuilt by grid search (E05); its curvature and condition number set how hard the estimation problem is."),
    code(
        """\
d5 = load("E05")
sigma_key = list(d5["surfaces"].keys())[0]
surf = d5["surfaces"][sigma_key]
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.pcolormesh(d5["beta_grid"], d5["gamma_grid"], surf["surface"], shading="auto", cmap="viridis")
ax.set_xlabel("β"); ax.set_ylabel("γ"); ax.set_title(f"J(β,γ), σ={sigma_key}, κ(JᵀJ)≈{surf['cond_jtj']:.1f}")
fig.colorbar(im, ax=ax)
plt.show()
"""
    ),
    md("## Optimizer shoot-out (E06)\n\nSuccess rate, mean iterations, and mean function evaluations for five optimizers over many random starts."),
    code(
        """\
d6 = load("E06")
for name, r in d6["results"].items():
    print(f"{name:28s}  success={r['success_rate']*100:5.1f}%  iters={r['mean_iterations'] or float('nan'):6.2f}  fevals={r['mean_fevals'] or float('nan'):7.1f}")
"""
    ),
    md(
        "## The centrepiece: solver-in-the-loop recovery bias (E07)\n\n"
        "For each solver, holding everything else fixed, bias in the recovered β at sigma=0 (noise-free) "
        "is caused *purely* by that solver's own truncation error. It should shrink at the solver's "
        "theoretical order as h -> 0 -- this is the concrete mechanism behind RQ2."
    ),
    code(
        """\
d7 = load("E07")
solvers = sorted({c["solver"] for c in d7["cells"]})
theta0 = (d7["cells"][0]["beta_true"], d7["cells"][0]["gamma_true"])
fig, ax = plt.subplots(figsize=(6, 4.5))
for solver in solvers:
    cells = [c for c in d7["cells"] if c["solver"] == solver and c["sigma_frac"] == 0.0
             and (c["beta_true"], c["gamma_true"]) == theta0]
    cells.sort(key=lambda c: c["h"])
    hs = [c["h"] for c in cells]
    bias = [abs(c["bias"][0]) for c in cells]
    ax.loglog(hs, bias, "o-", color=SOLVER_COLOR.get(solver, "gray"), label=solver)
ax.set_xlabel("fitting step size h"); ax.set_ylabel("|bias| in β̂ (noise-free)")
ax.legend(); ax.set_title("Solver-induced bias in recovered β")
plt.show()
"""
    ),
]

# ---------------------------------------------------------------- RQ3 ----
rq3 = [
    md(
        "# RQ3 -- Noise & sampling robustness\n\n"
        "*How robust is parameter recovery to observation noise, sparse sampling, and initial "
        "conditions, and does that hold across a range of (β, γ)?*\n\n"
        "Narrates `E08_noise.py`, `E09_sampling.py`, `E10_initial_conditions.py`, and `E11_uq.py`, "
        "and ends with the project's headline finding from `E13_crossover.py`."
    ),
    code(COMMON_SETUP),
    md("## Noise degradation (E08)"),
    code(
        """\
d8 = load("E08")
sigmas = [c["sigma_frac"] for c in d8["curve"]]
rmse = [c["rmse"][0] for c in d8["curve"]]
lo = [c["rmse_ci_lo"][0] for c in d8["curve"]]
hi = [c["rmse_ci_hi"][0] for c in d8["curve"]]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(sigmas, rmse, "o-")
ax.fill_between(sigmas, lo, hi, alpha=0.2)
ax.set_xlabel("σ"); ax.set_ylabel("RMSE(β̂)"); ax.set_title("Noise degradation with bootstrap CI")
plt.show()
"""
    ),
    md("## Sampling & window truncation (E09)\n\nIdentifiability degrades sharply once the observation window ends before the epidemic peak."),
    code(
        """\
d9 = load("E09")
import pandas as pd
df = pd.DataFrame(d9["cells"])
df["rmse_beta"] = df["rmse"].apply(lambda r: r[0])
pivot = df.pivot_table(index="window_frac", columns="dt", values="rmse_beta")
display(pivot)
"""
    ),
    md("## Uncertainty quantification, four ways (E11)\n\nAsymptotic, bootstrap, Monte Carlo, and MCMC overlaid; MCMC's Gelman-Rubin R̂ confirms chain convergence."),
    code(
        """\
d11 = load("E11")
print("rho(beta,gamma) =", d11["asymptotic"]["rho"])
print("MCMC R-hat =", d11["mcmc"]["rhat"], " ESS(beta,gamma) =", d11["mcmc"]["ess_beta"], d11["mcmc"]["ess_gamma"])
fig, ax = plt.subplots(figsize=(5.5, 5))
ell = np.array(d11["asymptotic"]["ellipse"])
boot = np.array(d11["bootstrap"]["thetas"])
mcmc = np.array(d11["mcmc"]["pooled_samples"])
ax.plot(ell[:, 0], ell[:, 1], "-", label="asymptotic 95%")
ax.scatter(boot[:, 0], boot[:, 1], s=4, alpha=0.4, label="bootstrap")
ax.scatter(mcmc[:, 0], mcmc[:, 1], s=4, alpha=0.4, label="MCMC")
ax.scatter(*d11["theta_true"], color="black", marker="x", s=80, label="truth")
ax.set_xlabel("β"); ax.set_ylabel("γ"); ax.legend()
plt.show()
"""
    ),
    md(
        "## The headline finding: crossover σ*\n\n"
        "Solver bias (deterministic, set by h) vs noise-induced std (rises with σ). Their "
        "intersection is σ*(h)."
    ),
    code(
        """\
d13 = load("E13")
row = d13["rows"][0]
bias = abs(row["solver_bias"][0])
sigmas = [c["sigma_frac"] for c in row["sigma_curve"] if c["sigma_frac"] > 0]
stds = [c["std"][0] for c in row["sigma_curve"] if c["sigma_frac"] > 0]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(sigmas, stds, "o-", label="std across replicates")
ax.axhline(bias, ls="--", color="crimson", label="solver bias(h)")
sigma_star = row["crossover_sigma_star"].get("beta")
if sigma_star:
    ax.axvline(sigma_star, ls=":", color="green", label=f"σ*={sigma_star:.4f}")
ax.set_xlabel("σ"); ax.set_ylabel("|error| in β̂"); ax.legend()
ax.set_title(f"Crossover σ* (solver={d13['solver']}, h={row['h']})")
plt.show()
print(d13["conclusion"])
"""
    ),
]

# ---------------------------------------------------------------- RQ4 ----
rq4 = [
    md(
        "# RQ4 -- Extensions: model variants and real data\n\n"
        "*Does the RQ1-RQ3 ordering survive a harder model or real data?*\n\n"
        "Narrates `E12_variants.py` (SEIR / SIRS / SIR+vaccination) and `E14_realdata.py` (Eyam 1666 -- "
        "see `data/raw/provenance.md` for a data-source caveat before citing these numbers)."
    ),
    code(COMMON_SETUP),
    md("## Model variants (E12)\n\nConvergence order on SEIR/SIRS/SIR-V, measured against a tight RK45 reference (no closed form exists for these models)."),
    code(
        """\
d12 = load("E12")
for v in d12["variants"]:
    print(f"--- {v['model']} (R0={v['r0']:.2f}) ---")
    for solver, res in v["solvers"].items():
        print(f"  {solver:8s} observed order p̂ = {res['observed_order']:.2f}")
"""
    ),
    md("## Real data: Eyam 1666 (E14)"),
    code(
        """\
d14 = load("E14")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(d14["days"], d14["s_observed"], "ko-", label="observed")
for solver, r in d14["results"].items():
    ax.plot(d14["days"], r["predicted_s"], "-", color=SOLVER_COLOR.get(solver, "gray"), label=f"fit ({solver}), R0̂={r['r0']:.2f}")
ax.set_xlabel("day"); ax.set_ylabel("S(t)"); ax.legend()
plt.show()
"""
    ),
    md(
        "## Honest limitations\n\n"
        "- The Eyam dataset's digitized counts were reconstructed from memory of commonly-reproduced "
        "teaching tables and have not been verified against the primary source in this session "
        "(see `data/raw/provenance.md`).\n"
        "- SIR is a poor model for a small, closed population with no births/deaths and complete "
        "quarantine -- Eyam is a demonstration that the pipeline *runs* on real data, not a claim "
        "that SIR is the right model for it."
    ),
]

write_notebook(NB_DIR / "RQ1_solvers.ipynb", rq1)
write_notebook(NB_DIR / "RQ2_recovery.ipynb", rq2)
write_notebook(NB_DIR / "RQ3_robustness.ipynb", rq3)
write_notebook(NB_DIR / "RQ4_extensions.ipynb", rq4)

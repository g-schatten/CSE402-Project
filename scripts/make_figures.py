"""Generates every report figure from results/*.json (PLAN.md section 8:
"figures are never pasted in"). Run after `make experiments`. Each figure
is saved as both .pdf (for LaTeX) and .svg (for quick viewing) into
figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import COMPARTMENT_COLOR, SOLVER_COLOR, apply_style

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

apply_style()


def load(exp_id: str) -> dict | None:
    path = RESULTS / f"{exp_id}.json"
    if not path.exists():
        print(f"  [skip] {exp_id}.json not found -- run `make experiment ID={exp_id}` first")
        return None
    return json.loads(path.read_text())


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf (+ .svg)")


def fig_e01_convergence():
    d = load("E01")
    if d is None:
        return
    regime = d["regimes"][0]
    fig, ax = plt.subplots(figsize=(5, 4))
    for solver, res in regime["solvers"].items():
        ax.loglog(res["h"], res["error_final_state"], "o-", color=SOLVER_COLOR.get(solver, "gray"), label=f"{solver} (p̂={res['observed_order']:.2f})")
    ax.set_xlabel("step size h")
    ax.set_ylabel(f"error at t={d['t_final']}")
    ax.set_title(f"E01: Convergence -- {regime['name']}")
    ax.legend()
    save(fig, "e01_convergence")


def fig_e02_invariants():
    d = load("E02")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for solver, per_h in d["results"].items():
        for h_str, entry in per_h.items():
            axes[0].semilogy(entry["t"], np.maximum(entry["mass_error"], 1e-18), color=SOLVER_COLOR.get(solver, "gray"), alpha=0.6, lw=1)
            axes[1].semilogy(entry["t"], np.maximum(entry["phase_invariant_drift"], 1e-18), color=SOLVER_COLOR.get(solver, "gray"), alpha=0.6, lw=1)
    axes[0].set_title("Mass error |S+I+R-N|\n(uninformative: roundoff at any h)")
    axes[1].set_title("Phase-invariant drift |Q(t)-Q(0)|\n(scales with solver order)")
    for ax in axes:
        ax.set_xlabel("t")
    save(fig, "e02_invariants")


def fig_e03_stability():
    d = load("E03")
    if d is None:
        return
    fig, axes = plt.subplots(1, len(d["solvers"]), figsize=(4 * len(d["solvers"]), 4), squeeze=False)
    betas, gammas = d["beta_grid"], d["gamma_grid"]
    for ax, (solver, res) in zip(axes[0], d["solvers"].items()):
        im = ax.pcolormesh(betas, gammas, np.array(res["first_failure_h"]).T, shading="auto", cmap="viridis")
        ax.set_title(f"{solver}: largest stable h")
        ax.set_xlabel("β")
        ax.set_ylabel("γ")
        fig.colorbar(im, ax=ax)
    save(fig, "e03_stability_frontier")


def fig_e04_structural():
    d = load("E04")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for solver, res in d["solvers"].items():
        axes[0].loglog(res["h"], np.maximum(res["peak_i_error"], 1e-12), "o-", color=SOLVER_COLOR.get(solver, "gray"), label=solver)
        axes[1].loglog(res["h"], np.maximum(res["peak_t_error"], 1e-12), "o-", color=SOLVER_COLOR.get(solver, "gray"), label=solver)
    axes[0].set_title("Peak height error"); axes[0].set_xlabel("h")
    axes[1].set_title("Peak time error"); axes[1].set_xlabel("h")
    axes[0].legend()
    save(fig, "e04_structural_accuracy")


def fig_e07_recovery_bias():
    d = load("E07")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for solver in {c["solver"] for c in d["cells"]}:
        cells = [c for c in d["cells"] if c["solver"] == solver and c["sigma_frac"] == 0.0 and c["beta_true"] == d["cells"][0]["beta_true"]]
        cells.sort(key=lambda c: c["h"])
        hs = [c["h"] for c in cells]
        bias = [abs(c["bias"][0]) if c["bias"][0] is not None else np.nan for c in cells]
        ax.loglog(hs, bias, "o-", color=SOLVER_COLOR.get(solver, "gray"), label=solver)
    ax.set_xlabel("fitting step size h")
    ax.set_ylabel("|bias| in β̂ (noise-free)")
    ax.set_title("E07: solver-induced bias in recovered β")
    ax.legend()
    save(fig, "e07_recovery_bias")


def fig_e08_noise_degradation():
    d = load("E08")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sigmas = [c["sigma_frac"] for c in d["curve"]]
    rmse_beta = [c["rmse"][0] if c["rmse"][0] is not None else np.nan for c in d["curve"]]
    lo = [c["rmse_ci_lo"][0] for c in d["curve"]]
    hi = [c["rmse_ci_hi"][0] for c in d["curve"]]
    ax.plot(sigmas, rmse_beta, "o-", color="#2f7fd6")
    ax.fill_between(sigmas, lo, hi, alpha=0.2, color="#2f7fd6")
    ax.set_xlabel("noise level σ (fraction of max I)")
    ax.set_ylabel("RMSE(β̂)")
    ax.set_title("E08: noise degradation with bootstrap CI")
    save(fig, "e08_noise_degradation")


def fig_e11_uncertainty():
    d = load("E11")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    ell = np.array(d["asymptotic"]["ellipse"])
    boot = np.array(d["bootstrap"]["thetas"])
    mcmc = np.array(d["mcmc"]["pooled_samples"])
    ax.plot(ell[:, 0], ell[:, 1], "-", color="#2f7fd6", label="asymptotic 95%")
    ax.scatter(boot[:, 0], boot[:, 1], s=4, alpha=0.4, color="#c98a1f", label="bootstrap")
    ax.scatter(mcmc[:, 0], mcmc[:, 1], s=4, alpha=0.3, color="#22a06b", label="MCMC")
    ax.scatter(*d["theta_true"], color="black", marker="x", s=60, label="truth")
    ax.set_xlabel("β"); ax.set_ylabel("γ")
    ax.set_title(f"Four UQ routes overlaid (ρ={d['asymptotic']['rho']:.2f})")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    beta_grid = d["profile_likelihood"]["beta_grid"]
    ax2.semilogy(beta_grid, np.maximum(d["profile_likelihood"]["cost_full_window"], 1e-8), "-", color="#22a06b", label="full window")
    ax2.semilogy(beta_grid, np.maximum(d["profile_likelihood"]["cost_truncated_window"], 1e-8), "-", color="#d43f4e", label="truncated (pre-peak)")
    ax2.axvline(d["theta_true"][0], color="black", ls=":", lw=1)
    ax2.set_xlabel("β (γ profiled out)"); ax2.set_ylabel("min cost")
    ax2.set_title("Profile likelihood: identifiability collapse")
    ax2.legend(fontsize=8)
    save(fig, "e11_uncertainty")


def fig_e09_sampling():
    d = load("E09")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    dts = sorted({c["dt"] for c in d["cells"]})
    for dt in dts:
        cells = sorted([c for c in d["cells"] if c["dt"] == dt], key=lambda c: c["window_frac"])
        wf = [c["window_frac"] for c in cells]
        rmse = [c["rmse"][0] if c["rmse"][0] is not None else np.nan for c in cells]
        ax.semilogy(wf, rmse, "o-", label=f"dt={dt}")
    ax.set_xlabel("observation window fraction (of ~6x peak time)")
    ax.set_ylabel("RMSE(β̂)")
    ax.set_title("E09: identifiability collapses before the epidemic peak")
    ax.legend()
    save(fig, "e09_sampling_window")


def fig_e13_crossover():
    d = load("E13")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    row = d["rows"][0]
    bias = abs(row["solver_bias"][0])
    sigmas = [c["sigma_frac"] for c in row["sigma_curve"] if c["sigma_frac"] > 0]
    stds = [c["std"][0] for c in row["sigma_curve"] if c["sigma_frac"] > 0]
    ax.loglog(sigmas, stds, "o-", color="#2f7fd6", label="std across replicates")
    ax.axhline(bias, color="#d43f4e", ls="--", label="solver bias(h)")
    sigma_star = row["crossover_sigma_star"].get("beta")
    if sigma_star:
        ax.axvline(sigma_star, color="#22a06b", ls=":", label=f"σ* = {sigma_star:.4f}")
    ax.set_xlabel("σ")
    ax.set_ylabel("|error| in β̂")
    ax.set_title(f"E13: crossover σ* (h={row['h']}, solver={d['solver']})")
    ax.legend()
    save(fig, "e13_crossover")


def main():
    print("Generating figures from results/*.json ...")
    fig_e01_convergence()
    fig_e02_invariants()
    fig_e03_stability()
    fig_e04_structural()
    fig_e07_recovery_bias()
    fig_e08_noise_degradation()
    fig_e09_sampling()
    fig_e11_uncertainty()
    fig_e13_crossover()
    print("Done.")


if __name__ == "__main__":
    main()

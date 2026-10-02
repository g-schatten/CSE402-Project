"""Figures for the final report (report/B_02.tex).

Reads the experiment artifacts in results/*.json (run
`python -m experiments.runner --all --profile quick` first) and writes
column-width PDFs into report/figures/. Two figures recompute a few
cheap quantities directly from core/sirlab: the teaser curve, and the
4-step-size noise-free bias sweep (12 fits) that extends E07's two step sizes.

Run from the repository root:
    python report/make_report_figures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sirlab.estimate.base import ForwardModel  # noqa: E402
from sirlab.estimate.lm import levenberg_marquardt  # noqa: E402
from sirlab.models.sir import SIR  # noqa: E402
from sirlab.observe import prevalence  # noqa: E402
from sirlab.reference import SIRReference  # noqa: E402
from sirlab.solvers import integrate  # noqa: E402

RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1).
# Contrast on white is < 3:1 for slots 3-5, so every series also gets a
# distinct marker and a legend entry.
SOLVER_STYLE = {
    "euler": ("#2a78d6", "o", "Euler"),
    "heun": ("#eb6834", "s", "Heun"),
    "rk4": ("#1baf7a", "^", "RK4"),
    "backward_euler": ("#eda100", "D", "Backward Euler"),
    "trapezoidal": ("#e87ba4", "v", "Trapezoidal"),
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
COL_W = 3.35  # ACM sigconf column width, inches

N, I0, BETA, GAMMA = 1000.0, 1.0, 0.3, 0.1


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "mathtext.fontset": "cm",
        }
    )


def load(exp_id: str) -> dict:
    with open(RESULTS / f"{exp_id}.json") as fh:
        return json.load(fh)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    png_dir = os.environ.get("REPORT_FIG_PNG")  # optional preview copies
    if png_dir:
        fig.savefig(Path(png_dir) / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_teaser() -> None:
    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=N - I0, i0=I0)
    t = np.linspace(0, 100, 401)
    exact = ref.trajectory(t)[:, 1]
    model, y0, theta = SIR(N), np.array([N - I0, I0, 0.0]), np.array([BETA, GAMMA])
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.plot(t, exact, color=INK, lw=2.0, label="exact (gold standard)")
    for solver, ls in [("euler", "--"), ("rk4", ":")]:
        c, _, lab = SOLVER_STYLE[solver]
        res = integrate(model, y0, theta, (0.0, 100.0), solver=solver, h=1.0)
        ax.plot(res.t, res.y[:, 1], color=c, ls=ls, lw=1.6, label=f"{lab}, $h=1$ day")
    ax.set_xlabel("day")
    ax.set_ylabel("infectious $I(t)$")
    ax.set_ylim(0, 335)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, columnspacing=1.0, handlelength=2.2)
    save(fig, "teaser")


def fig_convergence() -> None:
    d = load("E01")
    regime = d["regimes"][0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for solver, r in regime["solvers"].items():
        c, m, lab = SOLVER_STYLE[solver]
        ax.loglog(r["h"], r["error_final_state"], marker=m, color=c, label=f"{lab} ($\\hat p={r['observed_order']:.2f}$)")
    ax.set_xticks([0.125, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1"])
    ax.minorticks_off()
    ax.set_xlabel("step size $h$ (days)")
    ax.set_ylabel(r"$\|y_h(40)-y(40)\|_2$")
    ax.set_ylim(1e-9, 5e2)
    ax.legend(loc="lower right", ncol=1)
    save(fig, "convergence")


def fig_invariants() -> None:
    d = load("E02")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.8), sharex=True)
    for solver, per_h in d["results"].items():
        c, m, lab = SOLVER_STYLE[solver]
        hs = sorted(float(h) for h in per_h)
        mass = [per_h[str(h) if str(h) in per_h else f"{h}"]["max_mass_error"] for h in hs]
        drift = [per_h[str(h) if str(h) in per_h else f"{h}"]["max_phase_drift"] for h in hs]
        axes[0].loglog(hs, np.maximum(mass, 1e-16), marker=m, color=c, label=lab)
        axes[1].loglog(hs, drift, marker=m, color=c, label=lab)
    axes[0].set_ylim(1e-16, 1)
    axes[1].set_ylim(1e-11, 1)
    axes[0].set_title("(a) mass error")
    axes[1].set_title("(b) phase-invariant drift")
    axes[0].set_ylabel(r"$\max_t|S+I+R-N|$")
    axes[1].set_ylabel(r"$\max_t|Q(t)-Q(0)|$")
    for ax in axes:
        ax.set_xlabel("step size $h$")
        ax.set_xticks([0.1, 0.5, 2.0])
        ax.set_xticklabels(["0.1", "0.5", "2"])
        ax.minorticks_off()
    axes[0].legend(loc="upper right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "invariants")


def fig_bias() -> None:
    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=N - I0, i0=I0)
    t_obs = np.arange(1.0, 60.0 + 1e-9, 1.0)
    obs = ref.trajectory(t_obs)[:, 1]
    hs = [0.5, 0.25, 0.1, 0.05]
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    rows = {}
    for solver, p in [("euler", 1), ("heun", 2), ("rk4", 4)]:
        bias = []
        for h in hs:
            fwd = ForwardModel(SIR(N), np.array([N - I0, I0, 0.0]), t_obs,
                               lambda res, t: prevalence(res, i_index=1, t_obs=t), solver=solver, h=h)
            fwd.obs_state_index = 1
            fit = levenberg_marquardt(fwd, obs, np.array([BETA * 1.2, GAMMA * 0.8]), max_iter=100)
            bias.append(abs(fit.theta_hat[0] - BETA))
        rows[solver] = bias
        c, m, lab = SOLVER_STYLE[solver]
        ax.loglog(hs, bias, marker=m, color=c, label=f"{lab} (order {p})")
        # reference slope h^p through the smallest-h point
        hh = np.array([0.05, 0.5])
        ax.loglog(hh, 0.2 * bias[-1] * (hh / 0.05) ** p, color=c, lw=0.8, ls=":")
    print("noise-free |bias beta|:", {k: [f"{b:.3e}" for b in v] for k, v in rows.items()})
    ax.set_xticks(hs)
    ax.set_xticklabels(["1/2", "1/4", "1/10", "1/20"])
    ax.minorticks_off()
    ax.set_xlabel("fitting step size $h$ (days)")
    ax.set_ylabel(r"$|\hat\beta-\beta_{\rm true}|$ (noise-free)")
    ax.legend(loc="lower right")
    save(fig, "bias_vs_h")


def fig_noise_sampling() -> None:
    e08, e09 = load("E08"), load("E09")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9))
    ax = axes[0]
    curve = [c for c in e08["curve"] if c["sigma_frac"] > 0]
    s = [c["sigma_frac"] for c in curve]
    ax.fill_between(s, [c["rmse_ci_lo"][0] for c in curve], [c["rmse_ci_hi"][0] for c in curve],
                    color=SOLVER_STYLE["euler"][0], alpha=0.18, lw=0, label="95% bootstrap CI")
    ax.plot(s, [c["rmse"][0] for c in curve], marker="o", color=SOLVER_STYLE["euler"][0], label=r"RMSE$(\hat\beta)$")
    ax.set_xlabel(r"noise $\sigma_{\rm frac}$")
    ax.set_ylabel(r"RMSE$(\hat\beta)$")
    ax.set_title("(a) noise (E08, RK4)")
    ax.legend(loc="upper left")
    ax = axes[1]
    dts = sorted({c["dt"] for c in e09["cells"]})
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    for dt, col, mk in zip(dts, colors, markers):
        cells = sorted([c for c in e09["cells"] if c["dt"] == dt], key=lambda c: c["window_frac"])
        ax.plot([c["window_frac"] for c in cells], [c["rmse"][0] for c in cells], marker=mk, color=col,
                label=f"$\\Delta t={dt:g}$")
    ax.set_xlabel(r"window ($\times\,6t_{\rm peak}$)")
    ax.set_title("(b) sampling (E09)")
    ax.legend(loc="upper right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "noise_sampling")


def fig_uq() -> None:
    d = load("E11")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.0))
    ax = axes[0]
    boot = np.array(d["bootstrap"]["thetas"])
    mcmc = np.array(d["mcmc"]["pooled_samples"])
    ell = np.array(d["asymptotic"]["ellipse"])
    ax.scatter(mcmc[:, 0], mcmc[:, 1], s=2, color="#1baf7a", alpha=0.35, lw=0, label="MCMC")
    ax.scatter(boot[:, 0], boot[:, 1], s=5, color="#eb6834", alpha=0.8, lw=0, label="bootstrap")
    ax.plot(ell[:, 0], ell[:, 1], color="#2a78d6", lw=1.3, label="asymptotic 95%")
    ax.plot(*d["theta_true"], marker="x", color=INK, ms=6, mew=1.4, ls="none", label="truth")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(f"(a) UQ routes ($\\rho={d['asymptotic']['rho']:.2f}$)")
    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(axis="x", labelrotation=30)
    ax = axes[1]
    pl = d["profile_likelihood"]
    bg = np.array(pl["beta_grid"])
    for key, col, mk, lab in [("cost_full_window", "#2a78d6", "o", "full window"),
                              ("cost_truncated_window", "#eb6834", "s", r"pre-peak ($0.3\,t_{\rm peak}$)")]:
        y = np.array(pl[key])
        ax.semilogy(bg, y / y.min(), marker=mk, ms=2.5, color=col, label=lab)
    ax.axvline(d["theta_true"][0], color=MUTED, ls=":", lw=0.8)
    ax.set_xlabel(r"$\beta$ ($\gamma$ profiled out)")
    ax.set_ylabel("cost / min cost")
    ax.set_title("(b) profile likelihood")
    ax.set_ylim(0.7, 3e3)
    ax.legend(loc="upper center", fontsize=5.5)
    fig.tight_layout(w_pad=0.5, rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6, markerscale=1.8,
               handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    save(fig, "uq")


def fig_crossover() -> None:
    d = load("E13")
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    offsets = [(-42, 5), (6, -12), None]
    for row, col, mk, off in zip(d["rows"], colors, markers, offsets):
        sig = [c["sigma_frac"] for c in row["sigma_curve"]]
        std = [c["std"][0] for c in row["sigma_curve"]]
        bias = abs(row["solver_bias"][0])
        ax.loglog(sig, std, marker=mk, color=col, label=f"spread, $h={row['h']:g}$")
        ax.axhline(bias, color=col, ls="--", lw=1.0)
        s_star = row["crossover_sigma_star"].get("beta")
        if s_star:
            ax.plot([s_star], [bias], marker="*", ms=8, color=col, mec=INK, mew=0.5, ls="none")
            ax.annotate(f"$\\sigma^*={s_star:.3f}$", xy=(s_star, bias), xytext=off,
                        textcoords="offset points", fontsize=6.5, color=INK)
    ax.plot([], [], color=MUTED, ls="--", lw=1.0, label="solver bias $|b(h)|$ (dashed)")
    ax.annotate(r"$h=0.5$: $\sigma^*>0.2$", xy=(0.03, abs(d["rows"][-1]["solver_bias"][0])), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xlabel(r"noise level $\sigma_{\rm frac}$")
    ax.set_ylabel(r"error in $\hat\beta$ (Euler)")
    ax.legend(loc="lower right")
    save(fig, "crossover")


def fig_eyam() -> None:
    d = load("E14")
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    ax.plot(d["days"], d["s_observed"], "o", color=INK, ms=4, label="observed $S$ (Eyam 1666)")
    for solver, ls in [("rk4", "-"), ("euler", "--")]:
        c, m, lab = SOLVER_STYLE[solver]
        r = d["results"][solver]
        ax.plot(d["days"], r["predicted_s"], ls=ls, marker=m, ms=3, color=c,
                label=f"{lab} fit, $\\hat{{\\mathcal{{R}}}}_0={r['r0']:.2f}$")
    ax.set_xlabel("days after 18 June 1666")
    ax.set_ylabel("susceptible $S$")
    ax.legend(loc="upper right")
    save(fig, "eyam")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    style()
    fig_teaser()
    fig_convergence()
    fig_invariants()
    fig_bias()
    fig_noise_sampling()
    fig_uq()
    fig_crossover()
    fig_eyam()

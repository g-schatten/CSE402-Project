"""Figures for the supervisor slides (supervisor_slides/slides.tex).

Everything is recomputed from the project's own code (core/sirlab and the
experiment modules, quick profile), so the slide numbers match the report.
Run from the repository root:

    python supervisor_slides/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_style import COMPARTMENT_COLOR, SOLVER_COLOR, apply_style  # noqa: E402

from experiments import E01_convergence, E13_crossover  # noqa: E402
from sirlab.estimate.base import ForwardModel  # noqa: E402
from sirlab.estimate.lm import levenberg_marquardt  # noqa: E402
from sirlab.models.sir import SIR  # noqa: E402
from sirlab.observe import prevalence  # noqa: E402
from sirlab.reference import SIRReference  # noqa: E402
from sirlab.solvers import integrate  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
N, I0 = 1000.0, 1.0
BETA, GAMMA = 0.3, 0.1
LABEL = {"euler": "Euler", "heun": "Heun", "rk4": "RK4", "backward_euler": "Backward Euler", "trapezoidal": "Trapezoidal"}


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_epidemic() -> None:
    """Exact curve vs Euler/RK4 at h = 1 day: the solver draws a wrong curve."""
    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=N - I0, i0=I0)
    t = np.linspace(0, 100, 401)
    exact = ref.trajectory(t)[:, 1]
    model, y0, theta = SIR(N), np.array([N - I0, I0, 0.0]), np.array([BETA, GAMMA])
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(t, exact, color="#131a24", lw=2.2, label="Exact (gold standard)")
    for solver in ["euler", "rk4"]:
        res = integrate(model, y0, theta, (0.0, 100.0), solver=solver, h=1.0)
        ax.plot(res.t, res.y[:, 1], color=SOLVER_COLOR[solver], lw=1.6, ls="--", label=f"{LABEL[solver]}, $h=1$ day")
    ax.set_xlabel("day")
    ax.set_ylabel("infectious $I(t)$")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "epidemic_curve")


def fig_convergence() -> None:
    """E01, R0 = 3 regime: global error at T = 40 vs h against the gold standard."""
    cfg = yaml.safe_load(open(ROOT / "configs" / "e01.yaml"))
    cfg["_profile"] = "quick"
    regime = E01_convergence.run(cfg)["regimes"][0]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for solver, r in regime["solvers"].items():
        ax.loglog(r["h"], r["error_final_state"], "o-", color=SOLVER_COLOR[solver], lw=1.6, ms=4,
                  label=f"{LABEL[solver]} ($\\hat p$={r['observed_order']:.2f})")
    ax.set_xticks([0.125, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1"])
    ax.minorticks_off()
    ax.set_xlabel("step size $h$ (days)")
    ax.set_ylabel("error at $t=40$")
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "convergence")


def fig_bias() -> None:
    """E07 mechanism: noise-free data, |beta_hat - beta| vs fitting step size h."""
    ref = SIRReference(beta=BETA, gamma=GAMMA, n_total=N, s0=N - I0, i0=I0)
    t_obs = np.arange(1.0, 60.0 + 1e-9, 1.0)
    obs = ref.trajectory(t_obs)[:, 1]
    hs = [0.5, 0.25, 0.1, 0.05]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for solver, p in [("euler", 1), ("heun", 2), ("rk4", 4)]:
        bias = []
        for h in hs:
            fwd = ForwardModel(SIR(N), np.array([N - I0, I0, 0.0]), t_obs,
                               lambda res, t: prevalence(res, i_index=1, t_obs=t), solver=solver, h=h)
            fwd.obs_state_index = 1
            fit = levenberg_marquardt(fwd, obs, np.array([BETA * 1.2, GAMMA * 0.8]), max_iter=100)
            bias.append(abs(fit.theta_hat[0] - BETA))
        print(solver, dict(zip(hs, bias)))
        ax.loglog(hs, bias, "o-", color=SOLVER_COLOR[solver], lw=1.6, ms=4, label=f"{LABEL[solver]} (order {p})")
    ax.set_xticks(hs)
    ax.set_xticklabels(["1/2", "1/4", "1/10", "1/20"])
    ax.minorticks_off()
    ax.set_xlabel("fitting step size $h$ (days)")
    ax.set_ylabel(r"$|\hat\beta - \beta_{\mathrm{true}}|$, noise-free data")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "bias_vs_h")


def fig_crossover() -> None:
    """E13 (quick profile): Euler bias vs noise-induced spread; sigma* marked."""
    cfg = yaml.safe_load(open(ROOT / "configs" / "e13.yaml"))
    cfg["_profile"] = "quick"
    rows = E13_crossover.run(cfg)["rows"]
    for row in rows:
        print("h", row["h"], "bias", row["solver_bias"], "sigma*", row["crossover_sigma_star"])
    row = next(r for r in rows if r["h"] == 0.1)
    sig = np.array([c["sigma_frac"] for c in row["sigma_curve"]])
    std = np.array([c["std"][0] for c in row["sigma_curve"]])
    bias = abs(row["solver_bias"][0])
    s_star = row["crossover_sigma_star"]["beta"]

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    lo, hi = sig.min() * 0.8, sig.max() * 1.25
    ax.axvspan(lo, s_star, color="#d43f4e", alpha=0.07, lw=0)
    ax.axvspan(s_star, hi, color="#2f7fd6", alpha=0.07, lw=0)
    ax.loglog(sig, std, "o-", color="#2f7fd6", lw=1.8, ms=4.5, label="noise spread (std over 25 datasets)")
    ax.axhline(bias, color="#d43f4e", ls="--", lw=1.6, label="solver bias (noise-free fit)")
    ax.axvline(s_star, color="#131a24", ls=":", lw=1.2)
    ax.annotate(f"$\\sigma^* = {s_star:.3f}$", xy=(s_star, bias), xytext=(6, -16), textcoords="offset points", fontsize=9)
    ax.text(0.30, 0.06, "solver bias\ndominates", transform=ax.transAxes, fontsize=8, color="#a32532")
    ax.text(0.97, 0.06, "noise\ndominates", transform=ax.transAxes, fontsize=8, color="#1f5fa8", ha="right")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"noise level $\sigma$ (fraction of peak $I$)")
    ax.set_ylabel(r"error in $\hat\beta$ (Euler, $h=0.1$)")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "crossover")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    apply_style()
    fig_epidemic()
    fig_convergence()
    fig_bias()
    fig_crossover()

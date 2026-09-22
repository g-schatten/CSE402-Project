# Final presentation — outline

Build the actual .pptx from this outline with the `pptx` skill (or your preferred slide tool),
pulling figures from `figures/*.pdf` (export to PNG at build time) and numbers from `results/*.json`.
Target ~15 slides, live demo of the web dashboard in the middle rather than static screenshots.

1. **Title** — Numerical Analysis of the SIR Epidemic Model: Solver Accuracy, Parameter Recovery,
   and Robustness to Noise. CSE 402, Section B, Group 2.

2. **The hook** — one sentence: *there is a noise level σ\* below which your choice of ODE solver
   determines whether your epidemic parameter estimates are trustworthy, and above which it doesn't
   matter at all.* Show the E13 crossover figure immediately, before any methods.

3. **The gap in the base paper** — Capaldi et al. (2012) estimate SIR parameters and quantify their
   uncertainty, but treat the ODE solver as a black box. We ask what's inside that box.

4. **Research questions** — RQ1 (solver behavior), RQ2 (solver → parameter propagation), RQ3
   (noise/sampling robustness). One slide, matches the Overview page of the dashboard.

5. **Method in one slide** — the semi-analytic gold standard (phase relation + quadrature, eq. 2.1–2.2),
   the five solvers, the least-squares + sensitivity-equation estimator. Keep equations minimal;
   this is the slide people forget fastest, so it should carry the least novel information.

6. **The mass-conservation trap** — a genuinely surprising mid-talk beat: "we checked S+I+R=N like
   everyone does — it's satisfied to roundoff even when the solve is completely wrong." Show E02.

7. **Live demo: Solver Arena** — switch to the dashboard, drag h, watch Euler/Heun/RK4 diverge from
   the gold standard and the convergence-order slope update live.

8. **RQ1 results** — convergence order table (E01), stability frontier (E03).

9. **Live demo: Fitting Studio** — watch Gauss-Newton and Nelder-Mead descend the same cost
   landscape from the same start point.

10. **RQ2 results** — the E07 bias-vs-h log-log plot, annotated with the observed slopes (≈1, 2, 4).

11. **RQ3 results** — noise degradation (E08), sampling/window truncation and the identifiability
    collapse (E09), the four-way UQ overlay (E11).

12. **The headline finding, in full** — the E13 crossover figure again, now explained: below σ*,
    solver choice dominates; above it, statistically indistinguishable. State σ* for the reference
    (β, γ, h) case.

13. **Extensions** — SEIR/SIRS/vaccination (E12) preserve the ordering; Eyam 1666 (E14) as a
    real-data demonstration, with the data-provenance caveat stated plainly.

14. **Limitations** — quick-profile numbers in this talk; full-profile run intended as a follow-up;
    synthetic-data design; single base-model family tested at full depth.

15. **Conclusion + team** — one-line takeaway, contributions, thanks, link to the dashboard and repo.

**Backup slides:** dashboard screenshots (in case of no live network/projector issues), the LU/QR
condition-number identifiability point (β–γ correlation), MCMC diagnostics (R̂, ESS, trace plots),
full E07 factorial table.

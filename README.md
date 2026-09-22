# SIRLab — Numerical Analysis of the SIR Epidemic Model

**CSE 402: Numerical Analysis, Simulation & Modeling — Section B, Group 2**

A rigorous numerical-analysis study of the SIR epidemic model, paired with a polished interactive
web dashboard. We treat the ODE solver itself as the central experimental variable — extending
Capaldi et al. (2012), which estimates SIR parameters but treats the solver as a black box.

**The headline finding:** there exists a *crossover noise level σ\** at which solver-induced bias
in the recovered (β, γ) is exactly matched by noise-induced variance. Below σ\*, the choice of ODE
solver is the dominant error source; above it, solver refinement buys nothing distinguishable from
noise. See [`PLAN.md`](PLAN.md) for the full research plan, and the **Crossover σ\*** page of the
dashboard for the measured curve.

## Repository layout

```
core/sirlab/       hand-written numerical library (models, solvers, linalg, estimation, UQ)
experiments/       the 14-experiment registry (E01-E14) + runner
configs/           one YAML per experiment (every knob lives here)
results/           generated JSON artifacts (git-ignored; regenerate with `make experiments`)
figures/           generated .pdf/.svg report figures (never pasted in by hand)
tests/             pytest suite (unit, order-verification, estimator, UQ, hygiene)
web/               Vite + React + TypeScript interactive dashboard
scripts/           figure generation, TS/Python parity fixtures
notebooks/         narrative Jupyter notebooks per research question
data/raw/          Eyam 1666 dataset + provenance note
report/            LaTeX report
deck/              presentation outline
```

## Quickstart

```bash
make setup                       # venv + Python deps + npm install
make test                        # Python test suite (50 tests)
make experiments                 # run all 14 experiments (quick profile, ~10-15 min)
make web                         # start the dashboard dev server at localhost:5173
make figures                     # regenerate report figures from results/*.json
```

Run one experiment at higher fidelity: `make experiment ID=E07 PROFILE=full` (see PLAN.md §5 —
the `full` profile matches the paper-scale factorial and is intended for an offline/overnight run,
not an interactive session).

## Design principles

- **`core/sirlab` never imports SciPy or matplotlib** (enforced by `tests/test_hygiene.py`). Every
  solver, optimizer, and root-finder is hand-written; SciPy/NumPy's own routines appear only in
  tests, as an independent cross-check.
- **The semi-analytic gold standard** (`core/sirlab/reference.py`) gives an SIR trajectory accurate
  to machine precision with no ODE solver in the loop, so every accuracy claim in this project is
  measured against ground truth, not against "a finer numerical solve."
- **The web dashboard ships its own solver implementations** (`web/src/numerics/`), parity-tested
  against the Python core to ~1e-9 (`web/tests/parity.test.ts`), so every slider recomputes live in
  the browser while heavier factorial sweeps load from precomputed JSON.
- **Every result is reproducible.** Each experiment is seeded, content-hash cached
  (`sirlab.io.manifest`), and traceable to a specific artifact in `results/`.

## Status

Core numerics, all 14 experiments (quick profile), and the 10-page dashboard are implemented and
tested. The `full` profile (paper-scale factorial sizes, e.g. E07's ~1.9M-fit design) has not been
run in this session — see PLAN.md §9 for the build order and §12 for open items, including a data
provenance caveat on the Eyam 1666 dataset (`data/raw/provenance.md`).

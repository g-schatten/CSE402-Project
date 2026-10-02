# SIRLab — Numerical Analysis of the SIR Epidemic Model

**CSE 402: Numerical Analysis, Simulation & Modeling — Section B, Group 02**

| Member | Student ID |
|---|---|
| Saif Uz Zaman | 2105088 |
| Sayaad Muzahid Masfi | 2105066 |
| Ibtida bin Ahmed | 2105061 |
| Sakif Naieb Raiyan | 2105065 |
| Aurchi Chowdhury | 2105083 |

A numerical-analysis study of the SIR epidemic model, paired with an interactive web dashboard.
Our base paper, Capaldi et al. (2012), estimates SIR parameters by least squares but treats the
ODE solver as a black box. We make the solver and its step size the central experimental variables.

**Headline finding.** Solver truncation error passes into the fitted (β, γ) with the solver's own
convergence order, and it competes with noise-induced spread. The two are equal at a *crossover
noise level σ\**. Below σ\* the solver dominates the estimation error; above it, noise does. For
forward Euler, σ\* ≈ 5.8% of peak prevalence at h = 0.1 day and 15.6% at h = 0.25.

**Final report:** [`report/B_02.pdf`](report/B_02.pdf) (source: [`report/B_02.tex`](report/B_02.tex),
ACM `sigconf` template).

## Research questions

From the project proposal
([`presentations/SectionB_Group2_SIR_Project_Proposal.pptx`](presentations/SectionB_Group2_SIR_Project_Proposal.pptx)):

- **RQ1. Solver comparison:** how do forward Euler, Heun and RK4 differ in accuracy, convergence
  order and stability on the SIR system?
- **RQ2. Parameter recovery:** how does the choice of solver propagate into (β, γ) recovered by
  least squares?
- **RQ3. Noise and sampling robustness:** how robust is recovery to noise, sampling frequency,
  initial conditions and the true (β, γ)?
- **RQ4. Extension:** do the findings hold for SEIR / SIRS / SIR-with-vaccination and for real data
  (Eyam, 1666)?

## Repository layout

```
core/sirlab/       hand-written numerical library: models, solvers, linalg (LU, QR, eigen),
                   quadrature, roots, estimation, uncertainty quantification, gold standard
experiments/       the 14-experiment registry (E01-E14) and its runner
configs/           one YAML per experiment, each with a `quick` and a `full` profile
results/           generated JSON results (git-ignored; regenerate with the runner)
tests/             pytest suite (82 tests)
web/               Vite + React + TypeScript dashboard (10 pages) with its own TS solvers
scripts/           quick-look figures, TS/Python parity fixtures, notebook generator
notebooks/         one Jupyter notebook per research question (RQ1-RQ4)
data/raw/          Eyam 1666 dataset and its source (provenance.md)
report/            final report B_02.pdf and its sources: B_02.tex, refs.bib, ACM template
                   files, figures/ and make_report_figures.py
presentations/     project proposal, final presentation (pptx + html), supervisor slides
PLAN.md            the project's original research plan (code docstrings cite its sections)
```

## Experiments

| ID | Name | RQ | ID | Name | RQ |
|---|---|---|---|---|---|
| E01 | Convergence & order | 1 | E08 | Noise degradation | 3 |
| E02 | Invariant drift | 1 | E09 | Sampling & window | 3 |
| E03 | Stability frontier | 1 | E10 | Initial conditions | 3 |
| E04 | Structural accuracy | 1 | E11 | Uncertainty quantification | 3 |
| E05 | Cost landscape | 3 | E12 | Model variants | 4 |
| E06 | Optimizer shoot-out | 3 | E13 | Crossover σ\* | 2+3 |
| E07 | Solver-in-the-loop recovery | 2 | E14 | Real data (Eyam 1666) | 4 |

## Quickstart

Requires Python ≥ 3.11, and Node.js for the dashboard. These commands work on any OS:

```bash
pip install -e ".[dev]"                              # sirlab + numpy, pyyaml; dev: scipy, matplotlib, pandas, pytest, hypothesis
pytest tests/ -q                                     # 82 tests
python -m experiments.runner --all --profile quick   # all 14 experiments, ~17 min with 8 workers
python -m experiments.runner E07 --profile full      # one experiment at paper scale (slow)
python report/make_report_figures.py                 # report figures from results/*.json

mkdir -p web/public/data && cp results/*.json web/public/data/   # results for the dashboard
cd web && npm ci
npx vitest run                                       # TS/Python parity test (9 cases)
npm run dev                                          # dashboard at localhost:5173
```

The notebooks read `results/*.json` and open in Jupyter (`pip install jupyter`). On Linux/macOS
the [`Makefile`](Makefile) wraps the same steps (`make setup`, `make test`, `make experiments`,
`make figures`, `make web`, `make report`). The report compiles with pdfLaTeX + BibTeX, either on
Overleaf (upload the `report/` folder) or locally with `cd report && latexmk -pdf B_02.tex`.

## Design principles

- **`core/sirlab` never imports SciPy or matplotlib** (enforced by `tests/test_hygiene.py`). Every
  solver, optimizer, root finder and eigenvalue routine is hand-written. SciPy/NumPy routines appear
  only in the tests, as independent cross-checks. Random numbers come from NumPy's PCG64 generator.
- **A semi-analytic gold standard** (`core/sirlab/reference.py`) gives the SIR trajectory to near
  machine precision without any ODE solver. It uses the exact phase-plane reduction, adaptive
  Simpson quadrature and safeguarded Newton. Every accuracy claim is measured against it, not
  against "a finer numerical solve".
- **The dashboard ships its own solvers** (`web/src/numerics/`), parity-tested against the Python
  core (`web/tests/parity.test.ts`: solver states to 1e-9, gold standard to 1e-6). The Model Lab,
  Solver Arena and Stability pages recompute live; the other pages load precomputed results.
- **Every result is reproducible.** Each experiment is seeded (base seed `k` for `Ek`, with
  per-replicate CRC32-derived seeds). Each run is recorded in `results/manifest.json` (config hash,
  git commit, seed, wall time, host) and skipped when its config is unchanged.

## Data

`data/raw/eyam_1666.csv` holds the susceptible and infective counts for the Eyam plague, 18 June to
20 October 1666, from Raggett (1982), *A stochastic model of the Eyam plague*, Journal of Applied
Statistics 9(2):212–225. The values follow the table as reproduced in arXiv:1603.03819 (Table 1)
and arXiv:1809.07139 (Table 4). See [`data/raw/provenance.md`](data/raw/provenance.md).

## Status

All 14 experiments, the test suite and the dashboard are implemented, and the tests pass. 

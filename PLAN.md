# CSE 402 — Numerical Analysis of the SIR Epidemic Model
## Master Plan (Section B, Group 2)

> **Status:** implemented (core library, 14 experiments at the `quick` profile, dashboard, final
> report in `report/`). This file is kept as the original research plan; the code docstrings cite
> its sections. Where it and the code differ, the code and `README.md` describe what was built.
> **Audience:** the 5-person group + the coding agent that will build this.
> **Source of truth for scope.** If code and this document disagree, fix one of them deliberately.

---

## 0. The one-paragraph version

We build **SIRLab**: a rigorous numerical-analysis study of the SIR epidemic model plus a
polished interactive web showcase of it. The scientific core is a hand-written Python library
(Euler / Heun / RK4 / adaptive RK45 / backward Euler, forward sensitivity equations,
least-squares parameter estimation, and uncertainty quantification) that runs a registry of
reproducible experiments and exports machine-readable result artifacts. The showcase is a
static React + TypeScript dashboard that ships its **own** implementation of the same solvers,
so every slider recomputes live in the browser, while the heavy factorial sweeps are loaded
from the precomputed artifacts. A parity test proves the two implementations agree to ~1e-12.

**The headline result we are chasing** (this is the thing that makes the project more than a
lab exercise):

> There exists a **crossover noise level** σ\* at which solver-induced bias in the recovered
> (β, γ) is exactly matched by noise-induced variance. Below σ\*, your choice of ODE solver
> is the dominant error source and Euler is scientifically indefensible. Above σ\*, Euler at a
> sane step size is *statistically indistinguishable* from RK4 and paying for RK4 buys nothing.
> We locate σ\*(h, Δt, β, γ) empirically and explain it.

That single sentence is RQ1, RQ2 and RQ3 fused into one defensible, quotable finding.

---

## 1. Research questions and what counts as an answer

From the proposal, restated as falsifiable claims with concrete acceptance criteria.

### RQ1 — Solver behaviour
*How do Forward Euler, Heun and RK4 differ in accuracy, observed convergence order and
numerical stability on the SIR system?*

**Answer = ** a work-precision diagram plus a table of *observed* convergence orders
(p̂ ≈ 1, 2, 4 to within ±0.05 over the asymptotic regime), a stability boundary in (h, β, γ)
where Forward Euler first produces a negative compartment or diverges, and a demonstration
that N-conservation is a *useless* accuracy diagnostic here while the phase invariant is a
good one (see §2.5 — this is a real and slightly counter-intuitive finding).

### RQ2 — Solver → parameter propagation
*How does solver choice propagate into (β̂, γ̂) recovered by least squares?*

**Answer = ** for a fixed noise-free synthetic dataset, a curve of |β̂ − β_true| versus solver
step size h for each solver, showing that estimation bias inherits the solver's convergence
order (slope 1 / 2 / 4 on log-log). Plus the *asymmetry* result: β and γ are not equally
contaminated, because the cost functional's valley is aligned with the β/γ ratio.

### RQ3 — Robustness to noise and sampling
*How robust is recovery to observation noise, sparse sampling, initial conditions and the
region of (β, γ) space?*

**Answer = ** RMSE surfaces over (σ, Δt), the identifiability failure when the observation
window is truncated before the epidemic peak, the β–γ correlation coefficient and its
condition-number explanation, and the crossover σ\* described above.

### RQ4 (extension) — Does any of this survive contact with a harder model or real data?
Repeat the core pipeline on SEIR / SIRS / SIR+vaccination, and on the Eyam 1666 plague
outbreak. **Answer = ** either "yes, the ordering holds" or "no, and here is the mechanism".
Both are publishable-quality outcomes for a course project.

---

## 2. Mathematical specification

Everything in this section must be implemented exactly as written. Equation numbers are
referenced by the module specs in §4.

### 2.1 Models

All models are autonomous first-order systems ẏ = f(y; θ) with a conserved total population N.

**SIR** (state y = (S, I, R), θ = (β, γ)):

```
dS/dt = −β S I / N
dI/dt = +β S I / N − γ I
dR/dt = +γ I
```

**SEIR** (y = (S, E, I, R), θ = (β, σ_inc, γ)):

```
dS/dt = −β S I / N
dE/dt = +β S I / N − σ_inc E
dI/dt = +σ_inc E − γ I
dR/dt = +γ I
```

**SIRS** — SIR plus waning immunity at rate ξ: add `+ξR` to dS/dt and `−ξR` to dR/dt.

**SIR + vaccination** (y = (S, I, R, V), θ = (β, γ, ν)): add `−νS` to dS/dt and
`dV/dt = +νS`. V is absorbing and counts toward N.

Derived quantities: `R₀ = β/γ` (SIR, SIRS, and SEIR — the exposed stage does not change R₀,
only the timing), effective reproduction number `R_eff(t) = R₀ · S(t)/N`.

### 2.2 The semi-analytic gold standard (methodological backbone — do this first)

We do **not** measure solver error against "RK4 at a very small h". That is circular. For SIR
we have an exact reduction.

Divide dS/dR:  `dS/dR = −(β/(γN)) S`, giving the **phase invariant**

```
(2.1)   S(t) = S₀ · exp( −(β/(γN)) · (R(t) − R₀ᵢₙᵢₜ) )
        equivalently   Q(t) := ln S(t) + (β/(γN))·R(t)  is constant in t
```

Substituting into `dR/dt = γI = γ(N − R − S)` gives a **scalar separable ODE**, so

```
(2.2)   t(R) = ∫₀^R  dr / [ γ · ( N − r − S₀ · exp(−(β/(γN))·r) ) ]
```

The integrand is smooth and strictly positive on [0, R∞). Evaluate (2.2) with adaptive
Gauss–Kronrod quadrature to ~1e-14, then invert `t(R)` by Newton's method to obtain R(t), and
recover S from (2.1) and I from `I = N − S − R`.

**This yields a reference trajectory accurate to machine precision using no ODE solver at
all.** All RQ1 error measurements are made against it. Implement, verify, and freeze this
before writing any experiment. (Supporting reference: Prodanov 2021, *Entropy* 23(1), 59.)

Three further **closed-form** quantities give solver-independent checks:

```
(2.3)  Peak:       S at peak = N/R₀,  I_max = I₀ + S₀ − N/R₀ + (N/R₀)·ln( N / (R₀·S₀) )
(2.4)  Final size: ln(S₀/S∞) = R₀ · (N − S∞)/N     — solve for S∞ by Newton/bisection
(2.5)  Mass:       S + I + R = N  for all t
```

Peak *time* has no closed form; it comes from the gold standard and is a headline accuracy
metric ("Euler at h = 0.5 misplaces the epidemic peak by 1.7 days").

### 2.3 Solvers (all hand-written, no SciPy in the solve path)

| # | Method | Order | f-evals/step | Notes |
|---|--------|-------|--------------|-------|
| 1 | Forward Euler | 1 | 1 | `y_{n+1} = y_n + h f(y_n)` |
| 2 | Heun (explicit trapezoid / RK2) | 2 | 2 | predictor `ỹ = y_n + h f(y_n)`, corrector `y_{n+1} = y_n + (h/2)(f(y_n) + f(ỹ))` |
| 3 | Classical RK4 | 4 | 4 | standard k₁..k₄ |
| 4 | Dormand–Prince RK45 (adaptive) | 5(4) | 6 (FSAL) | reference-quality, adaptive h, PI step controller |
| 5 | Backward Euler (implicit) | 1 | Newton iters | **A-stable**; Newton system solved by our own LU with partial pivoting |
| 6 | Trapezoidal / Crank–Nicolson (implicit) | 2 | Newton iters | optional, same machinery |

Solvers 5–6 exist for the stability deep-dive and to justify the Week 5–6 LU decomposition
topic: each Newton step solves `(I − hJ)δ = −G(y)` with our LU routine.

**Fixed operation order is mandatory** across the Python and TypeScript implementations so the
parity test can hold to 1e-12. Write the stage combination exactly as
`y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)` in both — not `y + h*k1/6 + ...`.

### 2.4 Forward sensitivity equations (RQ2 backbone)

Estimation needs ∂y/∂θ. We do **not** use finite differences. We augment the state with the
sensitivities and integrate them *with the same solver*, so solver error enters the Jacobian
exactly the way it enters the trajectory — which is precisely what RQ2 is asking about.

For parameter θⱼ, with `s_j = ∂y/∂θⱼ`:

```
(2.6)   d(s_j)/dt = J_y f · s_j + ∂f/∂θⱼ,      s_j(0) = 0  (if y₀ is θ-independent)
```

For SIR:

```
        ⎡ −βI/N     −βS/N        0 ⎤                 ⎡ −SI/N ⎤            ⎡  0 ⎤
J_y f = ⎢ +βI/N   +βS/N − γ      0 ⎥    ∂f/∂β =      ⎢ +SI/N ⎥   ∂f/∂γ =  ⎢ −I ⎥
        ⎣   0        +γ          0 ⎦                 ⎣   0   ⎦            ⎣ +I ⎦
```

Analytic Jacobians must also be implemented for SEIR / SIRS / vaccination, and **unit-tested
against complex-step differentiation** (`Im(f(y + ih e_k))/h`, exact to machine precision, no
subtractive cancellation) — not against finite differences.

### 2.5 Observation operators and noise models

Two observation types, both required (the difference matters and is a finding in itself):

- **Prevalence**: `obs_k = I(t_k)` — what the base paper fits.
- **Incidence**: `obs_k = ∫_{t_{k−1}}^{t_k} β S I / N dt` — what real surveillance reports.
  Computed by composite **Simpson / trapezoid** on the solver's own grid (Week 11 topic).
  Include the "reporting fraction" ρ ∈ (0,1] so only ρ·incidence is observed.

Noise models applied to the clean observation vector:

| Model | Form | Use |
|---|---|---|
| Additive Gaussian | `obs·(1) + N(0, σ²)`, σ = ρ_n · max(I) | matches the base paper, easy to sweep |
| Proportional Gaussian | `obs·(1 + N(0, σ²))` | heteroscedastic, more realistic |
| Poisson | `Poisson(obs)` | correct for count data; variance is *not* a free knob |
| Negative binomial | mean obs, dispersion k | overdispersed real surveillance data |

**Important subtlety to document:** under Poisson noise the noise level is not independently
tunable — it is set by the epidemic size. This makes N (population size) an implicit
noise-level knob, which is a nice thing to show interactively.

### 2.6 Estimation

Cost functional, with optional weights:

```
(2.7)   J(θ) = Σ_k w_k · ( obs_k − h(y(t_k; θ)) )²,     w_k = 1 or 1/max(obs_k, 1) (Poisson-ish)
```

Optimizers to implement, all hand-written (this is the Week 9–10 payload):

1. **Grid search** — coarse 2-D scan; also produces the J(β,γ) landscape used everywhere in
   the UI. Cache the whole grid; it is a first-class result artifact, not a by-product.
2. **Golden-section line search** inside coordinate descent — the textbook 1-D method,
   demonstrated on the same landscape.
3. **Nelder–Mead** — derivative-free baseline; its simplex is beautiful to animate.
4. **Gauss–Newton** — `(JᵀJ) δ = −Jᵀ r`, J from §2.4, normal equations solved by **our LU**.
5. **Levenberg–Marquardt** — `(JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀ r` with λ adaptation. This is the
   production estimator; GN is the teaching one.

Also solve the normal equations a second way (Householder **QR** on J directly) and report the
condition numbers `κ(J)` vs `κ(JᵀJ) = κ(J)²`. This is a genuine numerical-analysis point:
near-degenerate β–γ identifiability squares the conditioning damage when you form JᵀJ.

Parameter positivity is handled by optimizing in log-space (`θ = exp(u)`), which also makes the
landscape better conditioned. Report both parameterizations at least once.

### 2.7 Uncertainty quantification

Four routes to a confidence region for (β, γ), to be **overlaid on one figure** — the money
plot of the UQ section:

1. **Asymptotic / linearized**: `Cov(θ̂) ≈ σ̂² (JᵀJ)⁻¹`, σ̂² = J(θ̂)/(n − p). Draw the 95 %
   ellipse from its eigendecomposition. Report ρ(β̂, γ̂).
2. **Residual bootstrap**: resample residuals B = 1000 times, refit, get the cloud.
3. **Monte Carlo over fresh noise realizations**: M = 500 new synthetic datasets from the same
   truth. This is the only one that measures true *bias*, because it knows the ground truth.
4. **Metropolis–Hastings MCMC**: Gaussian likelihood, σ sampled as a nuisance parameter,
   adaptive proposal covariance after burn-in. Report trace plots, acceptance rate (target
   ~0.234 for the 2-D random-walk), integrated autocorrelation time, effective sample size, and
   **Gelman–Rubin R̂ across 4 chains** (a project that reports R̂ looks serious; one that shows
   a single trace does not).

Plus **profile likelihood** for β and γ: fix one, re-optimize the other, plot J vs the fixed
parameter. A flat profile is the visual signature of non-identifiability, and it will appear
clearly in the truncated-window experiment.

### 2.8 Diagnostics

```
Observed order (with reference):   p̂ = log( e(h₁)/e(h₂) ) / log( h₁/h₂ )
Observed order (Richardson, none): p̂ = log( ‖y_h − y_{h/2}‖ / ‖y_{h/2} − y_{h/4}‖ ) / log 2
Error norms:  global at T, max-over-time (L∞), time-L², per-compartment
Work metric:  cumulative f-evaluations (Euler 1, Heun 2, RK4 4, DP54 6 per accepted step)
Invariants:   mass error |S+I+R−N|;  phase-invariant drift |Q(t) − Q(0)|  from (2.1)
Stability:    max over the trajectory of h·|λ| for eigenvalues λ of J_y f; positivity violations
```

**Deliberate finding to state in the report:** because the SIR right-hand side sums to zero
identically, *every* explicit Runge–Kutta method conserves S+I+R to roundoff, at any step size,
including an Euler solve that is otherwise catastrophically wrong. Mass conservation therefore
has **zero diagnostic value** here, while the phase invariant (2.1) drifts at the solver's
convergence order and is a genuine accuracy proxy. Most naive treatments get this wrong.

---

## 3. Repository architecture

```
CSE402-Project/
├── PLAN.md                      ← this file
├── README.md                    project front page: what, how to run, headline figures
├── Makefile                     `make setup | core | experiments | figures | web | report | all`
├── pyproject.toml               package metadata + pinned deps
├── .github/workflows/ci.yml     pytest + parity test + web build on push
│
├── core/sirlab/                 the numerical library — pure, no file I/O, no plotting
│   ├── models/       base.py  sir.py  seir.py  sirs.py  vaccination.py
│   ├── solvers/      base.py  euler.py  heun.py  rk4.py  rk45.py  implicit.py
│   ├── linalg/       lu.py  qr.py            (hand-written, partial pivoting / Householder)
│   ├── roots/        newton.py  bisection.py  (final-size equation, t(R) inversion)
│   ├── quadrature/   adaptive.py  simpson.py  (gold standard + incidence)
│   ├── reference.py  §2.2 semi-analytic gold standard + (2.3)(2.4) closed forms
│   ├── sensitivity.py                         §2.4 augmented system
│   ├── observe.py                             §2.5 operators + noise
│   ├── estimate/     grid.py  golden.py  neldermead.py  gaussnewton.py  lm.py
│   ├── uq/           asymptotic.py  bootstrap.py  montecarlo.py  mcmc.py  profile.py
│   ├── diagnostics.py                         §2.8
│   └── io/           schema.py  export.py  manifest.py
│
├── experiments/      E01_convergence.py … E14_realdata.py  +  registry.py  +  runner.py
├── configs/          e01.yaml … e14.yaml   (every knob lives here, never in code)
├── results/          generated artifacts — manifest.json + per-experiment .json/.parquet
├── figures/          generated .pdf/.svg for the report
├── tests/            unit + property + golden-fixture + parity tests
├── notebooks/        RQ1_solvers.ipynb  RQ2_recovery.ipynb  RQ3_robustness.ipynb  RQ4_extensions.ipynb
├── data/raw/         eyam_1666.csv  bombay_1905.csv  (+ provenance.md with sources)
│
├── web/                         Vite + React 18 + TypeScript static dashboard
│   ├── src/numerics/            the TypeScript twin of core (solvers, models, estimators)
│   ├── src/workers/             Web Workers so solving never blocks the UI thread
│   ├── src/design/              tokens, theme, primitives
│   ├── src/charts/              D3-scale + React-SVG chart components (Canvas for hot paths)
│   ├── src/pages/               the 10 sections of §6
│   ├── src/state/               Zustand stores
│   ├── public/data/             copy of results/ exports, committed for static deploy
│   └── tests/parity.test.ts     TS ↔ Python golden-fixture agreement
│
├── report/           B_02.tex (ACM sigconf), refs.bib, figures/, make_report_figures.py
└── presentations/    proposal, final presentation, supervisor slides
```

**Hard rule:** `core/sirlab` never imports matplotlib, never writes files, never reads configs.
It takes numbers and returns numbers. Everything else is a consumer. This is what makes the
test suite, the notebooks, the experiment runner and the parity fixtures all possible.

**SciPy policy:** SciPy may be used *only* to cross-check our own routines in tests
(`solve_ivp`, `least_squares`, `lu_factor`). It must never appear in `core/sirlab` or in any
result-producing path. Add a test that greps for this.

---

## 4. Python core — module specs

Each entry: responsibility, key signature, acceptance test.

### 4.1 `models/base.py`
```python
class Model(Protocol):
    name: str; state_names: list[str]; param_names: list[str]
    def rhs(self, t: float, y: NDArray, theta: NDArray) -> NDArray: ...
    def jacobian_y(self, t, y, theta) -> NDArray: ...      # ∂f/∂y
    def jacobian_theta(self, t, y, theta) -> NDArray: ...  # ∂f/∂θ, shape (n_state, n_param)
    def initial_state(self, cfg) -> NDArray: ...
```
**Accept:** `jacobian_y` and `jacobian_theta` match complex-step differentiation to < 1e-12
for 1000 random states across all four models.

### 4.2 `solvers/base.py`
```python
@dataclass
class SolveResult:
    t: NDArray; y: NDArray          # y shape (n_steps, n_state)
    n_fev: int; n_steps: int; n_rejected: int
    h_history: NDArray | None
    diverged: bool; negativity_events: int

def integrate(model, y0, theta, t_span, h, solver: str, dense_output_at=None) -> SolveResult
```
Every solver takes and returns the same shapes. `dense_output_at` interpolates onto observation
times (linear for Euler/Heun, cubic Hermite for RK4, the DP54 dense-output polynomial for RK45)
— **do not** force observation times to lie on the solver grid; decoupling h from Δt is exactly
what RQ3 needs.

**Accept:** on `y' = λy` each solver reproduces its textbook stability function; observed order
on SIR against the §2.2 gold standard is 1.00 / 2.00 / 4.00 ± 0.05.

### 4.3 `linalg/lu.py`
Doolittle LU with partial pivoting, `lu_factor` / `lu_solve` / `det` / `cond_estimate`.
**Accept:** matches `scipy.linalg.lu_factor` to 1e-12 on 500 random well-conditioned matrices;
correctly raises on singular input; used successfully inside backward Euler.

### 4.4 `reference.py`
Implements §2.2 exactly. `reference_trajectory(theta, y0, N, t_eval) -> NDArray`, plus
`peak_analytic`, `final_size`, `phase_invariant`.
**Accept:** self-consistency — its own output satisfies (2.1) to 1e-13, (2.3) peak height to
1e-10, and agrees with RK45 at rtol=atol=1e-13 to < 1e-10. **This test gates everything else.**

### 4.5 `estimate/*`
Common interface:
```python
@dataclass
class FitResult:
    theta_hat: NDArray; cost: float; n_iter: int; n_fev: int; converged: bool
    path: NDArray          # every iterate, for the animated UI
    jacobian: NDArray | None; cov: NDArray | None; message: str
```
`path` is mandatory — the Fitting Studio page animates it.
**Accept:** on noise-free synthetic data, all five optimizers recover (β, γ) to < 1e-6 when the
solver is RK45-tight; GN/LM converge in < 15 iterations from a 50 %-off start.

### 4.6 `uq/mcmc.py`
Adaptive Metropolis (Haario), 4 chains, configurable burn-in/thin.
**Accept:** on a Gaussian toy posterior with known covariance, the recovered covariance is
within 5 %; R̂ < 1.01 on the real SIR posterior after the configured run length.

### 4.7 `io/`
- `schema.py` — dataclasses mirroring the JSON contracts in §5, with a `to_json`/validate pair.
- `manifest.py` — every artifact records: experiment id, config **content hash**, git SHA,
  library version, RNG seed, wall time, host. Reruns with an unchanged hash are skipped.
- `export.py` — writes `results/<exp_id>.json` plus `.parquet` for anything > 50k rows,
  and **downsamples curves for the web** (the browser does not need 20 000 points; use
  Douglas–Peucker or uniform-in-arc-length decimation to ~600 points, and say so in the file).

---

## 5. Experiment registry

Every experiment is a file in `experiments/`, a YAML in `configs/`, and an entry in
`registry.py`. `make experiments` runs them all in dependency order with a multiprocessing pool
and a progress bar; `python -m experiments.runner E07 --force` runs one.

| id | Name | Factors swept | Primary output |
|----|------|---------------|----------------|
| **E01** | Convergence & order | solver × h (geometric ladder, 12 values) × 3 (β,γ) regimes | global/L∞/L² error vs h; fitted p̂; work-precision |
| **E02** | Invariant drift | solver × h × t | mass error, phase-invariant drift, positivity violations over time |
| **E03** | Stability frontier | solver × h × (β, γ) grid 40×40 | boolean stable/unstable map; first-failure h; h·λ_max heatmap |
| **E04** | Structural accuracy | solver × h | peak height error, **peak time error**, final-size error vs (2.3)/(2.4) |
| **E05** | Cost landscape | (β, γ) grid 200×200 × noise level × observation type | J(β,γ) surface, valley axis, curvature, κ(JᵀJ) |
| **E06** | Optimizer shoot-out | 5 optimizers × 20 random starts × 2 parameterizations (linear/log) | iterations, f-evals, success rate, convergence paths |
| **E07** | ★ **Solver-in-the-loop recovery** | solver(3) × h(6) × σ(6) × Δt(5) × (β,γ)(9) × replicates(200) | β̂, γ̂ per trial → bias, variance, RMSE. **The centrepiece.** |
| **E08** | Noise degradation | σ ladder × solver, M = 500 | RMSE(β̂), RMSE(γ̂) vs σ, with bootstrap CIs on the RMSE itself |
| **E09** | Sampling & window | Δt ladder × window truncation (25 %…100 % of epidemic) | RMSE vs #observations; identifiability collapse before the peak |
| **E10** | Initial conditions | I₀ known vs estimated as a 3rd parameter × I₀ value | added variance from estimating I₀; κ of the 3-param JᵀJ |
| **E11** | Uncertainty quantification | asymptotic vs bootstrap vs Monte Carlo vs MCMC | overlaid 95 % regions, ρ(β̂,γ̂), profile likelihoods, R̂/ESS |
| **E12** | Model variants | model(4) × solver(3) × h | does the RQ1/RQ2 ordering survive higher dimension & stiffer timescales |
| **E13** | ★ **Crossover σ\*** | fine σ grid × h grid, solver-bias vs noise-std decomposition | σ\*(h) curve — **the headline finding** |
| **E14** | Real data | Eyam 1666 (+ Bombay 1905) × solver × observation model | fitted (β̂, γ̂), R₀, CIs, residual diagnostics vs literature values |

**E07 sizing.** 3·6·6·5·9·200 ≈ 1.94 M fits. Each fit is a few hundred RHS evaluations — this is
minutes-to-an-hour on a multiprocessing pool with a NumPy-vectorised RHS, not a supercomputer
job. Build it with a `--quick` profile (÷10 on replicates and grids) so development iterations
are seconds, and a `--full` profile for the final run. Every figure must render from both.

**Reproducibility contract.** One seed per (experiment, factor-combination, replicate), derived
by `SeedSequence(master_seed).spawn(...)`. Re-running any single cell reproduces its numbers
exactly without running its neighbours. Record the master seed in the manifest.

---

## 6. The showcase — interactive web dashboard

### 6.1 Architecture

```
core/ (Python)  ──experiments──▶  results/*.json  ──copy──▶  web/public/data/
      │                                                            │
      │ golden fixtures                                            ▼
      └──────────────▶  web/tests/parity.test.ts  ◀──  web/src/numerics/ (TS twin)
                                                            │
                                          live sliders ──▶ Web Worker ──▶ charts
```

Two kinds of content on every page:
- **Live**: recomputed in-browser on every slider drag by the TS twin, in a Web Worker,
  throttled to animation frames. Anything a single solve can produce.
- **Precomputed**: loaded from `public/data/*.json`. Anything requiring thousands of fits.

The user should never be able to tell which is which — that is the design goal.

### 6.2 Design language

Non-negotiables for "sleek and modern" that actually mean something:

- **Dark-first**, with a real light theme; both driven by CSS custom properties on `:root`,
  never by duplicated component styles. Every chart reads correctly in both.
- **One type scale** (a modular scale, e.g. 1.25 ratio), one geometric sans for UI
  (Inter / Geist), one mono for numbers (JetBrains Mono / Geist Mono). **Tabular figures on
  every number that updates live** — otherwise values jitter horizontally as you drag a slider,
  which is the single most common tell of an amateur dashboard.
- **Motion with purpose**: Framer Motion for layout transitions and page enters; *no* animation
  on data updates during a drag (it lags the pointer and reads as sluggish). Animate on
  discrete state changes only. Respect `prefers-reduced-motion`.
- **A restrained palette**: one semantic colour per compartment, fixed globally
  (S / E / I / R / V), one per solver (Euler / Heun / RK4 / RK45 / BE), fixed globally. A reader
  who learns the colours on page 2 must not have to relearn them on page 6. Use the `dataviz`
  skill's palette method; verify contrast in both themes.
- **Charts**: D3 for scales/shapes, React for the DOM. SVG for anything under ~2 000 marks,
  Canvas for the landscape heatmap and the bootstrap/MCMC clouds. Shared crosshair + linked
  hover across small multiples. Every axis labelled with units. Log axes explicitly marked.
- **Every figure is exportable** (SVG / PNG button) — these become report figures.
- **Layout**: a persistent left rail for navigation, a sticky parameter dock, and content in a
  max-width column. Must work down to 768 px (the projector at the defence is 4:3 — check it).
- **KaTeX** for every equation, rendered from the same LaTeX source strings as the report.

### 6.3 Pages

1. **Overview** — animated hero: the epidemic curve drawing itself, a live R₀ dial, and an
   SVG compartment-flow diagram whose arrow thicknesses pulse with the instantaneous flux
   βSI/N and γI. One scroll-triggered sentence per research question. Ends with the E13
   headline stated in one line.

2. **Model Lab** — choose SIR / SEIR / SIRS / Vaccination. Sliders for every parameter, N, I₀,
   T. Three linked views: time series (stacked-area toggle), **phase portrait** (S–I plane with
   the invariant curve (2.1) drawn as the exact solution — watch a coarse Euler solve peel off
   it), and R_eff(t) with the epidemic-threshold line at 1. Analytic peak and final size shown
   as ghost markers so the live solve can be seen hitting or missing them.

3. **Solver Arena** — the teaching centrepiece. Pick h; race Euler / Heun / RK4 against the
   §2.2 gold standard. Three panels: overlaid trajectories, |error|(t) on a log axis, and the
   log-log convergence plot with the fitted slope annotated live. A **step-inspector**: zoom to
   a single step and draw the k₁…k₄ slope vectors as arrows with their weights — this is the
   figure that makes RK4 click for an audience. Toggle x-axis between *step size* and *f-evals*
   to reveal that RK4 wins on cost-adjusted accuracy too.

4. **Stability Playground** — crank h upward and watch Forward Euler oscillate, go negative,
   then diverge, with the step count to first failure and the compartment that broke it called
   out. Side panel: the complex stability regions of each method with the trajectory's
   h·λ eigenvalues plotted as moving dots crossing the boundary. Backward Euler toggle for the
   contrast. Loaded from E03 for the (β, γ) frontier map.

5. **Fitting Studio** — generate a noisy dataset with sliders (noise model, σ, Δt, window,
   observation type), then run an optimizer and **watch it descend** on the live J(β,γ)
   heatmap: the iterate path animates, contours redraw, residuals update below. Switch
   optimizer to compare paths on the same landscape (Nelder–Mead's crawling simplex versus
   Gauss–Newton's three long jumps is a genuinely satisfying side-by-side). Switch the
   *internal* solver to expose RQ2 directly: the estimate lands somewhere else.

6. **Uncertainty** — the four-way overlay from §2.7 on one axes, toggleable layer by layer.
   MCMC trace plots that animate as the chain is (re)played from the stored samples, the
   β–γ correlation ellipse with its number, and the two profile-likelihood curves side by side,
   one sharp and one flat.

7. **Experiment Explorer** — the E07 factorial, exposed. Faceted small multiples with
   cross-filtering: click a solver, the whole grid filters; brush a σ range, everything
   responds. This is where a reader convinces themselves we actually ran the sweep.

8. **Crossover** — one page, one idea, E13. Solver-bias curve and noise-std curve on the same
   axes, their intersection marked and annotated, with a slider for h that moves σ\*.

9. **Real Data** — Eyam 1666 with its story, the fit, residuals, our R̂₀ against the literature
   value, and an honest panel on what the SIR model gets wrong about a real outbreak.

10. **Methods & Team** — the full equation set in KaTeX, the derivation of (2.2), the solver
    tableaux, references, the five names, and links to the repo/report.

### 6.4 Parity

`scripts/gen_fixtures.py` writes `web/tests/fixtures/*.json`: for a grid of (model, θ, y₀, h,
T, solver), the full Python trajectory at float64. `parity.test.ts` runs the TS twin over the
same inputs and asserts max relative difference < 1e-12 (solvers) and < 1e-9 (estimators, which
involve iteration counts and may legitimately differ in the last ulp). CI fails on drift. This
test is what lets us claim the dashboard is showing the same science as the report.

---

## 7. Testing and reproducibility

| Layer | What |
|---|---|
| Unit | every linalg / quadrature / root-finding routine vs SciPy or closed form |
| Structural | Jacobians vs complex-step (§2.4); conservation laws; positivity for small h |
| **Order verification** | the canonical numerical-analysis test: observed p̂ vs theoretical for all solvers on 3 problems (scalar linear, SIR, SEIR) |
| Gold standard | `reference.py` self-consistency + agreement with tight RK45 (§4.4) |
| Property-based | Hypothesis: for any valid (β, γ, y₀), S is non-increasing, R is non-decreasing, all compartments in [0, N] |
| Estimator | exact recovery on noise-free data; unbiasedness over M replicates on noisy data (a one-sample t-test on the bias) |
| Golden | experiment outputs pinned by hash for a `--quick` profile, so refactors cannot silently change numbers |
| Parity | §6.4 |
| Hygiene | no SciPy in `core/sirlab`; no matplotlib in `core/sirlab`; every config key consumed |

`make all` from a clean clone regenerates every number, every figure and the PDF. That is the
reproducibility claim, and it should be stated in the report's abstract.

---

## 8. Report and deck

**Report** (`report/B_02.tex`, LaTeX, ACM `sigconf` template as required by the course). The
structure follows the faculty's final-report guidelines: Title (group, section, members and
IDs) · Abstract (150–250 words) · Introduction & Base Paper Review · Methodology ·
Implementation & Experiments · Results & Discussion (RQ1 / RQ2 / RQ3 / crossover / RQ4) ·
Conclusion & Future Work · References, with a link to this repository.

Figures are **never** pasted in. `report/figures/` is generated by
`python report/make_report_figures.py` from `results/`, and LaTeX includes them by path. A stale
figure should be impossible.

Matplotlib style file shared with the web palette so the report and the dashboard are visibly
the same project. PDF/vector output, Type 1 fonts, no rasterised text.

**Deck** (`presentations/final/`): ~15 slides, same palette, built around a live demo of the dashboard rather
than screenshots of it. Structure: hook (the crossover claim) → the gap in the base paper →
method in one slide → live demo → three results → limitations → conclusion. Screenshots go in
the backup slides in case the projector fights us.

---

## 9. Build order

Strictly sequential at the phase level; parallel within a phase.

**Phase 0 — Foundation.** venv + pyproject + Makefile + CI skeleton + repo tree. numpy,
matplotlib, pandas, pytest, hypothesis, pyyaml, tqdm; scipy as a *dev* dependency only.

**Phase 1 — Numerical bedrock.** models + Jacobians (complex-step tested), linalg/LU, QR,
quadrature, root-finding, **`reference.py` (§2.2)**. *Nothing else starts until the gold
standard passes its self-consistency test.*

**Phase 2 — Solvers.** All six, dense output, diagnostics, order-verification tests.
→ Unlocks **E01–E04**. First real results exist here.

**Phase 3 — Estimation.** Sensitivity system, observation operators, noise models, the five
optimizers. → Unlocks **E05–E07**.

**Phase 4 — UQ.** Asymptotic, bootstrap, Monte Carlo, MCMC, profile likelihood.
→ Unlocks **E08–E11, E13**.

**Phase 5 — Extensions.** SEIR / SIRS / vaccination; Eyam data ingestion.
→ Unlocks **E12, E14**.

**Phase 6 — Web foundation.** Vite/React/TS scaffold, design tokens, theme, chart primitives,
the TS twin, Web Workers, fixtures + parity test. Can start in parallel with Phase 3 — the
solver fixtures it needs exist after Phase 2.

**Phase 7 — Web pages.** In order: Model Lab → Solver Arena → Fitting Studio → Stability →
Uncertainty → Experiment Explorer → Crossover → Real Data → Overview → Methods. (Overview is
built late deliberately: it is a summary, and you cannot summarise what does not exist.)

**Phase 8 — Notebooks, figures, report, deck.** Notebooks narrate what the experiments already
computed; they must not recompute anything.

**Phase 9 — Polish.** Full-profile experiment run, accessibility pass, 4:3 projector check,
deploy to GitHub Pages, rehearse.

### Suggested ownership lanes (5 members)

| Lane | Scope | Suggested owner |
|---|---|---|
| A · Numerics core | Phases 1–2, order verification, gold standard | Saif Uz Zaman |
| B · Estimation & UQ | Phases 3–4, E05–E11, E13 | Ibtida bin Ahmed |
| C · Web engineering | Phases 6–7, TS twin, parity, charts | Sakif Naieb Raiyan |
| D · Experiments & data | Runner, configs, E12/E14, real-data provenance, figures | Aurchi Chowdhury |
| E · Report, deck, demo | Phase 8, narrative, defence rehearsal, integration QA | Sayaad Muzahid Masfi (presenter) |

Every member writes tests for their own lane. Lane E also owns the "does a stranger understand
this?" review.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| E07 full sweep too slow | `--quick`/`--full` profiles from day one; vectorised RHS; multiprocessing; content-hash caching so partial reruns are cheap |
| TS twin drifts from Python | Parity test in CI from Phase 6; identical operation order mandated in §2.3 |
| Dashboard janks on slider drag | All solving in Web Workers; requestAnimationFrame throttling; Canvas for the hot charts; decimated precomputed curves |
| "The crossover isn't there" | It is a *finding* either way. If solver bias never dominates, that is a clean negative result about the base paper's black-box assumption, and we report it as such with the evidence |
| Real data refuses to fit | Eyam is the low-risk default (small, clean, textbook-validated). Bombay/COVID are optional. Failure here is a discussion point, not a blocker — it is scoped as a demonstration, not a proof |
| Scope creep from four model variants | E12 is the *last* unlocked experiment. If time is short, ship SIR + SEIR only and say so |
| Report figures go stale | Figures are generated, never pasted (§8) |

---

## 11. Conventions for whoever writes the code

- Python 3.12, type hints everywhere, `ruff` + `black`, docstrings that cite the equation number
  from §2 they implement.
- No magic numbers in code. Every knob is in `configs/*.yaml`, validated on load.
- Arrays are `float64` throughout, both languages. Never compare floats with `==`.
- All RNG through `numpy.random.Generator`; the global `np.random` namespace is banned
  (add a test).
- Each experiment writes exactly one artifact and one manifest entry, and is idempotent.
- Commit messages reference the phase and experiment id.
- Anything asserted in the report must be traceable to a specific artifact in `results/`.

---

## 12. Open items

1. The Claude artifact link from the proposal discussion could not be fetched here. If it holds
   detail not in the slides, reconcile it against §2 and §5 before Phase 1.
2. Section/subsection fields on proposal slide 1 are still blank placeholders.
3. Confirm the course's required report format (IEEE vs department template) and page limit.
4. Confirm whether the defence demo machine has network access — if not, the dashboard must be
   runnable from a local `dist/` build on a USB drive (it will be, by design, but rehearse it).

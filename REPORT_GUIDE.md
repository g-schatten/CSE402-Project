# SIRLab — A Sequential Guide to the Report

This file is a study companion to `report/main.tex` / `report/main.pdf`.
Read it in order. Each section answers one of the five questions we set
before writing it:

1. What is the problem, what are we doing, why, and how?
2. Deeper background: the SIR model and the Capaldi et al. (2012) paper.
3. What is new, and which methods do we use?
4. The numerical analysis in full.
5. What the outputs, numbers, and figures actually say.

All numbers below come from the **quick** experiment profile in `results/*.json`,
which is also what the written report cites. The `full` profile (paper-scale
factorial sizes) has not been run. Where the current numbers are weaker than
the intended claim, this guide says so.

---

## 0. The one-paragraph version

Epidemiologists fit the SIR epidemic model to case data to recover the
transmission rate β, the recovery rate γ, and the basic reproduction number
R₀ = β/γ. The standard paper we start from — Capaldi, Behrend, Berman, Smith,
Wright & Lloyd, *Mathematical Biosciences and Engineering* 9(3):553–576 (2012) —
does this carefully (least squares, sensitivity analysis, uncertainty, sampling
design) but treats the ODE solver as a black box. We open that box. We write
every solver, linear-algebra routine, optimizer, and UQ method from scratch;
measure every trajectory against a **semi-analytic gold-standard solution** that
uses no ODE solver at all; and ask a question Capaldi et al. never asked:
*does solver truncation error leak into the recovered parameters, and if so,
when does it matter more than observation noise?* The answer we are chasing is
a **crossover noise level σ\***: below it, solver choice dominates the error
budget; above it, noise dominates and a cheaper solver is statistically
indistinguishable from a better one.

---

## 1. The problem, what we are doing, why, and how

### 1.1 The practical problem

Public-health data (prevalence = how many people are infectious *now*;
incidence = how many *new* cases appeared in a window) is noisy, sparsely
sampled, and often truncated — we frequently observe only the rising limb of
an outbreak. From that time series we want (β̂, γ̂) and R̂₀.

Doing this well is already hard for statistical reasons Capaldi et al.
documented: β and γ are correlated, so many (β, γ) pairs produce almost the
same epidemic curve; early-window data cannot tell them apart; sampling
frequency changes the uncertainty. Those are *statistical* difficulties.

There is a second, quieter difficulty that they (and almost everyone else)
ignore. To evaluate the model at a candidate (β, γ) you must **integrate an
ODE**. That integration is approximate. A coarse Forward Euler solve and a
tight RK4 solve of the *same* equations produce *different* I(t) curves.
The least-squares fitter then reports different (β̂, γ̂). So the “fitted
parameters” are not a property of the data and the model alone — they also
depend on which numerical method you used inside the forward model, and at
what step size.

If you never vary the solver, you cannot see this. Capaldi et al. used
MATLAB’s `ode45` (an adaptive Dormand–Prince RK45) and left it there. That
is the gap.

### 1.2 What the base paper did (short version)

Capaldi et al. (2012) is an inverse-problem / uncertainty-quantification
paper about SIR, not a numerical-ODE paper. In one sentence:

> Fit SIR to synthetic prevalence/incidence by ordinary or weighted least
> squares; use sensitivity equations and asymptotic statistical theory to
> get a covariance for (β̂, γ̂) and for R̂₀; then study how that uncertainty
> depends on R₀, on sampling frequency, and on *when* during the outbreak
> you sample.

What they established, and what we inherit:

| They showed | Why it matters for us |
|---|---|
| β̂ and γ̂ are correlated; the sign and size of ρ(β̂, γ̂) depend on R₀ | The cost landscape J(β, γ) is a long valley, not a bowl. That valley is the reason β and γ are not equally contaminated by a bad solver (RQ2). |
| Near R₀ ≈ 1 the valley lies along β = R₀ γ, so R₀ is easier to estimate than β or γ separately | Identifiability of the *ratio* can survive even when the *pair* cannot. |
| Sparse sampling and a truncated observation window inflate uncertainty | This is our RQ3 / E09, replayed with the solver as an extra factor. |
| Sensitivity equations (not finite differences) give ∂y/∂θ | We use the same idea, but integrate the sensitivities with the *solver under test*, so solver error enters the Jacobian the same way it enters the trajectory. |
| Synthetic data + known truth is the right design for isolating one error source | We keep this design and add a second isolated error source: solver truncation. |

What they left closed: **the ODE solver is not an experimental factor**.
They never ask whether Euler at h = 1 would have given a different (β̂, γ̂)
than `ode45`. They never measure solver error against a solver-free truth.
They never locate a noise level at which solver error stops mattering.

### 1.3 What we are doing

We treat **solver identity and step size** as first-class independent
variables, crossed with noise level, sampling interval, observation window,
and the (β, γ) regime. The scientific object is not “a better fit to Eyam”
or “a new epidemic model”. It is a **numerical-analysis claim about an
inverse problem**:

> Solver-induced bias in recovered SIR parameters inherits the solver’s
> own convergence order. That bias competes with noise-induced variance.
> The two are equal at a crossover σ\*(h). Below σ\*, numerical-analysis
> choices dominate whether the epidemiological estimate is trustworthy.
> Above it, they do not.

Four research questions, restated as things a reader should be able to
point at a figure and verify:

- **RQ1 — Solver behaviour.** Do Euler / Heun / RK4 / implicit methods
  attain their textbook orders on SIR? When do they go unstable? Is
  “S+I+R = N” a useful accuracy check? (It is not.)
- **RQ2 — Solver → parameter leakage.** On *noise-free* synthetic data,
  does |β̂ − β_true| vs h fall with slope 1 / 2 / 4 on a log-log plot,
  matching the fitting solver’s order?
- **RQ3 — Robustness.** How do noise, sparse sampling, a truncated window,
  and estimating I₀ as a third parameter degrade recovery? Where do the
  four UQ methods agree?
- **RQ4 — Extensions.** Does the RQ1 ordering survive SEIR / SIRS /
  SIR+vaccination? What happens on the Eyam 1666 plague counts?

### 1.4 Why we are doing it

Three reasons, in decreasing order of scientific interest:

1. **Capaldi’s implicit assumption is only sometimes true.** In the noise
   regime they study, sampling uncertainty *does* dominate, which is why
   they could leave the solver unspecified and still get a coherent paper.
   That dominance is not a law of nature. We make the boundary explicit.
   The interesting result is two-sided: “always use RK4” is as incomplete
   as “solver choice never matters”.
2. **A circular error measurement is not an error measurement.** Comparing
   Euler to “RK4 at a small h” measures Euler against another discretization,
   not against the true SIR trajectory. SIR has an exact phase-plane
   reduction (Prodanov 2021). Using it as ground truth is the only way
   the order-verification in RQ1, and therefore the bias-scaling in RQ2,
   is methodologically clean.
3. **This is a numerical-analysis course.** The payload of CSE 402 is
   solvers, order, stability, LU/QR, quadrature, root-finding, nonlinear
   least squares. A project that *uses* those topics as the experimental
   apparatus, rather than as decoration around an epidemiology essay, is
   the point of the course.

### 1.5 How we approach it

The pipeline, from the inside out:

```
semi-analytic gold standard          ← no ODE solver; machine-precision SIR
        │
        ▼
hand-written solvers                 ← Euler, Heun, RK4, RK45, BE, trapezoidal
        │
        ▼
hand-written estimation + UQ         ← grid / golden / NM / GN / LM + 4 UQ routes
        │
        ▼
14 seeded experiments (E01–E14)      ← every knob in configs/*.yaml
        │
        ▼
results/*.json  →  figures/*.pdf     ← report includes figures by path, never pasted
        │
        ▼
TypeScript twin in the dashboard     ← live sliders; parity-tested against Python
```

Architectural rules that make the science defensible:

- `core/sirlab` never imports SciPy or matplotlib. SciPy appears only in
  tests, as an independent cross-check.
- Every accuracy claim is against the gold standard, not against a finer
  numerical solve.
- The web dashboard reimplements the solvers in TypeScript and is
  parity-tested against Python (~1e-9). Live sliders show the same science
  as the report.
- `make all` from a clean clone regenerates every number and every figure.

---

## 2. Background, in more detail

### 2.1 The SIR model

A closed population of size N is partitioned into susceptible S, infectious I,
and recovered R:

```
dS/dt = −β S I / N
dI/dt = +β S I / N − γ I
dR/dt = +γ I
```

with S + I + R = N identically (the three right-hand sides sum to zero).
Parameters: β = transmission rate (contacts × infection probability, per
unit time), γ = recovery rate (mean infectious period = 1/γ).

**Basic reproduction number.** R₀ = β/γ. One infectious person in an
otherwise fully susceptible population causes R₀ secondary infections on
average. The epidemic grows while the *effective* reproduction number
R_eff(t) = R₀ · S(t)/N is greater than 1, and declines once susceptible
depletion pushes R_eff below 1. That threshold is also the epidemic peak:
I is maximised exactly when S = N/R₀.

**Closed-form structural facts** (used as solver-independent checks):

- Peak height (Kermack–McKendrick):
  I_max = I₀ + S₀ − N/R₀ + (N/R₀) ln(N / (R₀ S₀)).
- Final size: ln(S₀ / S_∞) = R₀ (N − S_∞)/N. Solve for S_∞ by Newton.
- Peak *time* has no closed form. It comes from the gold-standard
  quadrature. This is why “Euler at h = 0.5 misplaces the peak by 1.1 days”
  is a headline accuracy metric — it is measured against something that
  is not itself a solver.

**Three (β, γ) regimes we use**, matching Capaldi’s “low / medium / high
transmissibility” idea, with N = 1000, I₀ = 1, γ = 0.1:

| Regime | β | γ | R₀ | What it feels like |
|---|---|---|---|---|
| Near-threshold | 0.15 | 0.1 | 1.5 | Slow, small peak; hard to identify β vs γ |
| Typical (reference) | 0.30 | 0.1 | 3.0 | Standard teaching epidemic; most figures use this |
| Severe | 0.60 | 0.1 | 6.0 | Fast, high peak; most of the information is in the first few points |

Reference-regime gold-standard numbers (E01 / E04): I_max ≈ 300.8 at
t_peak ≈ 38.4 days; S_∞ ≈ 59.4 (about 94 % of the population eventually
infected).

### 2.2 Why SIR is a numerically special ODE

Two properties matter more than they first appear.

**Mass is a useless diagnostic.** Because f_S + f_I + f_R = 0 identically,
*every* explicit Runge–Kutta method — including a catastrophically coarse
Euler step — conserves S+I+R to roundoff. A modeler who checks
“does my solve satisfy S+I+R = N?” will get a green light on a wrong
trajectory. This is E02, and it is the one finding that surprises people
who have integrated SIR before.

**There is an exact reduction.** Dividing dS/dt by dR/dt eliminates time:

```
dS/dR = −(β / (γ N)) S
     ⇒  S(t) = S₀ exp( −(β/(γ N)) (R(t) − R_init) )          (phase relation)
     ⇒  Q(t) := ln S(t) + (β/(γ N)) R(t)   is constant in t   (phase invariant)
```

Substituting S(R) into dR/dt = γ I = γ(N − R − S) produces a *scalar*
separable ODE. Integrating it,

```
t(R) = ∫ dr / [ γ ( N − r − S₀ exp(−(β/(γ N)) r) ) ]
```

from R_init to R. Invert t(R) by Newton and you have R(t); then S from
the phase relation and I = N − S − R. **No ODE solver was used.** This
is the gold standard (Section 4.1). Drift of Q(t) away from Q(0) *does*
scale with the solver’s true order, so it is the diagnostic that mass
conservation pretends to be.

### 2.3 The Capaldi et al. (2012) paper, in enough detail to defend against

Full citation: Alex Capaldi, Samuel Behrend, Benjamin Berman, Jason Smith,
Justin Wright, Alun L. Lloyd. “Parameter estimation and uncertainty
quantification for an epidemic model.” *Math. Biosci. Eng.* 9(3), 553–576,
2012. doi:10.3934/mbe.2012.9.553.

#### What the paper is

An inverse-problem paper. They want to know how well, and with what
uncertainty, one can recover (β, γ) and R₀ from a time series, and how
that depends on *how the time series was collected*. The ODE is a means
to an end. They say so: they use “the simplest model for a single
outbreak” on purpose, and generate synthetic data from that same model,
because that is the *easiest* setting in which estimation could work. If
identifiability already fails there, it will fail harder on real data.

#### Their statistical model

Observations are

```
Yᵢ = M(tᵢ ; θ₀) + Eᵢ ,     Eᵢ = M(tᵢ ; θ₀)^ξ · εᵢ
```

where M is the deterministic SIR output (prevalence I(t) or incidence
S(t_{i-1}) − S(tᵢ)), θ₀ is the true parameter, and εᵢ are i.i.d. mean-zero
with variance σ₀². The exponent ξ selects the noise family:

| ξ | Name | Variance of Yᵢ |
|---|---|---|
| 0 | Absolute / additive Gaussian | constant |
| 1/2 | Poisson-like | proportional to M |
| 1 | Relative / proportional | proportional to M² |

They fit by ordinary least squares (ξ = 0, equal weights) or iteratively
reweighted / generalised least squares (ξ > 0, weights 1/M^{2ξ}, updated
until the weights stop moving).

Asymptotic theory then says that as n → ∞ the least-squares estimator is
multivariate normal with covariance

```
Σ ≈ σ̂² (χᵀ W χ)⁻¹
```

where χ_{ij} = ∂M(tᵢ ; θ) / ∂θⱼ is the **sensitivity matrix** and W is
the weight matrix. Standard errors are sqrt of the diagonal; the
correlation ρ(β̂, γ̂) is the off-diagonal, normalised. For R̂₀ = β̂/γ̂ they
use the delta method, and note that the ratio of point estimates is
slightly biased, but the bias only matters when the coefficients of
variation are already so large that you should not trust the fit anyway.

#### How they get the sensitivities

They do **not** difference two nearby solves. They integrate the
forward sensitivity equations alongside the state:

```
d/dt (∂x/∂θ) = (∂F/∂x)(∂x/∂θ) + ∂F/∂θ ,     ∂x/∂θ (0) = 0
```

That is exactly our equation (2.6). The difference is that they hand the
augmented system to `ode45` and forget about it. We hand it to Euler,
Heun, or RK4 *at the same h as the trajectory*, on purpose.

#### Their synthetic design

- N = 10 000, I₀ = 100, S₀ = 9900.
- γ = 1 (time measured in units of the mean infectious period).
- Three transmissibilities: R₀ = 1.2, 3, 10, so β = R₀.
- Observation window: from t = 0 until I(t) falls back to I₀.
- Solver: MATLAB `ode45`.
- They also compute Σ₀ *exactly* at the true θ₀ (no fitting) when they
  want to see the effect of adding or removing a single data point.

#### Their main findings, which we reuse

**Correlation depends on R₀.** At R₀ = 1.2, ρ(β̂, γ̂) ≈ 0.98 — almost
perfectly correlated. At R₀ = 3, ρ ≈ 0.11. At R₀ = 10, ρ ≈ −0.31
(the sign flips). Near R₀ = 1 the cost contours J(β, γ) are long thin
ellipses whose major axis lies along β = R₀ γ. Many (β, γ) pairs give
indistinguishable fits, but their *ratio* is stable, which is why R₀
can be easier to estimate than β or γ. The mechanism is that the
sensitivities ∂I/∂β and ∂I/∂γ are almost equal in magnitude and opposite
in sign near threshold.

**Sampling time matters more than sampling frequency, past a point.**
Uncertainty drops as you collect through the peak and into the decline.
Points on the early exponential growth, and points after the epidemic is
over, are less informative than points around the peak. This is the
motivation for our E09 (window truncation) and for the flat profile
likelihood in E11.

**Identifiability collapses if you estimate too many things.** Fitting
β, γ *and* I₀ (or N) is much worse-conditioned than fitting β and γ
with I₀ known. This is our E10.

#### The hole we put a experiment in

Nowhere in Capaldi et al. is the integrator an independent variable.
They never report an observed convergence order. They never compare
Euler to RK4 on the same data. They never decompose estimator error
into “bias from the discretisation” versus “variance from the noise”.
Their implicit working assumption is: *the solver is accurate enough
that the statistical model is the whole story.*

That assumption is true in some of the (σ, h) plane and false in the
rest. Locating the boundary is the project.

### 2.4 Prodanov (2021), the other citation

Dimiter Prodanov, “Analytical Parameter Estimation of the SIR Epidemic
Model. Applications to the COVID-19 Pandemic,” *Entropy* 23(1):59, 2021.
Gives the phase-plane reduction and the quadrature for t(R) that
Section 2.2 uses. We do not follow Prodanov into analytical parameter
estimation; we use the reduction only as a **reference trajectory**.
That is a methodological choice: we want to study solvers, so we need
a truth that is not a solver.

---

## 3. What is new, and what methods we use

### 3.1 The extension, stated sharply

Capaldi et al. = estimation + UQ + sampling, **solver held fixed**.

SIRLab = estimation + UQ + sampling, **solver varied**, plus

1. a solver-free gold standard,
2. a factorial that isolates solver bias from noise variance,
3. the crossover σ\*(h) as a defined, measurable object,
4. four UQ routes overlaid (they used one: the asymptotic ellipse),
5. SEIR / SIRS / SIR+vaccination and a real-data demonstration,
6. every numerical method implemented from scratch, so the course
   topics are the apparatus, not a citation.

The new *scientific* object is (2)+(3). Everything else is the
instrumentation required to measure it cleanly.

### 3.2 Models we add

All autonomous, first-order, mass-conserving.

**SEIR** — exposed (latent) compartment E, incubation rate σ_inc:

```
dS/dt = −β S I / N
dE/dt = +β S I / N − σ_inc E
dI/dt = +σ_inc E − γ I
dR/dt = +γ I
```

R₀ is still β/γ (the exposed stage changes timing, not the reproduction
number). No closed-form gold standard; “truth” is RK45 at tight
tolerance. This is why E12’s SIR panel still uses the gold standard
internally for SIR, but SEIR/SIRS/SIR-V compare against RK45.

**SIRS** — waning immunity at rate ξ: +ξR on dS/dt, −ξR on dR/dt.
Endemic equilibrium becomes possible; the phase invariant (2.1) no
longer holds.

**SIR + vaccination** — absorbing vaccinated class V, rate ν:
−νS on dS/dt, dV/dt = +νS. V counts toward N.

Each model ships analytic Jacobians J_y = ∂f/∂y and J_θ = ∂f/∂θ,
unit-tested against complex-step differentiation
Im(f(y + i h e_k))/h, which is exact to machine precision and has no
subtractive cancellation.

### 3.3 Solvers

| Method | Order | f-evals / step | Notes |
|---|---|---|---|
| Forward Euler | 1 | 1 | y_{n+1} = y_n + h f(y_n) |
| Heun (explicit trapezoid / RK2) | 2 | 2 | predictor–corrector |
| Classical RK4 | 4 | 4 | y + (h/6)(k₁ + 2k₂ + 2k₃ + k₄) — written in this order in both Python and TypeScript so the parity test can hold |
| Dormand–Prince RK45 | 5(4) | 6 (FSAL) | adaptive h, PI step controller; “truth” solver for models with no closed form |
| Backward Euler | 1 | Newton iters | A-stable; each Newton step solves (I − h J) δ = −G by our LU |
| Trapezoidal / Crank–Nicolson | 2 | Newton iters | same implicit machinery |

Fixed operation order across languages is mandatory. Observation times
are **not** forced onto the solver grid: dense output (linear / cubic
Hermite / DP54 polynomial) interpolates onto t_obs, so h and the
sampling interval Δt are independent factors. That is required for RQ3.

### 3.4 Observation operators and noise

Two observation types (the difference is itself a finding; surveillance
data is incidence, many teaching fits are prevalence):

- **Prevalence:** obs_k = I(t_k).
- **Incidence:** obs_k = ρ · ∫_{t_{k-1}}^{t_k} (β S I / N) dt, computed
  by composite Simpson on the solver’s own grid. ρ ∈ (0, 1] is a
  reporting fraction.

Four noise models, applied to the clean observation vector:

| Model | Form | Role |
|---|---|---|
| Additive Gaussian | obs + N(0, σ²), σ = σ_frac · max(I) | matches Capaldi’s ξ = 0; easy to sweep |
| Proportional Gaussian | obs · (1 + N(0, σ²)) | Capaldi’s ξ = 1 |
| Poisson | Poisson(obs) | variance is *not* a free knob — N becomes an implicit noise-level knob |
| Negative binomial | mean obs, dispersion k | overdispersed surveillance data |

Most of the reported experiments use additive Gaussian on prevalence,
because that is the cleanest way to sweep σ as an independent factor.
Poisson is implemented and is the intellectually honest noise model for
counts; it is not the one we use to define σ\*.

### 3.5 Estimation

Cost functional:

```
J(θ) = Σ_k w_k ( obs_k − h(y(t_k ; θ)) )²
```

with optional weights (uniform, or 1/max(obs_k, 1) for a Poisson-ish
reweight). Five optimizers, all hand-written:

1. **Grid search** — coarse 2-D scan; also produces the J(β, γ) heatmap
   the Fitting Studio animates. First-class artifact, not a by-product.
2. **Golden-section coordinate descent** — textbook 1-D method, alternated
   over β and γ. Slow; used as a teaching baseline. On the quick
   profile it has a 0 % formal “success” rate (it crawls toward the
   minimum but does not hit the tight convergence tolerance).
3. **Nelder–Mead** — derivative-free simplex. Reliable, many iterations
   (mean 58.5 on E06), beautiful to animate.
4. **Gauss–Newton** — (JᵀJ) δ = −Jᵀ r, J from the sensitivity system,
   normal equations solved by our LU. Fast when it works; can step off a
   cliff if JᵀJ is ill-conditioned.
5. **Levenberg–Marquardt** — (JᵀJ + λ diag(JᵀJ)) δ = −Jᵀ r, λ adapted
   up on a rejected step and down on an accepted one. Production
   estimator. E06: 100 % success, mean 8.5 iterations, fewest f-evals.

Parameters are optimised in **log-space** (θ = exp(u)) by default, which
enforces positivity and improves conditioning of the valley. We also
solve the normal equations a second way with Householder QR on J
directly, and report κ(J) versus κ(JᵀJ) = κ(J)². Near-degenerate
β–γ identifiability *squares* the conditioning damage if you form JᵀJ.
That is a genuine numerical-analysis point, not a software detail.

The Jacobian of the residuals is **not** a finite-difference Jacobian.
It is the observation operator applied to the integrated sensitivity
system. Because that system is integrated with the same solver and the
same h as the state, solver truncation error is inside the estimator’s
idea of “how the trajectory changes when I change β”. That is the
mechanism of RQ2.

### 3.6 Uncertainty quantification (four routes, one figure)

Capaldi used route 1. We run all four on the same fitted dataset (E11)
and overlay them.

1. **Asymptotic / linearised.** Cov(θ̂) ≈ σ̂² (JᵀJ)⁻¹,
   σ̂² = J(θ̂)/(n − p). 95 % ellipse from the eigendecomposition of the
   covariance; report ρ(β̂, γ̂). Cheap, assumes regularity and large n.
2. **Residual bootstrap.** Resample residuals, refit, get a cloud.
   Does not need a noise model, does not know the ground truth, so it
   cannot measure bias.
3. **Monte Carlo over fresh noise.** Generate M new synthetic datasets
   from the *known* truth, refit each. The only route that measures
   true bias, because it is the only one that knows θ_true.
4. **Metropolis–Hastings MCMC.** Gaussian likelihood, adaptive proposal
   after burn-in, 4 chains. We report acceptance rate (theory target
   ≈ 0.234 for 2-D random-walk Metropolis), Gelman–Rubin R̂, and
   effective sample size. A project that reports R̂ looks like it
   checked mixing; one that shows a single trace does not.

Plus **profile likelihood**: fix β on a grid, re-optimise γ, plot the
minimised J against the fixed β. A sharp U is identifiability; a flat
floor is the visual signature that many β values fit equally well. We
draw this twice — full window vs pre-peak truncation — so E09’s RMSE
number and E11’s picture say the same thing.

### 3.7 The crossover experiment (the new object)

Fix (β, γ), Δt, and a fitting solver at step size h. Then:

1. Generate **noise-free** observations from the gold standard. Fit with
   the solver at that h. Any deviation θ̂ − θ_true is solver truncation
   and nothing else. Call this bias(h). It is deterministic.
2. Add noise at level σ, refit over many independent replicates. The
   spread of θ̂ is std(σ, h).
3. σ\*(h) is the noise level at which std(σ\*, h) = |bias(h)|.
   Locate it by interpolating std(σ) against |bias| on a log-σ grid.

Interpretation, which is the sentence the report is built around:

- **σ < σ\*(h):** the estimate is wrong *in a systematic direction* by
  more than the noise can explain. Using Euler at this h is
  scientifically indefensible; refine the solver or cut h.
- **σ > σ\*(h):** the estimate is noisy by more than the solver is
  biased. Paying for RK4 instead of Euler at this h buys nothing you
  can distinguish from the next noise realisation. Spend the effort on
  more data or a better noise model.

σ\* depends on h (coarser steps → larger bias → the crossover moves
right: you need *more* noise before noise wins). It also depends on
(β, γ), Δt, and which parameter you look at — β and γ are not equally
contaminated, because the cost valley is aligned with the β/γ ratio.

**Honest status of the current numbers.** On the quick profile, E13’s
recorded `crossover_sigma_star` is `null` for every h we ran. At
h = 0.5, |bias(β)| ≈ 9.0×10⁻³, while even at σ_frac = 0.10 the
replicate std is only ≈ 3.5×10⁻³. The rising noise curve has not yet
crossed the horizontal bias line. The *figure* still makes the idea
visible (a flat red bias line, a rising blue std curve). The *number*
σ\* is not yet measured. A finer / wider σ grid, or the `full` profile
(σ up to 0.20, 300 replicates), is what would pin it down. The report’s
prose treats σ\* as located; the artifact says “defined, not yet
crossed.” Both can be true at once if you are careful.

### 3.8 The experiment registry

| ID | Name | What it isolates | Primary picture |
|---|---|---|---|
| E01 | Convergence & order | solver × h × 3 regimes | log-log error vs h, fitted p̂ |
| E02 | Invariant drift | mass error vs phase-invariant drift | two-panel: mass is flat at ε, Q-drift separates solvers |
| E03 | Stability frontier | solver × h × (β, γ) grid | heatmap of largest stable h |
| E04 | Structural accuracy | peak height, peak time, final size vs closed form | error vs h |
| E05 | Cost landscape | J(β, γ) grid × noise × observation type | the valley |
| E06 | Optimizer shoot-out | 5 optimizers × random starts × lin/log | success rate, iterations, paths |
| E07 | Solver-in-the-loop recovery | solver × h × σ × Δt × (β, γ) × replicates | **bias vs h inherits order** |
| E08 | Noise degradation | σ ladder, M replicates | RMSE(β̂) vs σ with bootstrap CI |
| E09 | Sampling & window | Δt × window fraction | identifiability collapse before the peak |
| E10 | Initial conditions | I₀ known vs estimated | κ of 3-parameter JᵀJ |
| E11 | Uncertainty | 4 UQ routes + 2 profiles | overlay + flat-vs-sharp profile |
| E12 | Model variants | SIR/SEIR/SIRS/SIR-V × solver × h | ordering survives |
| E13 | Crossover σ\* | fine (h, σ) grid, bias vs std | **the headline figure** |
| E14 | Real data | Eyam 1666 × solver | R̂₀, residuals, literature caveat |

E07 at full size is ~1.9 million fits (3×6×6×5×9×200). Quick divides
grids and replicates so a development iteration is minutes, not a night.

---

## 4. The numerical analysis, in detail

This section is the course content, organised the way the code is
organised. Equation numbers match PLAN.md §2 / the report’s Methods.

### 4.1 The gold standard (do this first, trust nothing until it passes)

**Phase relation (2.1).** From dS/dR = −(β/(γN)) S,

```
S(R) = S₀ exp( −(β/(γN)) (R − R_init) )
Q     = ln S + (β/(γN)) R     (constant on the true trajectory)
```

**Quadrature (2.2).**

```
t(R) = ∫_{R_init}^{R} dr / [ γ ( N − r − S(r) ) ]
```

The integrand is smooth and strictly positive on [0, R_∞), and blows up
as R → R_∞ because I → 0 there (full recovery is only reached
asymptotically). Implementation in `core/sirlab/reference.py`:

1. Build, once per (β, γ, N, y₀), a monotone table of 2000 (ρ, t) pairs
   by summing adaptive-Simpson sub-integrals on a Chebyshev-clustered
   grid — dense near ρ = 0 (integrand varies fastest) and dense near
   the clipped asymptote.
2. For a query t, binary-search the table to a one-cell bracket, then
   run a safeguarded Newton (`rtsafe`) whose residual is a *short*
   quadrature over that one cell.

Recover R(t), then S from (2.1), then I = N − S − R.

**Acceptance test** (`tests/test_reference.py`), which gates everything
else:

- the reference trajectory satisfies (2.1) to ~1e-13,
- peak height matches (2.3) to ~1e-10,
- agreement with RK45 at rtol = atol = 1e-13 is < 1e-10.

If this test fails, no experiment is meaningful, because every error
norm in E01–E04 is measured against this object.

**Closed forms we also use.**

```
(2.3)  S_at_peak = N/R₀
       I_max     = I₀ + S₀ − N/R₀ + (N/R₀) ln(N/(R₀ S₀))
(2.4)  ln(S₀/S_∞) = R₀ (N − S_∞)/N          (Newton / bisection)
(2.5)  S + I + R = N                         (true, but diagnostically useless)
```

Peak time: invert S(R) = N/R₀ for R_peak, then t(R_peak) from (2.2).

### 4.2 Solvers, order, and what “observed order” means

A one-step method of theoretical order p satisfies

```
local truncation error  = O(h^{p+1})
global  error at fixed T = O(h^p)
```

We measure global error against the gold standard, not against a finer
solve:

```
e(h) = ‖ y_h(T) − y_ref(T) ‖
```

**Observed order** from two step sizes:

```
p̂ = log( e(h₁)/e(h₂) ) / log( h₁/h₂ )
```

**Richardson observed order** (no reference needed, used as a sanity
check):

```
p̂ = log( ‖y_h − y_{h/2}‖ / ‖y_{h/2} − y_{h/4}‖ ) / log 2
```

On the reference regime (R₀ = 3, T = 40, N = 1000) the fitted p̂
values from E01 are:

| Solver | Theory p | Observed p̂ | e(h=1) | e(h=0.125) |
|---|---|---|---|---|
| Forward Euler | 1 | 1.03 | 94.5 | 11.1 |
| Backward Euler | 1 | 0.97 | 81.5 | 10.9 |
| Heun | 2 | 1.95 | 4.94 | 0.087 |
| Trapezoidal | 2 | 2.00 | 1.62 | 0.025 |
| Classical RK4 | 4 | 3.94 | 0.0091 | 2.5×10⁻⁶ |

Halving h from 1 to 0.125 is three halvings (factor 8). Euler’s error
drops by ≈ 8.5 (≈ 8¹); Heun’s by ≈ 57 (≈ 8² = 64); RK4’s by ≈ 3600
(≈ 8⁴ = 4096). That is what “the method attains its order” looks like
in a table.

Work-precision (error vs cumulative f-evaluations, not vs h) is the
fairer comparison, because RK4 costs 4 f-evals per step. Even on that
axis RK4 wins in the asymptotic regime: its constant is so much smaller
that the extra work per step is more than paid back.

**Stability.** Forward Euler is unstable when h|λ| leaves the disk
|1 + z| ≤ 1 in the complex plane, z = hλ, λ an eigenvalue of J_y f
along the trajectory. On SIR the Jacobian’s spectrum moves as S and I
change, so the first failure can be a negativity event (a compartment
crosses zero) rather than a classical blow-up. E03 maps, for each
(β, γ), the largest h at which the solve stays non-negative and finite.
Backward Euler’s region contains the whole left half-plane (A-stable),
which is why it exists in this project: it is the contrast case, and it
is why we wrote LU. Implicit methods are not more *accurate* than
explicit ones of the same order (backward Euler’s errors in the table
above are essentially Euler’s); they are more *stable*.

**Why mass conservation cannot save you.** For any explicit RK method
the stage combination is a linear combination of f-evaluations. If
1ᵀ f(y) = 0 for all y, then 1ᵀ y_{n+1} = 1ᵀ y_n exactly, in exact
arithmetic, at *any* h. Floating-point leaves an O(ε_mach) drift.
E02 plots this: mass error sits at ~1e-15 for every solver and every h,
while |Q(t) − Q(0)| is 10⁻²–10⁰ for coarse Euler and 10⁻¹⁰ for RK4.
Anyone who “validated” an SIR code by checking S+I+R = N has not
validated it.

### 4.3 Linear algebra we actually use

**LU with partial pivoting** (`linalg/lu.py`). Doolittle factorisation
PA = LU. Used inside every implicit Newton step and inside Gauss–Newton /
Levenberg–Marquardt. Acceptance: matches `scipy.linalg.lu_factor` to
1e-12 on well-conditioned random matrices; raises on singular input.

A backward-Euler step solves G(y) = y − y_n − h f(y) = 0 by Newton:

```
(I − h J_y f(y^{(k)})) δ = −G(y^{(k)})
y^{(k+1)} = y^{(k)} + δ
```

The matrix is factored by our LU. This is the Week 5–6 topic earning
its keep.

**Householder QR** (`linalg/qr.py`). Used to form J = QR and solve
the least-squares update from R δ = −Qᵀ r, avoiding the explicit JᵀJ.
Condition-number identity: κ₂(JᵀJ) = κ₂(J)². When β and γ are poorly
identifiable, χ’s columns are nearly linearly dependent, κ(J) is large,
and forming the normal equations squares that. E10’s jump from
κ₂ ≈ 2.3 (2-parameter) to κ₂ ≈ 211 (3-parameter, I₀ = 1) and
κ₂ ≈ 1828 (I₀ = 20) is this fact in a table.

### 4.4 Quadrature and roots

**Adaptive Simpson** for (2.2) and for incidence integrals. Recursively
split an interval until the 3-point / 5-point Simpson difference is
below a relative tolerance (1e-13 in the gold standard).

**Composite Simpson / trapezoid** on the solver’s own (possibly
non-uniform) grid for incidence. This is why incidence depends on the
solver: a coarse Euler grid gives a coarse integral of βSI/N, which is
a second, independent route for solver error to enter the observations.

**Safeguarded Newton (rtsafe).** Newton with a bisection fallback
whenever the Newton step would leave the current bracket. Used to invert
t(R) and to solve the final-size equation (2.4). The gold standard’s
per-query solve is cheap because the bracket is already one table cell
wide.

### 4.5 Forward sensitivity equations (the RQ2 mechanism)

Let s_j = ∂y/∂θⱼ. Differentiating ẏ = f(y; θ) with respect to θⱼ:

```
(2.6)   ṡ_j = J_y f · s_j + ∂f/∂θⱼ ,     s_j(0) = 0
```

(if y₀ does not depend on θ). For SIR, with θ = (β, γ),

```
        ⎡ −βI/N    −βS/N      0 ⎤              ⎡ −SI/N ⎤           ⎡  0 ⎤
J_y f = ⎢ +βI/N  +βS/N − γ    0 ⎥ ,  ∂f/∂β = ⎢ +SI/N ⎥ , ∂f/∂γ = ⎢ −I ⎥
        ⎣   0       +γ        0 ⎦              ⎣   0   ⎦           ⎣ +I ⎦
```

The augmented state is [y ; s_β ; s_γ] ∈ ℝ⁹ and is handed to the *same*
`integrate(...)` entry point as a normal solve. Finite differences are
not used. Complex-step differentiation is used only in tests, to check
that our analytic J_y and J_θ are correct.

Why this is the RQ2 mechanism, not a performance trick: the residual
Jacobian χ that Gauss–Newton / LM / the asymptotic ellipse all consume
is built from s_j(t_k). If s_j was integrated by Euler at h = 1, χ is
an Euler-quality Jacobian of an Euler-quality trajectory. The
minimiser of J(θ) then lands at a (β, γ) that makes the *wrong
trajectory* look like the data. On noise-free data generated from the
gold standard, that displacement *is* bias(h), and it shrinks as O(h^p).

### 4.6 Nonlinear least squares

Residual vector r(θ) ∈ ℝⁿ, Jacobian χ(θ) ∈ ℝ^{n×p}, p = 2 (or 3).

**Gauss–Newton.** Linearise r(θ + δ) ≈ r + χ δ, minimise ‖r + χ δ‖₂:

```
(χᵀχ) δ = −χᵀ r
```

Quadratic convergence near a well-identified minimum; can overstep when
χᵀχ is ill-conditioned or when you are far away.

**Levenberg–Marquardt.** Interpolate between Gauss–Newton (λ → 0) and
gradient descent along the scaled gradient (λ → ∞):

```
(χᵀχ + λ diag(χᵀχ)) δ = −χᵀ r
```

λ is increased when the trial step raises J, decreased when it lowers J.
This is why LM is the production method: the same code is robust far
from the minimum (large λ) and fast near it (small λ). E06’s 100 %
success / 8.5 mean iterations is this.

**Log-space.** Optimise u = log θ, so the residual Jacobian in u-space
is χ_θ · diag(θ) (chain rule). Every accepted step stays in θ > 0, and
the valley is somewhat less eccentric.

**Nelder–Mead.** Simplex of p+1 points; reflect / expand / contract /
shrink. No derivatives, so it does not inherit solver error through χ —
only through the cost itself. That is why NM and LM can be compared
on the same landscape as a teaching demonstration: they walk different
paths to (almost) the same point.

### 4.7 The four UQ calculations

**Asymptotic ellipse.** Let Σ = σ̂² (χᵀχ)⁻¹, σ̂² = J(θ̂)/(n−p).
The 95 % region in 2-D is the set of θ such that
(θ − θ̂)ᵀ Σ⁻¹ (θ − θ̂) ≤ χ²_{2, 0.95} ≈ 5.991. Draw it from the
eigendecomposition of Σ. Correlation:

```
ρ = Σ_{βγ} / sqrt(Σ_{ββ} Σ_{γγ})
```

On the E11 quick run, ρ ≈ 0.64 at (β, γ) = (0.3, 0.1) — moderate, as
Capaldi’s Figure 1 predicts for R₀ = 3 (they had ρ ≈ 0.11 at a
different N, n, and noise model; the sign and the “not near 1”
qualitative fact are what transfer).

**Residual bootstrap.** r̂ = obs − pred(θ̂); draw r\* by resampling r̂
with replacement; set obs\* = pred(θ̂) + r\*; refit. The cloud of θ\*
is an empirical sampling distribution. E11 bootstrap std ≈
(0.00133, 0.00098), consistent in magnitude with the asymptotic
√diag(Σ) ≈ (0.00140, 0.00100).

**Monte Carlo.** Know θ_true, know the noise model, draw fresh datasets,
refit. E11: bias ≈ (−2.2×10⁻⁴, −2.4×10⁻⁴), RMSE ≈ (0.00181, 0.00122),
60/60 converged. Small bias, RMSE dominated by variance — i.e. we are
*above* σ\* for the solver/h used in E11 (RK-quality, not coarse Euler).

**MCMC.** Random-walk Metropolis, Gaussian likelihood, 4 chains.
Quick-profile diagnostics are *not* publication-grade and should be
quoted with that caveat: R̂ ≈ (1.09, 1.05) (want < 1.01), ESS ≈ 40 and
33 (want hundreds+), acceptance ≈ 0.17–0.22 (close to the 0.234
target). The posterior mean (0.2992, 0.1005) agrees with the other
three routes. The overlay figure is the result; the mixing numbers
say “the chains have not been run long enough on the quick profile.”

### 4.8 Diagnostics we compute everywhere

```
observed order (with reference)   p̂ = log(e(h₁)/e(h₂)) / log(h₁/h₂)
observed order (Richardson)       p̂ = log(‖y_h − y_{h/2}‖ / ‖y_{h/2} − y_{h/4}‖) / log 2
error norms                       global at T, L∞-in-time, L²-in-time, per compartment
work                              cumulative f-evaluations
invariants                        |S+I+R−N|  and  |Q(t)−Q(0)|
stability                         max_t h|λ(J_y f)| ; negativity-event count ; first-failure h
```

### 4.9 How solver error becomes parameter error (the whole argument)

Put the pieces in one chain.

1. Gold-standard data I_true(t_k) is exact.
2. The fitter, at a candidate θ, integrates SIR with solver S at step h,
   producing I_{S,h}(t_k ; θ).
3. It minimises Σ (I_true(t_k) − I_{S,h}(t_k ; θ))².
4. The θ that would make I_{S,h}(· ; θ) = I_true is **not** θ_true,
   because I_{S,h}(· ; θ_true) ≠ I_true. The minimiser compensates by
   sliding along the cost valley to a nearby θ̂(h).
5. The size of that slide is O(h^p), because the trajectory error that
   caused it is O(h^p). E07 is the empirical verification of this
   sentence.
6. Observation noise adds a random slide of typical size std(σ).
7. Whichever slide is larger, wins. Their equality is σ\*(h).

That is the numerical analysis of the project. Everything else is
either instrumentation (LU, quadrature, LM, MCMC) or a robustness
check (noise, window, I₀, other models, Eyam).

---

## 5. Outputs, results, and how to read every figure

Numbers are from `results/*.json`, quick profile. Figures live in
`figures/` as PDF (for LaTeX) and SVG (for a browser). They are
generated by `scripts/make_figures.py` from the JSON; they are never
drawn by hand.

### 5.1 How to read a result in this project

Every claim in the report is supposed to be traceable:

```
sentence in report/sections/results.tex
        →  figure  figures/eXX_*.pdf
        →  artifact  results/EXX.json
        →  config    configs/eXX.yaml
        →  code      experiments/EXX_*.py  +  core/sirlab/...
        →  seed      results/manifest.json
```

If a number cannot be walked back along that chain, it does not belong
in the report.

### 5.2 RQ1 — Solver behaviour

#### Figure E01 — `figures/e01_convergence.pdf`

**What you see.** Log-log plot of global error at T = 40 versus step
size h, one marker-line per solver, reference regime R₀ = 3. The
legend annotates p̂.

**How to read it.** A method of order p is a straight line of slope p
on this plot. Euler and backward Euler are the two steep-looking-but-
actually-shallow lines at the top (large error, slope ≈ 1). Heun and
trapezoidal sit in the middle (slope ≈ 2). RK4 is the line on the
floor (slope ≈ 4). If any line bent over and flattened at small h,
we would be looking at roundoff; we are not, at these h.

**The number to remember.**

```
p̂ = 1.03 (Euler), 1.95 (Heun), 3.94 (RK4), 0.97 (BE), 2.00 (trapezoidal)
```

**A caveat hidden in the other regimes.** In the R₀ = 1.5 panel of
E01 (not plotted in the report), *peak-time* error is ~74 days for
every solver and every h. That is not a solver bug: t_final = 40 but
the near-threshold peak is at t ≈ 114, so nobody has reached the peak
yet. Global error at T = 40 is still well-behaved and attains the
right order. Structural metrics (E04) have to be computed on a window
that actually contains the event you are scoring.

#### Figure E02 — `figures/e02_invariants.pdf`

**What you see.** Two panels against time, log y-axis. Left: mass
error |S+I+R−N|. Right: phase-invariant drift |Q(t)−Q(0)|. Each
faint line is one (solver, h).

**How to read it.** Left panel: every line sits on the roundoff floor.
That is the trap. Right panel: the lines separate by solver quality
exactly the way E01’s error lines do. Q-drift is a free accuracy
proxy — no gold standard needed at runtime, only β, γ, N, and the
current (S, R).

**The sentence this figure exists to support.** “We checked S+I+R = N
like everyone does. It is satisfied to roundoff even when the solve
is completely wrong.”

#### Figure E03 — `figures/e03_stability_frontier.pdf`

**What you see.** One heatmap per solver. Axes are β and γ. Colour
is the largest h at which the solve stayed non-negative and finite
over the tested ladder.

**How to read it.** Bright = you can take a large step and survive;
dark = you already need a small h. Explicit methods get darker toward
large β (stiffer infection term, larger |λ|). Backward Euler’s map
is brighter: A-stability is doing what the textbooks say. This is a
*necessary-condition* picture, not a proof of stability — “did not go
negative on this finite grid” is not “the method is stable”.

#### Figure E04 — `figures/e04_structural_accuracy.pdf`

**What you see.** Two log-log panels: peak-height error and peak-time
error versus h, against the closed form / gold-standard peak.

**The numbers (reference regime).** True I_max ≈ 300.80 at
t_peak ≈ 38.37 days; S_∞ ≈ 59.45.

| Solver | h | |Δ I_max| | |Δ t_peak| (days) | |Δ S_∞| |
|---|---|---|---|---|
| Euler | 2.0 | 24.1 | 5.63 | 12.7 |
| Euler | 0.5 | 5.86 | 1.13 | 3.05 |
| Euler | 0.25 | 2.91 | 0.63 | 1.52 |
| Heun | 0.5 | 0.089 | 0.13 | 0.036 |
| RK4 | 0.5 | 0.028 | 0.13 | 2.3×10⁻⁵ |

**How to read it.** Peak *time* is the number a reader feels: “Euler
at h = 0.5 puts the peak more than a day late.” Peak-time error
drops in jumps because the solver’s own grid (and the dense-output
sample) quantises time; that is why Heun and RK4 share some
t_peak-error values. Final-size error is the cleanest of the three
structurally, because S_∞ is a property of the phase portrait, not of
a particular instant.

### 5.3 RQ2 — Solver-in-the-loop recovery

#### Figure E07 — `figures/e07_recovery_bias.pdf`

**What you see.** Log-log |bias in β̂| versus fitting step size h, on
**noise-free** data, one line per solver. This is the mechanism
figure of the whole paper.

**The numbers the report quotes**, at (β, γ) = (0.3, 0.1). Cutting h
from 0.25 to 0.05 is a factor of 5:

| Solver | |bias β̂| at h=0.25 | at h=0.05 | ratio | theory (5^p) |
|---|---|---|---|---|
| Euler | 4.48×10⁻³ | 8.93×10⁻⁴ | 5.0 | 5 |
| Heun | 5.88×10⁻⁵ | 2.43×10⁻⁶ | 24.3 | 25 |
| RK4 | 7.17×10⁻⁹ | 1.18×10⁻¹¹ | 610 | 625 |

**How to read it.** If the lines were flat, solver error would not
reach the estimator (Capaldi’s implicit assumption, as a picture).
If the lines had the wrong slope, the leakage would be real but we
would not understand it. They have the *right* slope. That is the
claim: estimation bias is not a mysterious inverse-problem artefact;
it is the ODE method’s truncation error, pushed through the
least-squares map, and it inherits the method’s order almost exactly.

Variance on these noise-free cells is ~10⁻²¹ — i.e. every replicate
lands on the same θ̂. The displacement is bias, not noise.

Asymmetry (not plotted, visible in the JSON): |bias γ̂| is smaller
than |bias β̂| but the same order in h. The cost valley runs along
roughly-constant R₀, so a solver-wrong trajectory is compensated
mostly by sliding β and γ *together*. The ratio is more stable than
either parameter. Capaldi already told us this about *noise*; E07
says it is also true of *discretisation*.

### 5.4 RQ3 — Noise, sampling, I₀, UQ

#### Figure E08 — `figures/e08_noise_degradation.pdf`

**What you see.** RMSE(β̂) versus noise level σ_frac, with a bootstrap
confidence band on the RMSE curve itself (not on β̂).

**The numbers.**

| σ_frac | RMSE(β̂) | 95 % CI on RMSE |
|---|---|---|
| 0 | 1.18×10⁻¹¹ | (essentially a point) |
| 0.02 | 5.62×10⁻⁴ | [4.18, 7.12]×10⁻⁴ |
| 0.05 | 1.79×10⁻³ | [1.49, 2.09]×10⁻³ |
| 0.10 | 2.94×10⁻³ | [1.99, 3.94]×10⁻³ |

**How to read it.** At σ = 0 the RMSE *is* the solver bias of whatever
solver E08 used (tight), hence 10⁻¹¹. After that, RMSE grows roughly
linearly with σ, as a well-behaved least-squares estimator under
additive Gaussian noise should. The band widening at σ = 0.10 is the
quick profile’s small replicate count (M = 30) talking.

#### Figure E09 — `figures/e09_sampling_window.pdf`

**What you see.** RMSE(β̂) versus observation-window fraction (of a
horizon ~6× peak time), one curve per sampling interval Δt.

**The numbers.**

| Δt | window 25 % (pre-peak) | window 50 % | window 100 % |
|---|---|---|---|
| 0.5 | 1.19×10⁻³ (115 pts) | 0.98×10⁻³ | 0.89×10⁻³ (460 pts) |
| 1.0 | 1.80×10⁻³ (57 pts) | 1.47×10⁻³ | 1.48×10⁻³ (230 pts) |
| 2.0 | 3.01×10⁻³ (28 pts) | 1.97×10⁻³ | 1.51×10⁻³ (115 pts) |

**How to read it.** Two effects multiply. (i) Cutting the window to
the first quarter — well before t_peak ≈ 38 — roughly doubles RMSE
at Δt = 2, and still hurts at Δt = 0.5. You have not seen the peak,
so you have not seen the information that separates β from γ.
(ii) Coarser sampling (larger Δt) raises the whole curve. Capaldi’s
Section 6, with a number on it. The 25 %-window / Δt = 2 cell
(RMSE 3.0×10⁻³, 28 points) is the “you fitted an epidemic from the
early growth phase and a handful of observations” warning.

#### E10 — no report figure, but a result

Treating I₀ as a third unknown, rather than a known initial condition:

| I₀ | κ (2-param) | κ (3-param) | var(β̂) known I₀ | var(β̂) unknown I₀ |
|---|---|---|---|---|
| 1 | 2.29 | 211 | 1.89×10⁻⁶ | 3.55×10⁻⁵ |
| 5 | 2.20 | 671 | 3.33×10⁻⁶ | 4.43×10⁻⁵ |
| 20 | 2.59 | 1828 | 3.58×10⁻⁶ | 1.25×10⁻⁵ |

κ jumps two to three orders of magnitude. This is the
κ(JᵀJ) = κ(J)² point in a table: adding a parameter whose
sensitivity is nearly collinear with the existing ones does not just
add a variance term, it wrecks the existing ones. Capaldi flagged
this; E10 measures it with our LU-estimated condition numbers.

#### Figure E11 — `figures/e11_uncertainty.pdf`

**What you see.** Two panels.

*Left:* (β, γ) plane. Black × = truth (0.3, 0.1). Blue closed curve =
asymptotic 95 % ellipse. Amber cloud = residual-bootstrap θ̂’s. Green
cloud = thinned MCMC samples. (Monte Carlo’s cloud is in the JSON;
the figure script currently overlays asymptotic + bootstrap + MCMC.)

*Right:* profile likelihood for β (γ re-optimised). Green = full
window, a U with a minimum near 0.3. Red = truncated at 30 % of
t_peak (t ≈ 11.5 of 38.4), visibly flatter.

**The numbers.**

| Route | What it reports | Value |
|---|---|---|
| Point estimate (LM) | θ̂ | (0.2989, 0.1003) |
| Asymptotic | ρ(β̂, γ̂) | 0.635 |
| Asymptotic | Σ | [[1.97e-6, 8.90e-7], [8.90e-7, 9.97e-7]] |
| Bootstrap | mean, std | (0.2988, 0.1002), std (0.00133, 0.00098) |
| Monte Carlo | bias, RMSE | bias ≈ −2×10⁻⁴, RMSE (0.00181, 0.00122) |
| MCMC | posterior mean | (0.2992, 0.1005) |
| MCMC | R̂ | (1.091, 1.047) — not fully mixed |
| MCMC | ESS | 40 (β), 33 (γ) — short chains |
| MCMC | acceptance | 0.17–0.22 |

**How to read it.** Left panel: three independent machines drew
essentially the same puddle around the truth. That is the “our UQ
is not an artefact of one formula” check. Right panel: the red curve
is E09, drawn as a picture instead of an RMSE. A flatter profile
means a wider set of β values produce almost the same min-J, which
*is* non-identifiability.

Do not over-claim the MCMC diagnostics on this profile. R̂ > 1.01 and
ESS in the tens mean “the overlay is qualitatively right; do not quote
an ESS-based Monte Carlo error.”

### 5.5 The headline — crossover σ\*

#### Figure E13 — `figures/e13_crossover.pdf`

**What you see.** Log-log |error in β̂| versus σ, for Forward Euler at
the first h in the sweep (h = 0.5). Blue markers: std of β̂ across
replicates, rising with σ. Red dashed horizontal: the deterministic
solver bias at this h. A green vertical line for σ\* is drawn only if
the interpolation found a crossing — on the current artifact it does
not, so the figure is the two curves without a marked intersection.

**The numbers, Euler, (β, γ) = (0.3, 0.1).**

| h | \|bias β̂\| | std(β̂) at σ=0.01 | at σ=0.05 | at σ=0.10 | σ\* found? |
|---|---|---|---|---|---|
| 0.5 | 9.02×10⁻³ | 3.48×10⁻⁴ | 1.64×10⁻³ | 3.48×10⁻³ | no |
| 1.0 | 1.82×10⁻² | 3.30×10⁻⁴ | 1.95×10⁻³ | 2.76×10⁻³ | no |
| 2.0 | 3.71×10⁻² | 3.26×10⁻⁴ | 1.67×10⁻³ | 2.81×10⁻³ | no |

Bias scales ≈ linearly with h (Euler, p = 1): 0.009 → 0.018 → 0.037
as h doubles, as it should. Noise-induced std is almost *independent*
of h — at these step sizes the replicate cloud’s width is set by σ,
not by the discretisation. That is why a crossing *must* exist at
large enough σ: a flat-in-h noise floor will eventually overtake a
fixed bias. We simply did not push σ far enough on the quick grid
(max σ_frac = 0.10). Linear extrapolation of std ≈ 0.035 · σ_frac
against bias 0.009 at h = 0.5 puts a rough σ\*_β near σ_frac ≈ 0.26,
outside the sweep. The `full` profile goes to 0.20 with 300
replicates; that is the run that would turn this table’s “no” into a
number.

**How to read the figure anyway.** The *idea* is the intersection.
Everything below the red line and left of where the blue curve would
cross is the “solver-dominated” half-plane: your estimate is more
wrong because of Euler than because of noise. Everything above/right
is the “noise-dominated” half-plane: Capaldi’s world, where not
specifying the solver was reasonable. The contribution is to draw
the boundary, not to pick a side.

### 5.6 RQ4 — Extensions

#### E12 — model variants (no dedicated report figure; numbers in JSON)

Observed orders at T = 60, against RK45-tight “truth” for the
non-SIR models:

| Model | Euler p̂ | Heun p̂ | RK4 p̂ | e_Euler(h=1) | e_RK4(h=1) |
|---|---|---|---|---|---|
| SIR | 1.04 | 1.95 | 3.94 | 17.7 | 0.0029 |
| SEIR | 1.01 | 1.97 | 3.99 | 56.3 | 0.00080 |
| SIRS | 1.03 | 1.95 | 3.93 | 17.4 | 0.0017 |
| SIR-V | 1.00 | 1.95 | 3.94 | 35.5 | 0.0032 |

**How to read it.** The ordering Euler ≪ Heun ≪ RK4, and the orders
themselves, survive a latent period, waning immunity, and a
vaccination sink. The finding is not an artefact of SIR’s exact
reduction. Absolute errors *do* change (SEIR at Euler/h=1 is much
worse than SIR — an extra stiff-ish timescale σ_inc), but the
*ordering* does not. That is all E12 is asked to show.

#### E14 — Eyam 1666

**The data.** Village of Eyam, Derbyshire, 1665–66 plague. Famous
because the village self-quarantined, so N ≈ 261 is a plausible closed
population. CSV in `data/raw/eyam_1666.csv`, 8 (day, S) pairs from
1666-06-19 (S = 254, so I₀ = 7) to 1666-09-25 (S = 83). Commonly
attributed to Raggett (1982), *Bull. IMA* 18:221–226.

**The fit (RK4).** β̂ ≈ 0.157, γ̂ ≈ 0.094, R̂₀ ≈ 1.67, residual SS ≈ 45.4.
Predicted S tracks the eight points to a few individuals (residuals
between about −3.7 and +4.2). The asymptotic ellipse is tight and
strongly correlated (off-diagonal of Σ comparable to the diagonal),
which is what Capaldi taught us to expect at R₀ near 1.7.

**How to read it, and the caveat you must say out loud.** This is a
*demonstration that the pipeline runs on a real table*, not a
historical finding. `data/raw/provenance.md` is explicit: the eight
pairs were reconstructed from memory of the widely copied teaching
table and have **not** been checked against Raggett (1982) in this
work. Until they are, do not quote 1.67 as “our Eyam R₀”. Swap in
a verified table and E14 does not need a code change — only a rerun.

### 5.7 Optimizers, for completeness (E06, no report figure)

8 random starts around the truth, noise-free-enough data:

| Optimizer | Success | Mean iters | Mean f-evals |
|---|---|---|---|
| Levenberg–Marquardt | 100 % | 8.5 | 19.8 |
| Nelder–Mead | 100 % | 58.5 | 111 |
| Gauss–Newton in log-space | 87.5 % | 9.5 | 35.8 |
| Grid, then GN | 75 % | 9.25 | 35.8 |
| Golden coordinate descent | 0 % * | 20 | 641 |

\*“0 % success” means it did not hit the tight convergence flag; the
example path in the JSON still walks from (0.45, 0.17) to
(0.303, 0.100), i.e. it is crawling the valley, just slowly. LM is
the production method for a reason. NM is the one you animate. Grid
search is the one that *draws* the valley.

### 5.8 What the discussion wants you to take away

Two points, and only two.

1. **Mass conservation is the wrong sanity check for SIR.** It is an
   algebraic identity of the vector field, inherited for free by every
   explicit RK method. The phase invariant is the check that carries
   information, and it is free.
2. **The crossover reframes the base paper rather than refuting it.**
   Capaldi et al. did not need to specify a solver because, in the
   noise regime they study, sampling uncertainty already dominates.
   That is a *region of the (σ, h) plane*, not a general fact. We
   drew (and, on a fuller sweep, would number) the boundary. Below
   it, the numerical-analysis choices this course teaches are the
   dominant reason an epidemiological estimate is or is not
   trustworthy. Above it, they are not, and the modeler’s effort
   belongs on data and on the noise model.

### 5.9 Threats to validity (do not skip these)

- **Profile scale.** Quick-profile replicate counts are small. Point
  estimates of RMSE and σ\* carry Monte Carlo noise. The *qualitative*
  orderings — especially E07’s slopes — are already extremely clean.
- **E13 did not numerically locate σ\* on this profile.** The
  definition is in the code; the intersection is not in the JSON.
  Extrapolation suggests σ\*_β(h=0.5) ≳ 0.2. Quote the figure as
  “the two curves, and where they would meet,” not as “σ\* = …”.
- **Synthetic-data circularity.** Core claims use data generated from
  the same SIR being fit. That is the correct design for isolating
  solver error. It cannot show that SIR is a good model of any real
  outbreak.
- **Eyam provenance.** See §5.6.
- **Single model family.** SEIR / SIRS / SIR-V are still small,
  non-stiff systems. Whether the *numerical value* of σ\* transfers
  to a stiff, high-dimensional epidemic model is untested. The
  *ordering* of solvers is what E12 supports.
- **MCMC length.** Quick-profile R̂ and ESS are not good enough to
  treat the MCMC cloud as a fully mixed posterior.

### 5.10 What “make all” actually rebuilds

```
make setup         # venv + Python deps + npm
make test          # pytest (hygiene, order, gold standard, estimators, UQ)
make experiments   # E01–E14, PROFILE=quick by default
make figures       # results/*.json → figures/*.pdf,*.svg
make report        # latexmk report/main.tex
make web           # dashboard at localhost:5173
```

A single experiment at paper scale: `make experiment ID=E13 PROFILE=full`.

---

## Appendix A — Map from this guide to the report

| Report section | This guide |
|---|---|
| Abstract + Introduction | §1 |
| Background | §2 |
| Methods | §3 and §4 |
| Results | §5.2–§5.6 |
| Discussion | §5.8 |
| Threats to validity | §5.9 |
| Conclusion | §1.3 and §5.8 |
| Appendix (profiles, reproducibility, team) | below |

**Team (from the report appendix).**

- Saif Uz Zaman — numerics core: SIR model and analytic Jacobians, the six solvers with
  dense output, the gold standard (adaptive Simpson + safeguarded Newton), LU/QR, E01–E04.
- Sayaad Muzahid Masfi — power method, inverse power method, and Hotelling deflation
  (`linalg/eigen.py`, used in E03, E05, E10, E11, E14) and their tests; E13 grid redesign;
  end-to-end runs on Windows and Linux; the report, RQ notebooks, and presentation.
- Ibtida bin Ahmed — sensitivity equations, observation and noise models, the five
  optimizers, the four UQ routes and profile likelihood, E05–E11 and E13.
- Sakif Naieb Raiyan — React/TypeScript dashboard (ten pages, d3-based SVG charts), the
  TypeScript numerics twin, and the Python/TypeScript parity test.
- Aurchi Chowdhury — experiment runner, configs, and content-hash manifest; JSON schema and
  web export; SEIR/SIRS/vaccination variants (E12); Eyam data provenance (E14); figures.

## Appendix B — Glossary

| Symbol / term | Meaning |
|---|---|
| β, γ | Transmission and recovery rates |
| R₀ | β/γ, basic reproduction number (also, unfortunately, used in the report for the initial R compartment — the gold-standard code calls that `r0_init`) |
| R_eff(t) | R₀ · S(t)/N |
| h | ODE step size |
| Δt, dt | Observation sampling interval (independent of h) |
| σ, σ_frac | Noise level; σ = σ_frac · max(I) |
| σ\* | Crossover: std(σ\*, h) = \|bias(h)\| |
| J(θ) | Least-squares cost |
| χ | Residual / observation Jacobian (sensitivity matrix) |
| κ | Condition number |
| p̂ | Observed convergence order |
| Q(t) | Phase invariant ln S + (β/(γN)) R |
| prevalence | I(t) |
| incidence | New cases in a window, ∫ βSI/N dt |
| gold standard | Solver-free SIR trajectory from (2.1)+(2.2) |
| quick / full | Experiment size profiles in every `configs/eXX.yaml` |

## Appendix C — Suggested reading order if you are presenting this

1. This file, §1 (five minutes).
2. Report Introduction + Background (the Capaldi gap).
3. This file, §2.2 and §4.1 (gold standard and the mass-conservation trap).
4. Figure E01, then E02, then E07, then E13 — in that order. Those four
   pictures *are* the talk.
5. Report Discussion, then Threats, then this file §5.9 so you do not
   over-claim σ\* or Eyam.
6. Live dashboard: Solver Arena (drag h) → Fitting Studio (watch LM vs
   Nelder–Mead) → Crossover page.

If someone only remembers one sentence, it should be this:

> There is a noise level below which your choice of ODE solver decides
> whether an SIR parameter estimate is trustworthy, and above which it
> does not matter — and that level is a number you can measure.

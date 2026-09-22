"""Adaptive Metropolis-Hastings MCMC (PLAN.md section 2.7, route 4 and
section 2.5 Week 9-10 syllabus item). Gaussian likelihood with sigma sampled
as a nuisance parameter (inverse-gamma conjugate update), Haario-style
adaptive proposal covariance after a burn-in window, target acceptance rate
~0.234 for 2-D random-walk Metropolis. Reports trace diagnostics: effective
sample size (via integrated autocorrelation time) and Gelman-Rubin R-hat
across multiple chains.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sirlab.estimate.base import ForwardModel


@dataclass
class MCMCResult:
    chains: NDArray[np.float64]  # (n_chains, n_samples, n_param) post-burn-in
    sigma_chains: NDArray[np.float64]  # (n_chains, n_samples)
    acceptance_rate: NDArray[np.float64]  # per chain
    log_post_chains: NDArray[np.float64]

    def pooled(self) -> NDArray[np.float64]:
        return self.chains.reshape(-1, self.chains.shape[-1])


def _log_likelihood(fwd: ForwardModel, theta: NDArray[np.float64], obs: NDArray[np.float64], sigma: float) -> float:
    if np.any(theta <= 0) or sigma <= 0:
        return -np.inf
    pred = fwd.predict(theta)
    n = len(obs)
    resid = obs - pred
    return float(-0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(resid**2) / sigma**2)


def _run_one_chain(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta0: NDArray[np.float64],
    *,
    n_samples: int,
    burn_in: int,
    prop_scale: NDArray[np.float64],
    rng: np.random.Generator,
    adapt_every: int = 100,
    adapt_start: int = 200,
    sigma_prior_shape: float = 2.0,
    sigma_prior_scale: float = 1.0,
):
    n_param = len(theta0)
    theta = theta0.copy()
    log_theta = np.log(theta)  # sample in log-space for positivity
    sigma = np.std(obs) * 0.5 + 1e-6
    total = n_samples + burn_in

    samples = np.empty((total, n_param))
    sigmas = np.empty(total)
    log_posts = np.empty(total)

    cov = np.diag(prop_scale**2)
    accepted = 0
    history = np.empty((total, n_param))

    log_post_cur = _log_likelihood(fwd, np.exp(log_theta), obs, sigma)

    for it in range(total):
        # adapt proposal covariance (Haario et al.) from the running sample history
        if it > adapt_start and it % adapt_every == 0:
            sd = 2.38**2 / n_param
            emp_cov = np.cov(history[:it].T) + 1e-8 * np.eye(n_param)
            cov = sd * emp_cov

        try:
            step = rng.multivariate_normal(np.zeros(n_param), cov)
        except np.linalg.LinAlgError:
            step = rng.normal(scale=prop_scale)
        log_theta_prop = log_theta + step
        log_post_prop = _log_likelihood(fwd, np.exp(log_theta_prop), obs, sigma)

        if np.log(rng.uniform()) < (log_post_prop - log_post_cur):
            log_theta = log_theta_prop
            log_post_cur = log_post_prop
            accepted += 1

        # Gibbs update of sigma^2 given theta (inverse-gamma conjugate for
        # Gaussian likelihood with known mean function)
        pred = fwd.predict(np.exp(log_theta))
        ssr = np.sum((obs - pred) ** 2)
        shape = sigma_prior_shape + len(obs) / 2.0
        scale = sigma_prior_scale + ssr / 2.0
        sigma2 = 1.0 / rng.gamma(shape, 1.0 / scale)
        sigma = np.sqrt(sigma2)
        log_post_cur = _log_likelihood(fwd, np.exp(log_theta), obs, sigma)

        history[it] = log_theta
        samples[it] = np.exp(log_theta)
        sigmas[it] = sigma
        log_posts[it] = log_post_cur

    return samples[burn_in:], sigmas[burn_in:], accepted / total, log_posts[burn_in:]


def run_mcmc(
    fwd: ForwardModel,
    obs: NDArray[np.float64],
    theta0: NDArray[np.float64],
    *,
    n_chains: int = 4,
    n_samples: int = 2000,
    burn_in: int = 1000,
    prop_scale: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> MCMCResult:
    rng = rng or np.random.default_rng()
    n_param = len(theta0)
    prop_scale = prop_scale if prop_scale is not None else 0.05 * np.ones(n_param)

    chains, sigma_chains, accept_rates, log_posts = [], [], [], []
    for c in range(n_chains):
        # overdispersed starting points (Gelman-Rubin needs this to be meaningful)
        start = theta0 * rng.uniform(0.6, 1.6, size=n_param)
        chain_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        samples, sigmas, acc, lp = _run_one_chain(
            fwd, obs, start, n_samples=n_samples, burn_in=burn_in, prop_scale=prop_scale, rng=chain_rng
        )
        chains.append(samples)
        sigma_chains.append(sigmas)
        accept_rates.append(acc)
        log_posts.append(lp)

    return MCMCResult(
        chains=np.array(chains),
        sigma_chains=np.array(sigma_chains),
        acceptance_rate=np.array(accept_rates),
        log_post_chains=np.array(log_posts),
    )


def gelman_rubin(chains: NDArray[np.float64]) -> NDArray[np.float64]:
    """R-hat per parameter, chains shape (n_chains, n_samples, n_param)."""
    m, n, p = chains.shape
    chain_means = chains.mean(axis=1)  # (m, p)
    grand_mean = chain_means.mean(axis=0)  # (p,)
    b = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)  # between-chain variance
    w = np.mean(chains.var(axis=1, ddof=1), axis=0)  # within-chain variance
    var_hat = (n - 1) / n * w + b / n
    return np.sqrt(var_hat / w)


def integrated_autocorr_time(x: NDArray[np.float64], *, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time via the initial positive sequence
    estimator (Geyer 1992), used for effective sample size = n / (2*tau)."""
    n = len(x)
    x = x - x.mean()
    if max_lag is None:
        max_lag = n // 2
    var0 = np.dot(x, x) / n
    if var0 == 0:
        return 1.0
    rho = np.empty(max_lag)
    for k in range(max_lag):
        rho[k] = np.dot(x[: n - k], x[k:]) / n / var0
    tau = 1.0
    k = 1
    while k + 1 < max_lag:
        pair_sum = rho[k] + rho[k + 1]
        if pair_sum < 0:
            break
        tau += 2 * pair_sum
        k += 2
    return max(tau, 1.0)


def effective_sample_size(x: NDArray[np.float64]) -> float:
    tau = integrated_autocorr_time(x)
    return len(x) / tau

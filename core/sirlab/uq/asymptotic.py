"""Asymptotic / linearized confidence region (PLAN.md section 2.7, route 1).

Cov(theta_hat) ~= sigma_hat^2 (J^T J)^-1,  sigma_hat^2 = J(theta_hat)/(n-p)

The 95% ellipse is drawn from the eigendecomposition of the covariance
matrix, computed by our power method with Hotelling deflation
(linalg/eigen.py). This is the cheapest UQ route and the one every other
route is compared against.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sirlab.linalg.eigen import symmetric_eigen


def confidence_ellipse(cov: NDArray[np.float64], theta_hat: NDArray[np.float64], *, confidence: float = 0.95, n_points: int = 200):
    """2-parameter 95% confidence ellipse via eigendecomposition of `cov`.
    Returns (ellipse_xy of shape (n_points, 2), chi2_quantile_used)."""
    from sirlab.uq._chi2 import chi2_quantile_2df

    chi2_val = chi2_quantile_2df(confidence)
    eigvals, eigvecs = symmetric_eigen(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    scaled = eigvecs @ (np.sqrt(eigvals * chi2_val)[:, None] * circle)
    return (theta_hat[:, None] + scaled).T, chi2_val


def correlation(cov: NDArray[np.float64]) -> float:
    """Correlation coefficient rho(beta_hat, gamma_hat) from the 2x2
    covariance matrix -- the beta-gamma identifiability diagnostic from the
    base paper (Capaldi et al. 2012), reproduced here."""
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

"""GARCH(1,1) with Normal or Student-t innovations, fitted by maximum likelihood.

sigma2_t = omega + alpha * e_{t-1}^2 + beta * sigma2_{t-1}
Implemented from scratch (no `arch` dependency) so every step is inspectable.
The variance recursion is vectorised with scipy.signal.lfilter.
Returns are scaled x100 internally for numerical stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln


@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    nu: float | None     # None -> Normal innovations
    loglik: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol_ann(self) -> float:
        return float(np.sqrt(self.omega / (1 - self.persistence) * 365) / 100)


def variance_path(e: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional variance for t = 0..n (last element = one-step-ahead forecast)."""
    s0 = np.var(e)
    x = omega + alpha * e**2
    y, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])
    return np.concatenate([[s0], y])


def _nll(params, e, dist):
    omega, alpha, beta = params[:3]
    if alpha + beta >= 0.9999:
        return 1e10
    s2 = variance_path(e, omega, alpha, beta)[:-1]
    if dist == "normal":
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + e**2 / s2)
    nu = params[3]
    z2 = e**2 / s2
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    return -np.sum(ll)


def fit(returns, dist: str = "t") -> GarchFit:
    e = 100 * (np.asarray(returns) - np.mean(returns))
    v = np.var(e)
    x0 = [v * 0.05, 0.08, 0.90] + ([6.0] if dist == "t" else [])
    bounds = [(1e-6, 10 * v), (1e-4, 0.5), (0.3, 0.9989)] + ([(2.1, 60.0)] if dist == "t" else [])
    res = minimize(_nll, x0, args=(e, dist), bounds=bounds, method="L-BFGS-B")
    p = res.x
    return GarchFit(p[0], p[1], p[2], p[3] if dist == "t" else None, -res.fun)


def forecast_vol(returns, g: GarchFit) -> float:
    """One-day-ahead conditional volatility (decimal, daily)."""
    e = 100 * (np.asarray(returns) - np.mean(returns))
    return float(np.sqrt(variance_path(e, g.omega, g.alpha, g.beta)[-1]) / 100)


def conditional_vol(returns, g: GarchFit) -> np.ndarray:
    e = 100 * (np.asarray(returns) - np.mean(returns))
    return np.sqrt(variance_path(e, g.omega, g.alpha, g.beta)[:-1]) / 100

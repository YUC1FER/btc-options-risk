"""Black-76 pricing and Greeks.

Deribit BTC options are European options on the BTC index, quoted against the
forward/futures price of the same expiry. We therefore price with Black-76 on
the forward F and set r = 0 (premium settled against the forward; discounting
is second-order at these tenors and is a documented simplification).

Units: USD per 1 BTC of notional (one Deribit contract = 1 BTC).
cp = +1 for calls, -1 for puts. T in years. sigma as a decimal (0.55 = 55%).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_T_MIN = 1e-10


def _d1d2(F, K, T, sigma):
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    T = np.maximum(T, _T_MIN)
    sigma = np.maximum(sigma, 1e-8)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sq
    return d1, d1 - sq


def price(F, K, T, sigma, cp):
    d1, d2 = _d1d2(F, K, T, sigma)
    cp = np.asarray(cp, dtype=float)
    F_, K_ = np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    val = cp * (F_ * norm.cdf(cp * d1) - K_ * norm.cdf(cp * d2))
    intrinsic = np.maximum(cp * (F_ - K_), 0.0)
    return np.where(np.asarray(T) <= _T_MIN, intrinsic, val)


def delta(F, K, T, sigma, cp):
    d1, _ = _d1d2(F, K, T, sigma)
    cp = np.asarray(cp, dtype=float)
    return cp * norm.cdf(cp * d1)


def gamma(F, K, T, sigma):
    d1, _ = _d1d2(F, K, T, sigma)
    T = np.maximum(np.asarray(T, dtype=float), _T_MIN)
    return norm.pdf(d1) / (np.asarray(F) * np.asarray(sigma) * np.sqrt(T))


def vega(F, K, T, sigma):
    """P&L per +1 vol point (0.01 in sigma)."""
    d1, _ = _d1d2(F, K, T, sigma)
    T = np.maximum(np.asarray(T, dtype=float), _T_MIN)
    return np.asarray(F) * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(F, K, T, sigma):
    """P&L per calendar day of time decay (r = 0, so identical for calls and puts)."""
    d1, _ = _d1d2(F, K, T, sigma)
    T = np.maximum(np.asarray(T, dtype=float), _T_MIN)
    return -np.asarray(F) * norm.pdf(d1) * np.asarray(sigma) / (2 * np.sqrt(T)) / 365.0


def implied_vol(target_price, F, K, T, cp, lo=1e-4, hi=10.0):
    """Invert Black-76 by Brent's method. Returns nan if no root in [lo, hi]."""
    f = lambda s: float(price(F, K, T, s, cp)) - target_price
    try:
        return brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        return float("nan")

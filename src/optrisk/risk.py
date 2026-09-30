"""Portfolio risk: VaR / Expected Shortfall under several methods, and stress tests.

All P&L is 1-day horizon, USD, full revaluation unless stated. Losses are positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .portfolio import Portfolio


def joint_shocks(spot: pd.Series, dvol: pd.Series) -> pd.DataFrame:
    """Daily log spot returns paired with same-day relative DVOL changes."""
    df = pd.concat([spot, dvol], axis=1, join="inner").dropna()
    out = pd.DataFrame({"r": np.log(df["close"]).diff(), "v": df["dvol"].pct_change()}).dropna()
    return out


def var_es(pnl, alpha: float = 0.99) -> tuple[float, float]:
    losses = -np.asarray(pnl)
    var = float(np.quantile(losses, alpha))
    es = float(losses[losses >= var].mean())
    return var, es


def historical(pf: Portfolio, shocks: pd.DataFrame, mode="sticky_strike", with_vol=True):
    F = pf.F0 * np.exp(shocks["r"].values)
    v = shocks["v"].values if with_vol else 0.0
    return pf.pnl(F, v, dt_days=1, mode=mode)


def monte_carlo(pf: Portfolio, shocks: pd.DataFrame, sigma_next: float, nu: float,
                n: int = 100_000, seed: int = 11, mode="sticky_strike"):
    """Spot: Student-t shocks scaled to the GARCH one-day forecast.
    Vol:  relative IV change = beta * r + eps, with beta and eps fitted by OLS on
    history -> preserves the empirical spot/vol dependency (leverage effect)."""
    rng = np.random.default_rng(seed)
    z = stats.t.rvs(nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
    r = sigma_next * z
    beta, intercept = np.polyfit(shocks["r"], shocks["v"], 1)
    resid_sd = np.std(shocks["v"] - (beta * shocks["r"] + intercept))
    v = intercept + beta * r + rng.normal(0, resid_sd, n)
    return pf.pnl(pf.F0 * np.exp(r), v, dt_days=1, mode=mode), {"beta": beta, "resid_sd": resid_sd}


def stress_grid(pf: Portfolio, spot_moves, vol_moves, mode="sticky_strike") -> pd.DataFrame:
    grid = {}
    for vm in vol_moves:
        grid[f"{vm:+.0%}"] = pf.pnl(pf.F0 * (1 + np.asarray(spot_moves)), vm, dt_days=1, mode=mode)
    return pd.DataFrame(grid, index=[f"{m:+.0%}" for m in spot_moves]).T  # rows: vol, cols: spot


def named_scenarios(pf: Portfolio, shocks: pd.DataFrame, mode="sticky_strike") -> pd.DataFrame:
    worst = shocks["r"].idxmin(); best = shocks["r"].idxmax(); vol_spike = shocks["v"].idxmax()
    sc = [
        (f"Worst historical day in sample ({worst:%Y-%m-%d})", np.exp(shocks.loc[worst, "r"]) - 1, shocks.loc[worst, "v"]),
        (f"Best historical day in sample ({best:%Y-%m-%d})", np.exp(shocks.loc[best, "r"]) - 1, shocks.loc[best, "v"]),
        (f"Largest DVOL spike in sample ({vol_spike:%Y-%m-%d})", np.exp(shocks.loc[vol_spike, "r"]) - 1, shocks.loc[vol_spike, "v"]),
        ("Hypothetical crash: spot -30%, IV +50%", -0.30, 0.50),
        ("Hypothetical melt-up: spot +25%, IV +20%", 0.25, 0.20),
        ("Hypothetical vol crush: spot 0%, IV -30%", 0.0, -0.30),
    ]
    rows = [{"scenario": n, "spot move": s, "IV move (rel.)": v,
             "P&L (USD)": float(pf.pnl(pf.F0 * (1 + s), v, dt_days=1, mode=mode)[0])} for n, s, v in sc]
    return pd.DataFrame(rows)

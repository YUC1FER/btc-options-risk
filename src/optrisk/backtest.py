"""Model validation and hedge research.

1. VaR backtesting on BTC daily returns (long 1 BTC):
   Historical Simulation, Normal, GARCH-Normal, GARCH-t, compared with the
   Kupiec unconditional-coverage test and the Christoffersen independence test.
2. Delta-hedge effectiveness: short 30-day ATM call sold at DVOL, delta-hedged
   at different frequencies over rolling historical windows, split by vol regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import pricing as bs
from . import vol


# ------------------------------------------------------------------ VaR tests
def kupiec(exceptions: np.ndarray, p: float) -> tuple[float, float]:
    n, x = len(exceptions), int(exceptions.sum())
    pi = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = ((n - x) * np.log(1 - pi) if x < n else 0.0) + (x * np.log(pi) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(exceptions: np.ndarray) -> tuple[float, float]:
    e = exceptions.astype(int)
    a, b = e[:-1], e[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    xlogy = lambda n, q: n * np.log(q) if n > 0 else 0.0
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def var_backtest(returns: pd.Series, alpha: float = 0.99, window: int = 500, refit_every: int = 20):
    r = returns.values
    q = 1 - alpha
    z_n = stats.norm.ppf(q)
    methods = ["Historical Simulation", "Normal", "GARCH-Normal", "GARCH-t"]
    var = {m: np.full(len(r), np.nan) for m in methods}
    g_n = g_t = None
    for t in range(window, len(r)):
        w = r[t - window:t]
        if (t - window) % refit_every == 0:
            g_n, g_t = vol.fit(w, "normal"), vol.fit(w, "t")
        var["Historical Simulation"][t] = -np.quantile(w, q)
        var["Normal"][t] = -(w.mean() + z_n * w.std(ddof=1))
        var["GARCH-Normal"][t] = -z_n * vol.forecast_vol(w, g_n)
        s_t = vol.forecast_vol(w, g_t)
        var["GARCH-t"][t] = -stats.t.ppf(q, g_t.nu) * np.sqrt((g_t.nu - 2) / g_t.nu) * s_t
    idx = returns.index[window:]
    var_df = pd.DataFrame({m: v[window:] for m, v in var.items()}, index=idx)
    realised = returns.iloc[window:]
    rows = []
    for m in methods:
        exc = (realised.values < -var_df[m].values)
        lr_k, p_k = kupiec(exc, q); lr_c, p_c = christoffersen(exc)
        rows.append({"model": m, "days": len(exc), "exceptions": int(exc.sum()),
                     "expected": round(q * len(exc), 1), "rate": exc.mean(),
                     "Kupiec p": p_k, "Christoffersen p": p_c,
                     "verdict": "accept" if (p_k > 0.05 and p_c > 0.05) else "reject"})
    return pd.DataFrame(rows), var_df, realised


# ---------------------------------------------------------- hedge effectiveness
def delta_hedge_study(spot: pd.Series, dvol: pd.Series, tenor: int = 30, step: int = 3,
                      freqs=(1, 2, 7)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sell 1 ATM call (K = S0) at the DVOL implied vol, hedge delta every h days.
    Assumes r = 0, no transaction costs, hedging with spot (perpetual) at daily closes."""
    df = pd.concat([spot, dvol], axis=1, join="inner").dropna()
    S, IV = df["close"].values, df["dvol"].values / 100
    rows = []
    for s in range(0, len(S) - tenor, step):
        path = S[s:s + tenor + 1]; K = path[0]; sig = IV[s]; T0 = tenor / 365
        prem = float(bs.price(path[0], K, T0, sig, 1)); payoff = max(path[-1] - K, 0)
        rv = np.std(np.diff(np.log(path)), ddof=1) * np.sqrt(365)
        rec = {"start": df.index[s], "iv": sig, "rv": rv, "unhedged": (prem - payoff) / K}
        for h in freqs:
            gains = 0.0
            for t in range(0, tenor, h):
                d = float(bs.delta(path[t], K, (tenor - t) / 365, sig, 1))
                gains += d * (path[min(t + h, tenor)] - path[t])
            rec[f"hedged_{h}d"] = (prem - payoff + gains) / K
        rows.append(rec)
    res = pd.DataFrame(rows)
    res["regime"] = pd.qcut(res["iv"], 3, labels=["low IV", "mid IV", "high IV"])
    summary = []
    for label, g in [("all", res)] + [(str(k), v) for k, v in res.groupby("regime", observed=True)]:
        u = g["unhedged"].var()
        for h in freqs:
            col = g[f"hedged_{h}d"]
            summary.append({"regime": label, "hedge freq": f"every {h}d", "windows": len(g),
                            "effectiveness": 1 - col.var() / u,
                            "mean P&L (% spot)": col.mean(), "P&L stdev (% spot)": col.std(),
                            "5% worst (% spot)": col.quantile(0.05)})
    return res, pd.DataFrame(summary)

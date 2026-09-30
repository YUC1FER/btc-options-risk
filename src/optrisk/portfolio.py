"""Portfolio representation, smile model and revaluation.

Two volatility-surface assumptions are supported for scenario revaluation:
- sticky_strike:    each option keeps its own implied vol when spot moves
- sticky_moneyness: implied vol is read off the smile at the new K/F
Comparing the two quantifies how much the VaR depends on the smile dynamics
assumption (a key modelling choice for a skewed BTC book).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import pricing as bs


@dataclass
class Smile:
    """Quadratic smile in log-moneyness k = ln(K/F): iv = a + b k + c k^2."""
    a: float
    b: float
    c: float

    @classmethod
    def fit(cls, K, F, iv, max_abs_k: float = 0.6) -> "Smile":
        k = np.log(np.asarray(K) / F)
        m = np.abs(k) < max_abs_k
        c2, c1, c0 = np.polyfit(k[m], np.asarray(iv)[m], 2)
        return cls(c0, c1, c2)

    def iv(self, K, F):
        k = np.log(np.asarray(K) / np.asarray(F))
        return np.maximum(self.a + self.b * k + self.c * k * k, 0.05)


@dataclass
class Position:
    name: str
    kind: str          # "option" | "future"
    qty: float         # contracts (1 contract = 1 BTC); negative = short
    K: float = np.nan
    T: float = 0.0
    cp: int = 0
    iv: float = np.nan


class Portfolio:
    def __init__(self, positions: list[Position], F0: float, smile: Smile | None = None):
        self.positions, self.F0, self.smile = positions, F0, smile

    # -------------------------------------------------------------- revaluation
    def value(self, F, vol_shift=0.0, dt_days: float = 0.0, mode: str = "sticky_strike"):
        F = np.atleast_1d(np.asarray(F, dtype=float))
        vs = np.broadcast_to(np.asarray(vol_shift, dtype=float), F.shape)
        total = np.zeros_like(F)
        for p in self.positions:
            if p.kind == "future":
                total += p.qty * (F - self.F0)
                continue
            T = max(p.T - dt_days / 365.0, 0.0)
            base_iv = p.iv if mode == "sticky_strike" else self.smile.iv(p.K, F)
            total += p.qty * bs.price(F, p.K, T, base_iv * (1 + vs), p.cp)
        return total

    def pnl(self, F, vol_shift=0.0, dt_days: float = 0.0, mode: str = "sticky_strike"):
        return self.value(F, vol_shift, dt_days, mode) - self.value(self.F0)[0]

    def taylor_pnl(self, F, vol_shift=0.0, dt_days: float = 0.0):
        """Delta-gamma-vega-theta approximation (sticky strike)."""
        F = np.atleast_1d(np.asarray(F, dtype=float)); dF = F - self.F0
        vs = np.broadcast_to(np.asarray(vol_shift, dtype=float), F.shape)
        total = np.zeros_like(F)
        for p in self.positions:
            if p.kind == "future":
                total += p.qty * dF
                continue
            d = bs.delta(self.F0, p.K, p.T, p.iv, p.cp); g = bs.gamma(self.F0, p.K, p.T, p.iv)
            v = bs.vega(self.F0, p.K, p.T, p.iv); th = bs.theta(self.F0, p.K, p.T, p.iv)
            total += p.qty * (d * dF + 0.5 * g * dF**2 + v * (p.iv * vs * 100) + th * dt_days)
        return total

    # ------------------------------------------------------------------- greeks
    def greeks(self) -> pd.DataFrame:
        rows = []
        for p in self.positions:
            if p.kind == "future":
                rows.append({"position": p.name, "qty": p.qty, "delta": p.qty,
                             "gamma": 0.0, "vega": 0.0, "theta": 0.0, "value": 0.0})
                continue
            args = (self.F0, p.K, p.T, p.iv)
            rows.append({"position": p.name, "qty": p.qty,
                         "delta": p.qty * float(bs.delta(*args, p.cp)),
                         "gamma": p.qty * float(bs.gamma(*args)),
                         "vega": p.qty * float(bs.vega(*args)),
                         "theta": p.qty * float(bs.theta(*args)),
                         "value": p.qty * float(bs.price(*args, p.cp))})
        df = pd.DataFrame(rows)
        df.loc[len(df)] = ["TOTAL", np.nan, *df[["delta", "gamma", "vega", "theta", "value"]].sum()]
        return df


def build_default(chain: pd.DataFrame, target_days: int = 30, size: float = 10.0) -> tuple[Portfolio, dict]:
    """Illustrative short-volatility book with a tail hedge, delta-neutral at inception:
    short ATM straddle, long 25-delta puts, residual delta flattened with the future.
    Built from the listed expiry closest to `target_days`."""
    exp_T = chain.groupby("expiry")["T"].first()
    expiry = (exp_T - target_days / 365).abs().idxmin()
    c = chain[chain["expiry"] == expiry]
    F, T = float(c["F"].median()), float(c["T"].iloc[0])
    smile = Smile.fit(c["K"], F, c["iv"])
    ivK = lambda K, cp: float(c[(c["K"] == K) & (c["cp"] == cp)]["iv"].iloc[0])

    K_atm = float(c.loc[(c["K"] - F).abs().idxmin(), "K"])
    puts = c[(c["cp"] == -1) & (c["K"] < F)].copy()
    puts["delta"] = bs.delta(F, puts["K"], T, puts["iv"], -1)
    K_25 = float(puts.loc[(puts["delta"] + 0.25).abs().idxmin(), "K"])

    pos = [
        Position("Short ATM call", "option", -size, K_atm, T, 1, ivK(K_atm, 1)),
        Position("Short ATM put", "option", -size, K_atm, T, -1, ivK(K_atm, -1)),
        Position("Long 25d put", "option", size, K_25, T, -1, ivK(K_25, -1)),
    ]
    pf = Portfolio(pos, F, smile)
    net_delta = float(pf.greeks().iloc[-1]["delta"])
    pf.positions.append(Position("Delta hedge (future)", "future", -net_delta))
    meta = {"expiry": str(expiry)[:10], "days": round(T * 365, 1), "F": F, "K_atm": K_atm, "K_25": K_25,
            "smile": smile}
    return pf, meta

"""Charts and a Markdown research report written to output/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "output"


def _fmt(df: pd.DataFrame, pct_cols=(), usd_cols=(), dec_cols=()) -> str:
    d = df.copy()
    for c in pct_cols:
        d[c] = d[c].map(lambda x: f"{x:.2%}")
    for c in usd_cols:
        d[c] = d[c].map(lambda x: f"{x:,.0f}")
    for c in dec_cols:
        d[c] = d[c].map(lambda x: f"{x:.3f}")
    return d.to_markdown(index=False)


def charts(res: dict) -> None:
    OUT.mkdir(exist_ok=True)
    # 1. P&L distribution
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pnl = res["pnl_hist"]; var = res["var_table"].iloc[0]["VaR 99%"]
    ax.hist(pnl, bins=80, color="#35618f", alpha=0.85)
    ax.axvline(-var, color="#b33", ls="--", label=f"99% VaR = {var:,.0f} USD")
    ax.set(title="1-day P&L distribution - historical full revaluation", xlabel="P&L (USD)", ylabel="days")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "pnl_distribution.png", dpi=140); plt.close(fig)
    # 2. Stress heatmap
    g = res["stress_grid"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    lim = np.abs(g.values).max()
    im = ax.imshow(g.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(g.shape[1]), g.columns); ax.set_yticks(range(g.shape[0]), g.index)
    ax.set(xlabel="spot move", ylabel="IV move (relative)", title="Stress P&L (USD, 1-day, full revaluation)")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]/1000:,.0f}k", ha="center", va="center", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "stress_heatmap.png", dpi=140); plt.close(fig)
    # 3. VaR backtest
    vdf, realised = res["var_series"], res["realised"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(realised.index, realised.values, color="#999", lw=0.6, label="daily return")
    for m, c in zip(["Historical Simulation", "GARCH-t"], ["#35618f", "#b33"]):
        ax.plot(vdf.index, -vdf[m], color=c, lw=1, label=f"-VaR 99% {m}")
    ax.set(title="VaR backtest - long 1 BTC", ylabel="log return"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "var_backtest.png", dpi=140); plt.close(fig)
    # 4. GARCH vs DVOL
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(res["garch_vol"].index, res["garch_vol"] * np.sqrt(365) * 100, label="GARCH-t conditional vol (ann.)")
    ax.plot(res["dvol"].index, res["dvol"], label="DVOL (30d implied)", alpha=0.8)
    ax.set(title="Realised-vol model vs implied vol", ylabel="vol %"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "garch_vs_dvol.png", dpi=140); plt.close(fig)
    # 5. Hedge effectiveness
    hs = res["hedge_summary"]
    piv = hs.pivot(index="regime", columns="hedge freq", values="effectiveness")
    ax = piv.plot.bar(figsize=(8, 4.2), rot=0, title="Delta-hedge effectiveness (1 - Var hedged / Var unhedged)")
    ax.set_ylabel("effectiveness"); plt.tight_layout(); plt.savefig(OUT / "hedge_effectiveness.png", dpi=140); plt.close()


def fat_tail_caveat(nu: float | None, threshold: float = 4.0) -> str | None:
    """Limitation sentence when the fitted Student-t is so heavy-tailed that tail estimates are
    unstable (the t-distribution's variance is infinite at nu <= 2, its 4th moment at nu <= 4)."""
    if nu is None or nu >= threshold:
        return None
    return (f"The fitted Student-t degrees of freedom (nu = {nu:.2f}) imply extremely heavy tails; "
            "variance is barely finite as nu approaches 2, so Expected Shortfall and Monte Carlo tail "
            "estimates are unstable and should be read as indicative.")


def markdown(res: dict) -> Path:
    m = res["meta"]; g = res["garch"]
    gk = res["greeks"].copy()
    lines = [
        "# BTC Options Portfolio - Risk Research Report",
        "",
        f"**Data source:** {res['source']}  ",
        f"**History:** {res['shocks'].index[0]:%Y-%m-%d} to {res['shocks'].index[-1]:%Y-%m-%d} "
        f"({len(res['shocks'])} daily joint spot/IV shocks)  ",
        f"**Portfolio expiry:** {m['expiry']} ({m['days']} days) | forward F = {m['F']:,.0f} | "
        f"ATM K = {m['K_atm']:,.0f} | 25-delta put K = {m['K_25']:,.0f}",
        "",
        "## 1. Portfolio and sources of risk",
        "Illustrative short-volatility book with a tail hedge: short ATM straddle, long 25-delta puts, "
        "delta flattened with the future. Greeks in USD (vega per vol point, theta per day).",
        "",
        _fmt(gk.fillna(0), dec_cols=["delta"],
             usd_cols=["cash gamma (USD per 1% move)", "gamma P&L 1% (USD)", "vega", "theta", "value"]),
        "",
        "Cash gamma = gamma x F^2 / 100, the change in USD delta for a 1% spot move; "
        "gamma P&L 1% = 0.5 x gamma x (0.01 x F)^2, the P&L from gamma alone for a 1% move in either direction.",
        "",
        f"Smile fit (iv = a + b k + c k^2, k = ln K/F): a = {m['smile'].a:.3f}, b = {m['smile'].b:.3f}, "
        f"c = {m['smile'].c:.3f}.",
        "",
        "## 2. Value-at-Risk and Expected Shortfall (1-day)",
        _fmt(res["var_table"], usd_cols=["VaR 99%", "ES 97.5%"]),
        "",
        "![P&L distribution](pnl_distribution.png)",
        "",
        "**Reading the comparison:**",
        *[f"- {x}" for x in res["findings_var"]],
        "",
        "## 3. Stress tests",
        _fmt(res["named"], pct_cols=["spot move", "IV move (rel.)"], usd_cols=["P&L (USD)"]),
        "",
        "![Stress heatmap](stress_heatmap.png)",
        "",
        "## 4. Volatility dynamics - GARCH(1,1)-t",
        f"alpha = {g.alpha:.3f}, beta = {g.beta:.3f}, persistence = {g.persistence:.3f}, "
        f"nu = {g.nu:.2f} (fat tails if < ~10), long-run vol = {g.long_run_vol_ann:.1%} annualised. "
        f"One-day-ahead forecast: {res['sigma_next']:.2%} ({res['sigma_next']*np.sqrt(365):.1%} annualised). "
        f"Spot/IV dependency (OLS of relative IV change on log return): beta = {res['mc_dep']['beta']:.2f}.",
        "",
        "![GARCH vs DVOL](garch_vs_dvol.png)",
        "",
        "## 5. Model validation - VaR backtest (long 1 BTC, 99%, rolling 500-day window)",
        _fmt(res["bt_table"], pct_cols=["rate"], dec_cols=["Kupiec p", "Christoffersen p"]),
        "",
        "Kupiec tests whether the exception rate matches 1%; Christoffersen tests whether exceptions cluster. "
        "p < 0.05 rejects the model.",
        "",
        "![VaR backtest](var_backtest.png)",
        "",
        "## 6. Hedge research - delta-hedging a short 30-day ATM call",
        _fmt(res["hedge_summary"], pct_cols=["effectiveness", "mean P&L (% spot)", "P&L stdev (% spot)", "5% worst (% spot)"]),
        "",
        "![Hedge effectiveness](hedge_effectiveness.png)",
        "",
        *[f"- {x}" for x in res["findings_hedge"]],
        "",
        "## 7. Assumptions and limitations",
        "- Black-76 with r = 0; no funding, basis or discounting effects.",
        "- One smile per expiry (quadratic in log-moneyness); IV scenario shocks applied as a parallel relative shift "
        "equal to the same-day DVOL change - skew and term-structure moves are not modelled separately.",
        "- Historical simulation assumes the sample contains the relevant tail; it is blind to regimes not in the window.",
        "- Hedge study: daily closes, no transaction costs or slippage, constant implied vol over the option's life, "
        "overlapping windows (observations are autocorrelated, so effective sample size is smaller than the count).",
        *[f"- {x}" for x in [fat_tail_caveat(g.nu)] if x],
        "- Illustrative portfolio, not a real trading position.",
        "",
        "## 8. Next research steps",
        "- Replace the parallel IV shock with PCA on the implied-vol surface (level / skew / curvature).",
        "- Filtered historical simulation (GARCH-scaled residuals) to make HS VaR regime-aware.",
        "- Add transaction costs and gamma-threshold (band) hedging to the hedge study.",
        "- Tail modelling with Extreme Value Theory (peaks-over-threshold) for ES.",
    ]
    OUT.mkdir(exist_ok=True)
    path = OUT / "REPORT.md"
    path.write_text("\n".join(lines))
    return path

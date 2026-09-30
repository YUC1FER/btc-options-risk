"""End-to-end run: data -> portfolio -> VaR/ES -> stress -> GARCH -> backtests -> report.

    python run.py                    # live Deribit data (falls back to data/ cache)
    python run.py --mode synthetic   # offline, seeded simulation
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from optrisk import backtest, data, portfolio, report, risk, vol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "synthetic"], default="live")
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--size", type=float, default=10.0, help="contracts per option leg")
    args = ap.parse_args()

    spot, dvol, chain, source = data.load(args.mode, args.days)
    print(f"[data] {source}")
    pf, meta = portfolio.build_default(chain, size=args.size)
    shocks = risk.joint_shocks(spot, dvol)

    # --- VaR / ES under competing assumptions
    g = vol.fit(shocks["r"].values, "t")
    sigma_next = vol.forecast_vol(shocks["r"].values, g)
    pnl_hs = risk.historical(pf, shocks, "sticky_strike")
    variants = {
        "Historical sim - full reval, sticky strike": pnl_hs,
        "Historical sim - full reval, sticky moneyness": risk.historical(pf, shocks, "sticky_moneyness"),
        "Historical sim - spot shocks only (no IV shocks)": risk.historical(pf, shocks, with_vol=False),
        "Historical sim - delta-gamma-vega-theta approximation": pf.taylor_pnl(pf.F0 * np.exp(shocks["r"].values), shocks["v"].values, 1),
    }
    pnl_mc, dep = risk.monte_carlo(pf, shocks, sigma_next, g.nu)
    variants["Monte Carlo - GARCH-t spot + regression IV shocks"] = pnl_mc
    rows = [{"method": k, "VaR 99%": risk.var_es(v, 0.99)[0], "ES 97.5%": risk.var_es(v, 0.975)[1]}
            for k, v in variants.items()]
    import pandas as pd
    var_table = pd.DataFrame(rows)
    base = var_table.iloc[0]["VaR 99%"]
    f_var = []
    for _, r in var_table.iloc[1:].iterrows():
        f_var.append(f"{r['method']}: VaR {r['VaR 99%']:,.0f} USD ({r['VaR 99%']/base-1:+.0%} vs baseline full-reval sticky-strike).")

    # --- stress
    grid = risk.stress_grid(pf, [-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3], [-0.3, -0.15, 0, 0.25, 0.5, 1.0])
    named = risk.named_scenarios(pf, shocks)

    # --- validation + hedge research
    bt_table, var_series, realised = backtest.var_backtest(shocks["r"])
    hedge_raw, hedge_summary = backtest.delta_hedge_study(spot, dvol)
    allrow = hedge_summary[hedge_summary["regime"] == "all"]
    best = allrow.loc[allrow["effectiveness"].idxmax()]
    vrp = (hedge_raw["iv"] - hedge_raw["rv"]).mean()
    f_hedge = [
        f"Best overall hedge frequency: {best['hedge freq']} (effectiveness {best['effectiveness']:.0%}).",
        f"Average implied minus realised vol at inception: {vrp:+.1%} - the sign of the variance risk premium "
        "the short-vol seller is paid to carry.",
    ]
    d1 = hedge_summary[hedge_summary["hedge freq"] == "every 1d"].set_index("regime")["effectiveness"]
    d7 = hedge_summary[hedge_summary["hedge freq"] == "every 7d"].set_index("regime")["effectiveness"]
    f_hedge.append(f"Daily hedging effectiveness: low IV {d1['low IV']:.0%}, high IV {d1['high IV']:.0%}. "
                   f"Weekly hedging: low IV {d7['low IV']:.0%}, high IV {d7['high IV']:.0%}. "
                   f"The cost of hedging less often is {d1['all'] - d7['all']:.1%} of effectiveness overall.")

    res = dict(source=source, meta=meta, greeks=pf.greeks(), shocks=shocks, pnl_hist=pnl_hs,
               var_table=var_table, findings_var=f_var, stress_grid=grid, named=named, garch=g,
               sigma_next=sigma_next, mc_dep=dep, bt_table=bt_table, var_series=var_series,
               realised=realised, hedge_summary=hedge_summary, findings_hedge=f_hedge,
               garch_vol=pd.Series(vol.conditional_vol(shocks["r"].values, g), index=shocks.index),
               dvol=dvol)
    report.charts(res)
    path = report.markdown(res)
    hedge_raw.to_csv(report.OUT / "hedge_windows.csv", index=False)
    print(var_table.to_string(index=False))
    print(bt_table.to_string(index=False))
    print(f"[done] report -> {path}")


if __name__ == "__main__":
    main()

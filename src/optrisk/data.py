"""Market data: Deribit public API (no key required) with a CSV cache and a
seeded synthetic fallback so the pipeline always runs end to end.

Series used
- BTC-PERPETUAL daily close           -> spot proxy for historical shocks
- DVOL (Deribit 30-day implied vol)   -> historical implied-vol shocks
- Live BTC option chain (mark IVs)    -> current smile and portfolio construction
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://www.deribit.com/api/v2/public/"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DAY_MS = 86_400_000


# ----------------------------------------------------------------------------- live
def _get(method: str, **params) -> dict:
    url = API + method + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        payload = json.load(r)
    if "result" not in payload:
        raise RuntimeError(f"Deribit error on {method}: {payload}")
    return payload["result"]


def _chunks(days: int, chunk: int = 300):
    end = int(time.time() * 1000)
    start = end - days * DAY_MS
    t = start
    while t < end:
        yield t, min(t + chunk * DAY_MS, end)
        t += chunk * DAY_MS


def fetch_spot(days: int) -> pd.Series:
    frames = []
    for s, e in _chunks(days):
        r = _get("get_tradingview_chart_data", instrument_name="BTC-PERPETUAL",
                 start_timestamp=s, end_timestamp=e, resolution="1D")
        if r.get("status") == "ok" and r["ticks"]:
            frames.append(pd.Series(r["close"], index=pd.to_datetime(r["ticks"], unit="ms")))
    s = pd.concat(frames)
    s = s[~s.index.duplicated()].sort_index()
    s.index = s.index.normalize()
    return s.rename("close")


def fetch_dvol(days: int) -> pd.Series:
    rows = []
    for s, e in _chunks(days):
        r = _get("get_volatility_index_data", currency="BTC",
                 start_timestamp=s, end_timestamp=e, resolution="1D")
        rows.extend(r.get("data", []))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    s = pd.Series(df["close"].values, index=pd.to_datetime(df["ts"], unit="ms")).sort_index()
    s.index = s.index.normalize()
    return s.rename("dvol")


def fetch_chain() -> pd.DataFrame:
    rows = _get("get_book_summary_by_currency", currency="BTC", kind="option")
    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        try:
            _, exp, strike, cp = r["instrument_name"].split("-")
            expiry = datetime.strptime(exp, "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
        except ValueError:
            continue
        if not r.get("mark_iv"):
            continue
        out.append({
            "instrument": r["instrument_name"],
            "expiry": expiry,
            "T": (expiry - now).total_seconds() / (365 * 86400),
            "K": float(strike),
            "cp": 1 if cp == "C" else -1,
            "iv": r["mark_iv"] / 100.0,
            "F": r["underlying_price"],
        })
    df = pd.DataFrame(out)
    return df[df["T"] > 1 / 365].reset_index(drop=True)


# ------------------------------------------------------------------------ synthetic
def synthetic(days: int = 1500, seed: int = 7):
    """GARCH(1,1)-t spot returns + mean-reverting log-DVOL with negative spot/vol
    correlation + a quadratic-smile option chain. For testing only: every output
    produced in this mode is labelled SYNTHETIC in the report."""
    rng = np.random.default_rng(seed)
    nu, a, b = 4.0, 0.10, 0.85
    var_lr = 0.60**2 / 365
    omega = var_lr * (1 - a - b)
    z = rng.standard_t(nu, days) * np.sqrt((nu - 2) / nu)
    r = np.empty(days); s2 = var_lr
    for t in range(days):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + a * r[t] ** 2 + b * s2
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize().tz_localize(None), periods=days, freq="D")
    path = np.exp(np.cumsum(r))
    spot = pd.Series(100_000 * path / path[-1], index=idx, name="close")  # rescaled to end at 100k

    lv = np.empty(days); lv[0] = np.log(60.0)
    for t in range(1, days):
        lv[t] = lv[t-1] + 0.05 * (np.log(60.0) - lv[t-1]) - 1.2 * r[t] + rng.normal(0, 0.035)
    dvol = pd.Series(np.exp(lv), index=idx, name="dvol")

    F = float(spot.iloc[-1]); atm = float(dvol.iloc[-1]) / 100
    rows = []
    now = pd.Timestamp.utcnow().floor("s")
    for d in (7, 14, 30, 60, 90, 180):
        T = d / 365
        for K in np.unique(np.round(F * np.linspace(0.5, 1.6, 45), -3)):
            k = np.log(K / F)
            iv = max(atm - 0.10 * k + 0.30 * k * k, 0.15)
            for cp in (1, -1):
                rows.append({"instrument": f"SYN-{d}D-{int(K)}-{'C' if cp == 1 else 'P'}",
                             "expiry": now + pd.Timedelta(days=d),
                             "T": T, "K": float(K), "cp": cp, "iv": iv, "F": F})
    return spot, dvol, pd.DataFrame(rows)


# ---------------------------------------------------------------------------- cache
def read_cache(folder: Path):
    """Read spot.csv, dvol.csv, chain.csv exactly as saved. 'round_trip' float parsing
    keeps every value (including each option's T) bit-for-bit, so a cached run
    reproduces the original run exactly."""
    folder = Path(folder)
    read = lambda name, **kw: pd.read_csv(folder / name, float_precision="round_trip", **kw)
    s = read("spot.csv", index_col=0, parse_dates=True).iloc[:, 0].rename("close")
    v = read("dvol.csv", index_col=0, parse_dates=True).iloc[:, 0].rename("dvol")
    c = read("chain.csv")
    c["expiry"] = pd.to_datetime(c["expiry"], utc=True)
    return s, v, c


def snapshot_time(chain: pd.DataFrame) -> pd.Timestamp:
    """Recover when the chain was pulled: T was saved as (expiry - pull time) in years."""
    pulled = chain["expiry"] - pd.to_timedelta(chain["T"] * 365 * 86400, unit="s")
    return pulled.median().round("s")


# --------------------------------------------------------------------------- loader
def load(mode: str = "live", days: int = 1500, seed: int = 7, data_dir: Path | None = None):
    """Returns (spot, dvol, chain, source_label).
    mode = "live" (Deribit API, falls back to data/), "synthetic" (seeded simulation),
    or "cache" (read a saved snapshot from data_dir; no network; `days` is ignored so the
    full saved history is used and results reproduce exactly)."""
    if mode == "synthetic":
        s, v, c = synthetic(days, seed)
        return s, v, c, "SYNTHETIC (seeded simulation - not market data)"
    if mode == "cache":
        s, v, c = read_cache(data_dir)
        return s, v, c, f"Deribit snapshot of {snapshot_time(c):%Y-%m-%d %H:%M} UTC (cached)"
    DATA_DIR.mkdir(exist_ok=True)
    try:
        s, v, c = fetch_spot(days), fetch_dvol(days), fetch_chain()
        s.to_csv(DATA_DIR / "spot.csv"); v.to_csv(DATA_DIR / "dvol.csv")
        c.to_csv(DATA_DIR / "chain.csv", index=False)
        return s, v, c, f"Deribit public API, pulled {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    except Exception as exc:  # network down: fall back to last cache
        if (DATA_DIR / "spot.csv").exists():
            s, v, c = read_cache(DATA_DIR)
            return s, v, c, f"Deribit cache in data/ (live fetch failed: {exc})"
        raise RuntimeError(f"Live fetch failed and no cache found: {exc}. "
                           "Run with --mode synthetic to test the pipeline offline.") from exc

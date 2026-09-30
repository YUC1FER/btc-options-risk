"""Sanity and validation tests: `python -m pytest -q`."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from optrisk import backtest, pricing as bs, vol  # noqa: E402

F, K, T, S = 100_000.0, 95_000.0, 30 / 365, 0.55


def test_put_call_parity():
    c, p = bs.price(F, K, T, S, 1), bs.price(F, K, T, S, -1)
    assert c - p == pytest.approx(F - K, rel=1e-10)


@pytest.mark.parametrize("cp", [1, -1])
def test_delta_matches_finite_difference(cp):
    h = 1.0
    fd = (bs.price(F + h, K, T, S, cp) - bs.price(F - h, K, T, S, cp)) / (2 * h)
    assert bs.delta(F, K, T, S, cp) == pytest.approx(fd, rel=1e-5)


def test_gamma_matches_finite_difference():
    h = 10.0
    fd = (bs.price(F + h, K, T, S, 1) - 2 * bs.price(F, K, T, S, 1) + bs.price(F - h, K, T, S, 1)) / h**2
    assert bs.gamma(F, K, T, S) == pytest.approx(fd, rel=1e-3)


def test_vega_and_theta_match_finite_difference():
    fd_v = (bs.price(F, K, T, S + 0.005, 1) - bs.price(F, K, T, S - 0.005, 1))
    assert bs.vega(F, K, T, S) == pytest.approx(fd_v, rel=1e-3)
    fd_t = bs.price(F, K, T - 1 / 365, S, 1) - bs.price(F, K, T, S, 1)
    assert bs.theta(F, K, T, S) == pytest.approx(fd_t, rel=2e-2)


def test_implied_vol_round_trip():
    px = float(bs.price(F, K, T, S, -1))
    assert bs.implied_vol(px, F, K, T, -1) == pytest.approx(S, abs=1e-8)


def test_garch_recovers_parameters():
    rng = np.random.default_rng(0)
    n, om, a, b = 4000, 0.02, 0.08, 0.90
    e = np.empty(n); s2 = om / (1 - a - b)
    for t in range(n):
        e[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = om + a * e[t] ** 2 + b * s2
    g = vol.fit(e / 100, "normal")
    assert g.alpha == pytest.approx(a, abs=0.03)
    assert g.beta == pytest.approx(b, abs=0.04)


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    rng = np.random.default_rng(1)
    good = rng.random(2000) < 0.01
    bad = rng.random(2000) < 0.04
    assert backtest.kupiec(good, 0.01)[1] > 0.05
    assert backtest.kupiec(bad, 0.01)[1] < 0.01


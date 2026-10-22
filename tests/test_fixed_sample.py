import numpy as np
import pytest

from universal_outlier.exponents import fit_exponent, local_slopes, wilson_interval
from universal_outlier.fixed_sample import exact_pe_binary, monte_carlo_pe
from universal_outlier.problem import Problem


def test_monte_carlo_matches_exact():
    pr = Problem.bernoulli(0.4, 0.6)
    rules = ["glt", "kl_avg", "mmd_avg", "lrt"]
    mc = monte_carlo_pe(pr, rules, [60], 100_000, np.random.default_rng(0))
    ex = exact_pe_binary(pr, rules, 60)
    for r in rules:
        p = ex[r]
        got = mc[mc.detector == r].pe_avg.item()
        assert got == pytest.approx(p, abs=4 * np.sqrt(p * (1 - p) / 300_000))


def test_hypotheses_symmetric():
    pr = Problem.bernoulli(0.3, 0.5)
    mc = monte_carlo_pe(pr, ["glt"], [40], 50_000, np.random.default_rng(1))
    pe = mc[[f"pe_k{k}" for k in range(3)]].to_numpy().ravel()
    assert pe.max() - pe.min() < 0.01


def test_fit_recovers_exponent():
    n = np.arange(100, 1001, 100)
    pe = 0.3 * np.exp(-0.01 * n)
    fit = fit_exponent(n, pe, trials=10**9)
    assert fit.c == pytest.approx(0.01, rel=1e-6)
    assert fit_exponent(n, pe, method="half_tail").c == pytest.approx(0.01, rel=1e-6)
    _, s = local_slopes(n, pe)
    np.testing.assert_allclose(s, 0.01)


def test_fit_ignores_floor_points():
    n = np.arange(10, 101, 10)
    pe = np.r_[np.exp(-0.05 * n[:6]), [1e-6, 0, 0, 0]]
    fit = fit_exponent(n, pe, trials=100_000, min_errors=10)
    assert fit.c == pytest.approx(0.05, rel=1e-6)


def test_wilson_zero_errors():
    lo, hi = wilson_interval(0, 1000)
    assert lo == 0 and 0 < hi < 0.01

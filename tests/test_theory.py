import pytest

from universal_outlier.problem import Problem
from universal_outlier.theory import sanov_exponent


@pytest.mark.parametrize("p", [(0.25, 0.75), (0.1, 0.3)])
def test_lrt_exponent_is_2B(p):
    pr = Problem.bernoulli(*p)
    e, _ = sanov_exponent("lrt", pr, grid=81)
    assert e == pytest.approx(pr.two_b, rel=2e-3)


def test_universal_rules_one_third_in_local_regime():
    pr = Problem.bernoulli(0.45, 0.55)
    for rule in ["glt", "mmd_avg"]:
        e, _ = sanov_exponent(rule, pr, grid=81)
        assert e / pr.two_b == pytest.approx(1 / 3, abs=0.005)


def test_universal_below_optimum():
    pr = Problem.bernoulli(0.1, 0.3)
    e, _ = sanov_exponent("glt", pr, grid=81)
    assert 0 < e < pr.two_b

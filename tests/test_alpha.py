import numpy as np
import pytest

from universal_outlier.alpha import expected_stop_theory, loglog_slope, null_noise_floor


def test_loglog_slope_exact():
    n = np.array([10, 100, 1000])
    s, _, r2 = loglog_slope(n, 3.0 / n)
    assert s == pytest.approx(-1.0) and r2 == pytest.approx(1.0)


def test_noise_floor_decays_like_one_over_n():
    df = null_noise_floor([50, 100, 200, 400], 300, 1.0, np.random.default_rng(0))
    s, _, _ = loglog_slope(df.n, df.mean_abs)
    assert s == pytest.approx(-1.0, abs=0.15)
    assert abs(df["mean"]).max() < 3 * df.mean_abs.max()  # unbiased: centred near 0


def test_expected_stop_theory():
    assert expected_stop_theory(2.0, 1.0, 0.25) == pytest.approx(8.0)
    assert expected_stop_theory(1.0, 0.5, 0.25) == pytest.approx(16.0)

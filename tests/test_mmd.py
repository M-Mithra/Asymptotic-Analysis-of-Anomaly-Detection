import numpy as np
import pytest

from universal_outlier import detectors
from universal_outlier.mmd import (kernel_matrix, mmd2_u, mmd2_u_paired, pairwise_mmd2_from_types,
                                   population_mmd2_gaussians)
from universal_outlier.problem import Problem


def test_counts_formula_matches_samples():
    """MMD from symbol counts equals the sample U-statistic on raw 0/1 data."""
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.3, 40).astype(float)
    y = rng.binomial(1, 0.6, 40).astype(float)
    types = np.array([[[1 - x.mean(), x.mean()], [1 - y.mean(), y.mean()]]])
    M = pairwise_mmd2_from_types(types, 40, kernel_matrix([0.0, 1.0], 1.0))
    assert M[0, 0, 1] == pytest.approx(mmd2_u(x, y, 1.0), abs=1e-12)


def test_unbiased_under_null():
    rng = np.random.default_rng(1)
    vals = [mmd2_u(rng.normal(size=30), rng.normal(size=30)) for _ in range(3000)]
    assert abs(np.mean(vals)) < 3 * np.std(vals) / np.sqrt(len(vals))


def test_paired_close_to_unpaired():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=200), rng.normal(1, 1, size=200)
    assert mmd2_u_paired(x, y) == pytest.approx(mmd2_u(x, y), abs=0.02)


def test_population_gaussian_mmd():
    # MC check of the closed form, and the value used in the NP-SEQ paper setup
    rng = np.random.default_rng(3)
    est = np.mean([mmd2_u(rng.normal(size=400), rng.normal(1.2, 1, size=400)) for _ in range(40)])
    exact = population_mmd2_gaussians(0, 1, 1.2, 1, 1.0)
    assert exact == pytest.approx(0.2238, abs=1e-4)
    assert est == pytest.approx(exact, abs=0.01)


def test_mmd_rules_equivalent_to_closest_pair_for_M3():
    pr = Problem.bernoulli(0.5, 0.5)
    rng = np.random.default_rng(4)
    t = rng.uniform(size=(5000, 3))
    T = np.stack([1 - t, t], axis=-1)
    a = np.argmax(detectors.get("mmd_avg")(T, None, pr), axis=1)
    b = np.argmax(detectors.get("closest_pair")(T, None, pr), axis=1)
    c = np.argmax(detectors.get("mmd_maxmin")(T, None, pr), axis=1)
    assert (a == b).all() and (b == c).all()

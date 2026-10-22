import numpy as np
import pytest

from universal_outlier.decision import decide, error_given_scores, tie_mask
from universal_outlier.divergences import (bernoulli_kl, bhattacharyya, chernoff_information,
                                           kl_divergence, kl_exact, total_variation)
from universal_outlier.problem import Problem, bernoulli_scenarios, sample_counts


def test_kl_basic():
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert kl_divergence(p, q) == pytest.approx(kl_exact(p, q), rel=1e-6)
    assert bernoulli_kl(0.5, 0.75) == pytest.approx(kl_exact(p, q))


def test_kl_broadcasts_pairwise():
    P = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])
    D = kl_divergence(P[:, None, :], P[None, :, :])
    assert D.shape == (3, 3)
    assert D[0, 2] == pytest.approx(kl_divergence(P[0], P[2]))


def test_bhattacharyya_and_tvd():
    p, q = np.array([0.4, 0.6]), np.array([0.6, 0.4])
    assert bhattacharyya(p, q) == pytest.approx(-np.log(2 * np.sqrt(0.24)))
    assert total_variation(p, q) == pytest.approx(0.2)
    # Chernoff information of a symmetric pair equals B
    assert chernoff_information(p, q) == pytest.approx(bhattacharyya(p, q), rel=1e-4)


def test_problem_bernoulli():
    pr = Problem.bernoulli(0.3, 0.7)
    assert pr.k == 2 and pr.M == 3
    assert pr.tvd == pytest.approx(0.4)
    with pytest.raises(ValueError):
        Problem(pi=[0.5, 0.6], mu=[0.5, 0.5])


def test_sample_counts_shape_and_means():
    pr = Problem.bernoulli(0.2, 0.8)
    rng = np.random.default_rng(0)
    c = sample_counts(pr, 50, 20000, outlier=1, rng=rng)
    assert c.shape == (20000, 3, 2)
    assert (c.sum(axis=-1) == 50).all()
    frac = c[..., 1].mean(axis=0) / 50
    np.testing.assert_allclose(frac, [0.2, 0.8, 0.2], atol=0.01)


def test_scenarios_symmetric():
    for name, sc in bernoulli_scenarios("standard").items():
        a, b = sc["p"]
        assert a + b == pytest.approx(1.0)
        assert np.all(np.diff(sc["n"]) > 0)


def test_tie_breaking_is_fair():
    scores = np.array([[1.0, 1.0, 0.0]] * 20000)
    rng = np.random.default_rng(1)
    d = decide(scores, rng)
    assert set(np.unique(d)) == {0, 1}
    assert abs((d == 0).mean() - 0.5) < 0.02
    np.testing.assert_allclose(error_given_scores(scores[:1], 0), [0.5])
    np.testing.assert_allclose(error_given_scores(scores[:1], 2), [1.0])
    assert tie_mask(np.array([[3.0, 1.0, 3.0 + 1e-14]])).tolist() == [[True, False, True]]

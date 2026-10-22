import numpy as np
import pytest

from universal_outlier import detectors
from universal_outlier.divergences import kl_divergence
from universal_outlier.problem import Problem

PR = Problem.bernoulli(0.4, 0.6)


def test_glt_matches_definition():
    t = np.array([0.7, 0.4, 0.45])
    T = np.stack([1 - t, t], axis=-1)
    s = detectors.get("glt")(T[None], 100, PR)[0]
    for i in range(3):
        rest = [j for j in range(3) if j != i]
        loo = T[rest].mean(axis=0)
        assert -s[i] == pytest.approx(sum(kl_divergence(T[j], loo) for j in rest))
    assert np.argmax(s) == 0


def test_glt_categorical():
    pr = Problem(pi=[0.7, 0.2, 0.1], mu=[0.2, 0.5, 0.3], M=5)
    T = np.tile(np.array(pr.pi), (1, 5, 1))
    T[0, 3] = pr.mu
    assert np.argmax(detectors.get("glt")(T, None, pr)[0]) == 3


def test_glt_registered():
    assert "glt" in detectors.available()

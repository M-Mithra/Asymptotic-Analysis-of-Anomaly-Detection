import numpy as np
import pytest

from universal_outlier import detectors
from universal_outlier.divergences import kl_divergence
from universal_outlier.problem import Problem

PR = Problem.bernoulli(0.5, 0.5)


def types_from(t):
    t = np.asarray(t, dtype=float)
    return np.stack([1 - t, t], axis=-1)


def test_kl_avg_matches_loop():
    T = types_from([[0.2, 0.5, 0.55]])
    s = detectors.get("kl_avg")(T, 100, PR)[0]
    for i in range(3):
        ref = np.mean([kl_divergence(T[0, i], T[0, j]) for j in range(3) if j != i])
        assert s[i] == pytest.approx(ref)
    assert np.argmax(s) == 0


def test_kl_centroid_picks_far_sequence():
    T = types_from([[0.5, 0.52, 0.9]])
    assert np.argmax(detectors.get("kl_centroid")(T, None, PR)[0]) == 2


def test_registered():
    keys = detectors.available()
    assert {"kl_avg", "kl_centroid"} <= set(keys)

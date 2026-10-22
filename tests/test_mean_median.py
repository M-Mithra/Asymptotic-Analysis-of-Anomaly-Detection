import numpy as np

from universal_outlier import detectors
from universal_outlier.problem import Problem


def test_mean_median_pick_outlier():
    pr = Problem(pi=[0.6, 0.3, 0.1], mu=[0.1, 0.3, 0.6], M=7)
    T = np.tile(np.array(pr.pi), (1, 7, 1))
    T[0, 5] = pr.mu
    for key in ("mean_based", "median_based"):
        assert np.argmax(detectors.get(key)(T, None, pr)[0]) == 5


def test_median_all_zero_fallback():
    pr = Problem(pi=[1 / 3] * 3, mu=[1 / 3] * 3)
    T = np.eye(3)[None]
    s = detectors.get("median_based")(T, None, pr)
    assert np.isfinite(s).all()

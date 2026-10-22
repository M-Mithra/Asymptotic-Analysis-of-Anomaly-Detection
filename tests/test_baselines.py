import numpy as np

from universal_outlier import detectors
from universal_outlier.problem import Problem

PR = Problem.bernoulli(0.4, 0.6)


def test_oracle_and_tvd_registered():
    keys = detectors.available()
    assert {"glt", "lrt", "tvd_avg", "tvd_centroid"} <= set(keys)
    assert not keys["lrt"].universal


def test_lrt_picks_highest_llr():
    t = np.array([0.35, 0.65, 0.4])
    T = np.stack([1 - t, t], axis=-1)
    assert np.argmax(detectors.get("lrt")(T[None], 50, PR)[0]) == 1


def test_tvd_centroid_is_distance_to_mean():
    t = np.array([0.2, 0.6, 0.7])
    T = np.stack([1 - t, t], axis=-1)
    s = detectors.get("tvd_centroid")(T[None], None, PR)[0]
    np.testing.assert_allclose(s, [0.45, 0.15, 0.3])

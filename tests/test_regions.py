import numpy as np

from universal_outlier.regions import AMBIGUOUS, NO_OUTLIER, classify, classify_cube, disagreement


def test_obvious_points():
    t = np.array([[0.9, 0.1, 0.12], [0.3, 0.3, 0.3], [0.2, 0.5, 0.8], [0.5, 0.5, 0.1]])
    lab = classify("glt", t)
    assert lab[0] == 1
    assert lab[1] == NO_OUTLIER
    assert lab[2] == AMBIGUOUS  # equally spaced: outlier 1 and 3 tie
    assert lab[3] == 3


def test_cube_symmetry():
    df = classify_cube("kl_centroid", step=0.1)
    frac = df.label.value_counts(normalize=True)
    assert abs(frac[1] - frac[2]) < 1e-12 and abs(frac[2] - frac[3]) < 1e-12


def test_mmd_and_tvd_centroid_same_partition():
    a = classify_cube("mmd_avg", step=0.05)
    b = classify_cube("tvd_centroid", step=0.05)
    assert disagreement(a, b) == 0.0

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


def test_ties_have_zero_volume():
    """Ambiguity is a measure-zero set for every rule: random points never tie,
    and the tie fraction on a grid shrinks with the step (a grid artefact)."""
    t = np.random.default_rng(0).uniform(size=(200_000, 3))
    for rule in ("mmd_avg", "glt", "kl_avg"):
        assert (classify(rule, t) == AMBIGUOUS).sum() == 0
    coarse = (classify_cube("mmd_avg", 0.05).label == AMBIGUOUS).mean()
    fine = (classify_cube("mmd_avg", 0.025).label == AMBIGUOUS).mean()
    assert fine < 0.7 * coarse

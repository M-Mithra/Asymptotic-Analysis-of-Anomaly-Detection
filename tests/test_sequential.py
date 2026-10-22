import numpy as np
import pytest

from universal_outlier.mmd import mmd2_u, mmd2_u_paired
from universal_outlier.sequential import (History, IncrementalMMD, SeqConfig, run_np_seq,
                                          run_np_seq_ua, subset_statistic)


def grow(X, gamma=1.0):
    b, S, n = X.shape
    eng, H = IncrementalMMD(b, S, gamma), History(b, S, capacity=2)
    for t in range(n):
        eng.update(X[:, :, t], H.data)
        H.append(X[:, :, t])
    return eng, H


def test_incremental_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 4, 25))
    eng, H = grow(X, 0.7)
    np.testing.assert_array_equal(H.data, X)
    P, Pp, Q = eng.pairwise(), eng.pairwise(paired=True), eng.pooled()
    for b in range(3):
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert P[b, i, j] == pytest.approx(mmd2_u(X[b, i], X[b, j], 0.7), abs=1e-12)
                    assert Pp[b, i, j] == pytest.approx(mmd2_u_paired(X[b, i], X[b, j], 0.7), abs=1e-12)
            rest = np.concatenate([X[b, j] for j in range(4) if j != i])
            assert Q[b, i] == pytest.approx(mmd2_u(X[b, i], rest, 0.7), abs=1e-12)


def test_keep_drops_trials():
    rng = np.random.default_rng(1)
    eng, H = grow(rng.normal(size=(5, 3, 6)))
    before = eng.pairwise()[[1, 3]]
    eng.keep(np.array([1, 3]))
    np.testing.assert_allclose(eng.pairwise(), before)


def test_np_seq_monotone_in_C():
    df = run_np_seq(SeqConfig(S=4), 1.0, [1.0, 4.0], 600, np.random.default_rng(2))
    assert df.E_N.iloc[1] > df.E_N.iloc[0]
    assert df.P_err.iloc[1] <= df.P_err.iloc[0]
    assert (df.timeouts == 0).all()


def test_expected_delay_scales_like_C_over_mmd2():
    cfg = SeqConfig(S=5, statistic="pooled")
    df = run_np_seq(cfg, 1.0, [12.0], 400, np.random.default_rng(3))
    assert df.E_N.item() / (12.0 / 0.2238) == pytest.approx(1.0, abs=0.15)


def test_subset_statistic_finds_planted_pair():
    S = 5
    M = np.full((1, S, S), 0.01)
    M[0, [0, 0, 3, 3], [1, 2, 1, 2]] = 0.5
    M[0, [1, 2, 1, 2], [0, 0, 3, 3]] = 0.5
    M[0, [0, 3], [4, 4]] = M[0, [4, 4], [0, 3]] = 0.5
    best, val, subsets = subset_statistic(M, A=2)
    assert subsets[int(best[0])] == (0, 3)


def test_np_seq_ua_runs():
    r = run_np_seq_ua(SeqConfig(S=5), A=2, n_anomalies=0, C=0.8, trials=60, rng=np.random.default_rng(4))
    assert 0 <= r["P_err"] <= 1 and r["E_N"] <= r["T0"]

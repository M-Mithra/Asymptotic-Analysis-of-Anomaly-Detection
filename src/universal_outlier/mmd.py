"""Maximum Mean Discrepancy (MMD) estimators with the Gaussian (RBF) kernel.

k(x, y) = exp(-gamma (x - y)^2). The papers write the kernel with a
bandwidth, exp(-(x - y)^2 / (2 sigma^2)), i.e. gamma = 1 / (2 sigma^2); the
NP-SEQ paper uses sigma^2 = 0.5, which is gamma = 1.

Three routes to the same unbiased statistic M_u^2:

* :func:`mmd2_u` from raw samples (any real-valued data),
* :func:`mmd2_u_from_counts` from symbol counts over a finite alphabet,
  which is exact for discrete data and O(k^2) instead of O(n^2),
* the incremental block sums in :mod:`universal_outlier.sequential`.
"""

from __future__ import annotations

import numpy as np

from .backend import array_module


def gamma_from_sigma2(sigma2: float) -> float:
    return 1.0 / (2.0 * sigma2)


def rbf(x, y, gamma: float):
    xp = array_module(x)
    return xp.exp(-gamma * (x - y) ** 2)


def mmd2_u(x, y, gamma: float = 1.0) -> float:
    """Unbiased MMD^2 between 1-D samples x (size n) and y (size m).

    Within-sample sums exclude the diagonal; the cross term averages over all
    n * m pairs (Gretton et al. 2012, eq. 3). This is the estimator used in
    all of the project's scripts (``mmd_u_statistic_1d``/``compute_mmd2_fast``).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0
    kxx = rbf(x[:, None], x[None, :], gamma)
    kyy = rbf(y[:, None], y[None, :], gamma)
    kxy = rbf(x[:, None], y[None, :], gamma)
    return float(
        (kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - m) / (m * (m - 1)) - 2 * kxy.sum() / (n * m)
    )


def mmd2_u_paired(x, y, gamma: float = 1.0) -> float:
    """Paired unbiased MMD^2 of the NP-SEQ paper (equal sizes).

    M_u^2 = 1/(n(n-1)) sum_{l != m} h(z_l, z_m) with
    h = k(x_l, x_m) + k(y_l, y_m) - k(x_l, y_m) - k(x_m, y_l).
    It differs from :func:`mmd2_u` only in excluding the l = m cross pairs.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n != len(y):
        raise ValueError("paired estimator needs equal sample sizes")
    if n < 2:
        return 0.0
    kxx = rbf(x[:, None], x[None, :], gamma)
    kyy = rbf(y[:, None], y[None, :], gamma)
    kxy = rbf(x[:, None], y[None, :], gamma)
    h = kxx + kyy - kxy - kxy.T
    return float((h.sum() - np.trace(h)) / (n * (n - 1)))


def kernel_matrix(alphabet, gamma: float, xp=np):
    """K[a, b] = k(alphabet[a], alphabet[b])."""
    v = xp.asarray(alphabet, dtype=float)
    return rbf(v[:, None], v[None, :], gamma)


def pairwise_mmd2_from_types(types, n, K):
    """All pairwise M_u^2 between M sequences given their types.

    ``types`` has shape (..., M, k). With counts c = n * type,
    sum_{l != m} k(x_l, x_m) = c^T K c - n, and the cross sum is c_i^T K c_j.
    Pass ``n=None`` for the n -> infinity limit (P_i - P_j)^T K (P_i - P_j),
    which is the population MMD^2 between the two empirical pmfs.
    """
    xp = array_module(types)
    G = xp.einsum("...ia,ab,...jb->...ij", types, K, types)  # P_i^T K P_j
    self_term = G.diagonal(axis1=-2, axis2=-1)
    if n is None:
        within = self_term
    else:
        within = (n * n * self_term - n) / (n * (n - 1))
    return within[..., :, None] + within[..., None, :] - 2.0 * G


def population_mmd2_gaussians(m1, s1, m2, s2, gamma: float = 1.0) -> float:
    """Closed-form MMD^2 between N(m1, s1^2) and N(m2, s2^2) for the RBF kernel.

    E k(X, Y) = (1 + 2 gamma (sx^2 + sy^2))^{-1/2}
                * exp(-gamma (mx - my)^2 / (1 + 2 gamma (sx^2 + sy^2))).
    """

    def ek(ma, sa, mb, sb):
        d = 1.0 + 2.0 * gamma * (sa**2 + sb**2)
        return np.exp(-gamma * (ma - mb) ** 2 / d) / np.sqrt(d)

    return float(ek(m1, s1, m1, s1) + ek(m2, s2, m2, s2) - 2 * ek(m1, s1, m2, s2))


def population_mmd2_mc(sample_p, sample_q, gamma: float = 1.0, size: int = 4000, seed: int = 0) -> float:
    """Monte-Carlo estimate of MMD^2(p, q) for arbitrary samplers ``f(rng, size)``."""
    rng = np.random.default_rng(seed)
    return mmd2_u(sample_p(rng, size), sample_q(rng, size), gamma)

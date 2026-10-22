"""Divergences between discrete probability mass functions.

All functions take pmfs along the *last* axis and broadcast over leading
axes, so ``kl_divergence(P[:, :, None, :], P[:, None, :, :])`` gives every
pairwise divergence of a batch of types in one call. Natural logarithms are
used throughout, so exponents are in nats (matching ``ln P_e``).
"""

from __future__ import annotations

import numpy as np

from .backend import array_module

#: Additive smoothing used by the empirical KL statistics. Empirical types
#: frequently contain zeros, and the original project scripts used the same
#: ``p + eps, q + eps`` smoothing, which keeps the statistics finite.
KL_EPS = 1e-10


def kl_divergence(p, q, eps: float = KL_EPS):
    """Smoothed KL divergence D(p || q) = sum p log(p / q)."""
    xp = array_module(p)
    p_s = p + eps
    q_s = q + eps
    return xp.sum(p_s * xp.log(p_s / q_s), axis=-1)


def kl_exact(p, q):
    """Unsmoothed D(p || q) with the convention 0 log 0 = 0 (inf if q=0<p)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / q), 0.0)
    return terms.sum(axis=-1)


def bernoulli_kl(t, p):
    """D(Bern(t) || Bern(p)) for arrays of success probabilities ``t``."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(t > 0, t * np.log(t / p), 0.0)
        b = np.where(t < 1, (1 - t) * np.log((1 - t) / (1 - p)), 0.0)
    return a + b


def total_variation(p, q):
    """Total variation distance 0.5 * sum |p - q|."""
    xp = array_module(p)
    return 0.5 * xp.sum(xp.abs(p - q), axis=-1)


def bhattacharyya(p, q) -> float:
    """Bhattacharyya distance B(p, q) = -ln sum sqrt(p q).

    The optimal error exponent for one outlier with known p and q is 2B
    (Li, Nitinawarat & Veeravalli 2014, Prop. 7).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(-np.log(np.sum(np.sqrt(p * q))))


def chernoff_information(p, q, grid: int = 2001) -> float:
    """Chernoff information max_l -ln sum p^l q^(1-l) (binary test exponent)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    lam = np.linspace(0.0, 1.0, grid)[:, None]
    vals = -np.log(np.sum(p ** lam * q ** (1 - lam), axis=1))
    return float(vals.max())

"""MMD based outlier rules (Gaussian kernel over the alphabet values)."""

from __future__ import annotations

from ..backend import array_module
from ..mmd import kernel_matrix, pairwise_mmd2_from_types
from . import offdiag_mean, offdiag_min, register


def _pairwise(types, n, problem):
    xp = array_module(types)
    K = kernel_matrix(problem.alphabet, problem.gamma, xp)
    return pairwise_mmd2_from_types(types, n, K)


@register("mmd_avg", "Max Avg MMD")
def max_avg_mmd(types, n, problem):
    """Outlier = argmax_i mean_{j != i} M_u^2(i, j) (``decision_max_avg_mmd``)."""
    return offdiag_mean(_pairwise(types, n, problem))


@register("mmd_maxmin", "Max-min MMD")
def maxmin_mmd(types, n, problem):
    """Outlier = argmax_i min_{j != i} M_u^2(i, j): the NP-SEQ statistic at fixed n."""
    return offdiag_min(_pairwise(types, n, problem))


@register("closest_pair", "Closest pair")
def closest_pair(types, n, problem):
    """Outlier = sequence left out of the pair with the closest sample means.

    Written as argmax_i min_{j != i} |mean_i - mean_j|, which for M = 3 is
    exactly the original ``decision_closest_pair``. The "MMD" cube scripts
    (rbf(t_j, t_k) on sample means) implement this same rule.
    """
    xp = array_module(types)
    means = types @ xp.asarray(problem.alphabet, dtype=float)
    dist = xp.abs(means[..., :, None] - means[..., None, :])
    return offdiag_min(dist)

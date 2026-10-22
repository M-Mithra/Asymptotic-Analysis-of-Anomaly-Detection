"""Reference rules: the oracle likelihood-ratio test and TVD rules."""

from __future__ import annotations

from ..backend import array_module
from ..divergences import total_variation
from . import offdiag_mean, others_mean, register


@register("lrt", "LRT (knows pi, mu)", universal=False)
def oracle_lrt(types, n, problem):
    """Maximum-likelihood test with known pi and mu; its exponent is 2 B(pi, mu).

    Outlier = argmax_i sum_y P_i(y) log(mu(y) / pi(y)). Not universal: it is
    the benchmark that the universal tests are compared against.
    """
    xp = array_module(types)
    eps = 1e-300
    llr = xp.log((xp.asarray(problem.mu) + eps) / (xp.asarray(problem.pi) + eps))
    return types @ llr


@register("tvd_avg", "Max Avg TVD")
def max_avg_tvd(types, n, problem):
    """Outlier = argmax_i mean_{j != i} TV(P_i, P_j) (MINMAX scripts)."""
    pairwise = total_variation(types[..., :, None, :], types[..., None, :, :])
    return offdiag_mean(pairwise)


@register("tvd_centroid", "TVD to centroid")
def tvd_to_centroid(types, n, problem):
    """Outlier = argmax_i TV(P_i, mean_{j != i} P_j) (rule of the TVD cube plots)."""
    return total_variation(types, others_mean(types))

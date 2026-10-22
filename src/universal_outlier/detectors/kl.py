"""KL-divergence based outlier rules."""

from __future__ import annotations

from ..divergences import kl_divergence
from . import offdiag_mean, others_mean, register


@register("kl_avg", "Max Avg KL")
def max_avg_kl(types, n, problem):
    """Outlier = argmax_i mean_{j != i} D(P_i || P_j) (``decision_max_avg_kl``)."""
    pairwise = kl_divergence(types[..., :, None, :], types[..., None, :, :])
    return offdiag_mean(pairwise)


@register("kl_centroid", "KL to centroid")
def kl_to_centroid(types, n, problem):
    """Outlier = argmax_i D(P_i || mean_{j != i} P_j) (rule of the KL cube plots)."""
    return kl_divergence(types, others_mean(types))

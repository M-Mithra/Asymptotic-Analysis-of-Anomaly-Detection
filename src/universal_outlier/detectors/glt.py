"""Universal generalised-likelihood test (GLT) of Li, Nitinawarat & Veeravalli."""

from __future__ import annotations

from ..divergences import kl_divergence
from . import others_mean, register


@register("glt", "Universal GLT")
def universal_glt(types, n, problem):
    """Outlier = argmin_i sum_{j != i} D(P_j || mean_{l != i} P_l).

    For a single outlier this is eq. (37) of Li et al. (2014): the hypothesis
    whose leave-one-out group is most homogeneous. Returned negated so that
    the largest score is the declared outlier.
    """
    M = types.shape[-2]
    loo = others_mean(types)  # (..., M, k): row i = mean of the others
    # D(P_j || loo_i) for all i, j
    d = kl_divergence(types[..., None, :, :], loo[..., :, None, :])  # (..., i, j)
    diag = d.diagonal(axis1=-2, axis2=-1)
    dispersion = d.sum(axis=-1) - diag
    return -dispersion

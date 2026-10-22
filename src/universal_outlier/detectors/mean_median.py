"""Mean- and median-based tests of Geiger, Koch, Mihaljevic & Toller (2026).

Both estimate the typical pmf from all M types and declare the sequence
farthest from that estimate in KL divergence (arXiv:2601.00712, eqs. 7, 8).
They cost O(M log M) instead of the GLT's search over outlier subsets.
"""

from __future__ import annotations

from ..backend import array_module
from ..divergences import kl_divergence
from . import register


@register("mean_based", "Mean-based")
def mean_based(types, n, problem):
    """Outlier = argmax_i D(P_i || pi_hat), pi_hat = mean of all M types."""
    pi_hat = types.mean(axis=-2, keepdims=True)
    return kl_divergence(types, pi_hat)


@register("median_based", "Median-based")
def median_based(types, n, problem):
    """Outlier = argmax_i D(P_i || pi_bar), pi_bar = normalised symbol-wise median."""
    xp = array_module(types)
    med = xp.median(types, axis=-2, keepdims=True)
    total = med.sum(axis=-1, keepdims=True)
    # All symbol medians can be 0 (e.g. types e1, e2, e3); fall back to the mean.
    pi_bar = xp.where(total > 0, med / xp.where(total > 0, total, 1.0), types.mean(axis=-2, keepdims=True))
    return kl_divergence(types, pi_bar)

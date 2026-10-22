"""Turning detector scores into decisions, with fair tie-breaking.

With discrete data, exact ties between candidate outliers are common (e.g.
all M counts equal). ``np.argmax`` silently resolves a tie in favour of the
lowest index; when the true outlier is index 0 that counts every tie as a
*correct* decision and biases P_e downwards. Here ties are broken uniformly
at random, and :func:`error_given_scores` gives the exact conditional error
probability of that randomised rule.
"""

from __future__ import annotations

from .backend import array_module

RTOL = 1e-9
ATOL = 1e-12


def tie_mask(scores, rtol: float = RTOL, atol: float = ATOL):
    """Boolean mask (..., M) of entries tied with the row maximum."""
    xp = array_module(scores)
    best = scores.max(axis=-1, keepdims=True)
    return scores >= best - (atol + rtol * xp.abs(best))


def decide(scores, rng):
    """Index of the declared outlier per row, ties broken uniformly at random."""
    xp = array_module(scores)
    mask = tie_mask(scores)
    priority = rng.random(scores.shape) * mask
    return xp.argmax(priority, axis=-1)


def error_given_scores(scores, outlier: int):
    """P(decision != outlier | scores) for the uniformly randomised rule."""
    mask = tie_mask(scores)
    n_tied = mask.sum(axis=-1)
    return 1.0 - mask[..., outlier] / n_tied

"""Large-deviation (Sanov) error exponents of the fixed-sample rules.

For M sequences with outlier at index 0, the empirical types (T_0, ..., T_{M-1})
satisfy a joint large-deviation principle with rate

    I(t) = D(t_0 || mu) + sum_{j >= 1} D(t_j || pi).

Every rule here decides from the types alone (in the n -> infinity limit), so
by Sanov's theorem its error probability decays with exponent

    E(rule) = inf { I(t) : t in closure of the rule's error region }.

For binary data each type is one number t_j = P_j(1), so the error region is
exactly the coloured region of the project's (t1, t2, t3) "cube" plots: the
exponent is the KL-"distance" from the true point (mu, pi, pi) to the nearest
cube region where the rule names the wrong sequence. For the oracle LRT this
gives 2 B(pi, mu), which the tests check.

The infimum is found on a grid, refined around the minimiser.
"""

from __future__ import annotations

import numpy as np

from . import detectors as det_registry
from .decision import tie_mask
from .divergences import bernoulli_kl, bhattacharyya
from .problem import Problem


def _rate(t, problem: Problem):
    p_mu, p_pi = problem.mu[1], problem.pi[1]
    r = bernoulli_kl(t[..., 0], p_mu)
    for j in range(1, t.shape[-1]):
        r = r + bernoulli_kl(t[..., j], p_pi)
    return r


def _error_region(det, t, problem):
    """Points whose (closure of the) decision is not 'outlier = 0'."""
    types = np.stack([1.0 - t, t], axis=-1)
    scores = det(types, None, problem)
    mask = tie_mask(scores, rtol=1e-7, atol=1e-10)
    # in the closure of the error region iff some j != 0 attains the maximum
    return mask[..., 1:].any(axis=-1)


def _search(det, problem, axes, chunk=2_000_000):
    """Minimum rate over the error region on the product grid ``axes``."""
    M = problem.M
    best, arg = np.inf, None
    a0 = axes[0]
    rest = np.stack([m.ravel() for m in np.meshgrid(*axes[1:], indexing="ij")], axis=1)
    step = max(1, chunk // len(rest))
    for s in range(0, len(a0), step):
        t0 = a0[s:s + step]
        t = np.concatenate([np.repeat(t0, len(rest))[:, None], np.tile(rest, (len(t0), 1))], axis=1)
        err = _error_region(det, t, problem)
        if not err.any():
            continue
        r = _rate(t[err], problem)
        i = int(np.argmin(r))
        if r[i] < best:
            best, arg = float(r[i]), t[err][i]
    return best, arg


def sanov_exponent(detector, problem: Problem, grid: int = 161, refine: int = 4,
                   refine_points: int = 41) -> tuple[float, np.ndarray]:
    """Large-deviation exponent of ``detector`` and the minimising types.

    Only binary alphabets and small M (cost grid^M) are supported.
    """
    if problem.k != 2:
        raise ValueError("sanov_exponent supports binary alphabets")
    det = det_registry.get(detector) if isinstance(detector, str) else detector
    M = problem.M
    eps = 1e-9
    axes = [np.linspace(eps, 1 - eps, grid)] * M
    best, arg = _search(det, problem, axes)
    width = 1.0 / (grid - 1)
    for _ in range(refine):
        if arg is None:
            break
        half = refine_points // 2
        width *= 3.0 / half
        axes = [np.clip(np.linspace(a - half * width, a + half * width, refine_points), eps, 1 - eps)
                for a in arg]
        b, a = _search(det, problem, axes)
        if b <= best:
            best, arg = b, a
    return best, arg


def exponent_table(problem: Problem, detectors, grid: int = 161) -> list[dict]:
    """Sanov exponents of several rules, with 2B and the efficiency ratio."""
    two_b = 2 * bhattacharyya(problem.pi, problem.mu)
    rows = []
    for d in detectors:
        e, t = sanov_exponent(d, problem, grid=grid)
        rows.append({"detector": d if isinstance(d, str) else d.key, "exponent": e,
                     "two_b": two_b, "ratio": e / two_b, "argmin_types": np.round(t, 4).tolist()})
    return rows

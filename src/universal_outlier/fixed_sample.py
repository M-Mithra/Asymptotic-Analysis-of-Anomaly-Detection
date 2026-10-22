"""Error probability P_e(n) of fixed-sample-size outlier tests.

Two estimators:

* :func:`monte_carlo_pe` - vectorised Monte Carlo over symbol counts. Each
  trial costs O(M^2 k) regardless of n, so 10^6 trials at n = 10 000 take
  seconds (the original scripts built n x n kernel matrices per trial).
  Ties are handled by conditioning on the scores (the exact error of the
  randomised tie-break), which also lowers the variance.
* :func:`exact_pe_binary` - exact P_e(n) for binary data by summing the
  multinomial probabilities of every count vector (tails below ``tail``
  dropped). No Monte-Carlo floor, so tiny error probabilities are resolved.

P_e(k) denotes the error probability when sequence k is the outlier;
P_e,max = max_k P_e(k) is the quantity in the definition of the exponent.
Every rule here is permutation invariant, so all P_e(k) are equal and
``pe_avg`` (their mean) is the lower-variance estimate of P_e,max.
"""

from __future__ import annotations

import time
from collections import Counter
from itertools import combinations_with_replacement
from math import factorial

import numpy as np
import pandas as pd

from . import detectors as det_registry
from .backend import free_memory
from .decision import error_given_scores
from .problem import Problem, sample_counts


def _resolve(detectors):
    return [det_registry.get(d) if isinstance(d, str) else d for d in detectors]


def monte_carlo_pe(
    problem: Problem,
    detectors,
    n_values,
    trials: int,
    rng,
    xp=np,
    chunk: int = 200_000,
    verbose: bool = False,
) -> pd.DataFrame:
    """Monte-Carlo P_e(n) for each detector and each outlier position.

    Returns a long DataFrame with one row per (n, detector): columns
    ``pe_k0 .. pe_k{M-1}``, ``pe_avg``, ``pe_max``, ``errors`` (expected error
    count summed over hypotheses), ``trials`` (per hypothesis) and ``seconds``.
    """
    dets = _resolve(detectors)
    rows = []
    for n in np.asarray(n_values, dtype=int):
        t0 = time.perf_counter()
        err = {d.key: np.zeros(problem.M) for d in dets}
        for k_true in range(problem.M):
            done = 0
            while done < trials:
                b = min(chunk, trials - done)
                counts = sample_counts(problem, int(n), b, k_true, rng, xp)
                types = counts / float(n)
                for d in dets:
                    e = error_given_scores(d(types, int(n), problem), k_true)
                    err[d.key][k_true] += float(e.sum())
                done += b
        elapsed = time.perf_counter() - t0
        for d in dets:
            pe_k = err[d.key] / trials
            row = {"n": int(n), "detector": d.key}
            row.update({f"pe_k{k}": pe_k[k] for k in range(problem.M)})
            row.update(pe_avg=pe_k.mean(), pe_max=pe_k.max(), errors=err[d.key].sum(),
                       trials=trials, seconds=elapsed / len(dets))
            rows.append(row)
        if verbose:
            summary = "  ".join(f"{d.key}={err[d.key].mean() / trials:.2e}" for d in dets)
            print(f"    n={int(n):<6d} {summary}  ({elapsed:.1f}s)", flush=True)
    free_memory()
    return pd.DataFrame(rows)


def exact_pe_binary(problem: Problem, detectors, n: int, tail: float = 1e-14,
                    max_rows: int = 1_500_000) -> dict[str, float]:
    """Exact P_e(n) (outlier at index 0) for a binary alphabet.

    Sums P(counts) * P(error | counts) over all count vectors whose binomial
    marginals lie inside the central ``1 - 2 tail`` mass. The result is exact
    up to an absolute error of about ``2 M tail``. Cost grows like n^{M/2}.
    """
    from scipy.stats import binom

    if problem.k != 2:
        raise ValueError("exact_pe_binary needs a binary alphabet")
    dets = _resolve(detectors)
    M = problem.M

    def support(p1):
        lo = int(binom.ppf(tail, n, p1))
        hi = int(binom.isf(tail, n, p1))
        c = np.arange(max(lo, 0), min(hi, n) + 1)
        return c, binom.pmf(c, n, p1)

    c_mu, w_mu = support(problem.mu[1])
    c_pi, w_pi = support(problem.pi[1])
    # The typical sequences are exchangeable and every rule is permutation
    # equivariant, so enumerate unordered count tuples with their multiplicity.
    idx = np.array(list(combinations_with_replacement(range(len(c_pi)), M - 1)))
    typ_counts = c_pi[idx]  # (G, M-1)
    typ_w = np.prod(w_pi[idx], axis=1) * _multiplicity(idx)
    G = len(typ_w)

    total = {d.key: 0.0 for d in dets}
    step = max(1, max_rows // G)
    for s in range(0, len(c_mu), step):
        c0, w0 = c_mu[s:s + step], w_mu[s:s + step]
        ones = np.concatenate(
            [np.repeat(c0, G)[:, None], np.tile(typ_counts, (len(c0), 1))], axis=1
        ).astype(float)
        types = np.stack([n - ones, ones], axis=-1) / n
        w = np.outer(w0, typ_w).ravel()
        for d in dets:
            total[d.key] += float(np.dot(w, error_given_scores(d(types, n, problem), 0)))
    return total


def _multiplicity(idx: np.ndarray) -> np.ndarray:
    """Number of distinct orderings of each sorted index tuple (multinomial)."""
    out = np.full(len(idx), factorial(idx.shape[1]), dtype=float)
    for row, tup in enumerate(idx):
        for c in Counter(tup.tolist()).values():
            out[row] /= factorial(c)
    return out


def exact_pe_curve(problem: Problem, detectors, n_values, tail: float = 1e-14) -> pd.DataFrame:
    """:func:`exact_pe_binary` over a list of n, as a long DataFrame.

    Values below ``1000 * tail`` are not resolved by the truncation and are
    returned as NaN.
    """
    rows = []
    for n in np.asarray(n_values, dtype=int):
        pe = exact_pe_binary(problem, detectors, int(n), tail)
        for key, val in pe.items():
            val = val if val >= 1000 * tail else np.nan
            rows.append({"n": int(n), "detector": key, "pe_avg": val, "pe_max": val})
    return pd.DataFrame(rows)

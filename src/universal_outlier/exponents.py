"""Estimating the error exponent c from P_e(n) ~ exp(-c n).

``fit_exponent(method="half_tail")`` reproduces the project's original
``estimate_c``: an ordinary least-squares line through ln P_e over the
larger-n half of the points with P_e > 0. Its weakness is that points with
only a handful of errors have huge relative noise, and they receive the
same weight as well-measured ones; that is the main source of the jagged
c / 2B curves in the original results.

``method="wls"`` (default) keeps only points with at least ``min_errors``
observed errors and weights them by inverse variance,
Var(ln p_hat) ~ (1 - p) / (N p), and reports a standard error for c.
For exact P_e (no sampling noise) use :func:`local_slopes`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExponentFit:
    c: float
    intercept: float
    se: float
    n_points: int

    @property
    def ok(self) -> bool:
        return np.isfinite(self.c)


def fit_exponent(n, pe, trials: int | None = None, method: str = "wls",
                 min_errors: float = 10.0, tail_fraction: float = 0.5) -> ExponentFit:
    n = np.asarray(n, dtype=float)
    pe = np.asarray(pe, dtype=float)
    nan = ExponentFit(np.nan, np.nan, np.nan, 0)

    if method == "half_tail":
        valid = np.flatnonzero(pe > 0)
        if valid.size == 0:
            return ExponentFit(np.inf, np.nan, np.nan, 0)
        if valid.size < 3:
            return nan
        use = valid[valid.size // 2:]
        if use.size < 2:
            return nan
        slope, icpt = np.polyfit(n[use], np.log(pe[use]), 1)
        return ExponentFit(-slope, icpt, np.nan, use.size)

    if method != "wls":
        raise ValueError(f"unknown method {method!r}")
    floor = min_errors / trials if trials else 0.0
    valid = np.flatnonzero((pe > max(floor, 0.0)) & (pe < 1))
    if valid.size < 2:
        return nan
    # keep the larger-n part of the usable range, but at least 3 points
    k = max(3, int(np.ceil(valid.size * tail_fraction)))
    use = valid[-k:] if valid.size >= 3 else valid
    x, y = n[use], np.log(pe[use])
    w = pe[use] * (trials or 1e6) / (1 - pe[use])  # inverse variance of ln p_hat
    W = w.sum()
    xm, ym = (w * x).sum() / W, (w * y).sum() / W
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return nan
    slope = (w * (x - xm) * (y - ym)).sum() / sxx
    icpt = ym - slope * xm
    se = np.sqrt(1.0 / sxx)  # weights are true inverse variances
    return ExponentFit(-slope, icpt, se, use.size)


def local_slopes(n, pe):
    """Finite-difference exponents -(ln P_e(n2) - ln P_e(n1)) / (n2 - n1).

    Returned at the midpoints. For exact P_e these converge to the true
    exponent as n grows (the polynomial prefactor only adds O(1/n)).
    """
    n = np.asarray(n, dtype=float)
    lp = np.log(np.asarray(pe, dtype=float))
    return 0.5 * (n[1:] + n[:-1]), -(lp[1:] - lp[:-1]) / (n[1:] - n[:-1])


def wilson_interval(errors: float, trials: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (works for 0 errors)."""
    p = errors / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

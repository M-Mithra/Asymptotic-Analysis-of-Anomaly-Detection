"""Threshold-decay exponent alpha of NP-SEQ (MTP-2).

The threshold is T_n = C / n^alpha. Its natural scale is set by how fast the
statistic of *typical* streams (the noise floor) decays:

* Under p = q, M_u^2 is a degenerate U-statistic, so n * M_u^2 converges in
  distribution to sum_k lambda_k (Z_k^2 - 1) (Gretton et al. 2012, Thm 12):
  the noise floor is O(1/n), i.e. it decays with exponent exactly 1.
* For an anomalous stream, M_u^2 -> MMD^2(p, q) > 0.

Hence alpha = 1 is the rate at which the threshold tracks the noise. For the
test with alpha the stopping time behaves like N ~ (C / MMD^2)^(1/alpha)
(for alpha = 1 this is Theorem 3 of the NP-SEQ paper: N / C -> 1 / MMD^2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sequential import History, IncrementalMMD, sampler


def null_noise_floor(n_values, trials: int, gamma: float, rng, spec: dict | None = None,
                     paired: bool = False, batch: int = 500) -> pd.DataFrame:
    """Statistics of M_u^2 between two independent streams from the same law.

    Returns, for each n: mean |M_u^2|, mean M_u^2 (unbiased, ~0) and quantiles of
    n * M_u^2 (which stabilise if the noise floor is O(1/n)).
    """
    spec = spec or {"dist": "normal", "loc": 0.0, "scale": 1.0}
    draw = sampler(spec)
    n_values = np.asarray(sorted(set(int(v) for v in n_values)))
    want = set(n_values.tolist())
    vals = {n: [] for n in n_values}
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        eng = IncrementalMMD(b, 2, gamma)
        H = History(b, 2)
        for n in range(1, n_values.max() + 1):
            new = draw(rng, (b, 2))
            eng.update(new, H.data)
            H.append(new)
            if n in want:
                vals[n].append(eng.pairwise(paired)[:, 0, 1].copy())
        done += b
    rows = []
    for n in n_values:
        v = np.concatenate(vals[n])
        q = np.quantile(n * v, [0.5, 0.9, 0.99])
        rows.append({"n": n, "mean_abs": np.abs(v).mean(), "mean": v.mean(),
                     "nM_q50": q[0], "nM_q90": q[1], "nM_q99": q[2]})
    return pd.DataFrame(rows)


def loglog_slope(x, y):
    """Least-squares slope, intercept and R^2 of log10 y on log10 x."""
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    slope, icpt = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + icpt)
    r2 = 1 - resid.var() / ly.var()
    return float(slope), float(icpt), float(r2)


def expected_stop_theory(C, alpha: float, mmd2: float):
    """First-order stopping time (C / MMD^2)^(1/alpha) of NP-SEQ(alpha)."""
    return (np.asarray(C, float) / mmd2) ** (1.0 / alpha)


def paper_delay_bound(neg_log10_p, mmd2: float, K: float = 1.0):
    """Upper bound E[N] <= 32 K^2 ln(1/P_max) / MMD^4 (NP-SEQ paper, eq. 5)."""
    return 32 * K**2 * np.log(10) * np.asarray(neg_log10_p, float) / mmd2**2

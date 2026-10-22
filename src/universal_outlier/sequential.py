"""NP-SEQ: sequential nonparametric outlier detection with MMD.

Sreenivasan & Bhashyam (IEEE SPL 2021). S real-valued streams, one sample per
stream per time step. At time n compute the unbiased MMD^2 between streams and
stop the first time the statistic Gamma_n exceeds the threshold T_n = C / n^alpha.

Statistics (``statistic=``):

* ``"maxmin"`` (paper, eq. 1-2): Gamma_n = max_i min_{j != i} M_u^2(i, j, n),
  candidate = the maximising stream.
* ``"pooled"`` (the project's scripts, ``compute_mmd2_fast(stream_i, others)``):
  Gamma_n = max_i M_u^2(stream i, union of the other S - 1 streams).

Engine
------
Recomputing every MMD from scratch costs O(S^2 n^2) per step. Instead the
engine keeps the kernel block sums

    B[a, c] = sum_{l, m <= n} k(x_{a,l}, x_{c,m}),   D[a, c] = sum_{l <= n} k(x_{a,l}, x_{c,l})

and updates them with the new samples in O(S^2 n) (the paper's sequential
update, generalised). Both statistics are closed-form in B and D, and the whole
computation is vectorised over a batch of independent trials, so thousands of
trials run together; trials that have stopped are dropped from the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .backend import array_module, to_numpy

# ---------------------------------------------------------------------------
# samplers


def sampler(spec: dict):
    """Build ``f(rng, shape) -> samples`` from a spec such as
    ``{"dist": "normal", "loc": 1.2, "scale": 1}``. Supported: normal,
    laplace (``scale`` is the Laplace b), uniform (``low``, ``high``).
    """
    dist = spec["dist"]
    if dist == "normal":
        return lambda rng, shape: rng.normal(spec.get("loc", 0.0), spec.get("scale", 1.0), shape)
    if dist == "laplace":
        return lambda rng, shape: rng.laplace(spec.get("loc", 0.0), spec.get("scale", 1.0), shape)
    if dist == "uniform":
        return lambda rng, shape: rng.uniform(spec.get("low", 0.0), spec.get("high", 1.0), shape)
    raise ValueError(f"unknown distribution {dist!r}")


# ---------------------------------------------------------------------------
# incremental MMD


class IncrementalMMD:
    """Kernel block sums of a batch of S-stream sequences, grown one step at a time."""

    def __init__(self, batch: int, S: int, gamma: float, xp=np):
        self.xp, self.gamma, self.S = xp, gamma, S
        self.B = xp.zeros((batch, S, S))
        self.D = xp.zeros((batch, S, S))
        self.n = 0

    def update(self, new, past):
        """Add samples ``new`` (batch, S) given the history ``past`` (batch, S, n)."""
        xp, g = self.xp, self.gamma
        k_new = xp.exp(-g * (new[:, :, None] - new[:, None, :]) ** 2)  # (b, S, S)
        if past.shape[-1]:
            k_np = xp.exp(-g * (new[:, :, None, None] - past[:, None, :, :]) ** 2)  # (b, a, c, l)
            s = k_np.sum(axis=-1)  # sum_l k(new_a, x_{c,l})
            self.B += s + s.transpose(0, 2, 1)
        self.B += k_new
        self.D += k_new
        self.n += 1

    def keep(self, idx):
        idx = self.xp.asarray(idx)
        self.B = self.B[idx]
        self.D = self.D[idx]

    def pairwise(self, paired: bool = False):
        """M_u^2(i, j) for all pairs, shape (batch, S, S); diagonal is 0."""
        xp, n = self.xp, self.n
        within = (xp.diagonal(self.B, axis1=1, axis2=2) - n) / (n * (n - 1))
        if paired:
            cross = (self.B - self.D) / (n * (n - 1))
            cross = cross + cross.transpose(0, 2, 1)
        else:
            cross = 2.0 * self.B / (n * n)
        out = within[:, :, None] + within[:, None, :] - cross
        idx = xp.arange(self.S)
        out[:, idx, idx] = 0.0
        return out

    def pooled(self):
        """M_u^2(stream i, union of the others) for each i, shape (batch, S)."""
        xp, n, S = self.xp, self.n, self.S
        m = (S - 1) * n
        diagB = xp.diagonal(self.B, axis1=1, axis2=2)
        total = self.B.sum(axis=(1, 2))[:, None]
        row = self.B.sum(axis=2)  # sum_c B[i, c]
        xx = (diagB - n) / (n * (n - 1))
        yy_sum = total - 2.0 * row + diagB  # sum over a, c != i
        yy = (yy_sum - m) / (m * (m - 1))
        xy = (row - diagB) / (n * m)
        return xx + yy - 2.0 * xy


def _maxmin(pairwise, xp):
    S = pairwise.shape[-1]
    eye = xp.eye(S, dtype=bool)
    mins = xp.where(eye, xp.inf, pairwise).min(axis=-1)
    cand = xp.argmax(mins, axis=-1)
    return cand, xp.take_along_axis(mins, cand[:, None], axis=1)[:, 0]


class History:
    """Growable (batch, S, n) sample buffer with amortised O(1) appends."""

    def __init__(self, batch: int, S: int, xp=np, capacity: int = 64):
        self.xp = xp
        self.buf = xp.empty((batch, S, capacity))
        self.n = 0

    def append(self, new):
        if self.n == self.buf.shape[-1]:
            grown = self.xp.empty(self.buf.shape[:2] + (2 * self.n,))
            grown[..., : self.n] = self.buf
            self.buf = grown
        self.buf[..., self.n] = new
        self.n += 1

    @property
    def data(self):
        return self.buf[..., : self.n]

    def keep(self, idx):
        self.buf = self.buf[self.xp.asarray(idx)]


# ---------------------------------------------------------------------------
# NP-SEQ with exactly one anomaly


@dataclass
class SeqConfig:
    S: int = 5
    p: dict = None  # typical distribution spec
    q: dict = None  # anomalous distribution spec
    gamma: float = 1.0  # = 1 / (2 sigma^2); paper uses sigma^2 = 0.5
    statistic: str = "maxmin"  # or "pooled"
    paired: bool = False  # paired U-statistic (paper) vs unpaired cross term
    n_min: int = 2
    max_steps: int = 2000  # time-out T0; trials that time out count as errors

    def __post_init__(self):
        self.p = self.p or {"dist": "normal", "loc": 0.0, "scale": 1.0}
        self.q = self.q or {"dist": "normal", "loc": 1.2, "scale": 1.0}


def _statistic(engine, cfg, xp):
    if cfg.statistic == "maxmin":
        return _maxmin(engine.pairwise(cfg.paired), xp)
    if cfg.statistic == "pooled":
        vals = engine.pooled()
        cand = xp.argmax(vals, axis=-1)
        return cand, xp.take_along_axis(vals, cand[:, None], axis=1)[:, 0]
    raise ValueError(f"unknown statistic {cfg.statistic!r}")


def run_np_seq(cfg: SeqConfig, alpha: float, C_values, trials: int, rng, xp=np,
               batch: int = 2000) -> pd.DataFrame:
    """Run NP-SEQ(alpha) for several threshold constants C on shared data.

    The C values share each simulated trial (one pass per batch), which is
    valid because the stopping rule for each C only reads the statistic path.
    Returns one row per C with ``E[N]``, ``P_err`` (anomaly misidentified or
    time-out), ``timeouts`` and a Wilson 95% interval for ``P_err``.
    """
    C_values = np.sort(np.asarray(C_values, dtype=float))
    samp_p, samp_q = sampler(cfg.p), sampler(cfg.q)
    S, T = cfg.S, cfg.max_steps
    n_C = len(C_values)
    stop_sum = np.zeros(n_C)
    errors = np.zeros(n_C)
    timeouts = np.zeros(n_C)

    done_trials = 0
    while done_trials < trials:
        b = min(batch, trials - done_trials)
        anomaly = rng.integers(0, S, size=b)
        is_anom = (np.arange(S)[None, :] == anomaly[:, None])
        X = History(b, S, xp)
        engine = IncrementalMMD(b, S, cfg.gamma, xp)
        alive = np.arange(b)  # trial ids still running for at least one C
        stopped = np.zeros((b, n_C), bool)
        stop_n = np.full((b, n_C), T, dtype=float)
        correct = np.zeros((b, n_C), bool)
        C_dev = xp.asarray(C_values)
        for n in range(1, T + 1):
            ia = is_anom[alive]
            new = np.where(ia, samp_q(rng, ia.shape), samp_p(rng, ia.shape))
            new = xp.asarray(new)
            engine.update(new, X.data)
            X.append(new)
            if n < max(cfg.n_min, 2):
                continue
            cand, gam = _statistic(engine, cfg, xp)
            cross = gam[:, None] > C_dev[None, :] / n**alpha  # (alive, n_C)
            cross = to_numpy(cross)
            cand_h = to_numpy(cand)
            fresh = cross & ~stopped[alive]
            if fresh.any():
                rows, cols = np.nonzero(fresh)
                tid = alive[rows]
                stopped[tid, cols] = True
                stop_n[tid, cols] = n
                correct[tid, cols] = cand_h[rows] == anomaly[tid]
            still = ~stopped[alive].all(axis=1)
            if not still.all():
                keep = np.flatnonzero(still)
                alive = alive[keep]
                X.keep(keep)
                engine.keep(keep)
            if alive.size == 0:
                break
        stop_sum += stop_n.sum(axis=0)
        errors += (~correct).sum(axis=0)
        timeouts += (~stopped).sum(axis=0)
        done_trials += b

    from .exponents import wilson_interval

    rows = []
    for i, C in enumerate(C_values):
        lo, hi = wilson_interval(errors[i], trials)
        rows.append({"alpha": alpha, "C": C, "E_N": stop_sum[i] / trials, "P_err": errors[i] / trials,
                     "P_lo": lo, "P_hi": hi, "errors": int(errors[i]), "timeouts": int(timeouts[i]),
                     "trials": trials})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NP-SEQ-UA: at most A anomalies


def subset_statistic(pairwise, A: int):
    """max over non-empty subsets I with |I| <= A of min_{i in I, j not in I} M(i, j).

    Returns (best subset index into ``subsets``, statistic, subsets).
    """
    xp = array_module(pairwise)
    S = pairwise.shape[-1]
    subsets = [c for r in range(1, A + 1) for c in combinations(range(S), r)]
    vals = []
    for I in subsets:
        rest = [j for j in range(S) if j not in I]
        block = pairwise[:, list(I)][:, :, rest]
        vals.append(block.reshape(block.shape[0], -1).min(axis=1))
    vals = xp.stack(vals, axis=1)
    best = xp.argmax(vals, axis=1)
    return best, xp.take_along_axis(vals, best[:, None], axis=1)[:, 0], subsets


def run_np_seq_ua(cfg: SeqConfig, A: int, n_anomalies: int, C: float, trials: int, rng,
                  T0: int | None = None, batch: int = 2000, xp=np) -> dict:
    """NP-SEQ-UA (alpha = 1/2, time-out T0) with ``n_anomalies`` true anomalies.

    Declares the max-min subset when Gamma_n > C / sqrt(n), and the empty set
    at the time-out T0 = B C^2. The paper needs T0 >= 4 C^2 / MMD^4 + 1, so the
    default B = 100 suits MMD^2 >= 0.2 (the Gaussian example has 0.224). With ``n_anomalies = 0``
    the error is a false alarm, otherwise a misidentification.
    """
    T0 = T0 or int(np.ceil(100 * C**2))
    samp_p, samp_q = sampler(cfg.p), sampler(cfg.q)
    S = cfg.S
    errors, stop_sum, done = 0, 0.0, 0
    while done < trials:
        b = min(batch, trials - done)
        truth = np.zeros((b, S), bool)
        for t in range(b):
            truth[t, rng.choice(S, n_anomalies, replace=False)] = True
        X = History(b, S, xp)
        engine = IncrementalMMD(b, S, cfg.gamma, xp)
        alive = np.arange(b)
        decided = np.zeros(b, bool)
        for n in range(1, T0 + 1):
            ta = truth[alive]
            new = xp.asarray(np.where(ta, samp_q(rng, ta.shape), samp_p(rng, ta.shape)))
            engine.update(new, X.data)
            X.append(new)
            if n < 2:
                continue
            best, gam, subsets = subset_statistic(engine.pairwise(cfg.paired), A)
            gam = to_numpy(gam)
            best = to_numpy(best)
            hit = gam > C / np.sqrt(n)
            for r in np.flatnonzero(hit):
                t = alive[r]
                declared = np.zeros(S, bool)
                declared[list(subsets[best[r]])] = True
                errors += int(not np.array_equal(declared, truth[t]))
                stop_sum += n
                decided[t] = True
            if hit.any():
                keep = np.flatnonzero(~hit)
                alive = alive[keep]
                X.keep(keep)
                engine.keep(keep)
            if alive.size == 0:
                break
        # time-out: declare the empty set
        errors += int((truth[alive].any(axis=1)).sum())
        stop_sum += T0 * alive.size
        done += b
    return {"A": A, "L": n_anomalies, "C": C, "T0": T0, "P_err": errors / trials,
            "E_N": stop_sum / trials, "trials": trials}

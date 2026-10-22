"""Problem definition for the fixed-sample discrete outlier problem.

M sequences of length n are observed over a finite alphabet. One of them
(the outlier) is i.i.d. ``mu``, the rest are i.i.d. ``pi``. Because every test
in this project depends on the data only through the empirical types, data
is generated directly as per-sequence symbol *counts* (multinomial draws),
which is statistically identical to drawing raw sequences and far cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .divergences import bhattacharyya, total_variation


@dataclass(frozen=True)
class Problem:
    """Typical pmf ``pi``, outlier pmf ``mu`` over ``alphabet`` values.

    ``gamma`` is the RBF kernel parameter k(x, y) = exp(-gamma (x - y)^2)
    used by the MMD rules (the project used gamma = 1).
    """

    pi: np.ndarray
    mu: np.ndarray
    M: int = 3
    alphabet: np.ndarray | None = None
    gamma: float = 1.0
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        pi = np.asarray(self.pi, dtype=float)
        mu = np.asarray(self.mu, dtype=float)
        if pi.shape != mu.shape or pi.ndim != 1:
            raise ValueError("pi and mu must be 1-D pmfs of equal length")
        if not (np.isclose(pi.sum(), 1) and np.isclose(mu.sum(), 1)):
            raise ValueError("pi and mu must each sum to 1")
        if self.M < 3:
            raise ValueError("need at least M = 3 sequences")
        alphabet = np.arange(len(pi), dtype=float) if self.alphabet is None else np.asarray(self.alphabet, float)
        object.__setattr__(self, "pi", pi)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "alphabet", alphabet)

    @property
    def k(self) -> int:
        return len(self.pi)

    @property
    def two_b(self) -> float:
        """Optimal exponent 2 B(pi, mu) of the test that knows pi and mu."""
        return 2.0 * bhattacharyya(self.pi, self.mu)

    @property
    def tvd(self) -> float:
        return float(total_variation(self.pi, self.mu))

    @classmethod
    def bernoulli(cls, p_pi: float, p_mu: float, **kw) -> "Problem":
        """Bernoulli problem with P(X=1) = p_pi (typical) and p_mu (outlier).

        The alphabet is {0, 1}, so pmfs are ordered (P(0), P(1)).
        """
        return cls(pi=[1 - p_pi, p_pi], mu=[1 - p_mu, p_mu], **kw)


def sample_counts(problem: Problem, n: int, trials: int, outlier: int, rng, xp=np):
    """Draw symbol counts of shape (trials, M, k) with sequence ``outlier`` ~ mu."""
    pi = xp.asarray(problem.pi)
    mu = xp.asarray(problem.mu)
    blocks = []
    for j in range(problem.M):
        blocks.append(rng.multinomial(n, mu if j == outlier else pi, size=trials))
    return xp.stack(blocks, axis=1)


def bernoulli_scenarios(preset: str = "standard", grid: str = "exponent",
                        span: tuple[float, float] = (4.0, 30.0)) -> dict[str, dict]:
    """Bernoulli scenarios used for the error-exponent study.

    Each entry maps a name to ``{"p": (p_pi, p_mu), "n": lengths}``. The pairs
    are symmetric about 1/2, so TVD = p_mu - p_pi.

    ``grid="exponent"`` (default) picks 10 lengths with n * 2B(pi, mu) spread
    over ``span``, so every scenario is observed equally deep into its
    exponential decay (P_e from ~0.3 down to ~1e-4 for M = 3).
    ``grid="legacy"`` reproduces the lengths of ``2410.py`` / ``gpu.py``. They
    grow like 1/TVD while 2B shrinks like TVD^2, so hard scenarios only reach
    n * 2B ~ 4, are far from the asymptotic regime, and their fitted exponents
    come out too large.
    """
    from .divergences import bhattacharyya

    def lengths(n0, n1, count=10):
        return np.unique(np.linspace(n0, n1, count).round().astype(int))

    grids = {
        # TVD: (start n, stop n)
        0.50: (20, 200), 0.40: (20, 200), 0.35: (30, 300), 0.30: (40, 400),
        0.25: (45, 450), 0.20: (50, 500), 0.15: (100, 1000), 0.10: (200, 2000),
        0.08: (300, 3000), 0.06: (400, 4000), 0.04: (500, 5000), 0.02: (1000, 10000),
    }
    if preset == "quick":
        keep = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
    elif preset == "standard":
        keep = list(grids)
    else:
        raise ValueError(f"unknown preset {preset!r}")
    if grid not in ("exponent", "legacy"):
        raise ValueError(f"unknown grid {grid!r}")
    out = {}
    for tvd in keep:
        p_pi, p_mu = round(0.5 - tvd / 2, 4), round(0.5 + tvd / 2, 4)
        if grid == "legacy":
            n0, n1 = grids[tvd]
        else:
            two_b = 2 * bhattacharyya([1 - p_pi, p_pi], [1 - p_mu, p_mu])
            n0, n1 = max(2.0, span[0] / two_b), span[1] / two_b
        out[f"TVD_{tvd:.2f}"] = {"p": (p_pi, p_mu), "n": lengths(n0, n1)}
    return out

"""Registry of universal outlier decision rules.

A detector maps empirical types of the M sequences to an *outlier score* per
sequence; the declared outlier is the sequence with the largest score (rules
that minimise a statistic, like the GLT, return its negative).

Signature of a score function::

    score(types, n, problem) -> scores

* ``types``   array (..., M, k): empirical pmf of each sequence
* ``n``       sequence length, or ``None`` for the n -> infinity limit used by
              the large-deviation exponents and the decision-region plots
* ``problem`` :class:`~universal_outlier.problem.Problem` (alphabet, kernel
              parameter, and pi / mu for oracle rules)

Detector modules in this package register themselves with :func:`register`
and are discovered automatically, so adding a rule means adding one file.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable

_REGISTRY: dict[str, "Detector"] = {}


@dataclass(frozen=True)
class Detector:
    key: str
    label: str
    score: Callable
    universal: bool = True  # False for oracle rules that use pi / mu
    description: str = ""

    def __call__(self, types, n, problem):
        return self.score(types, n, problem)


def register(key: str, label: str, universal: bool = True):
    """Decorator registering a score function under ``key``."""

    def wrap(fn):
        if key in _REGISTRY:
            raise ValueError(f"detector {key!r} registered twice")
        doc = (fn.__doc__ or "").strip().splitlines()
        _REGISTRY[key] = Detector(key, label, fn, universal, doc[0] if doc else "")
        return fn

    return wrap


def _discover():
    for mod in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{mod.name}")


def get(key: str) -> Detector:
    _discover()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown detector {key!r}; available: {sorted(_REGISTRY)}") from None


def available() -> dict[str, Detector]:
    _discover()
    return dict(sorted(_REGISTRY.items()))


def others_mean(types):
    """Leave-one-out mean of the types: entry i is mean_{j != i} types_j."""
    M = types.shape[-2]
    total = types.sum(axis=-2, keepdims=True)
    return (total - types) / (M - 1)


def offdiag_mean(pairwise):
    """Row means of an (..., M, M) matrix excluding the diagonal."""
    M = pairwise.shape[-1]
    diag = pairwise.diagonal(axis1=-2, axis2=-1)
    return (pairwise.sum(axis=-1) - diag) / (M - 1)


def offdiag_min(pairwise):
    """Row minima of an (..., M, M) matrix excluding the diagonal."""
    from ..backend import array_module

    xp = array_module(pairwise)
    M = pairwise.shape[-1]
    eye = xp.eye(M, dtype=bool)
    return xp.where(eye, xp.inf, pairwise).min(axis=-1)

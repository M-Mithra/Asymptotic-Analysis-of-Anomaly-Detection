"""Array backend selection.

Every numerical routine in the package is written against an array module
``xp`` so the same code can run on NumPy (CPU). Use :func:`get_backend` to
obtain the module and a random generator for a backend name.
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("cpu",)


def get_backend(name: str = "cpu", seed: int | None = None):
    """Return ``(xp, rng)`` for the requested backend.

    ``rng`` exposes the NumPy ``Generator`` interface used in this package:
    ``random``, ``multinomial``, ``normal``, ``laplace``, ``uniform``,
    ``integers``.
    """
    if name != "cpu":
        raise ValueError(f"Unknown backend {name!r}; available: {BACKENDS}")
    return np, np.random.default_rng(seed)


def array_module(x):
    """Return the array module (NumPy) that owns ``x``."""
    return np


def to_numpy(x) -> np.ndarray:
    """Copy an array to host memory as a NumPy array."""
    return np.asarray(x)


def free_memory() -> None:
    """Release cached device memory (no-op on CPU)."""

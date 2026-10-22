"""Array backend selection: NumPy on the CPU or CuPy on an NVIDIA GPU.

Every numerical routine in the package is written against an array module
``xp`` so the same code runs on either device. Use :func:`get_backend` to
obtain the module and a random generator for a backend name::

    xp, rng = get_backend("gpu", seed=0)   # requires `pip install cupy-cuda12x`

On a GPU the Monte-Carlo engine keeps whole batches of trials on the device;
only the per-n error totals are copied back to the host.
"""

from __future__ import annotations

import numpy as np

try:  # CuPy is optional; everything works without it.
    import cupy  # type: ignore
except ImportError:  # pragma: no cover - depends on the machine
    cupy = None

BACKENDS = ("cpu", "gpu")


class ArrayRNG:
    """NumPy-``Generator``-like sampler on top of ``<module>.random.RandomState``.

    CuPy has no ``Generator.multinomial``, so multinomial counts are drawn
    with the conditional-binomial method: for symbols a = 1..k,
    c_a ~ Bin(n - c_1 - ... - c_{a-1}, p_a / (p_a + ... + p_k)).
    The class works with any array module, which lets the CPU tests check it.
    """

    def __init__(self, xp, seed=None):
        self.xp = xp
        self.rs = xp.random.RandomState(seed)

    def random(self, shape):
        return self.rs.random_sample(shape)

    def integers(self, low, high, size=None):
        return self.rs.randint(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.rs.normal(loc, scale, size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        return self.rs.laplace(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self.rs.uniform(low, high, size)

    def multinomial(self, n, pvals, size):
        xp = self.xp
        pvals = np.asarray(to_numpy(pvals), dtype=float)
        out = xp.zeros((size, len(pvals)), dtype=xp.int64)
        remaining = xp.full(size, n, dtype=xp.int64)
        mass = 1.0
        for a, p in enumerate(pvals[:-1]):
            frac = min(max(p / mass, 0.0), 1.0) if mass > 0 else 0.0
            draw = self.rs.binomial(remaining, frac)
            out[:, a] = draw
            remaining = remaining - draw
            mass -= p
        out[:, -1] = remaining
        return out


def gpu_available() -> bool:
    if cupy is None:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - no driver
        return False


def get_backend(name: str = "cpu", seed: int | None = None):
    """Return ``(xp, rng)`` for ``"cpu"`` (NumPy) or ``"gpu"`` (CuPy)."""
    if name == "cpu":
        return np, np.random.default_rng(seed)
    if name == "gpu":
        if not gpu_available():
            raise RuntimeError("backend 'gpu' needs CuPy and a CUDA device "
                               "(pip install cupy-cuda12x); use --backend cpu")
        props = cupy.cuda.runtime.getDeviceProperties(0)
        print(f"Using GPU: {props['name'].decode() if isinstance(props['name'], bytes) else props['name']}")
        return cupy, ArrayRNG(cupy, seed)
    raise ValueError(f"Unknown backend {name!r}; available: {BACKENDS}")


def array_module(x):
    """Return the array module (NumPy or CuPy) that owns ``x``."""
    if cupy is not None:
        return cupy.get_array_module(x)
    return np


def to_numpy(x) -> np.ndarray:
    """Copy an array to host memory as a NumPy array."""
    if cupy is not None and isinstance(x, cupy.ndarray):
        return cupy.asnumpy(x)
    return np.asarray(x)


def free_memory() -> None:
    """Release cached device memory (no-op on CPU)."""
    if cupy is not None and gpu_available():
        cupy.get_default_memory_pool().free_all_blocks()

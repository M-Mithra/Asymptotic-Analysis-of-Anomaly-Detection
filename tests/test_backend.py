import numpy as np
import pytest

from universal_outlier.backend import ArrayRNG, get_backend, gpu_available, to_numpy
from universal_outlier.fixed_sample import monte_carlo_pe
from universal_outlier.problem import Problem


def test_conditional_binomial_multinomial():
    """The GPU sampler's multinomial (tested here on NumPy) has the right law."""
    rng = ArrayRNG(np, seed=0)
    p = np.array([0.5, 0.3, 0.2])
    c = rng.multinomial(40, p, size=50_000)
    assert c.shape == (50_000, 3) and (c.sum(axis=1) == 40).all()
    np.testing.assert_allclose(c.mean(axis=0), 40 * p, rtol=0.01)
    np.testing.assert_allclose(c.var(axis=0), 40 * p * (1 - p), rtol=0.05)


def test_engine_runs_with_array_rng():
    pr = Problem.bernoulli(0.3, 0.7)
    df = monte_carlo_pe(pr, ["glt", "mmd_avg"], [30], 20_000, ArrayRNG(np, 1), np)
    ref = monte_carlo_pe(pr, ["glt", "mmd_avg"], [30], 20_000, np.random.default_rng(1))
    np.testing.assert_allclose(df.pe_avg, ref.pe_avg, atol=0.01)


def test_unknown_backend():
    with pytest.raises(ValueError):
        get_backend("tpu")


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device / CuPy")
def test_gpu_matches_cpu():  # pragma: no cover - runs only on a GPU machine
    xp, rng = get_backend("gpu", 0)
    pr = Problem.bernoulli(0.4, 0.6)
    gpu = monte_carlo_pe(pr, ["glt", "kl_avg", "mmd_avg"], [100], 200_000, rng, xp)
    cpu = monte_carlo_pe(pr, ["glt", "kl_avg", "mmd_avg"], [100], 200_000, np.random.default_rng(0))
    np.testing.assert_allclose(to_numpy(gpu.pe_avg), cpu.pe_avg, atol=0.005)

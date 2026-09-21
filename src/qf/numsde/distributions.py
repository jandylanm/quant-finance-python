"""Sampling and densities for distributions used in the NumSDE exercises."""

import numpy as np


def cauchy(
    n: int,
    mu: float,
    lam: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return n i.i.d. samples of Cau(mu, lam) using the inversion method.

    Uses F^{-1}(u) = mu + lam * tan(pi * (u - 1/2)) applied to n uniforms.
    """
    # Input validation
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if lam <= 0:
        raise ValueError("lam must be strictly positive.")

    # PRNG
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: n uniform samples
    u = rng.random(n)

    # Step 2: apply the inverse CDF, vectorized over the whole array
    return mu + lam * np.tan(np.pi * (u - 0.5))


def cauchy_pdf(x: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Density of Cau(mu, lam) evaluated at x."""
    return lam / (np.pi * (lam**2 + (x - mu) ** 2))

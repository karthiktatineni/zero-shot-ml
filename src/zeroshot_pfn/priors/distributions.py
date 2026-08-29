"""
Vectorized marginal distribution samplers for tabular synthetic features.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sample_gaussian(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gaussian with random mean and standard deviation."""
    loc = rng.uniform(-3.0, 3.0)
    scale = rng.uniform(0.2, 3.0)
    return rng.normal(loc=loc, scale=scale, size=n_rows).astype(np.float32)


def sample_gaussian_mixture(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gaussian Mixture Model (2 to 5 components)."""
    k = rng.integers(2, 6)
    weights = rng.dirichlet(np.ones(k))
    means = rng.uniform(-5.0, 5.0, size=k)
    scales = rng.uniform(0.1, 1.5, size=k)

    components = rng.choice(k, size=n_rows, p=weights)
    samples = rng.normal(loc=means[components], scale=scales[components])
    return samples.astype(np.float32)


def sample_student_t(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Student-t distribution (heavy tails)."""
    df = rng.uniform(1.5, 10.0)
    samples = rng.standard_t(df=df, size=n_rows)
    # Clip extreme heavy-tail outliers before robust scaling
    samples = np.clip(samples, -50.0, 50.0)
    return samples.astype(np.float32)


def sample_uniform(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from continuous Uniform distribution."""
    low = rng.uniform(-10.0, 5.0)
    high = low + rng.uniform(1.0, 15.0)
    return rng.uniform(low=low, high=high, size=n_rows).astype(np.float32)


def sample_beta(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Beta distribution."""
    a = rng.uniform(0.2, 5.0)
    b = rng.uniform(0.2, 5.0)
    return rng.beta(a=a, b=b, size=n_rows).astype(np.float32)


def sample_gamma(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Gamma distribution (skewed positive)."""
    shape = rng.uniform(0.5, 5.0)
    scale = rng.uniform(0.2, 2.0)
    return rng.gamma(shape=shape, scale=scale, size=n_rows).astype(np.float32)


def sample_lognormal(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Log-normal distribution."""
    mean = rng.uniform(-1.0, 1.0)
    sigma = rng.uniform(0.2, 1.0)
    samples = rng.lognormal(mean=mean, sigma=sigma, size=n_rows)
    samples = np.clip(samples, 0.0, 100.0)
    return samples.astype(np.float32)


def sample_zipf_categorical(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample Zipf-like discrete categorical values encoded numerically."""
    n_categories = rng.integers(3, 12)
    a = rng.uniform(1.2, 2.5)  # Zipf exponent
    ranks = np.arange(1, n_categories + 1)
    probs = 1.0 / (ranks ** a)
    probs /= probs.sum()

    categories = rng.choice(n_categories, size=n_rows, p=probs)
    return categories.astype(np.float32)


def sample_dirichlet_categorical(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Sample categorical values with random Dirichlet class frequencies."""
    n_categories = rng.integers(2, 8)
    alpha = rng.uniform(0.2, 2.0, size=n_categories)
    probs = rng.dirichlet(alpha)
    categories = rng.choice(n_categories, size=n_rows, p=probs)
    return categories.astype(np.float32)


DISTRIBUTION_SAMPLERS: list[Callable[[int, np.random.Generator], np.ndarray]] = [
    sample_gaussian,
    sample_gaussian_mixture,
    sample_student_t,
    sample_uniform,
    sample_beta,
    sample_gamma,
    sample_lognormal,
    sample_zipf_categorical,
    sample_dirichlet_categorical,
]


def sample_features(n_rows: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a heterogeneous feature matrix where each column is drawn from a randomly
    selected marginal distribution.
    Returns array shaped [n_rows, n_features].
    """
    cols = []
    for _ in range(n_features):
        sampler = rng.choice(DISTRIBUTION_SAMPLERS)
        col = sampler(n_rows, rng)
        cols.append(col)
    return np.column_stack(cols).astype(np.float32)

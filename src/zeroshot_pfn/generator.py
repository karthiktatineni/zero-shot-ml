"""
Synthetic Prior Generator producing on-the-fly tabular classification episodes.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from zeroshot_pfn.config import PriorConfig
from zeroshot_pfn.data import Episode, get_rng
from zeroshot_pfn.preprocessing import robust_scale_features
from zeroshot_pfn.priors.distributions import sample_features
from zeroshot_pfn.priors.functions import sample_task_function


def inject_missingness(
    x: np.ndarray,
    missing_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate boolean missingness mask [n_rows, n_features] with a blend of MCAR and MAR.
    """
    n_rows, n_cols = x.shape
    if missing_rate <= 0.0:
        return np.zeros((n_rows, n_cols), dtype=bool)

    # 50% MCAR (Missing Completely at Random), 50% MAR (Missing at Random)
    missing_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for c in range(n_cols):
        if rng.uniform() < 0.5 or n_cols == 1:
            # MCAR: independent Bernoulli per cell
            col_mask = rng.uniform(0.0, 1.0, size=n_rows) < missing_rate
        else:
            # MAR: probability conditioned on another feature
            dep_col = rng.choice([idx for idx in range(n_cols) if idx != c])
            dep_vals = x[:, dep_col]
            # Logistic sigmoid of standardized dependent feature
            normed_dep = (dep_vals - np.mean(dep_vals)) / (np.std(dep_vals) + 1e-6)
            prob = 1.0 / (1.0 + np.exp(-normed_dep))
            # Rescale to match target missing rate
            prob = prob * (missing_rate / (np.mean(prob) + 1e-6))
            prob = np.clip(prob, 0.0, 0.9)
            col_mask = rng.uniform(0.0, 1.0, size=n_rows) < prob

        missing_mask[:, c] = col_mask

    return missing_mask


def sample_episode(
    config: PriorConfig | None = None,
    rng: np.random.Generator | None = None,
    max_features: int = 20,
    max_rows: int = 120,
    max_classes: int = 10,
    max_retries: int = 10,
) -> Episode:
    """
    Sample a complete synthetic tabular classification episode conforming to the Episode contract.
    Enforces class coverage in the support set via rejection sampling.
    """
    if config is None:
        config = PriorConfig()
    if rng is None:
        rng = get_rng(config.seed)

    # 1. Sample episode dimensions
    n_classes = int(rng.integers(config.n_classes_min, config.n_classes_max + 1))
    n_features = int(rng.integers(config.n_features_min, config.n_features_max + 1))
    n_support = int(rng.integers(config.n_support_min, config.n_support_max + 1))
    n_query = int(rng.integers(config.n_query_min, config.n_query_max + 1))

    # Clamp total rows to max_rows
    total_rows = n_support + n_query
    if total_rows > max_rows:
        scale = max_rows / total_rows
        n_support = max(config.n_support_min, int(n_support * scale))
        n_query = max(config.n_query_min, max_rows - n_support)
        total_rows = n_support + n_query

    # 2. Partition features into signal and noise features
    noise_ratio = float(rng.uniform(config.noise_feature_ratio_min, config.noise_feature_ratio_max))
    n_noise = round(n_features * noise_ratio)
    n_signal = max(1, n_features - n_noise)
    n_noise = n_features - n_signal


    # Sample feature matrices
    x_signal = sample_features(total_rows, n_signal, rng)
    if n_noise > 0:
        x_noise = sample_features(total_rows, n_noise, rng)
        x_active = np.column_stack([x_signal, x_noise])
    else:
        x_active = x_signal

    # Randomly permute columns so signal features are not always at the front
    perm = rng.permutation(n_features)
    x_active = x_active[:, perm]

    # 3. Sample task function and labels with rejection sampling for support class coverage
    label_noise = float(rng.uniform(config.label_noise_min, config.label_noise_max))
    is_query = np.zeros(total_rows, dtype=bool)
    is_query[n_support:] = True

    y = None
    for _ in range(max_retries):
        cand_y = sample_task_function(x_signal, n_classes, rng, label_noise=label_noise)
        support_y = cand_y[:n_support]
        unique_classes = np.unique(support_y)

        # Check that all classes are present with at least min_support_per_class
        if len(unique_classes) == n_classes:
            counts = np.bincount(support_y, minlength=n_classes)
            if np.all(counts >= config.min_support_per_class):
                y = cand_y
                break

    # Fallback to learnable quantile hyperplane task if retries fail (guarantees all classes present)
    if y is None:
        # Use a single feature for perfectly axis-aligned fallback
        scalar_score = x_signal[:, 0]
        quantiles = np.linspace(0.0, 1.0, n_classes + 1)[1:-1]
        thresholds = np.quantile(scalar_score, quantiles)
        y = np.digitize(scalar_score, thresholds).astype(np.int64)
        # Guarantee class coverage in support set
        y[:n_classes] = np.arange(n_classes)



    # 4. Inject Missingness
    missing_rate = float(rng.uniform(config.missing_rate_min, config.missing_rate_max))
    missing_active = inject_missingness(x_active, missing_rate, rng)

    # 5. Robust Scaling
    scaled_active = robust_scale_features(x_active, missing_mask=missing_active)

    # 6. Pad to fixed max dimensions (max_features = 20)
    x_padded = np.zeros((total_rows, max_features), dtype=np.float32)
    x_padded[:, :n_features] = scaled_active

    feature_mask = np.zeros(max_features, dtype=bool)
    feature_mask[:n_features] = True

    missing_mask = np.zeros((total_rows, max_features), dtype=bool)
    missing_mask[:, :n_features] = missing_active

    episode = Episode(
        x=x_padded,
        y=y,
        is_query=is_query,
        feature_mask=feature_mask,
        missing_mask=missing_mask,
        n_classes=n_classes,
        n_features=n_features,
    )
    episode.validate(max_features=max_features, max_rows=max_rows, max_classes=max_classes)
    return episode


class PriorIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset streaming synthetic Prior episodes on-the-fly.
    Supports PyTorch multi-worker DataLoader with deterministic seed offsets.
    """

    def __init__(
        self,
        config: PriorConfig | None = None,
        max_episodes: int | None = None,
        base_seed: int = 1337,
    ):
        super().__init__()
        self.config = config or PriorConfig()
        self.max_episodes = max_episodes
        self.base_seed = base_seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            seed = self.base_seed + worker_id * 10007
        else:
            seed = self.base_seed

        rng = get_rng(seed)
        count = 0

        while self.max_episodes is None or count < self.max_episodes:
            ep = sample_episode(self.config, rng=rng)
            ep_tensors = ep.to_tensors()
            yield {
                "x": ep_tensors.x,
                "y": ep_tensors.y,
                "is_query": ep_tensors.is_query,
                "feature_mask": ep_tensors.feature_mask,
                "missing_mask": ep_tensors.missing_mask,
                "n_classes": ep.n_classes,
                "n_features": ep.n_features,
            }
            count += 1
"""
Dataset management utilities for tabular benchmark evaluation and synthetic PFN training streams.
"""


from torch.utils.data import DataLoader

def collate_episodes(episodes: list) -> dict:
    import torch
    x_batch = torch.stack([ep.x if isinstance(ep.x, torch.Tensor) else torch.tensor(ep.x) for ep in episodes])
    y_batch = torch.stack([ep.y if isinstance(ep.y, torch.Tensor) else torch.tensor(ep.y) for ep in episodes])
    is_query_batch = torch.stack([ep.is_query if isinstance(ep.is_query, torch.Tensor) else torch.tensor(ep.is_query) for ep in episodes])
    feature_mask_batch = torch.stack([ep.feature_mask if isinstance(ep.feature_mask, torch.Tensor) else torch.tensor(ep.feature_mask) for ep in episodes])
    missing_mask_batch = torch.stack([ep.missing_mask if isinstance(ep.missing_mask, torch.Tensor) else torch.tensor(ep.missing_mask) for ep in episodes])

    return {
        "x": x_batch,
        "y": y_batch,
        "is_query": is_query_batch,
        "feature_mask": feature_mask_batch,
        "missing_mask": missing_mask_batch,
    }

def get_training_dataloader(
    config: "PriorConfig",
    batch_size: int,
    num_workers: int = 0,
    seed: int | None = None
) -> DataLoader:
    dataset = PriorIterableDataset(config=config, seed=seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True,
    )
    return loader

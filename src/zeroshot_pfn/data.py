"""
Data structures, Episode contracts, seeding, and dataset utilities for Zero-Shot Tabular PFN.
"""

from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from zeroshot_pfn.config import CheckpointMetadata, ModelConfig, PriorConfig, TrainConfig


@dataclass
class Episode:
    """Standard container for a single tabular classification episode (support + query rows)."""

    x: torch.Tensor | np.ndarray
    """Features tensor/array shaped [n_rows, max_features]."""

    y: torch.Tensor | np.ndarray
    """Class labels shaped [n_rows]."""

    is_query: torch.Tensor | np.ndarray
    """Boolean mask shaped [n_rows], True for query rows, False for support rows."""

    feature_mask: torch.Tensor | np.ndarray
    """Boolean mask shaped [max_features], True for active features, False for padding."""

    missing_mask: torch.Tensor | np.ndarray
    """Boolean mask shaped [n_rows, max_features], True for missing cells, False for observed."""

    n_classes: int
    """Number of active classes in this episode."""

    n_features: int
    """Number of active features in this episode."""

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_support(self) -> int:
        if isinstance(self.is_query, torch.Tensor):
            return int((~self.is_query).sum().item())
        return int((~self.is_query).sum())

    @property
    def n_query(self) -> int:
        if isinstance(self.is_query, torch.Tensor):
            return int(self.is_query.sum().item())
        return int(self.is_query.sum())

    def to_tensors(self, device: torch.device | str | None = None) -> Episode:
        """Convert array fields to PyTorch Tensors on target device."""
        def _to_tensor(val, dtype):
            if isinstance(val, torch.Tensor):
                t = val.to(dtype=dtype)
            else:
                t = torch.as_tensor(val, dtype=dtype)
            return t.to(device) if device is not None else t

        return Episode(
            x=_to_tensor(self.x, torch.float32),
            y=_to_tensor(self.y, torch.int64),
            is_query=_to_tensor(self.is_query, torch.bool),
            feature_mask=_to_tensor(self.feature_mask, torch.bool),
            missing_mask=_to_tensor(self.missing_mask, torch.bool),
            n_classes=self.n_classes,
            n_features=self.n_features,
        )

    def to_numpy(self) -> Episode:
        """Convert PyTorch Tensor fields to NumPy arrays."""
        def _to_numpy(val):
            if isinstance(val, torch.Tensor):
                return val.detach().cpu().numpy()
            return np.asarray(val)

        return Episode(
            x=_to_numpy(self.x).astype(np.float32),
            y=_to_numpy(self.y).astype(np.int64),
            is_query=_to_numpy(self.is_query).astype(bool),
            feature_mask=_to_numpy(self.feature_mask).astype(bool),
            missing_mask=_to_numpy(self.missing_mask).astype(bool),
            n_classes=self.n_classes,
            n_features=self.n_features,
        )

    def validate(self, max_features: int = 20, max_rows: int = 120, max_classes: int = 10) -> None:
        """Validate shapes, data types, and logical consistency of the episode."""
        n_rows = self.n_rows
        assert n_rows <= max_rows, f"Episode row count {n_rows} exceeds max_rows {max_rows}"
        assert self.n_features <= max_features, f"Active features {self.n_features} > {max_features}"
        assert self.n_classes <= max_classes, f"Active classes {self.n_classes} > {max_classes}"
        assert self.n_support > 0, "Episode must contain at least 1 support row"
        assert self.n_query > 0, "Episode must contain at least 1 query row"

        if isinstance(self.x, torch.Tensor):
            assert self.x.shape == (n_rows, max_features), f"x shape mismatch: {self.x.shape}"
            assert self.y.shape == (n_rows,), f"y shape mismatch: {self.y.shape}"
            assert self.is_query.shape == (n_rows,), f"is_query shape mismatch: {self.is_query.shape}"
            assert self.feature_mask.shape == (max_features,), f"feature_mask shape: {self.feature_mask.shape}"
            assert self.missing_mask.shape == (n_rows, max_features), f"missing_mask shape: {self.missing_mask.shape}"
            assert torch.all(torch.isfinite(self.x)), "Non-finite values found in features x"
            assert int(self.y.min().item()) >= 0 and int(self.y.max().item()) < self.n_classes
        else:
            assert self.x.shape == (n_rows, max_features), f"x shape mismatch: {self.x.shape}"
            assert self.y.shape == (n_rows,), f"y shape mismatch: {self.y.shape}"
            assert self.is_query.shape == (n_rows,), f"is_query shape mismatch: {self.is_query.shape}"
            assert self.feature_mask.shape == (max_features,), f"feature_mask shape: {self.feature_mask.shape}"
            assert self.missing_mask.shape == (n_rows, max_features), f"missing_mask shape: {self.missing_mask.shape}"
            assert np.all(np.isfinite(self.x)), "Non-finite values found in features x"
            assert int(self.y.min()) >= 0 and int(self.y.max()) < self.n_classes


def set_seed(seed: int | None = None) -> int:
    """
    Set deterministic seeds across Python standard library, NumPy, and PyTorch.
    Returns the resolved seed integer.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Create an independent NumPy random Generator for worker-isolated sampling."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def get_git_commit() -> str | None:
    """Retrieve current Git commit hash if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except (subprocess.SubprocessError, OSError):
        return None


def create_checkpoint_metadata(
    model_config: ModelConfig,
    prior_config: PriorConfig,
    train_config: TrainConfig,
    episode_count: int,
    step_count: int,
    metrics: dict[str, Any],
    seed: int,
    git_commit: str | None = None,
    timestamp: str | None = None,
) -> CheckpointMetadata:
    """Assemble standardized metadata object for checkpoint files."""
    from zeroshot_pfn.config import CheckpointMetadata

    if git_commit is None:
        git_commit = get_git_commit()
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return CheckpointMetadata(
        model_config=model_config,
        prior_config=prior_config,
        train_config=train_config,
        episode_count=episode_count,
        step_count=step_count,
        metrics=metrics,
        seed=seed,
        git_commit=git_commit,
        timestamp=timestamp,
    )

import pandas as pd
from pathlib import Path
from zeroshot_pfn.preprocessing import robust_scale_features

DATA_DIR = Path('data')

def load_tabular_benchmark(
    name: str, 
    n_support: int, 
    n_query: int,
    max_features: int = 20,
    rng: np.random.Generator | None = None
) -> Episode:
    """
    Load a real-world benchmark dataset and format it as an Episode.
    Expects CSVs in data/benchmark_name.csv with 'target' as label.
    """
    if rng is None:
        rng = np.random.default_rng()

    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    
    y_full = df['target'].values
    X_full = df.drop(columns=['target']).values
    
    # Stratified shuffle (simplified for benchmark script)
    idx = rng.permutation(len(y_full))
    X_full, y_full = X_full[idx], y_full[idx]
    
    # Scale features
    X_full = robust_scale_features(X_full)
    
    # Limit rows
    total_rows = n_support + n_query
    if len(y_full) > total_rows:
        X_full = X_full[:total_rows]
        y_full = y_full[:total_rows]
    else:
        # If dataset is smaller, adjust queries
        n_query = len(y_full) - n_support
        
    n_rows = len(y_full)
    actual_features = min(X_full.shape[1], max_features)
    
    x_padded = np.zeros((n_rows, max_features), dtype=np.float32)
    x_padded[:, :actual_features] = X_full[:, :actual_features]
    
    is_query = np.zeros(n_rows, dtype=bool)
    is_query[n_support:] = True
    
    feature_mask = np.zeros(max_features, dtype=bool)
    feature_mask[:actual_features] = True
    
    missing_mask = np.isnan(x_padded)
    x_padded[missing_mask] = 0.0
    
    n_classes = len(np.unique(y_full))
    
    return Episode(
        x=x_padded,
        y=y_full,
        is_query=is_query,
        feature_mask=feature_mask,
        missing_mask=missing_mask,
        n_classes=n_classes,
        n_features=actual_features
    )

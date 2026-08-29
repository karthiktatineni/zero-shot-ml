"""
Unit tests for data utilities, seeding reproducibility, and Episode container contracts.
"""

from __future__ import annotations

import numpy as np
import torch

from zeroshot_pfn.config import ModelConfig, PriorConfig, TrainConfig
from zeroshot_pfn.data import (
    Episode,
    create_checkpoint_metadata,
    get_git_commit,
    get_rng,
    set_seed,
)


def test_seed_reproducibility():
    s1 = set_seed(42)
    val_py1 = np.random.rand(5)
    val_torch1 = torch.rand(5)

    s2 = set_seed(42)
    val_py2 = np.random.rand(5)
    val_torch2 = torch.rand(5)

    assert s1 == 42 and s2 == 42
    np.testing.assert_allclose(val_py1, val_py2)
    torch.testing.assert_close(val_torch1, val_torch2)


def test_episode_validation_and_conversions():
    n_rows = 50
    max_features = 20
    n_classes = 3
    n_features = 8

    # Create synthetic numpy episode
    rng = get_rng(123)
    x = rng.standard_normal((n_rows, max_features)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_rows, dtype=np.int64)
    is_query = np.zeros(n_rows, dtype=bool)
    is_query[35:] = True  # 35 support, 15 query

    feature_mask = np.zeros(max_features, dtype=bool)
    feature_mask[:n_features] = True

    missing_mask = np.zeros((n_rows, max_features), dtype=bool)
    missing_mask[0, 0] = True

    ep = Episode(
        x=x,
        y=y,
        is_query=is_query,
        feature_mask=feature_mask,
        missing_mask=missing_mask,
        n_classes=n_classes,
        n_features=n_features,
    )

    assert ep.n_rows == 50
    assert ep.n_support == 35
    assert ep.n_query == 15
    ep.validate()

    # Test conversion to PyTorch tensors
    ep_torch = ep.to_tensors()
    assert isinstance(ep_torch.x, torch.Tensor)
    assert isinstance(ep_torch.y, torch.Tensor)
    assert isinstance(ep_torch.is_query, torch.Tensor)
    assert ep_torch.x.dtype == torch.float32
    assert ep_torch.y.dtype == torch.int64
    ep_torch.validate()

    # Test roundtrip back to numpy
    ep_back = ep_torch.to_numpy()
    assert isinstance(ep_back.x, np.ndarray)
    np.testing.assert_allclose(ep.x, ep_back.x)


def test_get_git_commit():
    commit = get_git_commit()
    # In a git repository, commit should be a 40-character hex string or None if git fails
    if commit is not None:
        assert isinstance(commit, str)
        assert len(commit) >= 7


def test_checkpoint_metadata_creation():
    m_cfg = ModelConfig()
    p_cfg = PriorConfig()
    t_cfg = TrainConfig()

    meta = create_checkpoint_metadata(
        model_config=m_cfg,
        prior_config=p_cfg,
        train_config=t_cfg,
        episode_count=1000,
        step_count=500,
        metrics={"val_loss": 0.45, "val_acc": 0.88},
        seed=1337,
    )

    assert meta.episode_count == 1000
    assert meta.step_count == 500
    assert meta.metrics["val_loss"] == 0.45
    assert meta.model_config.d_model == 128
    assert meta.timestamp is not None
    assert meta.version == "0.1.0"

    # Test dictionary export
    as_dict = meta.to_dict()
    assert as_dict["episode_count"] == 1000
    assert as_dict["model_config"]["d_model"] == 128

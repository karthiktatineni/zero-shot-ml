"""
Unit tests for configuration dataclasses, validation, and JSON serialization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from zeroshot_pfn.config import (
    CheckpointMetadata,
    InferenceConfig,
    ModelConfig,
    MonitorConfig,
    PriorConfig,
    TrainConfig,
    TrainControlConfig,
)


def test_model_config_defaults_and_validation():
    cfg = ModelConfig()
    cfg.validate()
    assert cfg.d_model == 128
    assert cfg.n_layers == 6
    assert cfg.n_heads == 4
    assert cfg.d_ff == 256
    assert cfg.max_features == 20
    assert cfg.max_classes == 10
    assert cfg.max_rows == 120
    assert cfg.fourier_features == 16
    assert cfg.fourier_scale == 2.0


def test_prior_config_defaults_and_validation():
    cfg = PriorConfig()
    cfg.validate()
    assert cfg.n_support_min == 20
    assert cfg.n_support_max == 80
    assert cfg.n_query_min == 10
    assert cfg.n_query_max == 40
    assert cfg.n_features_min == 4
    assert cfg.n_features_max == 20
    assert cfg.n_classes_min == 2
    assert cfg.n_classes_max == 10


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.batch_size == 8
    assert cfg.grad_accum_steps == 2
    assert cfg.effective_batch_size == 16
    assert cfg.total_episodes == 500000


def test_roundtrip_serialization_all_configs():
    configs = [
        ModelConfig(d_model=96, n_layers=4, n_heads=4, d_ff=192, dropout=0.1),
        PriorConfig(n_support_min=10, n_support_max=50, n_features_min=5, n_features_max=15),
        TrainConfig(batch_size=4, grad_accum_steps=4, lr=1e-4, device="cpu"),
        MonitorConfig(log_interval_steps=5, dashboard_refresh_seconds=1.5),
        TrainControlConfig(control_poll_interval_seconds=2.0, pause_on_request=False),
        InferenceConfig(max_support_rows=50, max_query_rows=25, n_ensembles=4),
        CheckpointMetadata(
            model_config=ModelConfig(d_model=64, n_layers=2),
            prior_config=PriorConfig(n_classes_max=5),
            train_config=TrainConfig(lr=2e-4),
            episode_count=5000,
            step_count=2500,
            metrics={"val_loss": 0.32, "val_acc": 0.91},
            seed=42,
            git_commit="abcdef123456",
            timestamp="2026-08-29T12:00:00Z",
            version="0.1.0",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, cfg in enumerate(configs):
            cls = type(cfg)
            # 1. Dict round-trip
            as_dict = cfg.to_dict()
            from_dict = cls.from_dict(as_dict)
            assert from_dict == cfg, f"Dict round-trip failed for {cls.__name__}"

            # 2. JSON string round-trip
            json_str = cfg.to_json()
            from_json = cls.from_json(json_str)
            assert from_json == cfg, f"JSON string round-trip failed for {cls.__name__}"

            # 3. File save & load round-trip
            file_path = Path(tmp_dir) / f"config_{idx}.json"
            cfg.save(file_path)
            from_file = cls.load(file_path)
            assert from_file == cfg, f"File save/load round-trip failed for {cls.__name__}"


def test_monitor_and_control_configs():
    m_cfg = MonitorConfig()
    assert m_cfg.enabled is True
    assert m_cfg.log_interval_steps == 10

    c_cfg = TrainControlConfig()
    assert c_cfg.pause_on_request is True
    assert c_cfg.stop_on_request is True

    i_cfg = InferenceConfig()
    assert i_cfg.max_support_rows == 80
    assert i_cfg.max_query_rows == 40

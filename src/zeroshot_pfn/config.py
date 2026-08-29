"""
Configuration dataclasses and serialization for the Zero-Shot Tabular PFN Engine.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass
class BaseConfig:
    """Base dataclass providing JSON and dictionary serialization utilities."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Instantiate configuration from a dictionary, ignoring extraneous keys."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Instantiate configuration from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load configuration from a JSON file."""
        target = Path(path)
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration for the Dual-Axis Transformer."""

    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 256
    max_features: int = 20
    max_classes: int = 10
    max_rows: int = 120
    dropout: float = 0.0
    fourier_features: int = 16
    fourier_scale: float = 2.0
    use_checkpointing: bool = False
    missingness_embedding: bool = True

    def validate(self) -> None:
        """Validate model configuration constraints."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.max_features > 0, "max_features must be positive"
        assert self.max_classes >= 2, "max_classes must be at least 2"
        assert self.max_rows > 0, "max_rows must be positive"


@dataclass
class PriorConfig(BaseConfig):
    """Prior generator configuration for synthetic episode creation."""

    n_support_min: int = 20
    n_support_max: int = 80
    n_query_min: int = 10
    n_query_max: int = 40
    n_features_min: int = 4
    n_features_max: int = 20
    n_classes_min: int = 2
    n_classes_max: int = 10
    noise_feature_ratio_min: float = 0.4
    noise_feature_ratio_max: float = 0.85
    label_noise_min: float = 0.0
    label_noise_max: float = 0.02
    missing_rate_min: float = 0.0
    missing_rate_max: float = 0.1
    min_support_per_class: int = 1
    seed: int | None = None

    def validate(self) -> None:
        """Validate prior configuration constraints."""
        assert self.n_support_min <= self.n_support_max
        assert self.n_query_min <= self.n_query_max
        assert self.n_features_min <= self.n_features_max
        assert self.n_classes_min <= self.n_classes_max
        assert 0.0 <= self.label_noise_min <= self.label_noise_max <= 1.0
        assert 0.0 <= self.missing_rate_min <= self.missing_rate_max <= 1.0


@dataclass
class TrainConfig(BaseConfig):
    """Training pipeline and optimization configuration."""

    batch_size: int = 8
    grad_accum_steps: int = 2
    lr: float = 3e-4
    min_lr: float = 1e-6
    warmup_steps: int = 2000
    total_episodes: int = 500000
    eval_interval: int = 1000
    checkpoint_interval: int = 25000
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "float16"
    seed: int = 1337

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps


@dataclass
class MonitorConfig(BaseConfig):
    """Real-time monitoring and reporting configuration."""

    enabled: bool = True
    live_jsonl_path: str = "reports/training_live.jsonl"
    state_json_path: str = "reports/training_state.json"
    csv_log_path: str = "reports/training_log.csv"
    summary_path: str = "reports/training_summary.md"
    log_interval_steps: int = 10
    dashboard_refresh_seconds: float = 2.0
    keep_last_events: int = 5000
    track_gpu: bool = True
    track_episode_shapes: bool = True
    track_checkpoints: bool = True


@dataclass
class TrainControlConfig(BaseConfig):
    """Process execution, pause/resume, and control file configuration."""

    run_root: str = "runs"
    control_poll_interval_seconds: float = 5.0
    pause_on_request: bool = True
    stop_on_request: bool = True
    checkpoint_before_pause: bool = True
    checkpoint_before_stop: bool = True
    latest_run_state_path: str = "runs/latest.json"


@dataclass
class InferenceConfig(BaseConfig):
    """Zero-shot inference and test-time evaluation configuration."""

    max_support_rows: int = 80
    max_query_rows: int = 40
    max_features: int = 20
    support_sampling: str = "stratified"
    n_ensembles: int = 1
    column_permutation_ensembles: int = 1
    device: str = "auto"


@dataclass
class CheckpointMetadata(BaseConfig):
    """Standardized metadata schema packaged with every model checkpoint."""

    model_config: ModelConfig
    prior_config: PriorConfig
    train_config: TrainConfig
    episode_count: int
    step_count: int
    metrics: dict[str, Any] = field(default_factory=dict)
    seed: int = 1337
    git_commit: str | None = None
    timestamp: str | None = None
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata and nested configurations to dictionary."""
        return {
            "model_config": self.model_config.to_dict(),
            "prior_config": self.prior_config.to_dict(),
            "train_config": self.train_config.to_dict(),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "metrics": self.metrics,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct CheckpointMetadata from dictionary with nested configs."""
        return cls(
            model_config=ModelConfig.from_dict(data.get("model_config", {})),
            prior_config=PriorConfig.from_dict(data.get("prior_config", {})),
            train_config=TrainConfig.from_dict(data.get("train_config", {})),
            episode_count=data.get("episode_count", 0),
            step_count=data.get("step_count", 0),
            metrics=data.get("metrics", {}),
            seed=data.get("seed", 1337),
            git_commit=data.get("git_commit"),
            timestamp=data.get("timestamp"),
            version=data.get("version", "0.1.0"),
        )

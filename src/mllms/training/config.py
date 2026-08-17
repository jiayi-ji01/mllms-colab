"""Pretraining configuration and YAML resolution."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any

from mllms.config import load_yaml


@dataclass
class TrainingConfig:
    """Model, data, optimization, and runtime settings."""

    data_dir: str = "data/processed"
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.model"
    output_dir: str = "outputs/gpt12_tinystories_clone"
    resume: str | None = None
    device: str = "auto"
    seed: int = 42
    p_clone: float = 0.5

    vocab_size: int = 4096
    block_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 1024
    dropout: float = 0.1
    bias: bool = True

    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    target_seen_tokens: int | None = 200_000_000
    target_epochs: float | None = None
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_ratio: float = 0.01
    warmup_steps: int | None = None
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 20
    checkpoint_interval: int = 1_000
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name in (
            "target_seen_tokens",
            "target_epochs",
            "max_steps",
            "warmup_steps",
            "early_stopping_patience",
        ):
            value = getattr(self, name)
            if value is not None:
                positive[name] = value
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.target_seen_tokens is None and self.target_epochs is None:
            if self.max_steps is None:
                raise ValueError("set target_seen_tokens, target_epochs, or max_steps")
        if self.target_seen_tokens is not None and self.target_epochs is not None:
            raise ValueError("set target_seen_tokens or target_epochs, not both")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not 0.0 <= self.p_clone <= 1.0:
            raise ValueError("p_clone must be between 0 and 1")
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be between 0 and learning_rate")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must satisfy 0 <= warmup_ratio < 1")
        if self.warmup_steps is not None and self.warmup_ratio != 0.0:
            raise ValueError("set either warmup_steps or warmup_ratio, not both")

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps * self.block_size

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def resolve_max_steps(self, train_dataset_tokens: int) -> int:
        if self.max_steps is not None:
            return self.max_steps
        target_tokens = (
            self.target_seen_tokens
            if self.target_seen_tokens is not None
            else math.ceil(float(self.target_epochs) * train_dataset_tokens)
        )
        return math.ceil(target_tokens / self.tokens_per_step)

    def resolve_warmup_steps(self, max_steps: int) -> int:
        if self.warmup_steps is not None:
            return self.warmup_steps
        return math.ceil(self.warmup_ratio * max_steps)


def load_training_config(path: Path) -> TrainingConfig:
    """Load the model/data/training/runtime sections from one experiment YAML."""
    document = load_yaml(path)
    flattened: dict[str, Any] = {}
    for key, value in document.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    valid_fields = {field.name for field in fields(TrainingConfig)}
    unknown = set(flattened) - valid_fields
    if unknown:
        raise ValueError(f"Unknown training config fields: {sorted(unknown)}")
    return TrainingConfig(**flattened)

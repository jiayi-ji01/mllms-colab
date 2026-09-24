"""Checkpoint serialization and exact training-state restoration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT
from mllms.training.config import TrainingConfig


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    step: int,
    epoch: float,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    best_validation_loss: float,
    validation_loss: float,
    tokens_seen: int,
    original_tokens_seen: int,
    clone_tokens_seen: int,
    evaluations_without_improvement: int,
    generator: torch.Generator,
) -> None:
    """Save all state required for an exact training resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "tokens_seen": tokens_seen,
        "original_tokens_seen": original_tokens_seen,
        "clone_tokens_seen": clone_tokens_seen,
        "evaluations_without_improvement": evaluations_without_improvement,
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    generator: torch.Generator,
    expected_model_config: GPTConfig,
    device: torch.device,
) -> tuple[int, float, float, int, int, int, int]:
    """Restore model, optimizer, scheduler, scaler, counters, and RNG state."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"scheduler", "scaler", "tokens_seen"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            "Checkpoint predates the current 12-layer training format and cannot "
            f"be resumed; missing fields: {sorted(missing)}"
        )
    if checkpoint.get("model_config") != asdict(expected_model_config):
        raise ValueError(
            "Checkpoint model_config does not match this run. Use a new output "
            "directory and do not resume the incompatible checkpoint."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    generator.set_state(checkpoint["generator_state"].cpu())
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        states = [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        torch.cuda.set_rng_state_all(states)

    return (
        int(checkpoint["step"]),
        float(checkpoint["best_validation_loss"]),
        float(checkpoint.get("validation_loss", float("inf"))),
        int(checkpoint["tokens_seen"]),
        int(checkpoint["original_tokens_seen"]),
        int(checkpoint["clone_tokens_seen"]),
        int(checkpoint["evaluations_without_improvement"]),
    )

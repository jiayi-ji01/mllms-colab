"""Checkpoint saving, exact training resume and inference loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch

from config import TrainingConfig
from model import GPT, GPTConfig


@dataclass
class TrainingState:
    step: int = 0
    best_validation_loss: float = float("inf")
    validation_loss: float = float("inf")
    tokens_seen: int = 0
    original_tokens_seen: int = 0
    clone_tokens_seen: int = 0
    evaluations_without_improvement: int = 0


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    progress: TrainingState,
    training_config: TrainingConfig,
    generator: torch.Generator,
    *,
    epoch: float,
) -> None:
    """Save all state required for an exact training resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        **asdict(progress),
        "epoch": epoch,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
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
    device: torch.device,
) -> TrainingState:
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
    if checkpoint.get("model_config") != asdict(model.config):
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

    return TrainingState(
        step=int(checkpoint["step"]),
        best_validation_loss=float(checkpoint["best_validation_loss"]),
        validation_loss=float(checkpoint.get("validation_loss", float("inf"))),
        tokens_seen=int(checkpoint["tokens_seen"]),
        original_tokens_seen=int(checkpoint["original_tokens_seen"]),
        clone_tokens_seen=int(checkpoint["clone_tokens_seen"]),
        evaluations_without_improvement=int(checkpoint["evaluations_without_improvement"]),
    )



def load_model_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict, GPT, GPTConfig]:
    """Load checkpoint metadata and a model in evaluation mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint must contain model and model_config")
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model, model_config

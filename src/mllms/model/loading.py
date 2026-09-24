"""Load inference models from the stable checkpoint dictionary schema."""

from pathlib import Path

import torch

from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT


def load_model_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict, GPT, GPTConfig]:
    """Load checkpoint metadata and a model in evaluation mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return checkpoint, model, model_config

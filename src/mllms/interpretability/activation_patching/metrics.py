"""Activation-patching score definitions."""

import torch


METRICS = ("delta_ld", "recovery")


def site_scores(
    patched_ld: torch.Tensor,
    corrupted_ld: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw LD change and normalized clean-behavior recovery."""
    delta_ld = patched_ld - corrupted_ld
    return delta_ld, delta_ld / denominator

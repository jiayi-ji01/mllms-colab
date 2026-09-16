"""Activation-patching score definitions."""

import torch


METRICS = ("patched_ld", "delta_ld", "recovery")


def site_scores(
    patched_ld: torch.Tensor,
    corrupted_ld: torch.Tensor,
    denominator: torch.Tensor,
    denominator_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw LD change and normalized clean-behavior recovery."""
    delta_ld = patched_ld - corrupted_ld
    recovery = torch.where(
        denominator.abs() >= denominator_epsilon,
        delta_ld / denominator,
        torch.full_like(delta_ld, float("nan")),
    )
    return delta_ld, recovery

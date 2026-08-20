"""Batched clean-to-corrupted activation interventions."""

from __future__ import annotations

import torch

from mllms.evaluation.sva.scoring import sequence_log_probability_difference
from mllms.interpretability.activation_patching.metrics import METRICS, site_scores
from mllms.model.transformer import GPT


COMPONENTS = ("resid_post", "attn_out", "mlp_out")
ALL_COMPONENTS = (*COMPONENTS, "head_out")


@torch.inference_mode()
def patch_batch(
    model: GPT,
    examples: list[dict],
    device: torch.device,
    max_positions: int | None,
    intervention_batch_size: int,
) -> list[dict]:
    """Patch one equal-length example batch using batched interventions."""
    clean = torch.tensor(
        [example["clean_ids"] for example in examples], device=device
    )
    corrupted = torch.tensor(
        [example["corrupted_ids"] for example in examples], device=device
    )
    clean_answers = torch.tensor(
        [
            example.get("clean_answer_ids", [example.get("clean_answer_id")])
            for example in examples
        ], device=device
    )
    corrupted_answers = torch.tensor(
        [
            example.get(
                "corrupted_answer_ids", [example.get("corrupted_answer_id")]
            )
            for example in examples
        ], device=device
    )

    _, clean_cache = model.run_with_cache(clean)
    clean_ld = sequence_log_probability_difference(
        model, clean, clean_answers, corrupted_answers
    )
    corrupted_ld = sequence_log_probability_difference(
        model, corrupted, clean_answers, corrupted_answers
    )
    denominator = clean_ld - corrupted_ld

    batch_size, sequence_length = clean.shape
    first_position = (
        0 if max_positions is None else max(0, sequence_length - max_positions)
    )
    positions = torch.arange(first_position, sequence_length, device=device)
    position_count = positions.numel()
    n_layers = model.config.n_layers
    n_heads = model.config.n_heads

    batch_scores = {
        metric: {
            component: torch.empty(
                batch_size,
                n_layers,
                position_count,
                *(() if component != "head_out" else (n_heads,)),
                device=device,
            )
            for component in ALL_COMPONENTS
        }
        for metric in METRICS
    }

    example_grid = torch.arange(device=device, end=batch_size).repeat_interleave(
        position_count
    )
    position_grid = positions.repeat(batch_size)
    for layer in range(n_layers):
        for component in COMPONENTS:
            name = f"blocks.{layer}.{component}"
            clean_activation = clean_cache[name]
            total = example_grid.numel()
            layer_delta = torch.empty(total, device=device)
            layer_recovery = torch.empty(total, device=device)
            for start in range(0, total, intervention_batch_size):
                stop = min(start + intervention_batch_size, total)
                example_indices = example_grid[start:stop]
                token_indices = position_grid[start:stop]
                rows = torch.arange(stop - start, device=device)

                def patch_component(
                    activation: torch.Tensor,
                    example_indices: torch.Tensor = example_indices,
                    token_indices: torch.Tensor = token_indices,
                    rows: torch.Tensor = rows,
                    clean_activation: torch.Tensor = clean_activation,
                ) -> torch.Tensor:
                    patched = activation.clone()
                    patched[rows, token_indices] = clean_activation[
                        example_indices, token_indices
                    ]
                    return patched

                with model.hooks({name: patch_component}):
                    patched_ld = sequence_log_probability_difference(
                        model,
                        corrupted[example_indices],
                        clean_answers[example_indices],
                        corrupted_answers[example_indices],
                    )
                delta_ld, recovery = site_scores(
                    patched_ld,
                    corrupted_ld[example_indices],
                    denominator[example_indices],
                )
                layer_delta[start:stop] = delta_ld
                layer_recovery[start:stop] = recovery
            batch_scores["delta_ld"][component][:, layer] = layer_delta.view(
                batch_size, position_count
            )
            batch_scores["recovery"][component][:, layer] = layer_recovery.view(
                batch_size, position_count
            )

        name = f"blocks.{layer}.head_out"
        clean_activation = clean_cache[name]
        head_example_grid = example_grid.repeat_interleave(n_heads)
        head_position_grid = position_grid.repeat_interleave(n_heads)
        head_grid = torch.arange(device=device, end=n_heads).repeat(
            example_grid.numel()
        )
        total = head_example_grid.numel()
        layer_delta = torch.empty(total, device=device)
        layer_recovery = torch.empty(total, device=device)
        for start in range(0, total, intervention_batch_size):
            stop = min(start + intervention_batch_size, total)
            example_indices = head_example_grid[start:stop]
            token_indices = head_position_grid[start:stop]
            head_indices = head_grid[start:stop]
            rows = torch.arange(stop - start, device=device)

            def patch_head(
                activation: torch.Tensor,
                example_indices: torch.Tensor = example_indices,
                token_indices: torch.Tensor = token_indices,
                head_indices: torch.Tensor = head_indices,
                rows: torch.Tensor = rows,
                clean_activation: torch.Tensor = clean_activation,
            ) -> torch.Tensor:
                patched = activation.clone()
                patched[rows, head_indices, token_indices] = clean_activation[
                    example_indices, head_indices, token_indices
                ]
                return patched

            with model.hooks({name: patch_head}):
                patched_ld = sequence_log_probability_difference(
                    model,
                    corrupted[example_indices],
                    clean_answers[example_indices],
                    corrupted_answers[example_indices],
                )
            delta_ld, recovery = site_scores(
                patched_ld,
                corrupted_ld[example_indices],
                denominator[example_indices],
            )
            layer_delta[start:stop] = delta_ld
            layer_recovery[start:stop] = recovery
        batch_scores["delta_ld"]["head_out"][:, layer] = layer_delta.view(
            batch_size, position_count, n_heads
        )
        batch_scores["recovery"]["head_out"][:, layer] = layer_recovery.view(
            batch_size, position_count, n_heads
        )

    return [
        {
            "sample_id": example["sample_id"],
            "sequence_length": sequence_length,
            "position_count": position_count,
            "clean_ld": float(clean_ld[index]),
            "corrupted_ld": float(corrupted_ld[index]),
            "denominator": float(denominator[index]),
            "scores": {
                metric: {
                    component: batch_scores[metric][component][index]
                    .float()
                    .cpu()
                    .numpy()
                    for component in ALL_COMPONENTS
                }
                for metric in METRICS
            },
        }
        for index, example in enumerate(examples)
    ]

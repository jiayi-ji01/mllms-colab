"""Activation interventions and result aggregation."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import torch

from model import GPT
from sva import sequence_log_probability_difference


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


COMPONENTS = ("resid_post", "attn_out", "mlp_out")


ALL_COMPONENTS = (*COMPONENTS, "head_out")


@torch.inference_mode()
def patch_batch(
    model: GPT,
    examples: list[dict],
    device: torch.device,
    max_positions: int | None,
    intervention_batch_size: int,
    target_examples: list[dict] | None = None,
    components: tuple[str, ...] | list[str] | None = None,
    denominator_epsilon: float = 1e-6,
) -> list[dict]:
    """Patch source clean activations into target corrupted executions.

    Omitting ``target_examples`` preserves the original within-language behavior.
    Cross-language calls must pair the same semantic examples at identical token
    positions; answer scoring and the recovery denominator always use the target
    language.
    """
    targets = examples if target_examples is None else target_examples
    selected_components = tuple(components or ALL_COMPONENTS)
    unknown_components = set(selected_components) - set(ALL_COMPONENTS)
    if unknown_components:
        raise ValueError(
            f"unknown patch components: {sorted(unknown_components)}"
        )
    if len(set(selected_components)) != len(selected_components):
        raise ValueError("patch components must not contain duplicates")
    if len(examples) != len(targets):
        raise ValueError("source and target batches must have the same size")
    if any(
        str(source["sample_id"]) != str(target["sample_id"])
        for source, target in zip(examples, targets)
    ):
        raise ValueError("source and target sample IDs must match in order")

    source_clean = torch.tensor(
        [example["clean_ids"] for example in examples], device=device
    )
    target_clean = torch.tensor(
        [example["clean_ids"] for example in targets], device=device
    )
    target_corrupted = torch.tensor(
        [example["corrupted_ids"] for example in targets], device=device
    )
    clean_answers = torch.tensor(
        [
            example.get("clean_answer_ids", [example.get("clean_answer_id")])
            for example in targets
        ], device=device
    )
    corrupted_answers = torch.tensor(
        [
            example.get(
                "corrupted_answer_ids", [example.get("corrupted_answer_id")]
            )
            for example in targets
        ], device=device
    )

    if source_clean.shape != target_corrupted.shape:
        raise ValueError(
            "source clean and target corrupted prompts must have aligned lengths"
        )
    if target_clean.shape != target_corrupted.shape:
        raise ValueError("target clean and corrupted prompts must have aligned lengths")

    cache_names = {
        f"blocks.{layer}.{component}"
        for layer in range(model.config.n_layers)
        for component in selected_components
    }
    _, clean_cache = model.run_with_cache(source_clean, names=cache_names)
    clean_ld = sequence_log_probability_difference(
        model, target_clean, clean_answers, corrupted_answers
    )
    corrupted_ld = sequence_log_probability_difference(
        model, target_corrupted, clean_answers, corrupted_answers
    )
    denominator = clean_ld - corrupted_ld

    batch_size, sequence_length = target_corrupted.shape
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
            for component in selected_components
        }
        for metric in METRICS
    }

    example_grid = torch.arange(device=device, end=batch_size).repeat_interleave(
        position_count
    )
    position_grid = positions.repeat(batch_size)
    for layer in range(n_layers):
        for component in COMPONENTS:
            if component not in selected_components:
                continue
            name = f"blocks.{layer}.{component}"
            clean_activation = clean_cache[name]
            total = example_grid.numel()
            layer_patched = torch.empty(total, device=device)
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
                        target_corrupted[example_indices],
                        clean_answers[example_indices],
                        corrupted_answers[example_indices],
                    )
                delta_ld, recovery = site_scores(
                    patched_ld,
                    corrupted_ld[example_indices],
                    denominator[example_indices],
                    denominator_epsilon,
                )
                layer_patched[start:stop] = patched_ld
                layer_delta[start:stop] = delta_ld
                layer_recovery[start:stop] = recovery
            batch_scores["patched_ld"][component][:, layer] = layer_patched.view(
                batch_size, position_count
            )
            batch_scores["delta_ld"][component][:, layer] = layer_delta.view(
                batch_size, position_count
            )
            batch_scores["recovery"][component][:, layer] = layer_recovery.view(
                batch_size, position_count
            )

        if "head_out" not in selected_components:
            continue
        name = f"blocks.{layer}.head_out"
        clean_activation = clean_cache[name]
        head_example_grid = example_grid.repeat_interleave(n_heads)
        head_position_grid = position_grid.repeat_interleave(n_heads)
        head_grid = torch.arange(device=device, end=n_heads).repeat(
            example_grid.numel()
        )
        total = head_example_grid.numel()
        layer_patched = torch.empty(total, device=device)
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
                    target_corrupted[example_indices],
                    clean_answers[example_indices],
                    corrupted_answers[example_indices],
                )
            delta_ld, recovery = site_scores(
                patched_ld,
                corrupted_ld[example_indices],
                denominator[example_indices],
                denominator_epsilon,
            )
            layer_patched[start:stop] = patched_ld
            layer_delta[start:stop] = delta_ld
            layer_recovery[start:stop] = recovery
        batch_scores["patched_ld"]["head_out"][:, layer] = layer_patched.view(
            batch_size, position_count, n_heads
        )
        batch_scores["delta_ld"]["head_out"][:, layer] = layer_delta.view(
            batch_size, position_count, n_heads
        )
        batch_scores["recovery"]["head_out"][:, layer] = layer_recovery.view(
            batch_size, position_count, n_heads
        )

    return [
        {
            "sample_id": targets[index]["sample_id"],
            "source_sample_id": examples[index].get(
                "source_sample_id", examples[index]["sample_id"]
            ),
            "sequence_length": sequence_length,
            "position_count": position_count,
            "clean_ld": float(clean_ld[index]),
            "corrupted_ld": float(corrupted_ld[index]),
            "target_clean_ld": float(clean_ld[index]),
            "target_corrupted_ld": float(corrupted_ld[index]),
            "denominator": float(denominator[index]),
            "scores": {
                metric: {
                    component: batch_scores[metric][component][index]
                    .float()
                    .cpu()
                    .numpy()
                    for component in selected_components
                }
                for metric in METRICS
            },
        }
        for index, example in enumerate(examples)
    ]


def _mean_and_count(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    total = np.nansum(values, axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float32),
        where=count > 0,
    )
    return mean.astype(np.float32), count


def aggregate_results(
    results: list[dict],
    n_layers: int,
    n_heads: int,
    max_positions: int | None,
) -> tuple[dict, dict, dict, np.ndarray]:
    """Right-align variable prompt lengths and average every patch site."""
    components = tuple(results[0]["scores"][METRICS[0]])
    width = max(result["position_count"] for result in results)
    if max_positions is not None:
        width = min(width, max_positions)
    count = len(results)
    per_example = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for component in components:
            shape = (
                (count, n_layers, width, n_heads)
                if component == "head_out"
                else (count, n_layers, width)
            )
            per_example[metric][component] = np.full(
                shape, np.nan, dtype=np.float32
            )

    for index, result in enumerate(results):
        used = min(width, result["position_count"])
        for metric in METRICS:
            for component in components:
                per_example[metric][component][index, :, -used:] = result[
                    "scores"
                ][metric][component][:, -used:]

    means = {metric: {} for metric in METRICS}
    counts = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for component in components:
            means[metric][component], counts[metric][component] = _mean_and_count(
                per_example[metric][component]
            )
    return means, counts, per_example, np.arange(-width, 0)


def save_language_results(
    output_dir: Path,
    means: dict,
    counts: dict,
    per_example: dict,
    relative_positions: np.ndarray,
    results: list[dict],
    metadata: dict,
) -> None:
    """Save per-example arrays, aggregate arrays, CSV, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = {
        f"{component}_{metric}": per_example[metric][component]
        for metric in METRICS
        for component in per_example[metric]
    }
    np.savez_compressed(
        output_dir / "per_example_scores.npz",
        **archive,
        relative_positions=relative_positions,
        sample_ids=np.asarray([result["sample_id"] for result in results]),
    )
    for metric in METRICS:
        for component in per_example[metric]:
            np.save(
                output_dir / f"{component}_{metric}_mean.npy",
                means[metric][component],
            )
            np.save(
                output_dir / f"{component}_{metric}_count.npy",
                counts[metric][component],
            )

    with (output_dir / "mean_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "component",
                "layer",
                "relative_position",
                "head",
                "mean_patched_ld",
                "mean_delta_ld",
                "mean_recovery",
                "num_examples",
            ]
        )
        for component in per_example["recovery"]:
            patched_values = means["patched_ld"][component]
            delta_values = means["delta_ld"][component]
            recovery_values = means["recovery"][component]
            count_values = counts["recovery"][component]
            if component == "head_out":
                for layer in range(delta_values.shape[0]):
                    for position_index, position in enumerate(relative_positions):
                        for head in range(delta_values.shape[2]):
                            writer.writerow(
                                [
                                    component,
                                    layer,
                                    int(position),
                                    head,
                                    float(patched_values[layer, position_index, head]),
                                    float(delta_values[layer, position_index, head]),
                                    float(recovery_values[layer, position_index, head]),
                                    int(count_values[layer, position_index, head]),
                                ]
                            )
            else:
                for layer in range(delta_values.shape[0]):
                    for position_index, position in enumerate(relative_positions):
                        writer.writerow(
                            [
                                component,
                                layer,
                                int(position),
                                "",
                                float(patched_values[layer, position_index]),
                                float(delta_values[layer, position_index]),
                                float(recovery_values[layer, position_index]),
                                int(count_values[layer, position_index]),
                            ]
                        )

    metadata = {
        **metadata,
        "examples": [
            {
                key: result[key]
                for key in (
                    "sample_id",
                    "source_sample_id",
                    "sequence_length",
                    "target_clean_ld",
                    "target_corrupted_ld",
                    "denominator",
                )
            }
            for result in results
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

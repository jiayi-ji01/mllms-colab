"""Aggregation and structured serialization for patching experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mllms.interpretability.activation_patching.interventions import ALL_COMPONENTS
from mllms.interpretability.activation_patching.metrics import METRICS


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
    width = max(result["position_count"] for result in results)
    if max_positions is not None:
        width = min(width, max_positions)
    count = len(results)
    per_example = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for component in ALL_COMPONENTS:
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
            for component in ALL_COMPONENTS:
                per_example[metric][component][index, :, -used:] = result[
                    "scores"
                ][metric][component][:, -used:]

    means = {metric: {} for metric in METRICS}
    counts = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for component in ALL_COMPONENTS:
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
        for component in ALL_COMPONENTS
    }
    np.savez_compressed(
        output_dir / "per_example_scores.npz",
        **archive,
        relative_positions=relative_positions,
        sample_ids=np.asarray([result["sample_id"] for result in results]),
    )
    for metric in METRICS:
        for component in ALL_COMPONENTS:
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
                "mean_delta_ld",
                "mean_recovery",
                "num_examples",
            ]
        )
        for component in ALL_COMPONENTS:
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
                    "sequence_length",
                    "clean_ld",
                    "corrupted_ld",
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

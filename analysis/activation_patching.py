"""Patch clean SVA activations into corrupted prompts at every model site."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.sva_utils import (
    LANGUAGES,
    final_logit_difference,
    map_pair,
    read_pairs,
    score_language_pairs,
    select_device,
)
from data_lib.cloned import ClonedMapper
from model.config import GPTConfig
from model.model import GPT
from tokenizer.tokenizer import load_tokenizer


COMPONENTS = ("resid_post", "attn_out", "mlp_out")
ALL_COMPONENTS = (*COMPONENTS, "head_out")
METRICS = ("delta_ld", "recovery")


def _site_scores(
    patched_ld: torch.Tensor,
    corrupted_ld: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_ld = patched_ld - corrupted_ld
    return delta_ld, delta_ld / denominator


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
        [example["clean_answer_id"] for example in examples], device=device
    )
    corrupted_answers = torch.tensor(
        [example["corrupted_answer_id"] for example in examples], device=device
    )

    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupted_logits, _ = model(corrupted)
    clean_ld = final_logit_difference(
        clean_logits, clean_answers, corrupted_answers
    )
    corrupted_ld = final_logit_difference(
        corrupted_logits, clean_answers, corrupted_answers
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

    # Components have shape [batch, token, d_model]. Each expanded row patches
    # exactly one (example, token) site from the matching clean activation.
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
                    patched_logits, _ = model(corrupted[example_indices])
                patched_ld = final_logit_difference(
                    patched_logits,
                    clean_answers[example_indices],
                    corrupted_answers[example_indices],
                )
                delta_ld, recovery = _site_scores(
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

        # Head outputs have shape [batch, head, token, head_dim].
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
                patched_logits, _ = model(corrupted[example_indices])
            patched_ld = final_logit_difference(
                patched_logits,
                clean_answers[example_indices],
                corrupted_answers[example_indices],
            )
            delta_ld, recovery = _site_scores(
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

    results = []
    for index, example in enumerate(examples):
        results.append(
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
        )
    return results


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
        for component in COMPONENTS:
            per_example[metric][component] = np.full(
                (count, n_layers, width), np.nan, dtype=np.float32
            )
        per_example[metric]["head_out"] = np.full(
            (count, n_layers, width, n_heads), np.nan, dtype=np.float32
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


def plot_heatmaps(
    means: dict[str, np.ndarray],
    relative_positions: np.ndarray,
    metric: str,
    output_path: Path,
) -> None:
    """Plot token heatmaps and the final-token attention-head heatmap."""
    titles = {
        "resid_post": "Residual Stream",
        "attn_out": "Attention Output",
        "mlp_out": "MLP Output",
        "head_out": "Attention Heads (final token)",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, component in zip(axes.flat, ALL_COMPONENTS):
        full_values = means[component]
        values = full_values[:, -1] if component == "head_out" else full_values
        finite = np.abs(values[np.isfinite(values)])
        limit = float(np.percentile(finite, 95)) if finite.size else 1.0
        limit = max(limit, 1e-6)
        image = axis.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(titles[component])
        axis.set_ylabel("Layer")
        if component == "head_out":
            axis.set_xlabel("Head")
            axis.set_xticks(np.arange(values.shape[1]))
        else:
            axis.set_xlabel("Token position relative to prompt end")
            ticks = np.unique(
                np.linspace(
                    0,
                    len(relative_positions) - 1,
                    min(8, len(relative_positions)),
                ).astype(int)
            )
            axis.set_xticks(ticks, relative_positions[ticks])
        figure.colorbar(image, ax=axis, label=f"Mean {metric}")
    figure.suptitle(f"SVA Activation Patching: {metric}", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_language_results(
    output_dir: Path,
    means: dict,
    counts: dict,
    per_example: dict,
    relative_positions: np.ndarray,
    results: list[dict],
    metadata: dict,
) -> None:
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
                                    float(
                                        recovery_values[layer, position_index, head]
                                    ),
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

    metadata["examples"] = [
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
    ]
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for metric in METRICS:
        plot_heatmaps(
            means[metric],
            relative_positions,
            metric,
            output_dir / f"activation_patching_{metric}.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/babylm_tokenizer/tokenizer.model"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs/causalgym_sva/sanity_pairs.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/causalgym_patching")
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--language", choices=("original", "clone", "both"), default="both"
    )
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--example-batch-size", type=int, default=8)
    parser.add_argument("--intervention-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-positions",
        type=int,
        help="Patch only the final N prompt tokens (default: every token).",
    )
    parser.add_argument("--sanity-margin", type=float, default=0.0)
    parser.add_argument("--denominator-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("max_examples", "example_batch_size", "intervention_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        raise ValueError("max-positions must be positive")
    if args.sanity_margin < 0 or args.denominator_epsilon <= 0:
        raise ValueError("sanity-margin must be non-negative and epsilon positive")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer)
    mapper = ClonedMapper(
        original_vocab_size=tokenizer.vocab_size(),
        pad_id=tokenizer.pad_id(),
    )
    if mapper.model_vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")

    records = read_pairs(args.data)
    # One joint filter keeps the original and clone experiments sample-identical.
    baseline_scores = {}
    baseline_rejected = []
    for language, language_id in LANGUAGES.items():
        scores, rejected = score_language_pairs(
            model,
            records,
            mapper,
            language_id,
            device,
            args.example_batch_size,
        )
        baseline_scores[language] = scores
        baseline_rejected.extend(
            {"language": language, **record} for record in rejected
        )

    sanity_records = []
    sanity_rejected = []
    for record in records:
        sample_id = str(record["sample_id"])
        if any(sample_id not in baseline_scores[name] for name in LANGUAGES):
            continue
        scores = {name: baseline_scores[name][sample_id] for name in LANGUAGES}
        passes = all(
            score["clean_ld"] > args.sanity_margin
            and score["corrupted_ld"] < -args.sanity_margin
            and abs(score["clean_ld"] - score["corrupted_ld"])
            >= args.denominator_epsilon
            for score in scores.values()
        )
        if passes:
            sanity_records.append(record)
        else:
            sanity_rejected.append({"sample_id": sample_id, "scores": scores})

    # Alternate number conditions so a limited formal run stays as balanced as
    # the available sanity-passed pool permits.
    by_type = {
        clean_type: [
            record
            for record in sanity_records
            if record.get("clean_type") == clean_type
        ]
        for clean_type in ("singular", "plural")
    }
    accepted_records = []
    offsets = {clean_type: 0 for clean_type in by_type}
    while len(accepted_records) < min(args.max_examples, len(sanity_records)):
        made_progress = False
        for clean_type in ("singular", "plural"):
            offset = offsets[clean_type]
            if offset >= len(by_type[clean_type]):
                continue
            accepted_records.append(by_type[clean_type][offset])
            offsets[clean_type] += 1
            made_progress = True
            if len(accepted_records) >= args.max_examples:
                break
        if not made_progress:
            break
    if not accepted_records:
        raise ValueError("No pair passed the joint original/clone SVA sanity check")

    requested_languages = (
        tuple(LANGUAGES) if args.language == "both" else (args.language,)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_sample_ids = [str(record["sample_id"]) for record in accepted_records]
    root_metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "data": str(args.data),
        "device": str(device),
        "languages": list(requested_languages),
        "num_input_pairs": len(records),
        "num_joint_sanity_pairs_available": len(sanity_records),
        "num_patched_pairs": len(accepted_records),
        "patched_clean_type_counts": {
            clean_type: sum(
                record.get("clean_type") == clean_type
                for record in accepted_records
            )
            for clean_type in ("singular", "plural")
        },
        "num_baseline_rejected": len(baseline_rejected),
        "num_sanity_rejected": len(sanity_rejected),
        "sample_ids": common_sample_ids,
        "sanity_definition": (
            "clean_ld > margin and corrupted_ld < -margin in both languages"
        ),
        "ld": "logit(clean_answer) - logit(corrupted_answer)",
        "delta_ld": "LD_patched - LD_corrupted",
        "recovery": (
            "(LD_patched - LD_corrupted) / "
            "(LD_clean - LD_corrupted)"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(root_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for language in requested_languages:
        mapped = [
            map_pair(
                record,
                mapper,
                LANGUAGES[language],
                model_config.block_size,
            )
            for record in accepted_records
        ]
        grouped: dict[int, list[dict]] = defaultdict(list)
        for example in mapped:
            grouped[len(example["clean_ids"])].append(example)

        results = []
        completed = 0
        for sequence_length in sorted(grouped):
            group = grouped[sequence_length]
            for start in range(0, len(group), args.example_batch_size):
                batch = group[start : start + args.example_batch_size]
                results.extend(
                    patch_batch(
                        model,
                        batch,
                        device,
                        args.max_positions,
                        args.intervention_batch_size,
                    )
                )
                completed += len(batch)
                print(
                    f"{language:8s} | {completed:4d}/{len(mapped)} pairs | "
                    f"sequence length {sequence_length}"
                )

        # Restore the common source order after efficient length-grouped batches.
        by_id = {result["sample_id"]: result for result in results}
        results = [by_id[sample_id] for sample_id in common_sample_ids]
        means, counts, per_example, relative_positions = aggregate_results(
            results,
            model_config.n_layers,
            model_config.n_heads,
            args.max_positions,
        )
        language_dir = args.output_dir / language
        save_language_results(
            language_dir,
            means,
            counts,
            per_example,
            relative_positions,
            results,
            {
                **root_metadata,
                "language": language,
                "n_layers": model_config.n_layers,
                "n_heads": model_config.n_heads,
                "max_positions": args.max_positions,
            },
        )
        print(f"{language} results: {language_dir}")

    print(
        f"Common sanity-passed pairs: {len(sanity_records):,}; "
        f"patched: {len(accepted_records):,}"
    )
    print(f"Patching results: {args.output_dir}")


if __name__ == "__main__":
    main()

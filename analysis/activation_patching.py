"""Run activation patching on clean/corrupted SVA prompt pairs.

Each JSONL row must contain ``clean_answer_id`` and
``corrupted_answer_id`` plus either:

1. ``clean_sentence`` and ``corrupted_sentence`` (prompt strings), or
2. ``clean_input_ids`` and ``corrupted_input_ids`` (base tokenizer IDs).

String prompts are SentencePiece-encoded and prefixed with EOS, matching the
project's BLiMP evaluation convention. ID prompts are used exactly as given.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_lib.cloned import ClonedMapper
from model.config import GPTConfig
from model.model import GPT
from tokenizer.tokenizer import load_tokenizer


COMPONENTS = ("resid_post", "attn_out", "mlp_out")


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def load_examples(
    path: Path,
    tokenizer,
    mapper: ClonedMapper,
    language: int,
    block_size: int,
    max_examples: int | None,
) -> tuple[list[dict], list[dict]]:
    """Load valid, position-aligned prompt pairs and report rejected rows."""
    examples = []
    rejected = []

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", line_number))
            try:
                if "clean_input_ids" in record:
                    clean_ids = list(map(int, record["clean_input_ids"]))
                    corrupted_ids = list(map(int, record["corrupted_input_ids"]))
                else:
                    clean_ids = [
                        tokenizer.eos_id(),
                        *tokenizer.encode(record["clean_sentence"], out_type=int),
                    ]
                    corrupted_ids = [
                        tokenizer.eos_id(),
                        *tokenizer.encode(record["corrupted_sentence"], out_type=int),
                    ]

                clean_answer_id = int(record["clean_answer_id"])
                corrupted_answer_id = int(record["corrupted_answer_id"])
                if len(clean_ids) != len(corrupted_ids):
                    raise ValueError("clean/corrupted token lengths differ")
                if not clean_ids:
                    raise ValueError("prompts are empty")
                if len(clean_ids) > block_size:
                    raise ValueError("prompt exceeds checkpoint block_size")
                base_ids = clean_ids + corrupted_ids + [
                    clean_answer_id,
                    corrupted_answer_id,
                ]
                if min(base_ids) < 0 or max(base_ids) >= mapper.original_vocab_size:
                    raise ValueError("IDs must be base SentencePiece vocabulary IDs")

                def map_language(ids: list[int]) -> list[int]:
                    return mapper.map_to_language(
                        torch.tensor(ids, dtype=torch.long), language
                    ).tolist()

                examples.append(
                    {
                        "sample_id": sample_id,
                        "clean_ids": map_language(clean_ids),
                        "corrupted_ids": map_language(corrupted_ids),
                        "clean_answer_id": map_language([clean_answer_id])[0],
                        "corrupted_answer_id": map_language(
                            [corrupted_answer_id]
                        )[0],
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                rejected.append({"sample_id": sample_id, "reason": str(error)})

            if max_examples is not None and len(examples) >= max_examples:
                break

    return examples, rejected


def logit_diff(
    logits: torch.Tensor,
    clean_answer_id: int,
    corrupted_answer_id: int,
) -> torch.Tensor:
    """Return clean-answer logit minus corrupted-answer logit per batch row."""
    final_logits = logits[:, -1]
    return (
        final_logits[:, clean_answer_id]
        - final_logits[:, corrupted_answer_id]
    )


@torch.inference_mode()
def patch_one_example(
    model: GPT,
    example: dict,
    device: torch.device,
    max_positions: int | None,
    denominator_epsilon: float,
) -> dict:
    """Patch all requested sites for one prompt pair using batched interventions."""
    clean = torch.tensor([example["clean_ids"]], device=device)
    corrupted = torch.tensor([example["corrupted_ids"]], device=device)
    clean_answer_id = example["clean_answer_id"]
    corrupted_answer_id = example["corrupted_answer_id"]

    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupted_logits, _ = model(corrupted)
    clean_diff = logit_diff(
        clean_logits, clean_answer_id, corrupted_answer_id
    ).item()
    corrupted_diff = logit_diff(
        corrupted_logits, clean_answer_id, corrupted_answer_id
    ).item()
    denominator = clean_diff - corrupted_diff
    if abs(denominator) < denominator_epsilon:
        raise ValueError(f"patch denominator is too small ({denominator:.3e})")

    sequence_length = clean.size(1)
    first_position = (
        0 if max_positions is None else max(0, sequence_length - max_positions)
    )
    positions = torch.arange(first_position, sequence_length, device=device)
    position_count = positions.numel()
    row_indices = torch.arange(position_count, device=device)
    scores: dict[str, np.ndarray] = {}

    for component in COMPONENTS:
        component_scores = torch.empty(
            model.config.n_layers, position_count, device=device
        )
        for layer in range(model.config.n_layers):
            name = f"blocks.{layer}.{component}"
            clean_activation = clean_cache[name]

            def patch_positions(
                activation: torch.Tensor,
                clean_activation: torch.Tensor = clean_activation,
            ) -> torch.Tensor:
                patched = activation.clone()
                patched[row_indices, positions] = clean_activation[0, positions]
                return patched

            with model.hooks({name: patch_positions}):
                patched_logits, _ = model(
                    corrupted.expand(position_count, -1)
                )
            patched_diff = logit_diff(
                patched_logits, clean_answer_id, corrupted_answer_id
            )
            component_scores[layer] = (
                patched_diff - corrupted_diff
            ) / denominator
        scores[component] = component_scores.cpu().numpy()

    head_count = model.config.n_heads
    head_rows = torch.arange(head_count, device=device)
    head_scores = torch.empty(
        model.config.n_layers, head_count, device=device
    )
    for layer in range(model.config.n_layers):
        name = f"blocks.{layer}.head_out"
        clean_activation = clean_cache[name]

        def patch_heads(
            activation: torch.Tensor,
            clean_activation: torch.Tensor = clean_activation,
        ) -> torch.Tensor:
            patched = activation.clone()
            patched[head_rows, head_rows, -1] = clean_activation[0, head_rows, -1]
            return patched

        with model.hooks({name: patch_heads}):
            patched_logits, _ = model(corrupted.expand(head_count, -1))
        patched_diff = logit_diff(
            patched_logits, clean_answer_id, corrupted_answer_id
        )
        head_scores[layer] = (patched_diff - corrupted_diff) / denominator

    return {
        **scores,
        "head_out": head_scores.cpu().numpy(),
        "sequence_length": sequence_length,
        "position_count": position_count,
        "clean_diff": clean_diff,
        "corrupted_diff": corrupted_diff,
        "denominator": denominator,
    }


def aggregate_results(
    results: list[dict],
    n_layers: int,
    n_heads: int,
    max_positions: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Right-align variable prompt lengths and compute nan-aware means."""
    width = max(result["position_count"] for result in results)
    if max_positions is not None:
        width = min(width, max_positions)
    count = len(results)
    per_example = {
        component: np.full((count, n_layers, width), np.nan, dtype=np.float32)
        for component in COMPONENTS
    }
    per_example["head_out"] = np.empty(
        (count, n_layers, n_heads), dtype=np.float32
    )

    for index, result in enumerate(results):
        used = min(width, result["position_count"])
        for component in COMPONENTS:
            per_example[component][index, :, -used:] = result[component][:, -used:]
        per_example["head_out"][index] = result["head_out"]

    means = {
        component: np.nanmean(values, axis=0)
        for component, values in per_example.items()
    }
    relative_positions = np.arange(-width, 0)
    return means, per_example, relative_positions


def save_results(
    output_dir: Path,
    means: dict[str, np.ndarray],
    per_example: dict[str, np.ndarray],
    relative_positions: np.ndarray,
    examples: list[dict],
    results: list[dict],
    rejected: list[dict],
    args: argparse.Namespace,
    checkpoint_step: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in means.items():
        np.save(output_dir / f"{name}_mean.npy", values)
    np.savez_compressed(
        output_dir / "per_example_scores.npz",
        **per_example,
        relative_positions=relative_positions,
        sample_ids=np.array([example["sample_id"] for example in examples]),
    )

    with (output_dir / "mean_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["component", "layer", "position_or_head", "patch_score"])
        for component in COMPONENTS:
            for layer, row in enumerate(means[component]):
                for position, score in zip(relative_positions, row):
                    writer.writerow([component, layer, int(position), float(score)])
        for layer, row in enumerate(means["head_out"]):
            for head, score in enumerate(row):
                writer.writerow(["head_out", layer, head, float(score)])

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "data": str(args.data),
        "device": str(device),
        "language": args.language,
        "num_examples": len(results),
        "num_rejected": len(rejected),
        "relative_positions": relative_positions.tolist(),
        "metric": "logit(clean_answer) - logit(corrupted_answer)",
        "patch_score": (
            "(patched_diff - corrupted_diff) / "
            "(clean_diff - corrupted_diff)"
        ),
        "examples": [
            {
                "sample_id": example["sample_id"],
                "sequence_length": result["sequence_length"],
                "clean_diff": result["clean_diff"],
                "corrupted_diff": result["corrupted_diff"],
                "denominator": result["denominator"],
            }
            for example, result in zip(examples, results)
        ],
        "rejected": rejected,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_heatmaps(
    means: dict[str, np.ndarray],
    relative_positions: np.ndarray,
    output_path: Path,
) -> None:
    """Save four activation-patching heatmaps in one figure."""
    titles = {
        "resid_post": "Residual Stream",
        "attn_out": "Attention Output",
        "mlp_out": "MLP Output",
        "head_out": "Attention Heads at Final Position",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    for axis, component in zip(axes.flat, (*COMPONENTS, "head_out")):
        values = means[component]
        finite = np.abs(values[np.isfinite(values)])
        limit = max(1.0, float(np.percentile(finite, 95))) if finite.size else 1.0
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
        axis.set_yticks(np.arange(values.shape[0]))

        if component == "head_out":
            axis.set_xlabel("Head")
            axis.set_xticks(np.arange(values.shape[1]))
        else:
            axis.set_xlabel("Token Position Relative to Final Prompt Token")
            tick_count = min(8, len(relative_positions))
            tick_indices = np.unique(
                np.linspace(0, len(relative_positions) - 1, tick_count).astype(int)
            )
            axis.set_xticks(tick_indices)
            axis.set_xticklabels(relative_positions[tick_indices])
        figure.colorbar(image, ax=axis, label="Mean Patch Score")

    figure.suptitle("Subject–Verb Agreement Activation Patching", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/training/best.pt")
    )
    parser.add_argument("--data", type=Path, default=Path("data/sva_pairs.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/activation_patching")
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--language", choices=("original", "clone"), default="original"
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--max-positions",
        type=int,
        help="Patch only the last N prompt positions (default: every position).",
    )
    parser.add_argument("--denominator-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max-examples must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        raise ValueError("max-positions must be positive")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = load_tokenizer()
    mapper = ClonedMapper(
        original_vocab_size=tokenizer.vocab_size(),
        pad_id=tokenizer.pad_id(),
    )
    if mapper.model_vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint and cloned tokenizer vocabulary sizes differ")

    language_id = 0 if args.language == "original" else 1
    examples, rejected = load_examples(
        args.data,
        tokenizer,
        mapper,
        language_id,
        model_config.block_size,
        args.max_examples,
    )
    if not examples:
        raise ValueError(f"No valid SVA pairs found in {args.data}")

    results = []
    accepted_examples = []
    for index, example in enumerate(examples, start=1):
        try:
            result = patch_one_example(
                model,
                example,
                device,
                args.max_positions,
                args.denominator_epsilon,
            )
        except ValueError as error:
            rejected.append(
                {"sample_id": example["sample_id"], "reason": str(error)}
            )
            continue
        results.append(result)
        accepted_examples.append(example)
        print(
            f"[{index}/{len(examples)}] {example['sample_id']} | "
            f"clean={result['clean_diff']:+.3f} | "
            f"corrupted={result['corrupted_diff']:+.3f}"
        )

    if not results:
        raise ValueError("All pairs had an undefined patching denominator")

    means, per_example, relative_positions = aggregate_results(
        results,
        model_config.n_layers,
        model_config.n_heads,
        args.max_positions,
    )
    save_results(
        args.output_dir,
        means,
        per_example,
        relative_positions,
        accepted_examples,
        results,
        rejected,
        args,
        int(checkpoint["step"]),
        device,
    )
    figure_path = args.output_dir / "activation_patching_heatmaps.png"
    plot_heatmaps(means, relative_positions, figure_path)

    print(f"\nDevice: {device}")
    print(f"Accepted / rejected: {len(results)} / {len(rejected)}")
    print(f"Raw arrays: {args.output_dir / 'per_example_scores.npz'}")
    print(f"Mean CSV: {args.output_dir / 'mean_scores.csv'}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()

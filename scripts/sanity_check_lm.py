"""Run next-token and validation-loss sanity checks for a trained GPT.

This script is evaluation-only: it never updates model parameters or optimizer
state. Run it from the repository root after installing the project.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Also support `python scripts/sanity_check_lm.py` without an editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_lib.cloned import ClonedMapper, TokenStream  # noqa: E402
from model.config import GPTConfig  # noqa: E402
from model.model import GPT  # noqa: E402
from tokenizer.tokenizer import load_tokenizer  # noqa: E402
from train import (  # noqa: E402
    TrainingConfig,
    _autocast_context,
    _load_config_file,
    _select_device,
    _select_precision,
)


DEFAULT_PREFIXES = [
    "The dog is",
    "She went to the",
    "There are many",
    "The boys are",
    "He likes to",
    "The cat sat on the",
    "I want to eat",
    "They were playing",
    "A little girl found",
    "The teacher told the",
    "My friend has",
    "We need to",
    "The sun is",
    "The children have",
    "It was a",
    "Once upon a time",
    "The woman opened the",
    "His father works",
    "The birds are flying",
    "This book is about",
    "The car stopped at",
    "You can see",
    "The baby started to",
    "Our house has",
    "The students were",
    "After dinner, we",
    "In the morning, she",
    "The man looked at",
    "Her mother gave her",
    "The water was",
    "These flowers are",
    "One day, the boy",
    "The dogs have",
    "I think that",
    "The family went",
    "The doctor said",
    "A large tree stood",
    "The people in the city",
    "When he arrived, the",
    "She could not",
    "The farmer had",
    "The room was full of",
    "They decided to",
    "The young woman was",
    "Everyone wanted to",
    "The train arrived at",
    "Because it was raining",
    "The story begins with",
    "The two friends walked",
    "There is a",
]

IGNORE_INDEX = -100


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Checkpoint must contain model and model_config")
    return checkpoint


def _resolve_training_config(
    config_path: Path | None,
    checkpoint: dict[str, Any],
) -> TrainingConfig:
    if config_path is not None:
        return TrainingConfig(**_load_config_file(config_path))
    values = checkpoint.get("training_config")
    if not isinstance(values, dict):
        raise ValueError("Use --config because checkpoint has no training_config")
    return TrainingConfig(**values)


def _validate_configuration(
    training_config: TrainingConfig,
    checkpoint_model_config: GPTConfig,
    tokenizer,
) -> None:
    expected = GPTConfig(
        vocab_size=2 * training_config.vocab_size,
        block_size=training_config.block_size,
        d_model=training_config.d_model,
        n_heads=training_config.n_heads,
        n_layers=training_config.n_layers,
        d_ff=training_config.d_ff,
        dropout=training_config.dropout,
        bias=training_config.bias,
    )
    if asdict(expected) != asdict(checkpoint_model_config):
        raise ValueError(
            "Config architecture does not match checkpoint model_config:\n"
            f"config={asdict(expected)}\n"
            f"checkpoint={asdict(checkpoint_model_config)}"
        )
    if tokenizer.vocab_size() != training_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary is {tokenizer.vocab_size()}, but config "
            f"requires {training_config.vocab_size}"
        )


def _read_prefixes(path: Path | None, limit: int) -> list[str]:
    if limit <= 0:
        raise ValueError("--num-prefixes must be positive")
    if path is None:
        prefixes = DEFAULT_PREFIXES
    else:
        if not path.is_file():
            raise FileNotFoundError(f"Prefix file not found: {path}")
        prefixes = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not prefixes:
        raise ValueError("No non-empty prefixes were found")
    return prefixes[:limit]


def _token_text(tokenizer, model_token_id: int, base_vocab_size: int) -> tuple[str, str]:
    base_token_id = model_token_id % base_vocab_size
    piece = tokenizer.id_to_piece(base_token_id)
    decoded = tokenizer.decode([base_token_id])
    return piece, decoded


@torch.no_grad()
def predict_next_tokens(
    model: GPT,
    tokenizer,
    clone_mapper: ClonedMapper,
    prefixes: list[str],
    languages: list[str],
    top_k: int,
    device: torch.device,
    precision: str,
) -> list[dict[str, Any]]:
    """Return true full-vocabulary top-k probabilities for each prefix."""
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    model.eval()
    records: list[dict[str, Any]] = []

    for prefix_index, prefix in enumerate(prefixes):
        base_ids = tokenizer.encode(prefix, out_type=int)
        if not base_ids:
            raise ValueError(f"Prefix produced no tokens: {prefix!r}")
        base_ids = base_ids[-model.config.block_size :]
        base_tensor = torch.tensor(base_ids, dtype=torch.long)

        for language in languages:
            language_id = 0 if language == "original" else 1
            input_ids = clone_mapper.map_to_language(
                base_tensor,
                language_id,
            ).unsqueeze(0).to(device)
            with _autocast_context(device, precision):
                logits, _ = model(input_ids)
            probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
            values, token_ids = torch.topk(
                probabilities,
                k=min(top_k, model.config.vocab_size),
            )

            for rank, (token_id, probability) in enumerate(
                zip(token_ids.tolist(), values.tolist()),
                start=1,
            ):
                piece, decoded = _token_text(
                    tokenizer,
                    token_id,
                    clone_mapper.original_vocab_size,
                )
                records.append(
                    {
                        "prefix_index": prefix_index,
                        "prefix": prefix,
                        "language": language,
                        "rank": rank,
                        "predicted_token": piece,
                        "decoded_token": decoded,
                        "predicted_token_id": token_id,
                        "predicted_token_space": (
                            "original"
                            if token_id < clone_mapper.original_vocab_size
                            else "clone"
                        ),
                        "probability": probability,
                    }
                )
    return records


def _write_prediction_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "prefix",
        "language",
        "rank",
        "predicted_token",
        "decoded_token",
        "predicted_token_id",
        "predicted_token_space",
        "probability",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})


def _print_prediction_examples(
    records: list[dict[str, Any]],
    num_prefixes: int = 10,
) -> None:
    print("\nNext-token predictions (first 10 prefixes)")
    shown = sorted({int(record["prefix_index"]) for record in records})[
        :num_prefixes
    ]
    for prefix_index in shown:
        matching = [
            record
            for record in records
            if record["prefix_index"] == prefix_index
        ]
        print(f"\n{matching[0]['prefix']!r}")
        for language in dict.fromkeys(record["language"] for record in matching):
            predictions = [
                record for record in matching if record["language"] == language
            ]
            formatted = ", ".join(
                f"{record['predicted_token']} ({record['probability']:.3%})"
                for record in predictions
            )
            print(f"  {language:>8}: {formatted}")


def _full_validation_batches(
    validation_stream: TokenStream,
    clone_mapper: ClonedMapper,
    language_id: int,
    batch_size: int,
    block_size: int,
):
    """Yield every adjacent validation-token pair exactly once."""
    if batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    num_pairs = len(validation_stream.tokens) - 1
    if num_pairs <= 0:
        raise ValueError("Validation token stream needs at least two tokens")

    starts = list(range(0, num_pairs, block_size))
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        base_inputs = torch.full(
            (len(batch_starts), block_size),
            clone_mapper.pad_id,
            dtype=torch.long,
        )
        base_targets = torch.full_like(base_inputs, clone_mapper.pad_id)
        valid_mask = torch.zeros_like(base_inputs, dtype=torch.bool)

        for row, start in enumerate(batch_starts):
            length = min(block_size, num_pairs - start)
            base_inputs[row, :length] = torch.from_numpy(
                np.array(
                    validation_stream.tokens[start : start + length],
                    dtype=np.int64,
                    copy=True,
                )
            )
            base_targets[row, :length] = torch.from_numpy(
                np.array(
                    validation_stream.tokens[start + 1 : start + length + 1],
                    dtype=np.int64,
                    copy=True,
                )
            )
            valid_mask[row, :length] = True

        input_ids = clone_mapper.map_to_language(base_inputs, language_id)
        targets = clone_mapper.map_to_language(base_targets, language_id)
        targets = targets.masked_fill(~valid_mask, IGNORE_INDEX)
        yield input_ids, targets, int(valid_mask.sum().item())


def _sampled_validation_batches(
    validation_stream: TokenStream,
    clone_mapper: ClonedMapper,
    language_id: int,
    batch_size: int,
    block_size: int,
    num_batches: int,
    seed: int,
):
    """Yield the deterministic validation sample used during training."""
    if num_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    for _ in range(num_batches):
        input_ids, targets, _ = validation_stream.get_batch(
            batch_size=batch_size,
            block_size=block_size,
            device="cpu",
            cloned_mapper=clone_mapper,
            language=language_id,
            generator=generator,
        )
        yield input_ids, targets, input_ids.numel()


@torch.no_grad()
def validation_loss(
    model: GPT,
    validation_stream: TokenStream,
    clone_mapper: ClonedMapper,
    language: str,
    batch_size: int,
    device: torch.device,
    precision: str,
    eval_batches: int | None,
    seed: int,
) -> dict[str, float | int]:
    """Compute token-weighted cross-entropy on sampled or full validation."""
    model.eval()
    language_id = 0 if language == "original" else 1
    total_negative_log_likelihood = 0.0
    total_valid_positions = 0

    if eval_batches is None:
        batches = _full_validation_batches(
            validation_stream,
            clone_mapper,
            language_id,
            batch_size,
            model.config.block_size,
        )
        num_chunks = math.ceil(
            (len(validation_stream.tokens) - 1) / model.config.block_size
        )
        progress_total = math.ceil(num_chunks / batch_size)
    else:
        batches = _sampled_validation_batches(
            validation_stream,
            clone_mapper,
            language_id,
            batch_size,
            model.config.block_size,
            eval_batches,
            seed,
        )
        progress_total = eval_batches
    for input_ids, targets, valid_positions in tqdm(
        batches,
        total=progress_total,
        desc=f"validation/{language}",
        leave=False,
    ):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with _autocast_context(device, precision):
            logits, _ = model(input_ids)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        total_negative_log_likelihood += float(loss_sum.item())
        total_valid_positions += valid_positions

    average_loss = total_negative_log_likelihood / total_valid_positions
    perplexity = math.exp(average_loss) if average_loss < 709 else float("inf")
    return {
        "average_cross_entropy": average_loss,
        "perplexity": perplexity,
        "valid_next_token_positions": total_valid_positions,
    }


def _evaluate_languages(
    model: GPT,
    model_name: str,
    validation_stream: TokenStream,
    clone_mapper: ClonedMapper,
    languages: list[str],
    batch_size: int,
    device: torch.device,
    precision: str,
    eval_batches: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for language in languages:
        metrics = validation_loss(
            model,
            validation_stream,
            clone_mapper,
            language,
            batch_size,
            device,
            precision,
            eval_batches,
            seed,
        )
        rows.append({"model": model_name, "language": language, **metrics})
        print(
            f"{model_name:>18} / {language:<8} | "
            f"loss {metrics['average_cross_entropy']:.4f} | "
            f"ppl {metrics['perplexity']:.2f} | "
            f"positions {metrics['valid_next_token_positions']:,}"
        )

    total_positions = sum(row["valid_next_token_positions"] for row in rows)
    average_loss = sum(
        row["average_cross_entropy"] * row["valid_next_token_positions"]
        for row in rows
    ) / total_positions
    rows.append(
        {
            "model": model_name,
            "language": "average",
            "average_cross_entropy": average_loss,
            "perplexity": math.exp(average_loss),
            "valid_next_token_positions": total_positions,
        }
    )
    return rows


def _write_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "language",
        "average_cross_entropy",
        "perplexity",
        "valid_next_token_positions",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix-file", type=Path)
    parser.add_argument("--num-prefixes", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument(
        "--eval-batches",
        type=int,
        help="Validation batches; defaults to the training config.",
    )
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Evaluate every validation position instead of a fixed sample.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("original", "clone"),
        default=["original", "clone"],
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument("--random-seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    training_config = _resolve_training_config(args.config, checkpoint)
    model_config = GPTConfig(**checkpoint["model_config"])

    tokenizer_path = args.tokenizer_path or Path(training_config.tokenizer_path)
    data_dir = args.data_dir or Path(training_config.data_dir)
    output_dir = args.output_dir or args.checkpoint.parent / "sanity_check_lm"
    eval_batch_size = args.eval_batch_size or training_config.micro_batch_size
    if args.full_validation and args.eval_batches is not None:
        raise ValueError("Use --eval-batches or --full-validation, not both")
    eval_batches = (
        None
        if args.full_validation
        else args.eval_batches or training_config.eval_batches
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_path)
    _validate_configuration(training_config, model_config, tokenizer)
    clone_mapper = ClonedMapper(
        original_vocab_size=training_config.vocab_size,
        p_clone=training_config.p_clone,
        pad_id=tokenizer.pad_id(),
    )
    validation_stream = TokenStream(data_dir, "validation")
    prefixes = _read_prefixes(args.prefix_file, args.num_prefixes)

    device = _select_device(args.device)
    if args.precision == "auto":
        precision = _select_precision(device)
    else:
        precision = args.precision
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("bf16/fp16 sanity checks require CUDA; use fp32")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Validation data: {validation_stream.path}")
    print(f"Device / precision: {device} / {precision}")
    print(f"Prefixes: {len(prefixes)}; top-k: {args.top_k}")
    if eval_batches is None:
        print("Validation mode: full token stream")
    else:
        positions = eval_batches * eval_batch_size * model_config.block_size
        print(
            f"Validation mode: {eval_batches} deterministic batches "
            f"({positions:,} positions per language/model)"
        )

    trained_model = GPT(model_config)
    trained_model.load_state_dict(checkpoint["model"])
    trained_model.to(device).eval()

    prediction_records = predict_next_tokens(
        trained_model,
        tokenizer,
        clone_mapper,
        prefixes,
        args.languages,
        args.top_k,
        device,
        precision,
    )
    prediction_path = output_dir / "next_token_predictions.csv"
    _write_prediction_csv(prediction_path, prediction_records)
    _print_prediction_examples(prediction_records)

    print("\nValidation loss / perplexity")
    validation_rows = _evaluate_languages(
        trained_model,
        "trained",
        validation_stream,
        clone_mapper,
        args.languages,
        eval_batch_size,
        device,
        precision,
        eval_batches,
        training_config.seed,
    )
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.manual_seed(args.random_seed)
    random_model = GPT(model_config).to(device).eval()
    validation_rows.extend(
        _evaluate_languages(
            random_model,
            "random_initialization",
            validation_stream,
            clone_mapper,
            args.languages,
            eval_batch_size,
            device,
            precision,
            eval_batches,
            training_config.seed,
        )
    )

    validation_path = output_dir / "validation_loss_perplexity.csv"
    _write_validation_csv(validation_path, validation_rows)
    averages = {
        row["model"]: row
        for row in validation_rows
        if row["language"] == "average"
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "tokenizer": str(tokenizer_path),
        "validation_data": str(validation_stream.path),
        "device": str(device),
        "precision": precision,
        "model_config": asdict(model_config),
        "num_prefixes": len(prefixes),
        "top_k": args.top_k,
        "validation_mode": (
            "full" if eval_batches is None else "deterministic_sample"
        ),
        "eval_batches": eval_batches,
        "validation": validation_rows,
        "trained_minus_random_loss": (
            averages["trained"]["average_cross_entropy"]
            - averages["random_initialization"]["average_cross_entropy"]
        ),
        "random_to_trained_perplexity_ratio": (
            averages["random_initialization"]["perplexity"]
            / averages["trained"]["perplexity"]
        ),
    }
    summary_path = output_dir / "sanity_check_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSaved outputs")
    print(f"  Next-token predictions: {prediction_path}")
    print(f"  Validation comparison: {validation_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

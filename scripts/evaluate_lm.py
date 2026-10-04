"""Evaluate loss and perplexity in the original and cloned languages."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import load_model_checkpoint
from config import TrainingConfig, load_training_config
from model import GPT, GPTConfig
from runtime import autocast_context, select_device, select_precision
from token_data import ClonedMapper, TokenStream, load_tokenizer


IGNORE_INDEX = -100


def _full_validation_batches(
    validation_stream: TokenStream,
    clone_mapper: ClonedMapper,
    language_id: int,
    batch_size: int,
    block_size: int,
):
    if batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    num_pairs = len(validation_stream.tokens) - 1
    if num_pairs <= 0:
        raise ValueError("Validation token stream needs at least two tokens")
    starts = list(range(0, num_pairs, block_size))
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        base_inputs = torch.full(
            (len(batch_starts), block_size), clone_mapper.pad_id, dtype=torch.long
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
            validation_stream, clone_mapper, language_id, batch_size, model.config.block_size
        )
        chunks = math.ceil(
            (len(validation_stream.tokens) - 1) / model.config.block_size
        )
        progress_total = math.ceil(chunks / batch_size)
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
        with autocast_context(device, precision):
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
    return {
        "average_cross_entropy": average_loss,
        "perplexity": math.exp(average_loss) if average_loss < 709 else float("inf"),
        "valid_next_token_positions": total_valid_positions,
    }


def evaluate_languages(
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
    rows = []
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


def _resolve_training_config(
    config_path: Path | None, checkpoint: dict[str, Any]
) -> TrainingConfig:
    if config_path is not None:
        return load_training_config(config_path)
    values = checkpoint.get("training_config")
    if not isinstance(values, dict):
        raise ValueError("Use --config because checkpoint has no training_config")
    return TrainingConfig(**values)


def _validate_configuration(
    training_config: TrainingConfig,
    checkpoint_model_config: GPTConfig,
    tokenizer,
) -> None:
    expected = training_config.model_config()
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


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Held-out token stream used for loss/perplexity.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=("original", "clone"),
        default=["original", "clone"],
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint, trained_model, model_config = load_model_checkpoint(args.checkpoint, device)
    training_config = _resolve_training_config(args.config, checkpoint)
    tokenizer_path = args.tokenizer_path or Path(training_config.tokenizer_path)
    data_dir = args.data_dir or Path(training_config.data_dir)
    output_dir = args.output_dir or args.checkpoint.parent / "sanity_check_lm"
    eval_batch_size = args.eval_batch_size or training_config.micro_batch_size
    if args.full_validation and args.eval_batches is not None:
        raise ValueError("Use --eval-batches or --full-validation, not both")
    eval_batches = (
        None if args.full_validation else args.eval_batches or training_config.eval_batches
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_path)
    _validate_configuration(training_config, model_config, tokenizer)
    clone_mapper = ClonedMapper(
        original_vocab_size=training_config.vocab_size,
        p_clone=training_config.p_clone,
        pad_id=tokenizer.pad_id(),
    )
    evaluation_stream = TokenStream(data_dir, args.split)
    precision = select_precision(device) if args.precision == "auto" else args.precision
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("bf16/fp16 evaluation requires CUDA; use fp32")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Evaluation data ({args.split}): {evaluation_stream.path}")
    print(f"Device / precision: {device} / {precision}")
    print("\nValidation loss / perplexity")
    validation_rows = evaluate_languages(
        trained_model,
        "trained",
        evaluation_stream,
        clone_mapper,
        args.languages,
        eval_batch_size,
        device,
        precision,
        eval_batches,
        training_config.seed,
    )
    for row in validation_rows:
        if row["language"] != "average":
            print(
                f"{row['model']:>18} / {row['language']:<8} | "
                f"loss {row['average_cross_entropy']:.4f} | "
                f"ppl {row['perplexity']:.2f} | "
                f"positions {row['valid_next_token_positions']:,}"
            )

    validation_path = output_dir / f"{args.split}_loss_perplexity.csv"
    validation_fields = [
        "model", "language", "average_cross_entropy", "perplexity",
        "valid_next_token_positions",
    ]
    _write_csv(validation_path, validation_fields, validation_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "tokenizer": str(tokenizer_path),
        "evaluation_split": args.split,
        "evaluation_data": str(evaluation_stream.path),
        "validation_data": str(evaluation_stream.path),
        "device": str(device),
        "precision": precision,
        "model_config": asdict(model_config),
        "evaluation_mode": "full" if eval_batches is None else "deterministic_sample",
        "validation_mode": "full" if eval_batches is None else "deterministic_sample",
        "eval_batches": eval_batches,
        "evaluation": validation_rows,
        "validation": validation_rows,
    }
    summary_path = output_dir / "sanity_check_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.split.title()} comparison: {validation_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

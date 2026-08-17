"""Run next-token and validation-loss sanity checks for a trained GPT."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch

from mllms.data.cloned_language import ClonedMapper
from mllms.data.token_stream import TokenStream
from mllms.evaluation.language_model import (
    evaluate_languages,
    predict_next_tokens,
    read_prefixes,
)
from mllms.model.config import GPTConfig
from mllms.model.transformer import GPT
from mllms.runtime import select_device, select_precision
from mllms.tokenizer.sentencepiece import load_tokenizer
from mllms.training.config import TrainingConfig, load_training_config


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


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _print_prediction_examples(records: list[dict], num_prefixes: int = 10) -> None:
    print("\nNext-token predictions (first 10 prefixes)")
    shown = sorted({int(record["prefix_index"]) for record in records})[:num_prefixes]
    for prefix_index in shown:
        matching = [
            record for record in records if record["prefix_index"] == prefix_index
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
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--full-validation", action="store_true")
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
    validation_stream = TokenStream(data_dir, "validation")
    prefixes = read_prefixes(args.prefix_file, args.num_prefixes)
    device = select_device(args.device)
    precision = select_precision(device) if args.precision == "auto" else args.precision
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("bf16/fp16 sanity checks require CUDA; use fp32")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Validation data: {validation_stream.path}")
    print(f"Device / precision: {device} / {precision}")
    print(f"Prefixes: {len(prefixes)}; top-k: {args.top_k}")

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
    prediction_fields = [
        "prefix", "language", "rank", "predicted_token", "decoded_token",
        "predicted_token_id", "predicted_token_space", "probability",
    ]
    _write_csv(prediction_path, prediction_fields, prediction_records)
    _print_prediction_examples(prediction_records)

    print("\nValidation loss / perplexity")
    validation_rows = evaluate_languages(
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
        evaluate_languages(
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
    for row in validation_rows:
        if row["language"] != "average":
            print(
                f"{row['model']:>18} / {row['language']:<8} | "
                f"loss {row['average_cross_entropy']:.4f} | "
                f"ppl {row['perplexity']:.2f} | "
                f"positions {row['valid_next_token_positions']:,}"
            )

    validation_path = output_dir / "validation_loss_perplexity.csv"
    validation_fields = [
        "model", "language", "average_cross_entropy", "perplexity",
        "valid_next_token_positions",
    ]
    _write_csv(validation_path, validation_fields, validation_rows)
    averages = {
        row["model"]: row for row in validation_rows if row["language"] == "average"
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
        "validation_mode": "full" if eval_batches is None else "deterministic_sample",
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
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Next-token predictions: {prediction_path}")
    print(f"Validation comparison: {validation_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

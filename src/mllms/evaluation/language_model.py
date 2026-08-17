"""Reusable next-token and validation-loss evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mllms.data.cloned_language import ClonedMapper
from mllms.data.token_stream import TokenStream
from mllms.model.transformer import GPT
from mllms.runtime import autocast_context


DEFAULT_PREFIXES = [
    "The dog is", "She went to the", "There are many", "The boys are",
    "He likes to", "The cat sat on the", "I want to eat", "They were playing",
    "A little girl found", "The teacher told the", "My friend has", "We need to",
    "The sun is", "The children have", "It was a", "Once upon a time",
    "The woman opened the", "His father works", "The birds are flying",
    "This book is about", "The car stopped at", "You can see",
    "The baby started to", "Our house has", "The students were",
    "After dinner, we", "In the morning, she", "The man looked at",
    "Her mother gave her", "The water was", "These flowers are",
    "One day, the boy", "The dogs have", "I think that", "The family went",
    "The doctor said", "A large tree stood", "The people in the city",
    "When he arrived, the", "She could not", "The farmer had",
    "The room was full of", "They decided to", "The young woman was",
    "Everyone wanted to", "The train arrived at", "Because it was raining",
    "The story begins with", "The two friends walked", "There is a",
]
IGNORE_INDEX = -100


def read_prefixes(path: Path | None, limit: int) -> list[str]:
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
    return tokenizer.id_to_piece(base_token_id), tokenizer.decode([base_token_id])


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
        base_tensor = torch.tensor(
            base_ids[-model.config.block_size :], dtype=torch.long
        )
        for language in languages:
            language_id = 0 if language == "original" else 1
            input_ids = clone_mapper.map_to_language(
                base_tensor, language_id
            ).unsqueeze(0).to(device)
            with autocast_context(device, precision):
                logits, _ = model(input_ids)
            probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
            values, token_ids = torch.topk(
                probabilities, k=min(top_k, model.config.vocab_size)
            )
            for rank, (token_id, probability) in enumerate(
                zip(token_ids.tolist(), values.tolist()), start=1
            ):
                piece, decoded = _token_text(
                    tokenizer, token_id, clone_mapper.original_vocab_size
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

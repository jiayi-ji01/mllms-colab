"""Shared helpers for controlled SVA evaluation and activation patching."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import torch

from data_lib.cloned import ClonedMapper


LANGUAGES = {"original": 0, "clone": 1}


def select_device(name: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device."""
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


def read_pairs(path: Path, max_pairs: int | None = None) -> list[dict]:
    """Read prepared controlled pairs while preserving their fixed order."""
    pairs = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {
                "sample_id",
                "task",
                "clean_input_ids",
                "corrupted_input_ids",
                "clean_answer_id",
                "corrupted_answer_id",
            }
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing fields: {sorted(missing)}"
                )
            pairs.append(record)
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    if not pairs:
        raise ValueError(f"No SVA pairs found in {path}")
    return pairs


def map_pair(
    record: dict,
    mapper: ClonedMapper,
    language_id: int,
    block_size: int,
) -> dict:
    """Map one base-token pair into an original or cloned ID space."""
    clean_ids = list(map(int, record["clean_input_ids"]))
    corrupted_ids = list(map(int, record["corrupted_input_ids"]))
    clean_answer = int(record["clean_answer_id"])
    corrupted_answer = int(record["corrupted_answer_id"])

    if not clean_ids or len(clean_ids) != len(corrupted_ids):
        raise ValueError("clean/corrupted prompts must have equal non-zero length")
    if len(clean_ids) > block_size:
        raise ValueError("prompt exceeds checkpoint context length")
    all_ids = clean_ids + corrupted_ids + [clean_answer, corrupted_answer]
    if min(all_ids) < 0 or max(all_ids) >= mapper.original_vocab_size:
        raise ValueError("pair contains IDs outside the base tokenizer vocabulary")
    if clean_answer == corrupted_answer:
        raise ValueError("clean and corrupted answers must be different tokens")

    def mapped(values: list[int]) -> list[int]:
        tensor = torch.tensor(values, dtype=torch.long)
        return mapper.map_to_language(tensor, language_id).tolist()

    return {
        "sample_id": str(record["sample_id"]),
        "task": str(record["task"]),
        "clean_type": str(record.get("clean_type", "unknown")),
        "clean_ids": mapped(clean_ids),
        "corrupted_ids": mapped(corrupted_ids),
        "clean_answer_id": mapped([clean_answer])[0],
        "corrupted_answer_id": mapped([corrupted_answer])[0],
    }


def final_logit_difference(
    logits: torch.Tensor,
    clean_answer_ids: torch.Tensor,
    corrupted_answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute LD = logit(clean answer) - logit(corrupted answer)."""
    final_logits = logits[:, -1]
    rows = torch.arange(final_logits.size(0), device=final_logits.device)
    return (
        final_logits[rows, clean_answer_ids]
        - final_logits[rows, corrupted_answer_ids]
    )


@torch.inference_mode()
def score_language_pairs(
    model,
    records: list[dict],
    mapper: ClonedMapper,
    language_id: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict], list[dict]]:
    """Score clean and corrupted prompts with one common LD direction."""
    mapped_pairs = []
    rejected = []
    for record in records:
        try:
            mapped_pairs.append(
                map_pair(record, mapper, language_id, model.config.block_size)
            )
        except ValueError as error:
            rejected.append(
                {"sample_id": str(record.get("sample_id")), "reason": str(error)}
            )

    grouped: dict[int, list[dict]] = defaultdict(list)
    for pair in mapped_pairs:
        grouped[len(pair["clean_ids"])].append(pair)

    scores: dict[str, dict] = {}
    for sequence_length in sorted(grouped):
        group = grouped[sequence_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            clean = torch.tensor(
                [pair["clean_ids"] for pair in batch],
                dtype=torch.long,
                device=device,
            )
            corrupted = torch.tensor(
                [pair["corrupted_ids"] for pair in batch],
                dtype=torch.long,
                device=device,
            )
            clean_answers = torch.tensor(
                [pair["clean_answer_id"] for pair in batch], device=device
            )
            corrupted_answers = torch.tensor(
                [pair["corrupted_answer_id"] for pair in batch], device=device
            )
            clean_logits, _ = model(clean)
            corrupted_logits, _ = model(corrupted)
            clean_ld = final_logit_difference(
                clean_logits, clean_answers, corrupted_answers
            ).cpu()
            corrupted_ld = final_logit_difference(
                corrupted_logits, clean_answers, corrupted_answers
            ).cpu()

            for pair, clean_value, corrupted_value in zip(
                batch, clean_ld.tolist(), corrupted_ld.tolist()
            ):
                scores[pair["sample_id"]] = {
                    "clean_ld": float(clean_value),
                    "corrupted_ld": float(corrupted_value),
                    "clean_correct": clean_value > 0.0,
                    # The corrupted context should prefer the alternate answer.
                    "corrupted_correct": corrupted_value < 0.0,
                    "pair_correct": clean_value > 0.0 and corrupted_value < 0.0,
                }
    return scores, rejected


def summarize_scores(records: Iterable[dict], language: str) -> dict[str, float | int]:
    """Summarize prompt accuracy and correctly oriented LD values."""
    rows = list(records)
    if not rows:
        raise ValueError("Cannot summarize an empty SVA result set")
    values = [row[language] for row in rows]
    prompt_correct = [
        correct
        for score in values
        for correct in (score["clean_correct"], score["corrupted_correct"])
    ]
    # Flip the corrupted LD so every margin is correct minus incorrect.
    oriented_ld = [
        margin
        for score in values
        for margin in (score["clean_ld"], -score["corrupted_ld"])
    ]
    return {
        "num_pairs": len(rows),
        "num_prompts": 2 * len(rows),
        "accuracy": sum(prompt_correct) / len(prompt_correct),
        "pair_accuracy": sum(score["pair_correct"] for score in values) / len(rows),
        "clean_accuracy": sum(score["clean_correct"] for score in values) / len(rows),
        "corrupted_accuracy": (
            sum(score["corrupted_correct"] for score in values) / len(rows)
        ),
        "mean_logit_difference": sum(oriented_ld) / len(oriented_ld),
        "mean_clean_ld": sum(score["clean_ld"] for score in values) / len(rows),
        "mean_corrupted_ld": (
            sum(score["corrupted_ld"] for score in values) / len(rows)
        ),
    }

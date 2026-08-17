"""Controlled SVA pair loading and cloned-language mapping."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mllms.data.cloned_language import ClonedMapper


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

"""Controlled SVA pair loading and cloned-language mapping."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mllms.data.cloned_language import ClonedMapper


def _retokenize_pair(record: dict, tokenizer) -> dict:
    """Align the canonical text fields with the experiment tokenizer."""
    required_text = {
        "clean_prompt",
        "corrupted_prompt",
        "clean_answer",
        "corrupted_answer",
    }
    missing = required_text - set(record)
    if missing:
        raise ValueError(f"cannot retokenize; missing fields: {sorted(missing)}")

    clean_prompt = str(record["clean_prompt"])
    corrupted_prompt = str(record["corrupted_prompt"])
    clean_ids = list(tokenizer.encode(clean_prompt, out_type=int))
    corrupted_ids = list(tokenizer.encode(corrupted_prompt, out_type=int))
    def answer_ids(prompt: str, prompt_ids: list[int], answer: str) -> list[int]:
        full_ids = list(tokenizer.encode(f"{prompt} {answer}", out_type=int))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("answer changes prompt tokenization")
        result = list(map(int, full_ids[len(prompt_ids) :]))
        if not result:
            raise ValueError(f"answer {answer!r} has no tokenizer tokens")
        return result

    eos_id = int(tokenizer.eos_id())
    if eos_id < 0:
        raise ValueError("tokenizer must define EOS")
    aligned = dict(record)
    aligned["clean_input_ids"] = [eos_id, *clean_ids]
    aligned["corrupted_input_ids"] = [eos_id, *corrupted_ids]
    aligned["clean_answer_ids"] = answer_ids(
        clean_prompt,
        clean_ids,
        str(record["clean_answer"]),
    )
    aligned["corrupted_answer_ids"] = answer_ids(
        corrupted_prompt,
        corrupted_ids,
        str(record["corrupted_answer"]),
    )
    aligned["prediction_position"] = len(clean_ids)
    return aligned


def read_pairs(
    path: Path,
    max_pairs: int | None = None,
    tokenizer=None,
) -> list[dict]:
    """Read pairs in fixed order and optionally align IDs to a tokenizer."""
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
            if tokenizer is not None:
                try:
                    record = _retokenize_pair(record, tokenizer)
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
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
    clean_answer_values = (
        record["clean_answer_ids"]
        if "clean_answer_ids" in record
        else [record["clean_answer_id"]]
    )
    corrupted_answer_values = (
        record["corrupted_answer_ids"]
        if "corrupted_answer_ids" in record
        else [record["corrupted_answer_id"]]
    )
    clean_answers = list(map(int, clean_answer_values))
    corrupted_answers = list(map(int, corrupted_answer_values))

    if not clean_ids or not corrupted_ids:
        raise ValueError("clean/corrupted prompts must have non-zero length")
    longest_prompt = max(len(clean_ids), len(corrupted_ids))
    if longest_prompt > block_size:
        raise ValueError("prompt exceeds checkpoint context length")
    if longest_prompt + max(len(clean_answers), len(corrupted_answers)) - 1 > block_size:
        raise ValueError("prompt and answer exceed checkpoint context length")
    all_ids = clean_ids + corrupted_ids + clean_answers + corrupted_answers
    if min(all_ids) < 0 or max(all_ids) >= mapper.original_vocab_size:
        raise ValueError("pair contains IDs outside the base tokenizer vocabulary")
    if clean_answers == corrupted_answers:
        raise ValueError("clean and corrupted answers must be different")

    def mapped(values: list[int]) -> list[int]:
        tensor = torch.tensor(values, dtype=torch.long)
        return mapper.map_to_language(tensor, language_id).tolist()

    return {
        "sample_id": str(record["sample_id"]),
        "task": str(record["task"]),
        "clean_type": str(record.get("clean_type", "unknown")),
        "clean_ids": mapped(clean_ids),
        "corrupted_ids": mapped(corrupted_ids),
        "clean_answer_ids": mapped(clean_answers),
        "corrupted_answer_ids": mapped(corrupted_answers),
    }

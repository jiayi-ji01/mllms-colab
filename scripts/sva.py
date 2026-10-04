"""SVA pair loading and sequence scoring."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
import json

import torch

from token_data import ClonedMapper


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


LANGUAGES = {"original": 0, "clone": 1}


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


def answer_log_probability(
    model,
    prompts: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the summed autoregressive log probability of each answer."""
    if answer_ids.ndim != 2 or answer_ids.size(1) == 0:
        raise ValueError("answer_ids must have shape [batch, nonzero answer length]")
    model_input = torch.cat((prompts, answer_ids[:, :-1]), dim=1)
    logits, _ = model(model_input)
    start = prompts.size(1) - 1
    answer_logits = logits[:, start : start + answer_ids.size(1)]
    return answer_logits.log_softmax(dim=-1).gather(
        -1, answer_ids.unsqueeze(-1)
    ).squeeze(-1).sum(dim=-1)


def sequence_log_probability_difference(
    model,
    prompts: torch.Tensor,
    clean_answer_ids: torch.Tensor,
    corrupted_answer_ids: torch.Tensor,
) -> torch.Tensor:
    """Return log P(clean answer | prompt) - log P(corrupted answer | prompt)."""
    return answer_log_probability(
        model, prompts, clean_answer_ids
    ) - answer_log_probability(model, prompts, corrupted_answer_ids)


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

    grouped: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for pair in mapped_pairs:
        grouped[(
            len(pair["clean_ids"]),
            len(pair["corrupted_ids"]),
            len(pair["clean_answer_ids"]),
            len(pair["corrupted_answer_ids"]),
        )].append(pair)

    scores: dict[str, dict] = {}
    for group_key in sorted(grouped):
        group = grouped[group_key]
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
                [pair["clean_answer_ids"] for pair in batch], device=device
            )
            corrupted_answers = torch.tensor(
                [pair["corrupted_answer_ids"] for pair in batch], device=device
            )
            clean_ld = sequence_log_probability_difference(
                model, clean, clean_answers, corrupted_answers
            ).cpu()
            corrupted_ld = sequence_log_probability_difference(
                model, corrupted, clean_answers, corrupted_answers
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

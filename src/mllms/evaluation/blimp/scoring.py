"""BLiMP request construction, scoring, filtering, and aggregation."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from mllms.model.transformer import GPT


Request = tuple[int, str, list[int], int]


def build_requests(records: list[dict], language: str, eos_id: int) -> list[Request]:
    requests = []
    for index, record in enumerate(records):
        ids = record[language]
        requests.extend(
            [
                (index, "sentence_good_logprob", [eos_id, *ids["sentence_good_ids"], eos_id], 0),
                (index, "sentence_bad_logprob", [eos_id, *ids["sentence_bad_ids"], eos_id], 0),
                (
                    index,
                    "correct_verb_logprob",
                    [eos_id, *ids["prefix_good_ids"], *ids["correct_verb_ids"]],
                    len(ids["prefix_good_ids"]),
                ),
                (
                    index,
                    "incorrect_verb_logprob",
                    [eos_id, *ids["prefix_bad_ids"], *ids["incorrect_verb_ids"]],
                    len(ids["prefix_bad_ids"]),
                ),
            ]
        )
    return requests


@torch.no_grad()
def score_requests(
    model: GPT,
    requests: list[Request],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float]]:
    results: list[dict[str, float]] = [
        defaultdict(float) for _ in range(max(request[0] for request in requests) + 1)
    ]
    groups: dict[int, list[Request]] = defaultdict(list)
    for request in requests:
        groups[len(request[2])].append(request)

    for sequence_length in sorted(groups):
        group = groups[sequence_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            sequences = torch.tensor(
                [request[2] for request in batch], dtype=torch.long, device=device
            )
            logits, _ = model(sequences[:, :-1])
            token_logprobs = F.log_softmax(logits, dim=-1).gather(
                dim=-1, index=sequences[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            positions = torch.arange(sequence_length - 1, device=device)
            score_starts = torch.tensor(
                [request[3] for request in batch], device=device
            )
            mask = positions.unsqueeze(0) >= score_starts.unsqueeze(1)
            scores = (token_logprobs * mask).sum(dim=1).cpu().tolist()
            for request, score in zip(batch, scores):
                index, name, _, _ = request
                results[index][name] = score
    return [dict(result) for result in results]


@torch.no_grad()
def score_verb_logits(
    model: GPT,
    records: list[dict],
    language: str,
    eos_id: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Compare one-token correct/incorrect verbs after the same prefix."""
    results: list[dict[str, float]] = [{} for _ in records]
    groups: dict[int, list[tuple[int, list[int], int, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        ids = record[language]
        sequence = [eos_id, *ids["prefix_good_ids"]]
        groups[len(sequence)].append(
            (
                index,
                sequence,
                ids["correct_verb_ids"][0],
                ids["incorrect_verb_ids"][0],
            )
        )

    for sequence_length in sorted(groups):
        group = groups[sequence_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            sequences = torch.tensor(
                [item[1] for item in batch], dtype=torch.long, device=device
            )
            logits, _ = model(sequences)
            final_logits = logits[:, -1]
            correct_ids = torch.tensor([item[2] for item in batch], device=device)
            incorrect_ids = torch.tensor([item[3] for item in batch], device=device)
            rows = torch.arange(len(batch), device=device)
            correct_logits = final_logits[rows, correct_ids]
            incorrect_logits = final_logits[rows, incorrect_ids]
            differences = correct_logits - incorrect_logits
            for item, correct, incorrect, difference in zip(
                batch,
                correct_logits.cpu().tolist(),
                incorrect_logits.cpu().tolist(),
                differences.cpu().tolist(),
            ):
                results[item[0]] = {
                    "correct_verb_logit": correct,
                    "incorrect_verb_logit": incorrect,
                    "logit_diff": difference,
                    "margin": difference,
                    "correct": difference > 0.0,
                }
    return results


def select_verb_logit_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Select rows where a literal two-logit comparison is defined."""
    accepted = []
    excluded = defaultdict(int)
    for record in records:
        ids = record["original"]
        if record["scoring_method"] != "one_prefix":
            excluded["different_prefixes"] += 1
        elif ids["prefix_good_ids"] != ids["prefix_bad_ids"]:
            excluded["different_tokenized_prefixes"] += 1
        elif len(ids["correct_verb_ids"]) != 1:
            excluded["correct_verb_is_not_one_token"] += 1
        elif len(ids["incorrect_verb_ids"]) != 1:
            excluded["incorrect_verb_is_not_one_token"] += 1
        else:
            accepted.append(record)
    return accepted, dict(excluded)


def summarize(rows: list[dict]) -> dict:
    original_correct = [row["original"]["correct"] for row in rows]
    clone_correct = [row["clone"]["correct"] for row in rows]
    count = len(rows)
    summary = {
        "num_examples": count,
        "original_accuracy": sum(original_correct) / count,
        "clone_accuracy": sum(clone_correct) / count,
        "original_mean_margin": sum(row["original"]["margin"] for row in rows) / count,
        "clone_mean_margin": sum(row["clone"]["margin"] for row in rows) / count,
        "original_clone_accuracy_gap": (sum(original_correct) - sum(clone_correct)) / count,
        "prediction_agreement_rate": sum(
            original == clone for original, clone in zip(original_correct, clone_correct)
        ) / count,
        "both_correct": sum(
            original and clone for original, clone in zip(original_correct, clone_correct)
        ),
        "both_incorrect": sum(
            not original and not clone
            for original, clone in zip(original_correct, clone_correct)
        ),
        "original_only_correct": sum(
            original and not clone
            for original, clone in zip(original_correct, clone_correct)
        ),
        "clone_only_correct": sum(
            clone and not original
            for original, clone in zip(original_correct, clone_correct)
        ),
    }
    if "sentence_correct" in rows[0]["original"]:
        summary["original_sentence_accuracy"] = sum(
            row["original"]["sentence_correct"] for row in rows
        ) / count
        summary["clone_sentence_accuracy"] = sum(
            row["clone"]["sentence_correct"] for row in rows
        ) / count
    if "logit_diff" in rows[0]["original"]:
        summary["original_mean_logit_diff"] = summary["original_mean_margin"]
        summary["clone_mean_logit_diff"] = summary["clone_mean_margin"]
    return summary

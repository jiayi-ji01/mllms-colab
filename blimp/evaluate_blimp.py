"""Evaluate a trained GPT on preprocessed BLiMP agreement pairs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data_lib.cloned import ClonedMapper
from model.config import GPTConfig
from model.model import GPT
from tokenizer.tokenizer import load_tokenizer


Request = tuple[int, str, list[int], int]


def build_requests(records: list[dict], language: str, eos_id: int) -> list[Request]:
    requests = []
    for index, record in enumerate(records):
        ids = record[language]
        requests.extend(
            [
                (
                    index,
                    "sentence_good_logprob",
                    [eos_id, *ids["sentence_good_ids"], eos_id],
                    0,
                ),
                (
                    index,
                    "sentence_bad_logprob",
                    [eos_id, *ids["sentence_bad_ids"], eos_id],
                    0,
                ),
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
    results: list[dict[str, float]] = [defaultdict(float) for _ in range(
        max(request[0] for request in requests) + 1
    )]
    groups: dict[int, list[Request]] = defaultdict(list)
    for request in requests:
        groups[len(request[2])].append(request)

    for sequence_length in sorted(groups):
        group = groups[sequence_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            sequences = torch.tensor(
                [request[2] for request in batch],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(sequences[:, :-1])
            token_logprobs = F.log_softmax(logits, dim=-1).gather(
                dim=-1,
                index=sequences[:, 1:].unsqueeze(-1),
            ).squeeze(-1)

            positions = torch.arange(sequence_length - 1, device=device)
            score_starts = torch.tensor(
                [request[3] for request in batch],
                device=device,
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
                [item[1] for item in batch],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(sequences)
            final_logits = logits[:, -1]
            correct_ids = torch.tensor(
                [item[2] for item in batch], device=device
            )
            incorrect_ids = torch.tensor(
                [item[3] for item in batch], device=device
            )
            row_indices = torch.arange(len(batch), device=device)
            correct_logits = final_logits[row_indices, correct_ids]
            incorrect_logits = final_logits[row_indices, incorrect_ids]
            differences = correct_logits - incorrect_logits

            for item, correct, incorrect, difference in zip(
                batch,
                correct_logits.cpu().tolist(),
                incorrect_logits.cpu().tolist(),
                differences.cpu().tolist(),
            ):
                index = item[0]
                results[index] = {
                    "correct_verb_logit": correct,
                    "incorrect_verb_logit": incorrect,
                    "logit_diff": difference,
                    # Keep the common field so existing tables remain usable.
                    "margin": difference,
                    "correct": difference > 0.0,
                }

    return results


def select_verb_logit_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Select BLiMP rows where a literal two-logit comparison is defined."""
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
        "original_mean_margin": sum(
            row["original"]["margin"] for row in rows
        ) / count,
        "clone_mean_margin": sum(row["clone"]["margin"] for row in rows) / count,
        "original_clone_accuracy_gap": (
            sum(original_correct) - sum(clone_correct)
        ) / count,
        "prediction_agreement_rate": sum(
            original == clone
            for original, clone in zip(original_correct, clone_correct)
        ) / count,
        "both_correct": sum(
            original and clone
            for original, clone in zip(original_correct, clone_correct)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/blimp/processed/agreement.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/training/best.pt"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.model"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blimp"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--scoring",
        choices=("conditional-logprob", "verb-logit"),
        default="conditional-logprob",
        help=(
            "Use the existing conditional score, or compare raw logits for "
            "one-token verb alternatives after an identical prefix."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.device == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    device = torch.device(device_name)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer)
    mapper = ClonedMapper(
        original_vocab_size=tokenizer.vocab_size(),
        pad_id=tokenizer.pad_id(),
    )
    if model_config.vocab_size != mapper.model_vocab_size:
        raise ValueError("checkpoint and cloned tokenizer vocabulary sizes differ")

    with args.data.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records:
        raise ValueError(f"No preprocessed examples found in {args.data}")

    excluded: dict[str, int] = {}
    if args.scoring == "verb-logit":
        records, excluded = select_verb_logit_records(records)
        if not records:
            raise ValueError("No BLiMP rows support a one-token verb-logit score")

    language_scores = {}
    for language_id, language in ((0, "original"), (1, "clone")):
        eos_id = mapper.map_to_language(
            torch.tensor([tokenizer.eos_id()]),
            language_id,
        ).item()
        if args.scoring == "verb-logit":
            language_scores[language] = score_verb_logits(
                model,
                records,
                language,
                eos_id,
                args.batch_size,
                device,
            )
        else:
            requests = build_requests(records, language, eos_id)
            language_scores[language] = score_requests(
                model,
                requests,
                args.batch_size,
                device,
            )

    results = []
    for index, record in enumerate(records):
        result = {
            key: record[key]
            for key in (
                "sample_id",
                "subtask",
                "sentence_good",
                "sentence_bad",
                "scoring_method",
                "prefix_good",
                "prefix_bad",
                "correct_verb",
                "incorrect_verb",
            )
        }
        for language in ("original", "clone"):
            scores = language_scores[language][index]
            if args.scoring == "verb-logit":
                result[language] = scores
                continue
            sentence_margin = (
                scores["sentence_good_logprob"]
                - scores["sentence_bad_logprob"]
            )
            margin = (
                scores["correct_verb_logprob"]
                - scores["incorrect_verb_logprob"]
            )
            result[language] = {
                **scores,
                "sentence_margin": sentence_margin,
                "sentence_correct": sentence_margin > 0.0,
                "margin": margin,
                "correct": margin > 0.0,
            }
        result["predictions_agree"] = (
            result["original"]["correct"] == result["clone"]["correct"]
        )
        results.append(result)

    by_subtask: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_subtask[result["subtask"]].append(result)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "device": str(device),
        "primary_metric": (
            "verb_logit_diff"
            if args.scoring == "verb-logit"
            else "conditional_verb_margin"
        ),
        "scoring": args.scoring,
        "excluded_examples": excluded,
        "overall": summarize(results),
        "by_subtask": {
            subtask: summarize(rows)
            for subtask, rows in sorted(by_subtask.items())
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = args.output_dir / "agreement_results.jsonl"
    summary_path = args.output_dir / "agreement_summary.json"
    with examples_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    overall = summary["overall"]
    print(f"Checkpoint: {args.checkpoint} (step {checkpoint['step']:,})")
    print(f"Device: {device}")
    print(f"Scoring: {args.scoring}")
    print(f"Examples: {overall['num_examples']:,}")
    if excluded:
        print(f"Excluded: {sum(excluded.values()):,} {excluded}")
    print(f"Original accuracy: {overall['original_accuracy']:.4f}")
    print(f"Clone accuracy: {overall['clone_accuracy']:.4f}")
    print(f"Accuracy gap: {overall['original_clone_accuracy_gap']:+.4f}")
    print(f"Prediction agreement: {overall['prediction_agreement_rate']:.4f}")
    metric_label = (
        "mean logit difference"
        if args.scoring == "verb-logit"
        else "mean margin"
    )
    print(f"Original {metric_label}: {overall['original_mean_margin']:.4f}")
    print(f"Clone {metric_label}: {overall['clone_mean_margin']:.4f}")
    print(
        "Both correct / both incorrect / original only / clone only: "
        f"{overall['both_correct']:,} / {overall['both_incorrect']:,} / "
        f"{overall['original_only_correct']:,} / "
        f"{overall['clone_only_correct']:,}"
    )
    print("\nBy subtask:")
    for subtask, metrics in summary["by_subtask"].items():
        print(
            f"{subtask}: original={metrics['original_accuracy']:.4f}, "
            f"clone={metrics['clone_accuracy']:.4f}, "
            f"gap={metrics['original_clone_accuracy_gap']:+.4f}"
        )
    print(f"\nPer-example results: {examples_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

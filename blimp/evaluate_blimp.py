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


def summarize(rows: list[dict]) -> dict:
    original_correct = [row["original"]["correct"] for row in rows]
    clone_correct = [row["clone"]["correct"] for row in rows]
    count = len(rows)

    return {
        "num_examples": count,
        "original_accuracy": sum(original_correct) / count,
        "clone_accuracy": sum(clone_correct) / count,
        "original_mean_margin": sum(
            row["original"]["margin"] for row in rows
        ) / count,
        "clone_mean_margin": sum(row["clone"]["margin"] for row in rows) / count,
        "original_sentence_accuracy": sum(
            row["original"]["sentence_correct"] for row in rows
        ) / count,
        "clone_sentence_accuracy": sum(
            row["clone"]["sentence_correct"] for row in rows
        ) / count,
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/blimp"))
    parser.add_argument("--batch-size", type=int, default=64)
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

    tokenizer = load_tokenizer()
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

    language_scores = {}
    for language_id, language in ((0, "original"), (1, "clone")):
        eos_id = mapper.map_to_language(
            torch.tensor([tokenizer.eos_id()]),
            language_id,
        ).item()
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
        "primary_metric": "conditional_verb_margin",
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
    print(f"Examples: {overall['num_examples']:,}")
    print(f"Original accuracy: {overall['original_accuracy']:.4f}")
    print(f"Clone accuracy: {overall['clone_accuracy']:.4f}")
    print(f"Accuracy gap: {overall['original_clone_accuracy_gap']:+.4f}")
    print(f"Prediction agreement: {overall['prediction_agreement_rate']:.4f}")
    print(f"Original mean margin: {overall['original_mean_margin']:.4f}")
    print(f"Clone mean margin: {overall['clone_mean_margin']:.4f}")
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

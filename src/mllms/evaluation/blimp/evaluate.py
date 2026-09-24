"""Evaluate a trained GPT on preprocessed BLiMP agreement pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from mllms.config import parse_configured_args
from mllms.data.cloned_language import ClonedMapper
from mllms.evaluation.blimp.scoring import (
    build_requests,
    score_requests,
    score_verb_logits,
    select_verb_logit_records,
    summarize,
)
from mllms.model.loading import load_model_checkpoint
from mllms.runtime import select_device
from mllms.tokenizer.sentencepiece import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--data",
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--scoring",
        choices=("conditional-logprob", "verb-logit"),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps")
    )
    return parse_configured_args(
        parser,
        Path("configs/evaluation/blimp.yaml"),
        ("evaluate",),
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = select_device(args.device)

    checkpoint, model, model_config = load_model_checkpoint(args.checkpoint, device)

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
            torch.tensor([tokenizer.eos_id()]), language_id
        ).item()
        if args.scoring == "verb-logit":
            language_scores[language] = score_verb_logits(
                model, records, language, eos_id, args.batch_size, device
            )
        else:
            requests = build_requests(records, language, eos_id)
            language_scores[language] = score_requests(
                model, requests, args.batch_size, device
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
                scores["sentence_good_logprob"] - scores["sentence_bad_logprob"]
            )
            margin = (
                scores["correct_verb_logprob"] - scores["incorrect_verb_logprob"]
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
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "tokenizer": str(args.tokenizer),
        "data": str(args.data),
        "device": str(device),
        "batch_size": args.batch_size,
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
        "mean logit difference" if args.scoring == "verb-logit" else "mean margin"
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
    print(f"Per-example results: {examples_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

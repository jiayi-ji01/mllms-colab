"""Evaluate controlled SVA pairs in original and cloned token spaces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from analysis.sva_utils import (
    LANGUAGES,
    read_pairs,
    score_language_pairs,
    select_device,
    summarize_scores,
)
from data_lib.cloned import ClonedMapper
from model.config import GPTConfig
from model.model import GPT
from tokenizer.tokenizer import load_tokenizer


def summarize_results(rows: list[dict]) -> dict:
    """Create overall and task-level original/clone summaries."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    def summary(group: list[dict]) -> dict:
        original = summarize_scores(group, "original")
        clone = summarize_scores(group, "clone")
        decisions = [
            (
                row["original"]["clean_correct"],
                row["original"]["corrupted_correct"],
                row["clone"]["clean_correct"],
                row["clone"]["corrupted_correct"],
            )
            for row in group
        ]
        agreement = sum(
            original_clean == clone_clean
            and original_corrupted == clone_corrupted
            for (
                original_clean,
                original_corrupted,
                clone_clean,
                clone_corrupted,
            ) in decisions
        ) / len(group)
        return {
            "num_pairs": len(group),
            "original": original,
            "clone": clone,
            "original_clone_accuracy_gap": (
                original["accuracy"] - clone["accuracy"]
            ),
            "pair_prediction_agreement": agreement,
            "joint_sanity_pairs": sum(row["joint_sanity_pass"] for row in group),
        }

    return {
        "overall": summary(rows),
        "by_task": {
            task: summary(group) for task, group in sorted(by_task.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/babylm_tokenizer/tokenizer.model"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/causalgym/sva_pairs.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/causalgym_sva"),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--sanity-margin",
        type=float,
        default=0.0,
        help="Require clean LD > margin and corrupted LD < -margin in both languages.",
    )
    parser.add_argument("--min-sanity-pairs", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be positive")
    if args.sanity_margin < 0.0:
        raise ValueError("sanity-margin must be non-negative")

    device = select_device(args.device)
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
    if mapper.model_vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")

    records = read_pairs(args.data, args.max_pairs)
    language_scores = {}
    rejected = []
    for language, language_id in LANGUAGES.items():
        scores, language_rejected = score_language_pairs(
            model,
            records,
            mapper,
            language_id,
            device,
            args.batch_size,
        )
        language_scores[language] = scores
        rejected.extend({"language": language, **row} for row in language_rejected)

    results = []
    sanity_records = []
    for record in records:
        sample_id = str(record["sample_id"])
        if any(sample_id not in language_scores[name] for name in LANGUAGES):
            continue
        result = {
            "sample_id": sample_id,
            "task": record["task"],
            "source_split": record.get("source_split"),
            "clean_type": record.get("clean_type"),
            "corrupted_type": record.get("corrupted_type"),
            "clean_prompt": record.get("clean_prompt"),
            "corrupted_prompt": record.get("corrupted_prompt"),
            "clean_answer": record.get("clean_answer"),
            "corrupted_answer": record.get("corrupted_answer"),
            "original": language_scores["original"][sample_id],
            "clone": language_scores["clone"][sample_id],
        }
        result["joint_sanity_pass"] = all(
            result[language]["clean_ld"] > args.sanity_margin
            and result[language]["corrupted_ld"] < -args.sanity_margin
            for language in LANGUAGES
        )
        results.append(result)
        if result["joint_sanity_pass"]:
            sanity_records.append({**record, "baseline": result})

    if not results:
        raise ValueError("No pairs could be evaluated")

    summaries = summarize_results(results)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "data": str(args.data),
        "device": str(device),
        "reported_metric": "LD = logit(correct) - logit(incorrect)",
        "patching_ld_direction": (
            "logit(clean_answer) - logit(corrupted_answer) for both runs"
        ),
        "accuracy_definition": "accuracy over both clean and corrupted prompts",
        "pair_accuracy_definition": "both counterfactual prompts are correct",
        "sanity_definition": (
            "clean_ld > sanity_margin and corrupted_ld < -sanity_margin "
            "for both original and clone"
        ),
        "sanity_margin": args.sanity_margin,
        "num_rejected": len(rejected),
        **summaries,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sva_results.jsonl"
    sanity_path = args.output_dir / "sanity_pairs.jsonl"
    summary_path = args.output_dir / "sva_summary.json"
    with results_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    with sanity_path.open("w", encoding="utf-8") as output:
        for record in sanity_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if rejected:
        (args.output_dir / "rejected_pairs.json").write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    overall = summary["overall"]
    print(f"Checkpoint step: {checkpoint['step']:,}")
    print(f"Evaluated pairs: {len(results):,}")
    for language in LANGUAGES:
        values = overall[language]
        print(
            f"{language:8s} accuracy={values['accuracy']:.4f} | "
            f"pair_accuracy={values['pair_accuracy']:.4f} | "
            f"mean_LD={values['mean_logit_difference']:+.4f}"
        )
    print(f"Joint sanity pairs: {len(sanity_records):,}")
    if len(sanity_records) < args.min_sanity_pairs:
        print(
            f"WARNING: only {len(sanity_records):,} pairs passed joint sanity; "
            f"the requested formal patching range starts at "
            f"{args.min_sanity_pairs:,}."
        )
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Patching input: {sanity_path}")


if __name__ == "__main__":
    main()

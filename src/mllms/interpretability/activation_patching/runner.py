"""Run clean-to-corrupted SVA activation patching in either token space."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from mllms.config import parse_configured_args
from mllms.data.cloned_language import ClonedMapper
from mllms.evaluation.sva.pairs import map_pair, read_pairs
from mllms.evaluation.sva.scoring import LANGUAGES, score_language_pairs
from mllms.interpretability.activation_patching.interventions import patch_batch
from mllms.interpretability.activation_patching.results import (
    aggregate_results,
    save_language_results,
)
from mllms.model.loading import load_model_checkpoint
from mllms.runtime import select_device
from mllms.tokenizer.sentencepiece import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
    )
    parser.add_argument(
        "--data",
        type=Path,
    )
    parser.add_argument(
        "--output-dir", type=Path
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument(
        "--language", choices=("original", "clone", "both")
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--example-batch-size", type=int)
    parser.add_argument("--intervention-batch-size", type=int)
    parser.add_argument(
        "--max-positions",
        type=int,
        help="Patch only the final N prompt tokens (default: every token).",
    )
    parser.add_argument("--sanity-margin", type=float)
    parser.add_argument("--denominator-epsilon", type=float)
    return parse_configured_args(
        parser,
        Path("configs/interpretability/activation_patching.yaml"),
        ("run",),
    )


def _select_balanced_records(
    records: list[dict],
    baseline_scores: dict,
    max_examples: int,
    sanity_margin: float,
    denominator_epsilon: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Apply one joint filter, then alternate singular/plural source conditions."""
    sanity_records = []
    sanity_rejected = []
    for record in records:
        sample_id = str(record["sample_id"])
        if len(record["clean_input_ids"]) != len(record["corrupted_input_ids"]):
            sanity_rejected.append({
                "sample_id": sample_id,
                "reason": "activation patching requires equal prompt token lengths",
            })
            continue
        if any(sample_id not in baseline_scores[name] for name in LANGUAGES):
            continue
        scores = {name: baseline_scores[name][sample_id] for name in LANGUAGES}
        passes = all(
            score["clean_ld"] > sanity_margin
            and score["corrupted_ld"] < -sanity_margin
            and abs(score["clean_ld"] - score["corrupted_ld"])
            >= denominator_epsilon
            for score in scores.values()
        )
        if passes:
            sanity_records.append(record)
        else:
            sanity_rejected.append({"sample_id": sample_id, "scores": scores})

    by_type = {
        clean_type: [
            record
            for record in sanity_records
            if record.get("clean_type") == clean_type
        ]
        for clean_type in ("singular", "plural")
    }
    accepted = []
    offsets = {clean_type: 0 for clean_type in by_type}
    while len(accepted) < min(max_examples, len(sanity_records)):
        made_progress = False
        for clean_type in ("singular", "plural"):
            offset = offsets[clean_type]
            if offset >= len(by_type[clean_type]):
                continue
            accepted.append(by_type[clean_type][offset])
            offsets[clean_type] += 1
            made_progress = True
            if len(accepted) >= max_examples:
                break
        if not made_progress:
            break
    return accepted, sanity_records, sanity_rejected


def main() -> None:
    args = parse_args()
    for name in ("max_examples", "example_batch_size", "intervention_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        raise ValueError("max-positions must be positive")
    if args.sanity_margin < 0 or args.denominator_epsilon <= 0:
        raise ValueError("sanity-margin must be non-negative and epsilon positive")

    device = select_device(args.device)
    checkpoint, model, model_config = load_model_checkpoint(args.checkpoint, device)

    tokenizer = load_tokenizer(args.tokenizer)
    mapper = ClonedMapper(
        original_vocab_size=tokenizer.vocab_size(),
        pad_id=tokenizer.pad_id(),
    )
    if mapper.model_vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")

    records = read_pairs(args.data, tokenizer=tokenizer)
    baseline_scores = {}
    baseline_rejected = []
    for language, language_id in LANGUAGES.items():
        scores, rejected = score_language_pairs(
            model,
            records,
            mapper,
            language_id,
            device,
            args.example_batch_size,
        )
        baseline_scores[language] = scores
        baseline_rejected.extend(
            {"language": language, **record} for record in rejected
        )

    accepted_records, sanity_records, sanity_rejected = _select_balanced_records(
        records,
        baseline_scores,
        args.max_examples,
        args.sanity_margin,
        args.denominator_epsilon,
    )
    if not accepted_records:
        raise ValueError("No pair passed the joint original/clone SVA sanity check")

    requested_languages = (
        tuple(LANGUAGES) if args.language == "both" else (args.language,)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_sample_ids = [str(record["sample_id"]) for record in accepted_records]
    root_metadata = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "model_config": asdict(model_config),
        "tokenizer": str(args.tokenizer),
        "data": str(args.data),
        "device": str(device),
        "languages": list(requested_languages),
        "num_input_pairs": len(records),
        "num_joint_sanity_pairs_available": len(sanity_records),
        "num_patched_pairs": len(accepted_records),
        "patched_clean_type_counts": {
            clean_type: sum(
                record.get("clean_type") == clean_type
                for record in accepted_records
            )
            for clean_type in ("singular", "plural")
        },
        "patched_task_counts": dict(
            Counter(record.get("task") for record in accepted_records)
        ),
        "patched_source_split_counts": dict(
            Counter(record.get("source_split") for record in accepted_records)
        ),
        "num_baseline_rejected": len(baseline_rejected),
        "num_sanity_rejected": len(sanity_rejected),
        "sample_ids": common_sample_ids,
        "random_seed": None,
        "selection_order": "input order with deterministic singular/plural alternation",
        "example_batch_size": args.example_batch_size,
        "intervention_batch_size": args.intervention_batch_size,
        "sanity_margin": args.sanity_margin,
        "denominator_epsilon": args.denominator_epsilon,
        "sanity_definition": (
            "clean_ld > margin and corrupted_ld < -margin in both languages"
        ),
        "ld": (
            "log P(clean answer sequence) - "
            "log P(corrupted answer sequence)"
        ),
        "delta_ld": "LD_patched - LD_corrupted",
        "recovery": (
            "(LD_patched - LD_corrupted) / (LD_clean - LD_corrupted)"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(root_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for language in requested_languages:
        mapped = [
            map_pair(
                record,
                mapper,
                LANGUAGES[language],
                model_config.block_size,
            )
            for record in accepted_records
        ]
        grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
        for example in mapped:
            grouped[(
                len(example["clean_ids"]),
                len(example["clean_answer_ids"]),
                len(example["corrupted_answer_ids"]),
            )].append(example)

        results = []
        completed = 0
        for group_key in sorted(grouped):
            group = grouped[group_key]
            for start in range(0, len(group), args.example_batch_size):
                batch = group[start : start + args.example_batch_size]
                results.extend(
                    patch_batch(
                        model,
                        batch,
                        device,
                        args.max_positions,
                        args.intervention_batch_size,
                    )
                )
                completed += len(batch)
                print(
                    f"{language:8s} | {completed:4d}/{len(mapped)} pairs | "
                    f"prompt/answer lengths {group_key}"
                )

        by_id = {result["sample_id"]: result for result in results}
        results = [by_id[sample_id] for sample_id in common_sample_ids]
        means, counts, per_example, relative_positions = aggregate_results(
            results,
            model_config.n_layers,
            model_config.n_heads,
            args.max_positions,
        )
        language_dir = args.output_dir / language
        save_language_results(
            language_dir,
            means,
            counts,
            per_example,
            relative_positions,
            results,
            {
                **root_metadata,
                "language": language,
                "n_layers": model_config.n_layers,
                "n_heads": model_config.n_heads,
                "max_positions": args.max_positions,
            },
        )
        print(f"{language} results: {language_dir}")

    print(
        f"Common sanity-passed pairs: {len(sanity_records):,}; "
        f"patched: {len(accepted_records):,}"
    )
    print(f"Patching results: {args.output_dir}")


if __name__ == "__main__":
    main()

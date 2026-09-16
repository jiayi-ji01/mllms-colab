"""Run clean-to-corrupted SVA activation patching in either token space."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from mllms.config import parse_configured_args
from mllms.data.cloned_language import ClonedMapper
from mllms.evaluation.sva.pairs import map_pair, read_pairs
from mllms.evaluation.sva.scoring import LANGUAGES, score_language_pairs
from mllms.interpretability.activation_patching.interventions import (
    ALL_COMPONENTS,
    patch_batch,
)
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
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=(
            "original-to-original",
            "clone-to-clone",
            "original-to-clone",
            "clone-to-original",
        ),
        help="Explicit source-to-target patching directions.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=ALL_COMPONENTS,
        help="Activation components to patch (default: all).",
    )
    parser.add_argument(
        "--prediction-position-only",
        action="store_true",
        help="Patch only the final prompt position.",
    )
    parser.add_argument(
        "--selection",
        choices=("joint-sanity", "all"),
        help="Select joint sanity-passed pairs or a fixed all-pairs cohort.",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=(
            "clean",
            "opposite-number",
            "same-number-shuffled",
            "opposite-number-shuffled",
        ),
        help="Source-activation controls to run (default: clean).",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--max-input-pairs",
        type=int,
        help="Read only the first N pairs for a smoke test.",
    )
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


def _select_fixed_records(
    records: list[dict],
    baseline_scores: dict,
    max_examples: int,
    *,
    require_shuffled_controls: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Choose a deterministic task/number-balanced cohort without sanity filtering."""
    eligible = []
    rejected = []
    for record in records:
        sample_id = str(record["sample_id"])
        if len(record["clean_input_ids"]) != len(record["corrupted_input_ids"]):
            rejected.append({
                "sample_id": sample_id,
                "reason": "activation patching requires equal prompt token lengths",
            })
            continue
        if any(sample_id not in baseline_scores[name] for name in LANGUAGES):
            rejected.append({
                "sample_id": sample_id,
                "reason": "baseline score unavailable in at least one language",
            })
            continue
        eligible.append(record)

    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in eligible:
        strata[(
            str(record.get("task", "unknown")),
            str(record.get("clean_type", "unknown")),
            str(record.get("clean_attractor_relation", "none")),
        )].append(record)
    if require_shuffled_controls:
        if max_examples % 4:
            raise ValueError(
                "max-examples must be divisible by four when shuffled controls are requested"
            )
        paired: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in eligible:
            paired[(str(record.get("task", "unknown")), len(record["clean_input_ids"]))][
                str(record.get("clean_type", "unknown"))
            ].append(record)
        by_task: dict[str, list[tuple[list[dict], list[dict]]]] = defaultdict(list)
        for (task, _), by_number in sorted(paired.items()):
            singular = by_number.get("singular", [])
            plural = by_number.get("plural", [])
            if len(singular) >= 2 and len(plural) >= 2:
                by_task[task].append((singular, plural))
        accepted = []
        offsets = {task: 0 for task in by_task}
        task_names = sorted(by_task)
        while len(accepted) + 4 <= max_examples:
            made_progress = False
            for task in task_names:
                options = by_task[task]
                if not options:
                    continue
                index = offsets[task] % len(options)
                group_round = offsets[task] // len(options)
                singular, plural = options[index]
                start = group_round * 2
                if start + 2 > len(singular) or start + 2 > len(plural):
                    offsets[task] += 1
                    continue
                accepted.extend([*singular[start : start + 2], *plural[start : start + 2]])
                offsets[task] += 1
                made_progress = True
                if len(accepted) >= max_examples:
                    break
            if not made_progress:
                break
        if len(accepted) < max_examples:
            rejected.append({
                "reason": "insufficient paired task/length/number strata for shuffled controls",
                "requested": max_examples,
                "selected": len(accepted),
            })
        return accepted, rejected

    accepted = []
    offsets = {key: 0 for key in strata}
    while len(accepted) < min(max_examples, len(eligible)):
        made_progress = False
        for key in sorted(strata):
            offset = offsets[key]
            if offset >= len(strata[key]):
                continue
            accepted.append(strata[key][offset])
            offsets[key] += 1
            made_progress = True
            if len(accepted) >= max_examples:
                break
        if not made_progress:
            break
    return accepted, rejected


def _resolve_directions(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.directions:
        return [tuple(value.split("-to-", maxsplit=1)) for value in args.directions]
    languages = tuple(LANGUAGES) if args.language == "both" else (args.language,)
    return [(language, language) for language in languages]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_control_sources(
    source_examples: list[dict],
    control: str,
) -> list[dict]:
    """Build deterministic source activations while retaining target sample IDs."""
    if control in {"clean", "opposite-number"}:
        controlled = []
        for example in source_examples:
            item = dict(example)
            item["source_sample_id"] = str(example["sample_id"])
            if control == "opposite-number":
                item["clean_ids"] = list(example["corrupted_ids"])
            controlled.append(item)
        return controlled

    opposite = control == "opposite-number-shuffled"
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for example in source_examples:
        groups[(
            str(example["task"]),
            len(example["clean_ids"]),
            str(example["clean_type"]),
        )].append(example)

    controlled = []
    offsets: dict[tuple[str, int, str], int] = defaultdict(int)
    for target in source_examples:
        desired_type = str(target["clean_type"])
        if opposite:
            desired_type = "plural" if desired_type == "singular" else "singular"
        key = (
            str(target["task"]),
            len(target["clean_ids"]),
            desired_type,
        )
        candidates = groups.get(key, [])
        if not candidates or (
            not opposite
            and len(candidates) == 1
            and str(candidates[0]["sample_id"]) == str(target["sample_id"])
        ):
            raise ValueError(
                f"cannot construct {control} control for {target['sample_id']}"
            )
        rotation = 0 if opposite else 1
        source = candidates[(offsets[key] + rotation) % len(candidates)]
        offsets[key] += 1
        item = dict(source)
        item["source_sample_id"] = str(source["sample_id"])
        item["sample_id"] = str(target["sample_id"])
        controlled.append(item)
    return controlled


def main() -> None:
    args = parse_args()
    for name in ("max_examples", "example_batch_size", "intervention_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_positions is not None and args.max_positions <= 0:
        raise ValueError("max-positions must be positive")
    if args.max_input_pairs is not None and args.max_input_pairs <= 0:
        raise ValueError("max-input-pairs must be positive")
    if args.prediction_position_only and args.max_positions not in (None, 1):
        raise ValueError(
            "prediction-position-only conflicts with max-positions other than 1"
        )
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

    records = read_pairs(args.data, args.max_input_pairs, tokenizer=tokenizer)
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

    sanity_selected, sanity_records, sanity_rejected = _select_balanced_records(
        records,
        baseline_scores,
        args.max_examples,
        args.sanity_margin,
        args.denominator_epsilon,
    )
    fixed_rejected = []
    if args.selection == "all":
        accepted_records, fixed_rejected = _select_fixed_records(
            records,
            baseline_scores,
            args.max_examples,
            require_shuffled_controls=any(
                "shuffled" in control for control in (args.controls or ())
            ),
        )
    else:
        accepted_records = sanity_selected
    if not accepted_records:
        raise ValueError("No pair is eligible for the requested patching selection")

    directions = _resolve_directions(args)
    components = tuple(args.components or ALL_COMPONENTS)
    controls = tuple(args.controls or ("clean",))
    max_positions = 1 if args.prediction_position_only else args.max_positions
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_sample_ids = [str(record["sample_id"]) for record in accepted_records]
    root_metadata = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "model_config": asdict(model_config),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": _sha256(args.tokenizer),
        "data": str(args.data),
        "dataset": str(records[0].get("dataset", "unknown")),
        "data_sha256": _sha256(args.data),
        "device": str(device),
        "directions": [f"{source}-to-{target}" for source, target in directions],
        "components": list(components),
        "controls": list(controls),
        "selection": args.selection,
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
        "num_fixed_selection_rejected": len(fixed_rejected),
        "selection_rejections": fixed_rejected,
        "sample_ids": common_sample_ids,
        "random_seed": None,
        "selection_order": (
            "deterministic task/number/attractor round-robin"
            if args.selection == "all"
            else "input order with deterministic singular/plural alternation"
        ),
        "control_pairing": (
            "deterministic rotation within task/prompt-length/number strata"
        ),
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

    mapped_by_language = {
        language: [
            map_pair(record, mapper, language_id, model_config.block_size)
            for record in accepted_records
        ]
        for language, language_id in LANGUAGES.items()
    }
    for source_language, target_language in directions:
        base_source_examples = mapped_by_language[source_language]
        target_examples = mapped_by_language[target_language]
        direction_name = f"{source_language}_to_{target_language}"
        for control in controls:
            source_examples = build_control_sources(base_source_examples, control)
            grouped: dict[
                tuple[int, int, int], list[tuple[dict, dict]]
            ] = defaultdict(list)
            for source, target in zip(source_examples, target_examples):
                group_key = (
                    len(target["corrupted_ids"]),
                    len(target["clean_answer_ids"]),
                    len(target["corrupted_answer_ids"]),
                )
                grouped[group_key].append((source, target))

            results = []
            completed = 0
            for group_key in sorted(grouped):
                group = grouped[group_key]
                for start in range(0, len(group), args.example_batch_size):
                    pair_batch = group[start : start + args.example_batch_size]
                    source_batch = [pair[0] for pair in pair_batch]
                    target_batch = [pair[1] for pair in pair_batch]
                    results.extend(
                        patch_batch(
                            model,
                            source_batch,
                            device,
                            max_positions,
                            args.intervention_batch_size,
                            target_examples=target_batch,
                            components=components,
                            denominator_epsilon=args.denominator_epsilon,
                        )
                    )
                    completed += len(pair_batch)
                    print(
                        f"{source_language:8s}->{target_language:8s} | "
                        f"{control:24s} | "
                        f"{completed:4d}/{len(source_examples)} pairs | "
                        f"prompt/answer lengths {group_key}"
                    )

            by_id = {result["sample_id"]: result for result in results}
            results = [by_id[sample_id] for sample_id in common_sample_ids]
            means, counts, per_example, relative_positions = aggregate_results(
                results,
                model_config.n_layers,
                model_config.n_heads,
                max_positions,
            )
            direction_dir = args.output_dir / direction_name / control
            control_condition = (
                ("within_language_" if source_language == target_language else "cross_language_")
                + control
            )
            save_language_results(
                direction_dir,
                means,
                counts,
                per_example,
                relative_positions,
                results,
                {
                    **root_metadata,
                    "source_language": source_language,
                    "target_language": target_language,
                    "control_condition": control_condition,
                    "n_layers": model_config.n_layers,
                    "n_heads": model_config.n_heads,
                    "max_positions": max_positions,
                },
            )
            print(f"{direction_name}/{control} results: {direction_dir}")

    print(
        f"Common sanity-passed pairs: {len(sanity_records):,}; "
        f"patched: {len(accepted_records):,}"
    )
    print(f"Patching results: {args.output_dir}")


if __name__ == "__main__":
    main()

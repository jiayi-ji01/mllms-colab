"""Locate BLiMP agreement targets and build original/clone token IDs."""

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import re

import torch

from data_lib.cloned import ClonedMapper
from model.config import GPTConfig
from tokenizer.tokenizer import load_tokenizer


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+|[^\w\s]")
TWO_PREFIX_SUBTASKS = {
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_2",
}


def surface_tokens(text: str) -> list[re.Match[str]]:
    return list(TOKEN_PATTERN.finditer(text))


def locate_targets(record: dict) -> dict:
    good_matches = surface_tokens(record["sentence_good"])
    bad_matches = surface_tokens(record["sentence_bad"])
    good_words = [match.group() for match in good_matches]
    bad_words = [match.group() for match in bad_matches]
    differences = [
        operation
        for operation in SequenceMatcher(
            a=good_words,
            b=bad_words,
            autojunk=False,
        ).get_opcodes()
        if operation[0] != "equal"
    ]

    if len(differences) != 1:
        raise ValueError("expected exactly one surface difference")
    tag, good_start, good_end, bad_start, bad_end = differences[0]
    if tag != "replace" or good_end - good_start != 1 or bad_end - bad_start != 1:
        raise ValueError("surface difference must replace one token")

    if record["subtask"] in TWO_PREFIX_SUBTASKS:
        if good_end >= len(good_matches) or bad_end >= len(bad_matches):
            raise ValueError("two-prefix sample has no shared target after difference")
        good_target = good_matches[good_end]
        bad_target = bad_matches[bad_end]
        if good_target.group() != bad_target.group():
            raise ValueError("two-prefix target must match in good and bad sentences")
        scoring_method = "two_prefix"
    else:
        good_target = good_matches[good_start]
        bad_target = bad_matches[bad_start]
        scoring_method = "one_prefix"

    return {
        "scoring_method": scoring_method,
        "difference_good": good_matches[good_start].group(),
        "difference_bad": bad_matches[bad_start].group(),
        "good_target_start": good_target.start(),
        "good_target_end": good_target.end(),
        "bad_target_start": bad_target.start(),
        "bad_target_end": bad_target.end(),
        "correct_verb": good_target.group(),
        "incorrect_verb": bad_target.group(),
    }


def encode_target(tokenizer, sentence: str, start: int, end: int) -> dict:
    sentence_ids = tokenizer.encode(sentence, out_type=int)
    prefix_ids = tokenizer.encode(sentence[:start], out_type=int)
    through_target_ids = tokenizer.encode(sentence[:end], out_type=int)

    if sentence_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("prefix is not aligned with sentence tokenization")
    if sentence_ids[: len(through_target_ids)] != through_target_ids:
        raise ValueError("target end is not aligned with sentence tokenization")

    target_ids = through_target_ids[len(prefix_ids) :]
    if not target_ids:
        raise ValueError("target verb has no tokens")

    return {
        "sentence_ids": sentence_ids,
        "prefix_ids": prefix_ids,
        "target_ids": target_ids,
        "target_token_start": len(prefix_ids),
    }


def map_ids(mapper: ClonedMapper, values: list[int], language: int) -> list[int]:
    return mapper.map_to_language(torch.tensor(values), language).tolist()


def preprocess_record(record: dict, tokenizer, mapper: ClonedMapper, block_size: int) -> dict:
    target = locate_targets(record)
    good = encode_target(
        tokenizer,
        record["sentence_good"],
        target["good_target_start"],
        target["good_target_end"],
    )
    bad = encode_target(
        tokenizer,
        record["sentence_bad"],
        target["bad_target_start"],
        target["bad_target_end"],
    )

    if max(len(good["sentence_ids"]), len(bad["sentence_ids"])) + 1 > block_size:
        raise ValueError("sentence exceeds checkpoint block_size")
    if max(
        len(good["prefix_ids"]) + len(good["target_ids"]),
        len(bad["prefix_ids"]) + len(bad["target_ids"]),
    ) > block_size:
        raise ValueError("conditional target exceeds checkpoint block_size")
    if target["scoring_method"] == "one_prefix" and good["prefix_ids"] != bad["prefix_ids"]:
        raise ValueError("one-prefix sample has different tokenized prefixes")

    original = {
        "sentence_good_ids": good["sentence_ids"],
        "sentence_bad_ids": bad["sentence_ids"],
        "prefix_good_ids": good["prefix_ids"],
        "prefix_bad_ids": bad["prefix_ids"],
        "correct_verb_ids": good["target_ids"],
        "incorrect_verb_ids": bad["target_ids"],
    }
    clone = {
        name: map_ids(mapper, ids, 1)
        for name, ids in original.items()
    }

    return {
        **record,
        "scoring_method": target["scoring_method"],
        "difference_good": target["difference_good"],
        "difference_bad": target["difference_bad"],
        "prefix_good": record["sentence_good"][: target["good_target_start"]],
        "prefix_bad": record["sentence_bad"][: target["bad_target_start"]],
        "correct_verb": target["correct_verb"],
        "incorrect_verb": target["incorrect_verb"],
        "correct_verb_token_count": len(good["target_ids"]),
        "incorrect_verb_token_count": len(bad["target_ids"]),
        "sentence_good_token_count": len(good["sentence_ids"]),
        "sentence_bad_token_count": len(bad["sentence_ids"]),
        "target_good_token_start": good["target_token_start"],
        "target_bad_token_start": bad["target_token_start"],
        "original": original,
        "clone": clone,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/blimp/raw/agreement.jsonl"))
    parser.add_argument(
        "--output",
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
    parser.add_argument("--examples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = GPTConfig(**checkpoint["model_config"])
    tokenizer = load_tokenizer(args.tokenizer)
    mapper = ClonedMapper(
        original_vocab_size=tokenizer.vocab_size(),
        pad_id=tokenizer.pad_id(),
    )
    if model_config.vocab_size != mapper.model_vocab_size:
        raise ValueError("checkpoint and cloned tokenizer vocabulary sizes differ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted: list[dict] = []
    rejected = Counter()

    with args.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                accepted.append(
                    preprocess_record(record, tokenizer, mapper, model_config.block_size)
                )
            except ValueError as error:
                rejected[str(error)] += 1

    with args.output.open("w", encoding="utf-8") as output:
        for record in accepted:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {args.output}")
    print(f"Accepted: {len(accepted):,}")
    print(f"Rejected: {sum(rejected.values()):,}")
    for reason, count in rejected.items():
        print(f"  {reason}: {count:,}")

    sample_count = min(args.examples, len(accepted))
    for record in random.Random(42).sample(accepted, sample_count):
        print("\n---")
        print(f"ID: {record['sample_id']}")
        print(f"Method: {record['scoring_method']}")
        print(f"Good prefix: {record['prefix_good']!r}")
        print(f"Bad prefix: {record['prefix_bad']!r}")
        print(
            f"Verb: {record['correct_verb']!r} / "
            f"{record['incorrect_verb']!r}"
        )
        print(
            f"Token starts: {record['target_good_token_start']} / "
            f"{record['target_bad_token_start']}"
        )
        print(
            f"Verb IDs: {record['original']['correct_verb_ids']} / "
            f"{record['original']['incorrect_verb_ids']}"
        )


if __name__ == "__main__":
    main()

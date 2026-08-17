"""Generate tokenizer-aligned Marvin--Linzen-style SVA minimal pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from mllms.config import parse_configured_args
from mllms.tokenizer.sentencepiece import load_tokenizer


CONDITIONS = ("simple", "pp_attractor", "object_relative", "subject_relative")

# Human-denoting regular nouns keep the generated sentences semantically plausible.
NOUN_LEMMAS = (
    "actor", "artist", "author", "baker", "banker", "captain", "chef",
    "clerk", "coach", "dancer", "doctor", "driver", "farmer", "guard",
    "judge", "lawyer", "leader", "manager", "nurse", "officer", "painter",
    "pilot", "player", "poet", "professor", "reporter", "sailor", "senator",
    "singer", "soldier", "student", "teacher", "tourist", "visitor", "waiter",
    "worker", "writer", "director", "engineer",
)
MAIN_VERB_LEMMAS = (
    "arrive", "dance", "laugh", "run", "smile", "speak", "wait", "walk",
    "work", "sleep", "swim", "travel", "sing", "stand", "sit", "leave",
    "return", "listen", "write", "talk", "play", "cook", "drive", "teach",
)
PAST_TRANSITIVE_VERBS = (
    "admired", "called", "followed", "greeted", "helped", "hired", "invited",
    "met", "praised", "questioned", "thanked", "visited", "watched",
    "welcomed", "noticed", "trusted",
)
SUBJECT_MODIFIERS = (None, "young", "old", "local", "famous")
PREPOSITIONS = ("near", "behind", "beside")


def regular_plural(noun: str) -> str:
    """Inflect the restricted regular-noun lexicon for number."""
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def third_person_singular(verb: str) -> str:
    """Inflect the restricted present-tense verb lexicon."""
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, out_type=int))


def _word_boundaries(tokenizer: Any, words: list[str]) -> tuple[list[int], list[int]]:
    """Return token IDs and cumulative boundaries after every whitespace word."""
    full = _encode(tokenizer, " ".join(words))
    boundaries = [0]
    for end in range(1, len(words) + 1):
        prefix = _encode(tokenizer, " ".join(words[:end]))
        if prefix != full[: len(prefix)]:
            raise ValueError("word boundary changes earlier tokenization")
        boundaries.append(len(prefix))
    return full, boundaries


def _answer_id(tokenizer: Any, prompt: str, answer: str) -> int:
    prompt_ids = _encode(tokenizer, prompt)
    full_ids = _encode(tokenizer, f"{prompt} {answer}")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("answer changes prompt tokenization")
    answer_ids = full_ids[len(prompt_ids) :]
    if len(answer_ids) != 1:
        raise ValueError(f"answer {answer!r} is not one tokenizer token")
    return answer_ids[0]


def compatible_lexicons(tokenizer: Any) -> dict[str, list[str]]:
    """Filter source lexicons using the tokenizer constraints of the experiment."""
    nouns = []
    for lemma in NOUN_LEMMAS:
        singular = _encode(tokenizer, lemma)
        plural = _encode(tokenizer, regular_plural(lemma))
        if singular and len(singular) == len(plural) and len(singular) <= 2:
            nouns.append(lemma)

    main_verbs = []
    for lemma in MAIN_VERB_LEMMAS:
        singular = _encode(tokenizer, third_person_singular(lemma))
        plural = _encode(tokenizer, lemma)
        if len(singular) == len(plural) == 1 and singular != plural:
            main_verbs.append(lemma)

    past_verbs = [verb for verb in PAST_TRANSITIVE_VERBS if len(_encode(tokenizer, verb)) == 1]
    if len(nouns) < 12 or len(main_verbs) < 8 or len(past_verbs) < 4:
        raise ValueError(
            "Tokenizer leaves too few compatible controlled-SVA lexical items: "
            f"nouns={len(nouns)}, main_verbs={len(main_verbs)}, "
            f"past_verbs={len(past_verbs)}"
        )
    return {"nouns": nouns, "main_verbs": main_verbs, "past_verbs": past_verbs}


def _split_lexicon(values: list[str], dev_fraction: float, rng: random.Random) -> tuple[list[str], list[str]]:
    shuffled = list(values)
    rng.shuffle(shuffled)
    dev_count = max(1, round(len(shuffled) * dev_fraction))
    return sorted(shuffled[:dev_count]), sorted(shuffled[dev_count:])


def _prompt_words(
    condition: str,
    subject: str,
    modifier: str | None,
    attractor: str | None,
    relation: str | None,
) -> tuple[list[str], int, int | None]:
    words = ["The"]
    if modifier is not None:
        words.append(modifier)
    subject_index = len(words)
    words.append(subject)
    attractor_index = None

    if condition == "pp_attractor":
        words.extend([str(relation), "the"])
        attractor_index = len(words)
        words.append(str(attractor))
    elif condition == "object_relative":
        words.extend(["that", "the"])
        attractor_index = len(words)
        words.extend([str(attractor), str(relation)])
    elif condition == "subject_relative":
        words.extend(["that", str(relation), "the"])
        attractor_index = len(words)
        words.append(str(attractor))
    elif condition != "simple":
        raise ValueError(f"Unknown controlled-SVA condition: {condition}")
    return words, subject_index, attractor_index


def make_record(
    tokenizer: Any,
    *,
    sample_id: str,
    split: str,
    condition: str,
    clean_number: str,
    attractor_number: str | None,
    subject_lemma: str,
    attractor_lemma: str | None,
    main_verb_lemma: str,
    modifier: str | None,
    relation: str | None,
) -> dict[str, Any]:
    """Build and validate one number-only clean/corrupted counterfactual."""
    if clean_number not in {"singular", "plural"}:
        raise ValueError("clean_number must be singular or plural")
    corrupted_number = "plural" if clean_number == "singular" else "singular"
    subject_forms = {
        "singular": subject_lemma,
        "plural": regular_plural(subject_lemma),
    }
    main_verb_forms = {
        "singular": third_person_singular(main_verb_lemma),
        "plural": main_verb_lemma,
    }
    attractor = None
    if attractor_lemma is not None:
        if attractor_number not in {"singular", "plural"}:
            raise ValueError("non-simple conditions require attractor_number")
        attractor = (
            attractor_lemma
            if attractor_number == "singular"
            else regular_plural(attractor_lemma)
        )

    clean_words, subject_index, attractor_index = _prompt_words(
        condition,
        subject_forms[clean_number],
        modifier,
        attractor,
        relation,
    )
    corrupted_words, corrupted_subject_index, corrupted_attractor_index = _prompt_words(
        condition,
        subject_forms[corrupted_number],
        modifier,
        attractor,
        relation,
    )
    if subject_index != corrupted_subject_index or attractor_index != corrupted_attractor_index:
        raise ValueError("clean/corrupted word regions are not aligned")
    differences = [
        index
        for index, (clean, corrupted) in enumerate(zip(clean_words, corrupted_words))
        if clean != corrupted
    ]
    if len(clean_words) != len(corrupted_words) or differences != [subject_index]:
        raise ValueError("clean/corrupted prompts must differ only at the subject")

    clean_prompt = " ".join(clean_words)
    corrupted_prompt = " ".join(corrupted_words)
    clean_ids, clean_boundaries = _word_boundaries(tokenizer, clean_words)
    corrupted_ids, corrupted_boundaries = _word_boundaries(tokenizer, corrupted_words)
    if clean_boundaries != corrupted_boundaries or len(clean_ids) != len(corrupted_ids):
        raise ValueError("clean/corrupted token boundaries differ")

    clean_answer = main_verb_forms[clean_number]
    corrupted_answer = main_verb_forms[corrupted_number]
    clean_answer_id = _answer_id(tokenizer, clean_prompt, clean_answer)
    corrupted_answer_id = _answer_id(tokenizer, corrupted_prompt, corrupted_answer)
    if clean_answer_id == corrupted_answer_id:
        raise ValueError("answer token IDs are identical")

    eos_id = int(tokenizer.eos_id())
    if eos_id < 0:
        raise ValueError("tokenizer must define EOS")
    subject_span = [
        1 + clean_boundaries[subject_index],
        1 + clean_boundaries[subject_index + 1],
    ]
    attractor_span = None
    if attractor_index is not None:
        attractor_span = [
            1 + clean_boundaries[attractor_index],
            1 + clean_boundaries[attractor_index + 1],
        ]

    return {
        "sample_id": sample_id,
        "dataset": "mllms_controlled_sva_v1",
        "split": split,
        "task": condition,
        "clean_type": clean_number,
        "corrupted_type": corrupted_number,
        "attractor_type": attractor_number,
        "clean_attractor_relation": (
            None if attractor_number is None else (
                "matched" if attractor_number == clean_number else "mismatched"
            )
        ),
        "subject_lemma": subject_lemma,
        "attractor_lemma": attractor_lemma,
        "main_verb_lemma": main_verb_lemma,
        "modifier": modifier,
        "relation": relation,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "clean_answer": clean_answer,
        "corrupted_answer": corrupted_answer,
        "clean_input_ids": [eos_id, *clean_ids],
        "corrupted_input_ids": [eos_id, *corrupted_ids],
        "clean_answer_id": clean_answer_id,
        "corrupted_answer_id": corrupted_answer_id,
        "subject_token_span": subject_span,
        "attractor_token_span": attractor_span,
        "prediction_position": len(clean_ids),
    }


def _relation_choice(condition: str, lexicon: dict[str, list[str]], rng: random.Random) -> str | None:
    if condition == "pp_attractor":
        return rng.choice(PREPOSITIONS)
    if condition in {"object_relative", "subject_relative"}:
        return rng.choice(lexicon["past_verbs"])
    return None


def generate_split(
    tokenizer: Any,
    *,
    split: str,
    lexicon: dict[str, list[str]],
    pairs_per_condition: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate an exactly balanced, duplicate-free split."""
    if pairs_per_condition <= 0 or pairs_per_condition % 4:
        raise ValueError("pairs_per_condition must be a positive multiple of four")
    records = []
    seen: set[tuple[Any, ...]] = set()
    strata = (
        ("singular", "singular"),
        ("singular", "plural"),
        ("plural", "singular"),
        ("plural", "plural"),
    )
    for condition in CONDITIONS:
        condition_records = []
        targets = (
            (("singular", None), pairs_per_condition // 2),
            (("plural", None), pairs_per_condition // 2),
        ) if condition == "simple" else tuple(
            (stratum, pairs_per_condition // 4) for stratum in strata
        )
        for (clean_number, attractor_number), target in targets:
            accepted = 0
            attempts = 0
            while accepted < target:
                attempts += 1
                if attempts > target * 500:
                    raise ValueError(
                        f"Could not generate {target} valid {split}/{condition}/"
                        f"{clean_number}/{attractor_number} records"
                    )
                subject = rng.choice(lexicon["nouns"])
                attractor = None
                if condition != "simple":
                    choices = [noun for noun in lexicon["nouns"] if noun != subject]
                    attractor = rng.choice(choices)
                main_verb = rng.choice(lexicon["main_verbs"])
                modifier = rng.choice(SUBJECT_MODIFIERS)
                relation = _relation_choice(condition, lexicon, rng)
                key = (
                    split, condition, clean_number, attractor_number, subject,
                    attractor, main_verb, modifier, relation,
                )
                if key in seen:
                    continue
                try:
                    record = make_record(
                        tokenizer,
                        sample_id=(
                            f"{split}:{condition}:"
                            f"{len(condition_records):04d}"
                        ),
                        split=split,
                        condition=condition,
                        clean_number=clean_number,
                        attractor_number=attractor_number,
                        subject_lemma=subject,
                        attractor_lemma=attractor,
                        main_verb_lemma=main_verb,
                        modifier=modifier,
                        relation=relation,
                    )
                except ValueError:
                    continue
                seen.add(key)
                condition_records.append(record)
                accepted += 1
        rng.shuffle(condition_records)
        for index, record in enumerate(condition_records):
            record["sample_id"] = f"{split}:{condition}:{index:04d}"
        records.extend(condition_records)
    rng.shuffle(records)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tokenizer_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dev-pairs-per-condition", type=int)
    parser.add_argument("--test-pairs-per-condition", type=int)
    parser.add_argument("--lexical-dev-fraction", type=float)
    parser.add_argument("--seed", type=int)
    return parse_configured_args(
        parser,
        Path("configs/evaluation/sva.yaml"),
        ("controlled",),
    )


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.lexical_dev_fraction <= 0.4:
        raise ValueError("lexical-dev-fraction must be between 0.1 and 0.4")
    tokenizer = load_tokenizer(args.tokenizer)
    compatible = compatible_lexicons(tokenizer)
    split_rng = random.Random(args.seed)
    dev_lexicon: dict[str, list[str]] = {}
    test_lexicon: dict[str, list[str]] = {}
    for name, values in compatible.items():
        dev, test = _split_lexicon(
            values,
            args.lexical_dev_fraction,
            split_rng,
        )
        dev_lexicon[name] = dev
        test_lexicon[name] = test

    dev_records = generate_split(
        tokenizer,
        split="dev",
        lexicon=dev_lexicon,
        pairs_per_condition=args.dev_pairs_per_condition,
        rng=random.Random(args.seed + 1),
    )
    test_records = generate_split(
        tokenizer,
        split="test",
        lexicon=test_lexicon,
        pairs_per_condition=args.test_pairs_per_condition,
        rng=random.Random(args.seed + 2),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "dev.jsonl", dev_records)
    _write_jsonl(args.output_dir / "test.jsonl", test_records)
    _write_jsonl(args.output_dir / "all.jsonl", [*dev_records, *test_records])
    metadata = {
        "dataset": "mllms_controlled_sva_v1",
        "config": str(args.config),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": _tokenizer_sha256(args.tokenizer),
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "construction": (
            "clean/corrupted prompts differ only in the number-inflected "
            "form of one subject lemma; correct/incorrect verbs are one token"
        ),
        "relation_label": (
            "clean_attractor_relation describes the clean prompt only; flipping "
            "the subject reverses matched/mismatched status in the corrupted prompt"
        ),
        "splits": {
            "dev": {
                "num_pairs": len(dev_records),
                "task_counts": Counter(row["task"] for row in dev_records),
                "lexicon": dev_lexicon,
            },
            "test": {
                "num_pairs": len(test_records),
                "task_counts": Counter(row["task"] for row in test_records),
                "lexicon": test_lexicon,
            },
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Compatible nouns: {len(compatible['nouns'])}")
    print(f"Compatible main verbs: {len(compatible['main_verbs'])}")
    print(f"Compatible past verbs: {len(compatible['past_verbs'])}")
    print(f"Development pairs: {len(dev_records):,}")
    print(f"Test pairs: {len(test_records):,}")
    print(f"Dataset: {args.output_dir}")


if __name__ == "__main__":
    main()

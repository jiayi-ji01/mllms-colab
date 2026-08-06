"""Build token-aligned subject–verb agreement pairs for activation patching."""

import argparse
import json
from pathlib import Path

from tokenizer.tokenizer import load_tokenizer


PAIR_BANK = (
    ("The boy", "The boys", "runs", "run"),
    ("The girl", "The girls", "walks", "walk"),
    ("The dog", "The dogs", "eats", "eat"),
    ("The cat", "The cats", "sleeps", "sleep"),
    ("The bird", "The birds", "sings", "sing"),
    ("The teacher", "The teachers", "smiles", "smile"),
    ("The child", "The children", "plays", "play"),
    ("The key to the cabinets", "The keys to the cabinet", "is", "are"),
    ("The dog near the houses", "The dogs near the house", "runs", "run"),
    ("The boy beside the trees", "The boys beside the tree", "walks", "walk"),
    ("The cat behind the doors", "The cats behind the door", "sleeps", "sleep"),
    ("The bird above the windows", "The birds above the window", "sings", "sing"),
    ("The teacher near the students", "The teachers near the student", "is", "are"),
    ("The farmer beside the horses", "The farmers beside the horse", "works", "work"),
    ("The doctor behind the nurses", "The doctors behind the nurse", "smiles", "smile"),
    ("The parent near the children", "The parents near the child", "waits", "wait"),
)


def encode_pair(tokenizer, prompt: str, answer: str) -> tuple[list[int], int]:
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    full_ids = tokenizer.encode(f"{prompt} {answer}", out_type=int)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("answer changes prompt tokenization")
    answer_ids = full_ids[len(prompt_ids) :]
    if len(answer_ids) != 1:
        raise ValueError("answer is not one token")
    return [tokenizer.eos_id(), *prompt_ids], answer_ids[0]


def build_pairs(tokenizer, limit: int) -> tuple[list[dict], list[str]]:
    pairs = []
    rejected = []
    for index, candidate in enumerate(PAIR_BANK):
        clean_prompt, corrupted_prompt, clean_answer, corrupted_answer = candidate
        try:
            clean_ids, clean_answer_id = encode_pair(
                tokenizer, clean_prompt, clean_answer
            )
            corrupted_ids, corrupted_answer_id = encode_pair(
                tokenizer, corrupted_prompt, corrupted_answer
            )
            if len(clean_ids) != len(corrupted_ids):
                raise ValueError("prompt token lengths differ")
            if clean_answer_id == corrupted_answer_id:
                raise ValueError("answer token IDs are identical")
            pairs.append(
                {
                    "sample_id": f"sva-{index:02d}",
                    "clean_sentence": clean_prompt,
                    "corrupted_sentence": corrupted_prompt,
                    "clean_input_ids": clean_ids,
                    "corrupted_input_ids": corrupted_ids,
                    "clean_answer": clean_answer,
                    "corrupted_answer": corrupted_answer,
                    "clean_answer_id": clean_answer_id,
                    "corrupted_answer_id": corrupted_answer_id,
                }
            )
        except ValueError as error:
            rejected.append(f"{clean_prompt!r}: {error}")
        if len(pairs) >= limit:
            break
    return pairs, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.model"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sva_pairs.jsonl"),
    )
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    pairs, rejected = build_pairs(tokenizer, args.max_examples)
    if not pairs:
        raise ValueError("No token-aligned SVA pairs were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for pair in pairs:
            output.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} pairs to: {args.output}")
    print(f"Rejected candidates: {len(rejected)}")
    for reason in rejected:
        print(f"  {reason}")


if __name__ == "__main__":
    main()

"""Tokenize prepared TinyStories splits into flat uint16 streams."""

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from tokenizer.tokenizer import load_tokenizer


WRITE_BUFFER_SIZE = 1_000_000


def tokenize_file(
    tokenizer: spm.SentencePieceProcessor,
    input_file: Path,
    output_file: Path,
    target_tokens: int | None = None,
) -> dict[str, int]:
    token_buffer: list[int] = []
    story_count = 0
    token_count = 0

    with (
        input_file.open("r", encoding="utf-8") as file,
        output_file.open("wb") as output_stream,
    ):
        for line in tqdm(file, desc=f"Tokenizing {input_file.name}"):
            story = line.strip()
            if not story:
                continue
            story_ids = tokenizer.encode(story, out_type=int)
            # Append the EOS token ID to the end of each story
            story_ids.append(tokenizer.eos_id())

            # add the tokenized story to the list of token IDs
            token_buffer.extend(story_ids)
            token_count += len(story_ids)
            story_count += 1

            # avoid memory issues by writing to file in chunks
            if len(token_buffer) >= WRITE_BUFFER_SIZE:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output_stream)
                token_buffer.clear()

            # Stop only after a complete story, so the count can be slightly
            # above the target but document boundaries remain intact.
            if target_tokens is not None and token_count >= target_tokens:
                break

        if token_buffer:
            np.asarray(token_buffer, dtype=np.uint16).tofile(output_stream)
            token_buffer.clear()

    print(f"\nSplit: {input_file.stem}")
    print(f"Stories: {story_count:,}")
    print(f"Actual base tokens: {token_count:,}")
    print(f"Original token space: {token_count:,}")
    print(f"Clone token space: {token_count:,}")
    print(f"Original + clone token spaces: {2 * token_count:,}")
    print(f"Saved to: {output_file}")
    return {"stories": story_count, "tokens": token_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.model"),
    )
    parser.add_argument(
        "--target-train-tokens",
        type=int,
        default=100_000_000,
        help="Stop train.bin after reaching this many actual encoded tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_train_tokens <= 0:
        raise ValueError("target-train-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)

    vocab_size = tokenizer.vocab_size()
    if vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Vocabulary size {vocab_size} is too large for uint16."
        )
    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"EOS ID: {tokenizer.eos_id()}")

    statistics = {}
    for split in ("train", "validation", "test"):
        input_file = args.input_dir / f"{split}.txt"
        if not input_file.is_file():
            raise FileNotFoundError(f"Prepared split not found: {input_file}")
        output_file = args.output_dir / f"{split}.bin"
        statistics[split] = tokenize_file(
            tokenizer,
            input_file,
            output_file,
            target_tokens=args.target_train_tokens if split == "train" else None,
        )

    train_tokens = statistics["train"]["tokens"]
    if train_tokens < args.target_train_tokens:
        raise RuntimeError(
            f"Prepared train text produced only {train_tokens:,} tokens; "
            f"target is {args.target_train_tokens:,}. Prepare more train text."
        )

    manifest = {
        "tokenizer": str(args.tokenizer),
        "base_vocab_size": vocab_size,
        "target_train_tokens": args.target_train_tokens,
        "splits": statistics,
        "train_language_token_spaces": {
            "original": train_tokens,
            "clone": train_tokens,
            "total": 2 * train_tokens,
        },
    }
    manifest_path = args.output_dir / "token_counts.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nToken count manifest: {manifest_path}")


if __name__ == "__main__":
    main()

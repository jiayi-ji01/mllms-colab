"""Train the SentencePiece tokenizer used by the language model."""

import argparse
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/train.txt"))
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer"),
    )
    parser.add_argument("--vocab-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vocab_size <= 0:
        raise ValueError("vocab-size must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(f"Training corpus not found: {args.input}")

    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=str(args.model_prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
    )

    print("Tokenizer training completed.")
    print(f"Model saved to: {args.model_prefix}.model")
    print(f"Vocabulary saved to: {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()

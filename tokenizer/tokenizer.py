"""SentencePiece tokenizer loading utilities."""

from pathlib import Path

import sentencepiece as spm


TOKENIZER_DIR = Path("artifacts/tokenizer")


def load_tokenizer(
    model_path: Path = TOKENIZER_DIR / "tokenizer.model",
) -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece model from disk."""
    path_to_model = Path(model_path)
    if not path_to_model.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {path_to_model}")
    return spm.SentencePieceProcessor(model_file=str(path_to_model))

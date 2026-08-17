"""Prepare deterministic English Wikipedia train/validation/test text splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import datasets
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


DATASET_ID = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
DATASET_REVISION = "1be2737543076fd154887e462434a01d5783896a"


def normalize_article(text: str) -> str:
    """Store one cleaned Wikipedia article per SentencePiece input line."""
    return " ".join(text.split())


def choose_split(identifier: str, seed: int, targets: dict[str, int]) -> str:
    """Assign an article deterministically in proportion to requested word budgets."""
    total = sum(targets.values())
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % total
    boundary = 0
    for split in ("train", "validation", "test"):
        boundary += targets[split]
        if bucket < boundary:
            return split
    raise RuntimeError("split assignment exceeded target range")


def prepare_rows(
    rows: Iterable[dict],
    output_dir: Path,
    targets: dict[str, int],
    seed: int,
) -> dict[str, dict[str, int | str]]:
    """Write hash-disjoint article splits until every word budget is reached."""
    expected = {"train", "validation", "test"}
    if set(targets) != expected or any(value <= 0 for value in targets.values()):
        raise ValueError("positive train/validation/test word targets are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: output_dir / f"{split}.txt" for split in expected}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite prepared Wikipedia splits: " + ", ".join(existing)
        )

    words = {split: 0 for split in expected}
    articles = {split: 0 for split in expected}
    seen_ids: set[str] = set()
    partial_paths = {
        split: path.with_suffix(path.suffix + ".partial")
        for split, path in paths.items()
    }
    streams = {
        split: path.open("w", encoding="utf-8")
        for split, path in partial_paths.items()
    }
    progress = tqdm(total=sum(targets.values()), unit="word", desc="Preparing Wikipedia")
    try:
        for row in rows:
            identifier = str(row.get("id", "")).strip()
            if not identifier or identifier in seen_ids:
                continue
            seen_ids.add(identifier)
            split = choose_split(identifier, seed, targets)
            if words[split] >= targets[split]:
                continue
            article = normalize_article(str(row.get("text", "")))
            if not article:
                continue
            article_words = len(article.split())
            streams[split].write(article + "\n")
            words[split] += article_words
            articles[split] += 1
            progress.update(article_words)
            if all(words[name] >= targets[name] for name in expected):
                break
        else:
            missing = {
                split: targets[split] - words[split]
                for split in expected
                if words[split] < targets[split]
            }
            raise RuntimeError(f"Wikipedia stream ended before targets: {missing}")
    finally:
        progress.close()
        for stream in streams.values():
            stream.close()

    for split in ("train", "validation", "test"):
        partial_paths[split].replace(paths[split])

    return {
        split: {
            "target_words": targets[split],
            "actual_words": words[split],
            "articles": articles[split],
            "text_file": paths[split].name,
        }
        for split in ("train", "validation", "test")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wikipedia/raw"))
    parser.add_argument("--train-words", type=int, default=100_000_000)
    parser.add_argument("--validation-words", type=int, default=1_000_000)
    parser.add_argument("--test-words", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=DATASET_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = {
        "train": args.train_words,
        "validation": args.validation_words,
        "test": args.test_words,
    }
    if any(value <= 0 for value in targets.values()):
        raise ValueError("Wikipedia word targets must be positive")
    resolved_revision = HfApi().dataset_info(
        DATASET_ID,
        revision=args.revision,
    ).sha
    wikipedia = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=resolved_revision,
    )
    statistics = prepare_rows(wikipedia, args.output_dir, targets, args.seed)
    manifest = {
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": resolved_revision,
        "license": ["CC BY-SA 3.0", "GFDL"],
        "seed": args.seed,
        "split_method": "SHA256(seed:article_id), proportional word-budget buckets",
        "datasets_version": datasets.__version__,
        "splits": statistics,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset: {DATASET_ID}/{DATASET_CONFIG}@{resolved_revision}")
    for split, values in statistics.items():
        print(
            f"{split:10s} {values['actual_words']:,} words, "
            f"{values['articles']:,} articles"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

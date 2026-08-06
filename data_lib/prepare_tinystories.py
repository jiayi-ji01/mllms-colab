"""Prepare deterministic TinyStories train, validation, and test splits."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import datasets
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "roneneldan/TinyStories"
# 固定官方数据版本，避免未来仓库更新改变实验数据
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

SEED = 42

TARGETS = {
    # The tokenizer stage truncates by actual encoded-token count. Preparing at
    # least 100M words guarantees enough source text without estimating tokens
    # from text-file size.
    "train": 100_000_000,
    "validation": 200_000,
    "test": 200_000,
}

# 将 don't、girl's 等带撇号形式计为一个 English word
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def count_words(text: str) -> int:
    """Count English words in one story."""
    return len(WORD_PATTERN.findall(text))


def normalize_for_sentencepiece(text: str) -> str:
    """
    SentencePiece 默认按行读取语料。

    将一个完整故事转为一行，但 JSONL 文件仍保留原始换行。
    """
    return " ".join(text.split())


def story_hash(text: str) -> str:
    """Create a stable identifier from story contents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_story(jsonl_file, text_file, text: str, source_split: str) -> int:
    words = count_words(text)
    identifier = story_hash(text)

    record = {
        "id": identifier,
        "source_split": source_split,
        "word_count": words,
        "text": text,
    }

    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    text_file.write(normalize_for_sentencepiece(text) + "\n")

    return words


def select_one_split(
    dataset,
    output_dir: Path,
    output_name: str,
    target_words: int,
    source_split: str,
    seen_hashes: set[str],
) -> dict:
    actual_words = 0
    story_count = 0
    duplicate_count = 0

    jsonl_path = output_dir / f"{output_name}.jsonl"
    text_path = output_dir / f"{output_name}.txt"

    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        text_path.open("w", encoding="utf-8") as text_file,
    ):
        progress = tqdm(
            total=target_words,
            unit="word",
            desc=f"Preparing {output_name}",
        )

        for row in dataset:
            text = row["text"].strip()

            if not text:
                continue

            identifier = story_hash(text)

            # 防止完全相同的故事出现在多个输出 split 中
            if identifier in seen_hashes:
                duplicate_count += 1
                continue

            words = count_words(text)

            if words == 0:
                continue

            written_words = write_story(
                jsonl_file=jsonl_file,
                text_file=text_file,
                text=text,
                source_split=source_split,
            )

            seen_hashes.add(identifier)
            actual_words += written_words
            story_count += 1
            progress.update(written_words)

            # 只在完整故事写完后停止，所以实际词数可能略高
            if actual_words >= target_words:
                break

        progress.close()

    if actual_words < target_words:
        raise RuntimeError(
            f"{output_name} only reached {actual_words:,} words; "
            f"target was {target_words:,}."
        )

    return {
        "source_split": source_split,
        "target_words": target_words,
        "actual_words": actual_words,
        "story_count": story_count,
        "duplicates_skipped": duplicate_count,
        "jsonl_file": jsonl_path.name,
        "text_file": text_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Remove it manually or choose another --output-dir."
        )

    output_dir.mkdir(parents=True)

    print(f"Downloading {DATASET_NAME}")
    print(f"Revision: {DATASET_REVISION}")
    print(f"Random seed: {SEED}")

    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
    )

    # 精确的全数据集洗牌，不使用 streaming shuffle buffer
    shuffled_train = dataset["train"].shuffle(seed=SEED)
    shuffled_validation = dataset["validation"].shuffle(seed=SEED)

    seen_hashes: set[str] = set()

    statistics = {}

    statistics["train"] = select_one_split(
        dataset=shuffled_train,
        output_dir=output_dir,
        output_name="train",
        target_words=TARGETS["train"],
        source_split="train",
        seen_hashes=seen_hashes,
    )

    # Validation 先取得前 200k+ words
    statistics["validation"] = select_one_split(
        dataset=shuffled_validation,
        output_dir=output_dir,
        output_name="validation",
        target_words=TARGETS["validation"],
        source_split="validation",
        seen_hashes=seen_hashes,
    )

    # 因为已选故事的 hash 在 seen_hashes 中，接下来会自动跳过它们
    statistics["test"] = select_one_split(
        dataset=shuffled_validation,
        output_dir=output_dir,
        output_name="test",
        target_words=TARGETS["test"],
        source_split="validation",
        seen_hashes=seen_hashes,
    )

    manifest = {
        "dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "word_definition": WORD_PATTERN.pattern,
        "datasets_version": datasets.__version__,
        "splits": statistics,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFinished:")
    for split_name, stats in statistics.items():
        print(
            f"{split_name:10s} "
            f"{stats['actual_words']:,} words, "
            f"{stats['story_count']:,} stories"
        )

    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()

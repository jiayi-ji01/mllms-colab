"""Download and assemble the official BabyLM 100M-word corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASET_ID = "cambridge-climb/BabyLM"
SOURCE_FILES = (
    "aochildes.txt",
    "bnc_spoken.txt",
    "cbt.txt",
    "children_stories.txt",
    "gutenberg.txt",
    "open_subtitles.txt",
    "qed.txt",
    "simple_wikipedia.txt",
    "switchboard.txt",
    "wikipedia.txt",
)
SPLITS = {
    "train": "100M",
    "validation": "dev",
    "test": "test",
}


def assemble_split(source_dir: Path, source_name: str, output: Path) -> dict:
    """Combine BabyLM domains while retaining newline document boundaries."""
    documents = 0
    words = 0
    source_counts: dict[str, dict[str, int]] = {}

    with output.open("w", encoding="utf-8") as destination:
        for filename in SOURCE_FILES:
            path = source_dir / "clean" / source_name / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing downloaded BabyLM file: {path}")

            file_documents = 0
            file_words = 0
            with path.open(encoding="utf-8") as source:
                for line in source:
                    document = " ".join(line.split())
                    if not document:
                        continue
                    destination.write(document + "\n")
                    file_documents += 1
                    file_words += len(document.split())

            documents += file_documents
            words += file_words
            source_counts[filename] = {
                "documents": file_documents,
                "words": file_words,
            }

    return {
        "documents": documents,
        "words": words,
        "source_files": source_counts,
        "text_file": output.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/babylm/raw"),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision; it is resolved to a commit SHA.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.output_dir / "source"
    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[
            f"clean/{source_name}/*.txt"
            for source_name in SPLITS.values()
        ],
        local_dir=source_dir,
    )

    statistics = {
        split: assemble_split(
            source_dir,
            source_name,
            args.output_dir / f"{split}.txt",
        )
        for split, source_name in SPLITS.items()
    }
    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": revision,
        "variant": "clean/100M with official dev and test",
        "word_definition": "whitespace-separated BabyLM words",
        "splits": statistics,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: {DATASET_ID}@{revision}")
    for split, values in statistics.items():
        print(
            f"{split:10s} {values['documents']:,} documents, "
            f"{values['words']:,} words"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

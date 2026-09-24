"""Download the BLiMP subject-verb agreement evaluation set."""

import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "nyu-mll/blimp"
DATASET_REVISION = "877fba0801ffb7cbd8c39c1ff314a46f053f6036"
OUTPUT_FILE = Path("data/blimp/raw/agreement.jsonl")

SUBTASKS = (
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for subtask in SUBTASKS:
            dataset = load_dataset(
                DATASET_NAME,
                subtask,
                split="train",
                revision=DATASET_REVISION,
            )
            counts[subtask] = len(dataset)

            for row in dataset:
                record = {
                    "sample_id": f"{subtask}:{row['pair_id']:04d}",
                    "subtask": subtask,
                    "sentence_good": row["sentence_good"],
                    "sentence_bad": row["sentence_bad"],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUTPUT_FILE}")
    for subtask, count in counts.items():
        print(f"{subtask}: {count:,}")
    print(f"Total: {sum(counts.values()):,}")


if __name__ == "__main__":
    main()

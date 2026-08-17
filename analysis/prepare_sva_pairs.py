"""Prepare balanced CausalGym-style SVA clean/corrupted prompt pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from huggingface_hub import hf_hub_download

from tokenizer.tokenizer import load_tokenizer


DATASET_ID = "aryaman/causalgym"
DATASET_REVISION = "95349c3a5e53e2506e8b212482ea6dd784978156"
SVA_TASKS = (
    "agr_sv_num_pp",
    "agr_sv_num_obj-relc",
    "agr_sv_num_subj-relc",
)
LABEL_PARADIGMS = (
    {"singular": "is", "plural": "are"},
    {"singular": "was", "plural": "were"},
    {"singular": "has", "plural": "have"},
)


def prompt_from_spans(spans: list[str]) -> str:
    """Join aligned CausalGym spans and replace its GPT-2 EOS marker."""
    prompt = "".join(spans)
    if prompt.startswith("<|endoftext|>"):
        prompt = prompt[len("<|endoftext|>") :]
    return " ".join(prompt.split())


def encode_prompt_answer(tokenizer, prompt: str, answer: str) -> tuple[list[int], int]:
    """Encode a prompt and require its answer to be one SentencePiece token."""
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    full_ids = tokenizer.encode(f"{prompt} {answer}", out_type=int)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("answer changes prompt tokenization")
    answer_ids = full_ids[len(prompt_ids) :]
    if len(answer_ids) != 1:
        raise ValueError(f"answer {answer!r} is not one tokenizer token")
    return [tokenizer.eos_id(), *prompt_ids], answer_ids[0]


def aligned_span_boundaries(tokenizer, clean: list[str], corrupted: list[str]) -> list[int]:
    """Require every controlled span to end at the same token position."""
    if len(clean) != len(corrupted):
        raise ValueError("clean/corrupted span counts differ")
    clean_boundaries = [
        len(tokenizer.encode(prompt_from_spans(clean[:end]), out_type=int))
        for end in range(1, len(clean) + 1)
    ]
    corrupted_boundaries = [
        len(tokenizer.encode(prompt_from_spans(corrupted[:end]), out_type=int))
        for end in range(1, len(corrupted) + 1)
    ]
    if clean_boundaries != corrupted_boundaries:
        raise ValueError("clean/corrupted span token boundaries differ")
    # Prepared prompts have one EOS token before the SentencePiece tokens.
    return [boundary + 1 for boundary in clean_boundaries]


def load_official_rows(splits: list[str], revision: str) -> list[dict]:
    """Download the small official JSON splits, pinned to a commit."""
    rows = []
    for split in splits:
        path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"{split}.json",
            revision=revision,
        )
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        for index, record in enumerate(document):
            if record.get("task") in SVA_TASKS:
                rows.append({**record, "source_split": split, "source_index": index})
    return rows


def prepare_candidate(tokenizer, record: dict) -> dict:
    """Turn a counterfactual CausalGym row into a token-aligned SVA pair."""
    clean_type = str(record["base_type"])
    corrupted_type = str(record["src_type"])
    if {clean_type, corrupted_type} != {"singular", "plural"}:
        raise ValueError("pair does not contrast singular and plural")

    # CausalGym sometimes pairs different auxiliary lemmas. Reusing a fixed
    # inflectional paradigm keeps LD focused on number rather than lexical choice.
    selector = (
        record["source_index"]
        + sum(map(ord, record["task"]))
        + sum(map(ord, record["source_split"]))
    ) % len(LABEL_PARADIGMS)
    paradigm = LABEL_PARADIGMS[selector]
    clean_answer = paradigm[clean_type]
    corrupted_answer = paradigm[corrupted_type]
    token_boundaries = aligned_span_boundaries(
        tokenizer, record["base"], record["src"]
    )
    clean_prompt = prompt_from_spans(record["base"])
    corrupted_prompt = prompt_from_spans(record["src"])
    clean_ids, clean_answer_id = encode_prompt_answer(
        tokenizer, clean_prompt, clean_answer
    )
    corrupted_ids, corrupted_answer_id = encode_prompt_answer(
        tokenizer, corrupted_prompt, corrupted_answer
    )
    if len(clean_ids) != len(corrupted_ids):
        raise ValueError("clean/corrupted tokenizer lengths differ")
    if clean_answer_id == corrupted_answer_id:
        raise ValueError("answer token IDs are identical")

    return {
        "sample_id": (
            f"{record['source_split']}:{record['task']}:"
            f"{record['source_index']:04d}"
        ),
        "dataset": DATASET_ID,
        "source_split": record["source_split"],
        "task": record["task"],
        "clean_type": clean_type,
        "corrupted_type": corrupted_type,
        "clean_prompt": clean_prompt,
        "corrupted_prompt": corrupted_prompt,
        "clean_answer": clean_answer,
        "corrupted_answer": corrupted_answer,
        "clean_input_ids": clean_ids,
        "corrupted_input_ids": corrupted_ids,
        "clean_answer_id": clean_answer_id,
        "corrupted_answer_id": corrupted_answer_id,
        "aligned_span_end_positions": token_boundaries,
        "official_base_label": str(record["base_label"]).strip(),
        "official_src_label": str(record["src_label"]).strip(),
    }


def balanced_sample(candidates: list[dict], count: int, seed: int) -> list[dict]:
    """Select exactly balanced clean singular/plural pairs across SVA tasks."""
    generator = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate["clean_type"], candidate["task"])].append(candidate)
    for bucket in buckets.values():
        generator.shuffle(bucket)

    targets = {"singular": count // 2, "plural": count - count // 2}
    selected = []
    for clean_type, target in targets.items():
        task_buckets = [buckets[(clean_type, task)] for task in SVA_TASKS]
        task_offsets = [0] * len(task_buckets)
        selected_for_type = 0
        while selected_for_type < target:
            made_progress = False
            for task_index, bucket in enumerate(task_buckets):
                if task_offsets[task_index] >= len(bucket):
                    continue
                selected.append(bucket[task_offsets[task_index]])
                task_offsets[task_index] += 1
                selected_for_type += 1
                made_progress = True
                if selected_for_type >= target:
                    break
            if not made_progress:
                raise ValueError(
                    f"Only found {sum(len(bucket) for bucket in task_buckets)} "
                    f"valid {clean_type} pairs; cannot select {target}"
                )
    generator.shuffle(selected)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/babylm_tokenizer/tokenizer.model"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/causalgym/sva_pairs.jsonl"),
    )
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        default="train,dev,test",
        help="Comma-separated official splits used to form the fixed pair bank.",
    )
    parser.add_argument("--revision", default=DATASET_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_pairs <= 0:
        raise ValueError("num-pairs must be positive")
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    unknown = set(splits) - {"train", "dev", "test"}
    if not splits or unknown:
        raise ValueError(f"Invalid CausalGym splits: {sorted(unknown)}")

    tokenizer = load_tokenizer(args.tokenizer)
    official_rows = load_official_rows(splits, args.revision)
    candidates = []
    rejected = Counter()
    for record in official_rows:
        try:
            candidates.append(prepare_candidate(tokenizer, record))
        except (KeyError, TypeError, ValueError) as error:
            rejected[str(error)] += 1
    selected = balanced_sample(candidates, args.num_pairs, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in selected:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": DATASET_ID,
        "dataset_revision": args.revision,
        "source_splits": splits,
        "tasks": list(SVA_TASKS),
        "seed": args.seed,
        "requested_pairs": args.num_pairs,
        "saved_pairs": len(selected),
        "clean_type_counts": Counter(row["clean_type"] for row in selected),
        "task_counts": Counter(row["task"] for row in selected),
        "source_split_counts": Counter(row["source_split"] for row in selected),
        "label_pair_counts": Counter(
            f"{row['clean_answer']}/{row['corrupted_answer']}" for row in selected
        ),
        "rejected_counts": rejected,
        "note": (
            "Prompts come from CausalGym; labels use matched is/are, was/were, "
            "or has/have paradigms so LD isolates number agreement."
        ),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Official SVA rows: {len(official_rows):,}")
    print(f"Tokenizer-aligned candidates: {len(candidates):,}")
    print(f"Saved pairs: {len(selected):,} -> {args.output}")
    print(f"Clean types: {dict(metadata['clean_type_counts'])}")
    print(f"Tasks: {dict(metadata['task_counts'])}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

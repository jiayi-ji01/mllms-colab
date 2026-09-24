"""Summarize prediction-position head effects against a patching control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mllms.interpretability.activation_patching.statistics import (
    holm_adjust,
    stratified_paired_bootstrap,
)


DISTRACTOR_TASKS = ("pp_attractor", "object_relative", "subject_relative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--control",
        choices=("opposite-number", "opposite-number-shuffled"),
        default="opposite-number-shuffled",
    )
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["8:3"],
        help="Prediction-position attention sites as LAYER:HEAD.",
    )
    return parser.parse_args()


def _task_lookup(path: Path) -> dict[str, str]:
    lookup = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                lookup[str(record["sample_id"])] = str(record["task"])
    return lookup


def _load_head_delta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path)
    positions = archive["relative_positions"]
    matches = np.flatnonzero(positions == -1)
    if matches.size != 1:
        raise ValueError(f"{path} must contain prediction position -1")
    values = archive["head_out_delta_ld"][:, :, int(matches[0]), :]
    return archive["sample_ids"].astype(str), values


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    task_lookup = _task_lookup(args.data)
    sites = []
    for value in args.sites:
        try:
            layer, head = (int(part) for part in value.split(":", maxsplit=1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid site {value!r}; expected LAYER:HEAD") from error
        sites.append((layer, head))
    rows = []
    direction_dirs = sorted(
        path for path in args.results_dir.glob("*_to_*") if path.is_dir()
    )
    for direction_index, direction_dir in enumerate(direction_dirs):
        observed_path = direction_dir / "clean/per_example_scores.npz"
        control_path = direction_dir / args.control / "per_example_scores.npz"
        if not observed_path.is_file() or not control_path.is_file():
            continue
        sample_ids, observed = _load_head_delta(observed_path)
        control_ids, control = _load_head_delta(control_path)
        if not np.array_equal(sample_ids, control_ids):
            raise ValueError(f"sample order differs for {direction_dir.name}")
        tasks = np.asarray([task_lookup[sample_id] for sample_id in sample_ids])
        groups = {
            "pooled_distractor": np.isin(tasks, DISTRACTOR_TASKS),
            **{task: tasks == task for task in ("simple", *DISTRACTOR_TASKS)},
        }
        for layer, head in sites:
            if not 0 <= layer < observed.shape[1] or not 0 <= head < observed.shape[2]:
                raise ValueError(
                    f"site {layer}:{head} is outside head array {observed.shape}"
                )
            site_rows = []
            for task_index, (task, mask) in enumerate(groups.items()):
                if not mask.any():
                    continue
                statistics = stratified_paired_bootstrap(
                    observed[mask, layer, head],
                    control[mask, layer, head],
                    tasks[mask],
                    iterations=args.iterations,
                    seed=(
                        args.seed
                        + direction_index * 100_000
                        + layer * 1_000
                        + head * 10
                        + task_index
                    ),
                )
                site_rows.append({
                    "direction": direction_dir.name,
                    "control": args.control,
                    "task": task,
                    "layer": layer,
                    "head": head,
                    **statistics,
                })
            distractor_rows = [
                row for row in site_rows if row["task"] in DISTRACTOR_TASKS
            ]
            adjusted = holm_adjust([
                float(row["bootstrap_p_two_sided"])
                for row in distractor_rows
            ])
            for row, value in zip(distractor_rows, adjusted):
                row["holm_p"] = value
            for row in site_rows:
                row.setdefault("holm_p", "")
            rows.extend(site_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "prediction_head_statistics.csv"
    fields = list(rows[0]) if rows else [
        "direction", "control", "task", "layer", "head",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "results_dir": str(args.results_dir),
        "data": str(args.data),
        "control": args.control,
        "iterations": args.iterations,
        "seed": args.seed,
        "sites": [f"{layer}:{head}" for layer, head in sites],
        "directions": [path.name for path in direction_dirs],
        "num_site_task_rows": len(rows),
        "statistics_csv": str(csv_path),
        "claim_rule": (
            "Both cross-language directions must beat the control for pooled "
            "distractors and replicate in at least two distractor tasks."
        ),
    }
    (args.output_dir / "statistics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Statistics: {csv_path}")


if __name__ == "__main__":
    main()

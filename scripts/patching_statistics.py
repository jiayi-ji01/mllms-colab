"""Paired patching statistics and trajectory uncertainty."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import numpy as np


def stratified_paired_bootstrap(
    observed: np.ndarray,
    control: np.ndarray,
    strata: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Estimate a paired effect and percentile CI with fixed stratum weights."""
    observed = np.asarray(observed, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    strata = np.asarray(strata)
    if observed.shape != control.shape or observed.shape != strata.shape:
        raise ValueError("observed, control, and strata must have the same shape")
    if observed.ndim != 1:
        raise ValueError("bootstrap inputs must be one-dimensional")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    valid = np.isfinite(observed) & np.isfinite(control)
    differences = observed[valid] - control[valid]
    valid_strata = strata[valid]
    if differences.size == 0:
        raise ValueError("bootstrap requires at least one finite pair")

    indices = [
        np.flatnonzero(valid_strata == value)
        for value in np.unique(valid_strata)
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = np.concatenate([
            rng.choice(group, size=group.size, replace=True) for group in indices
        ])
        draws[iteration] = differences[sampled].mean()
    probability_nonpositive = float(np.mean(draws <= 0.0))
    probability_nonnegative = float(np.mean(draws >= 0.0))
    return {
        "num_examples": int(differences.size),
        "mean_observed": float(observed[valid].mean()),
        "mean_control": float(control[valid].mean()),
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_p_two_sided": min(
            1.0,
            2.0 * min(probability_nonpositive, probability_nonnegative),
        ),
        "iterations": iterations,
        "seed": seed,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down family-wise-error adjusted p-values."""
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_intervals(values, strata, clusters=None, *, iterations=10_000, seed=42):
    """Return means and pointwise percentile intervals for one or more columns.

    Sample clusters within each task, retaining all rows in each sampled
    cluster. Within-task means are row-weighted; original task weights stay
    fixed. Without clusters this reduces to stratified row bootstrap. Missing
    values fail loudly instead of silently changing the shared cohort.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    strata = np.asarray(strata)
    if values.ndim != 2 or not len(values) or strata.shape != (len(values),):
        raise ValueError("values must be nonempty [examples, outcomes] with matching strata")
    if not np.isfinite(values).all() or iterations < 2:
        raise ValueError("finite values and at least two iterations are required")
    clusters = np.arange(len(values)) if clusters is None else np.asarray(clusters)
    if clusters.shape != strata.shape:
        raise ValueError("clusters must match strata")
    rng = np.random.default_rng(seed)
    draws = np.zeros((iterations, values.shape[1]))
    num_clusters = 0
    for task in np.unique(strata):
        indices = np.flatnonzero(strata == task)
        _, inverse = np.unique(clusters[indices], return_inverse=True)
        count = int(inverse.max()) + 1
        num_clusters += count
        sums = np.zeros((count, values.shape[1]))
        np.add.at(sums, inverse, values[indices])
        sizes = np.bincount(inverse).astype(float)
        for start in range(0, iterations, 250):
            stop = min(start + 250, iterations)
            weights = rng.multinomial(count, np.full(count, 1 / count), size=stop-start)
            means = (weights @ sums) / (weights @ sizes)[:, None]
            draws[start:stop] += means * (len(indices) / len(values))
    return {
        "mean": values.mean(axis=0),
        "ci_low": np.quantile(draws, .025, axis=0),
        "ci_high": np.quantile(draws, .975, axis=0),
        "num_examples": len(values),
        "num_clusters": num_clusters,
    }


def assert_same_ids(expected, actual, source):
    """Reject duplicates, reorderings and missing samples before paired analysis."""
    expected, actual = np.asarray(expected).astype(str), np.asarray(actual).astype(str)
    if len(set(actual)) != len(actual) or not np.array_equal(expected, actual):
        raise ValueError(f"sample IDs differ or repeat: {source}")


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

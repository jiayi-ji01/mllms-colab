"""Create figures and CSV summaries from experiment outputs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _latest_by_step(records: list[dict], record_type: str) -> list[dict]:
    by_step = {
        int(record["step"]): record
        for record in records
        if record.get("type") == record_type
    }
    return [by_step[step] for step in sorted(by_step)]


def _smooth(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.array(
        [array[max(0, index - window + 1) : index + 1].mean()
         for index in range(len(array))]
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training(run_dir: Path, output_dir: Path, smooth_window: int) -> None:
    records = _read_jsonl(run_dir / "train_log.jsonl")
    train_rows = _latest_by_step(records, "train")
    validation_rows = _latest_by_step(records, "validation")
    if not train_rows:
        raise ValueError("No training records found")

    steps = [row["step"] for row in train_rows]
    losses = [row["train_loss"] for row in train_rows]
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(steps, losses, alpha=0.25, label="Raw")
    axes[0, 0].plot(
        steps,
        _smooth(losses, smooth_window),
        linewidth=2,
        label=f"Moving average ({smooth_window})",
    )
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()

    if validation_rows:
        validation_steps = [row["step"] for row in validation_rows]
        for key, label in (
            ("original_loss", "Original"),
            ("clone_loss", "Clone"),
            ("average_loss", "Average"),
        ):
            axes[0, 1].plot(
                validation_steps,
                [row[key] for row in validation_rows],
                marker="o",
                label=label,
            )
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "No validation record yet", ha="center")
    axes[0, 1].set_title("Validation Loss")

    axes[1, 0].plot(steps, [row["learning_rate"] for row in train_rows])
    axes[1, 0].set_title("Learning Rate")

    axes[1, 1].plot(steps, [row["gradient_norm"] for row in train_rows])
    axes[1, 1].axhline(1.0, color="red", linestyle="--", label="Clip threshold")
    axes[1, 1].set_title("Gradient Norm Before Clipping")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Optimizer Step")
        axis.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 1].set_ylabel("Gradient Norm")

    figure_path = output_dir / "training_report.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    last_train = train_rows[-1]
    summary = {
        "last_step": last_train["step"],
        "last_train_loss": last_train["train_loss"],
        "tokens_seen": last_train["tokens_seen"],
        "original_tokens_seen": last_train["original_tokens_seen"],
        "clone_tokens_seen": last_train["clone_tokens_seen"],
        "clone_token_ratio": (
            last_train["clone_tokens_seen"] / last_train["tokens_seen"]
        ),
        "best_validation_step": "",
        "best_validation_loss": "",
        "last_original_loss": "",
        "last_clone_loss": "",
        "last_clone_loss_gap": "",
    }
    if validation_rows:
        best = min(validation_rows, key=lambda row: row["average_loss"])
        last_validation = validation_rows[-1]
        summary.update(
            {
                "best_validation_step": best["step"],
                "best_validation_loss": best["average_loss"],
                "last_original_loss": last_validation["original_loss"],
                "last_clone_loss": last_validation["clone_loss"],
                "last_clone_loss_gap": (
                    last_validation["clone_loss"]
                    - last_validation["original_loss"]
                ),
            }
        )
    summary_path = output_dir / "training_summary.csv"
    _write_csv(summary_path, [summary])

    print(f"Last step: {summary['last_step']:,}")
    print(f"Last train loss: {summary['last_train_loss']:.4f}")
    print(f"Clone token ratio: {summary['clone_token_ratio']:.4f}")
    if validation_rows:
        print(f"Best validation loss: {summary['best_validation_loss']:.4f}")
        print(f"Clone loss gap: {summary['last_clone_loss_gap']:+.4f}")
    print(f"Figure: {figure_path}")
    print(f"Table: {summary_path}")


def plot_blimp(results_dir: Path, output_dir: Path) -> None:
    summary = json.loads(
        (results_dir / "agreement_summary.json").read_text(encoding="utf-8")
    )
    rows = [{"subtask": "overall", **summary["overall"]}]
    rows.extend(
        {"subtask": name, **metrics}
        for name, metrics in summary["by_subtask"].items()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["subtask"] for row in rows]
    positions = np.arange(len(rows))
    height = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].barh(
        positions - height / 2,
        [row["original_accuracy"] for row in rows],
        height,
        label="Original",
    )
    axes[0].barh(
        positions + height / 2,
        [row["clone_accuracy"] for row in rows],
        height,
        label="Clone",
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_title("BLiMP Accuracy")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].legend()

    axes[1].barh(
        positions,
        [row["original_clone_accuracy_gap"] for row in rows],
        label="Original − Clone accuracy",
    )
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_title("Original–Clone Accuracy Gap")
    axes[1].set_xlabel("Accuracy Gap")
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    figure_path = output_dir / "blimp_report.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    table_rows = [
        {
            "subtask": row["subtask"],
            "examples": row["num_examples"],
            "original_accuracy": row["original_accuracy"],
            "clone_accuracy": row["clone_accuracy"],
            "accuracy_gap": row["original_clone_accuracy_gap"],
            "prediction_agreement": row["prediction_agreement_rate"],
            "original_mean_margin": row["original_mean_margin"],
            "clone_mean_margin": row["clone_mean_margin"],
        }
        for row in rows
    ]
    table_path = output_dir / "blimp_summary.csv"
    _write_csv(table_path, table_rows)

    overall = rows[0]
    print(f"Original accuracy: {overall['original_accuracy']:.4f}")
    print(f"Clone accuracy: {overall['clone_accuracy']:.4f}")
    print(f"Accuracy gap: {overall['original_clone_accuracy_gap']:+.4f}")
    print(f"Prediction agreement: {overall['prediction_agreement_rate']:.4f}")
    print(f"Figure: {figure_path}")
    print(f"Table: {table_path}")


def plot_sva(results_dir: Path, output_dir: Path) -> None:
    """Plot controlled SVA accuracy and correctly oriented logit margins."""
    summary = json.loads(
        (results_dir / "sva_summary.json").read_text(encoding="utf-8")
    )
    groups = [("overall", summary["overall"]), *summary["by_task"].items()]
    rows = []
    for group_name, group in groups:
        for language in ("original", "clone"):
            metrics = group[language]
            rows.append(
                {
                    "group": group_name,
                    "language": language,
                    "pairs": metrics["num_pairs"],
                    "accuracy": metrics["accuracy"],
                    "pair_accuracy": metrics["pair_accuracy"],
                    "mean_logit_difference": metrics["mean_logit_difference"],
                    "clean_accuracy": metrics["clean_accuracy"],
                    "corrupted_accuracy": metrics["corrupted_accuracy"],
                    "joint_sanity_pairs": group["joint_sanity_pairs"],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [name for name, _ in groups]
    positions = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for offset, language in ((-width / 2, "original"), (width / 2, "clone")):
        language_rows = [row for row in rows if row["language"] == language]
        axes[0].bar(
            positions + offset,
            [row["accuracy"] for row in language_rows],
            width,
            label=language.title(),
        )
        axes[1].bar(
            positions + offset,
            [row["mean_logit_difference"] for row in language_rows],
            width,
            label=language.title(),
        )
    axes[0].set_title("Controlled SVA Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("Mean Correct − Incorrect Logit Difference")
    axes[1].set_ylabel("Mean logit difference")
    axes[1].axhline(0.0, color="black", linewidth=1)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    figure_path = output_dir / "sva_report.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    table_path = output_dir / "sva_summary.csv"
    _write_csv(table_path, rows)

    overall = summary["overall"]
    for language in ("original", "clone"):
        metrics = overall[language]
        print(
            f"{language:8s} accuracy={metrics['accuracy']:.4f} | "
            f"mean_LD={metrics['mean_logit_difference']:+.4f}"
        )
    print(f"Joint sanity pairs: {overall['joint_sanity_pairs']:,}")
    print(f"Figure: {figure_path}")
    print(f"Table: {table_path}")


def plot_patching(
    results_dir: Path,
    output_dir: Path,
    top_k: int,
    language: str,
    metric: str,
) -> None:
    components = ("resid_post", "attn_out", "mlp_out", "head_out")
    language_dir = results_dir / language
    if not language_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {language} patching results: {language_dir}"
        )
    means = {
        component: np.load(language_dir / f"{component}_{metric}_mean.npy")
        for component in components
    }
    archive = np.load(language_dir / "per_example_scores.npz")
    relative_positions = archive["relative_positions"]
    output_dir.mkdir(parents=True, exist_ok=True)

    titles = {
        "resid_post": "Residual Stream",
        "attn_out": "Attention Output",
        "mlp_out": "MLP Output",
        "head_out": "Attention Heads",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, component in zip(axes.flat, components):
        full_values = means[component]
        values = full_values[:, -1] if component == "head_out" else full_values
        finite = np.abs(values[np.isfinite(values)])
        limit = float(np.percentile(finite, 95)) if finite.size else 1.0
        limit = max(limit, 1e-6)
        image = axis.imshow(
            values,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(titles[component])
        axis.set_ylabel("Layer")
        x_label = "Head" if component == "head_out" else "Relative position"
        axis.set_xlabel(x_label)
        if component == "head_out":
            axis.set_xticks(np.arange(values.shape[1]))
        else:
            ticks = np.unique(
                np.linspace(
                    0,
                    len(relative_positions) - 1,
                    min(8, len(relative_positions)),
                ).astype(int)
            )
            axis.set_xticks(ticks, relative_positions[ticks])
        figure.colorbar(image, ax=axis, label=f"Mean {metric}")

    figure.suptitle(f"{language.title()} SVA Patching: {metric}")
    figure_path = output_dir / f"patching_{language}_{metric}.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    sites = []
    for component, values in means.items():
        if component == "head_out":
            for layer in range(values.shape[0]):
                for position_index, position in enumerate(relative_positions):
                    for head in range(values.shape[2]):
                        score = values[layer, position_index, head]
                        if np.isfinite(score):
                            sites.append(
                                {
                                    "component": component,
                                    "layer": layer,
                                    "relative_position": int(position),
                                    "head": head,
                                    metric: float(score),
                                    "absolute_score": abs(float(score)),
                                }
                            )
        else:
            for layer, row in enumerate(values):
                for position_index, score in enumerate(row):
                    if np.isfinite(score):
                        sites.append(
                            {
                                "component": component,
                                "layer": layer,
                                "relative_position": int(
                                    relative_positions[position_index]
                                ),
                                "head": "",
                                metric: float(score),
                                "absolute_score": abs(float(score)),
                            }
                        )
    sites.sort(key=lambda row: row["absolute_score"], reverse=True)
    table_path = output_dir / f"patching_top_sites_{language}_{metric}.csv"
    _write_csv(table_path, sites[:top_k])

    print("Top patching sites:")
    for row in sites[: min(10, top_k)]:
        print(
            f"  {row['component']:10s} layer={row['layer']:2d} "
            f"position={row['relative_position']:3d} "
            f"head={str(row['head']):>2s} score={row[metric]:+.4f}"
        )
    print(f"Figure: {figure_path}")
    print(f"Table: {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="report", required=True)

    training = subparsers.add_parser("training")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--output-dir", type=Path)
    training.add_argument("--smooth-window", type=int, default=20)

    blimp = subparsers.add_parser("blimp")
    blimp.add_argument("--results-dir", type=Path, required=True)
    blimp.add_argument("--output-dir", type=Path)

    sva = subparsers.add_parser("sva")
    sva.add_argument("--results-dir", type=Path, required=True)
    sva.add_argument("--output-dir", type=Path)

    patching = subparsers.add_parser("patching")
    patching.add_argument("--results-dir", type=Path, required=True)
    patching.add_argument("--output-dir", type=Path)
    patching.add_argument("--top-k", type=int, default=50)
    patching.add_argument(
        "--language", choices=("original", "clone"), default="original"
    )
    patching.add_argument(
        "--metric", choices=("delta_ld", "recovery"), default="recovery"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report == "training":
        plot_training(
            args.run_dir,
            args.output_dir or args.run_dir,
            args.smooth_window,
        )
    elif args.report == "blimp":
        plot_blimp(
            args.results_dir,
            args.output_dir or args.results_dir,
        )
    elif args.report == "sva":
        plot_sva(
            args.results_dir,
            args.output_dir or args.results_dir,
        )
    else:
        plot_patching(
            args.results_dir,
            args.output_dir or args.results_dir,
            args.top_k,
            args.language,
            args.metric,
        )


if __name__ == "__main__":
    main()

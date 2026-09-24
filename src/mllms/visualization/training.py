"""Training-loss, validation-loss, learning-rate, and gradient plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mllms.visualization.common import latest_by_step, read_jsonl, smooth, write_csv


def plot_training(run_dir: Path, output_dir: Path, smooth_window: int) -> None:
    records = read_jsonl(run_dir / "train_log.jsonl")
    train_rows = latest_by_step(records, "train")
    validation_rows = latest_by_step(records, "validation")
    if not train_rows:
        raise ValueError("No training records found")
    steps = [row["step"] for row in train_rows]
    losses = [row["train_loss"] for row in train_rows]
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].plot(steps, losses, alpha=0.25, label="Raw")
    axes[0, 0].plot(
        steps,
        smooth(losses, smooth_window),
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
        "clone_token_ratio": last_train["clone_tokens_seen"] / last_train["tokens_seen"],
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
                "last_clone_loss_gap": last_validation["clone_loss"] - last_validation["original_loss"],
            }
        )
    summary_path = output_dir / "training_summary.csv"
    write_csv(summary_path, [summary])
    print(f"Figure: {figure_path}")
    print(f"Table: {summary_path}")

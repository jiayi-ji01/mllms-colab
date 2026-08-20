"""Controlled SVA performance plots."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mllms.visualization.common import write_csv


def plot_sva(results_dir: Path, output_dir: Path) -> None:
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
    axes[1].set_title("Mean Correct − Incorrect Sequence Log-Probability")
    axes[1].set_ylabel("Mean sequence log-probability difference")
    axes[1].axhline(0.0, color="black", linewidth=1)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    figure_path = output_dir / "sva_report.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    table_path = output_dir / "sva_summary.csv"
    write_csv(table_path, rows)
    print(f"Figure: {figure_path}")
    print(f"Table: {table_path}")

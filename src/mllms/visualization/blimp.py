"""BLiMP accuracy, margin, and original/clone comparison plots."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mllms.visualization.common import write_csv


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
    write_csv(table_path, table_rows)
    print(f"Figure: {figure_path}")
    print(f"Table: {table_path}")

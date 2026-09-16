"""Layer×token and layer×head plots from saved patching arrays."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mllms.visualization.common import write_csv


COMPONENTS = ("resid_post", "attn_out", "mlp_out", "head_out")


def plot_patching(
    results_dir: Path,
    output_dir: Path,
    top_k: int,
    language: str,
    metric: str,
    direction: str | None = None,
    control: str = "clean",
) -> None:
    language_dir = (
        results_dir / direction / control
        if direction is not None
        else results_dir / language
    )
    if not language_dir.is_dir():
        raise FileNotFoundError(f"Missing patching results: {language_dir}")
    available_components = [
        component
        for component in COMPONENTS
        if (language_dir / f"{component}_{metric}_mean.npy").is_file()
    ]
    if not available_components:
        raise FileNotFoundError(
            f"No {metric} component arrays found in {language_dir}"
        )
    means = {
        component: np.load(language_dir / f"{component}_{metric}_mean.npy")
        for component in available_components
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
    figure, axes = plt.subplots(
        1,
        len(available_components),
        figsize=(6 * len(available_components), 5),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, component in zip(axes.flat, available_components):
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
        axis.set_xlabel("Head" if component == "head_out" else "Relative position")
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
    label = direction or language
    if direction is not None:
        label = f"{direction}/{control}"
    figure.suptitle(f"{label} SVA Patching: {metric}")
    safe_label = label.replace("/", "_")
    figure_path = output_dir / f"patching_{safe_label}_{metric}.png"
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
                                "relative_position": int(relative_positions[position_index]),
                                "head": "",
                                metric: float(score),
                                "absolute_score": abs(float(score)),
                            }
                        )
    sites.sort(key=lambda row: row["absolute_score"], reverse=True)
    table_path = output_dir / f"patching_top_sites_{safe_label}_{metric}.csv"
    write_csv(table_path, sites[:top_k])
    print(f"Figure: {figure_path}")
    print(f"Table: {table_path}")

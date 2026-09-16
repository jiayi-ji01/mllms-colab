"""Paired, task-stratified statistics for activation-patching effects."""

from __future__ import annotations

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

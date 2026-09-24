"""Paired trajectory uncertainty, with optional prompt-cluster resampling.

Columns are outcomes on exactly the same ordered examples. Resampling weights
are shared across columns, so checkpoint contrasts preserve example pairing.
These intervals describe sampling uncertainty, not training-seed uncertainty.
"""

from __future__ import annotations

import numpy as np


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

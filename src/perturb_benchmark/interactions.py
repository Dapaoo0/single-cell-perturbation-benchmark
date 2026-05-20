"""Descriptive, non-circular interaction diagnostics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def interaction_residual(
    observed_double: np.ndarray,
    single_a: np.ndarray,
    single_b: np.ndarray,
) -> np.ndarray:
    """Return observed double response minus the additive single-gene response."""

    observed = np.asarray(observed_double, dtype=float)
    a = np.asarray(single_a, dtype=float)
    b = np.asarray(single_b, dtype=float)
    if observed.ndim != 1 or a.shape != observed.shape or b.shape != observed.shape:
        raise ValueError("interaction inputs must be aligned one-dimensional vectors")
    if not np.isfinite(np.concatenate([observed, a, b])).all():
        raise ValueError("interaction inputs must be finite")
    return observed - a - b


def assign_interaction_strata(
    magnitudes: pd.Series,
    *,
    n_strata: int = 3,
) -> pd.Series:
    """Assign deterministic equally sized strata from observed interaction magnitude."""

    if magnitudes.empty or magnitudes.index.has_duplicates:
        raise ValueError("interaction magnitudes require unique condition labels")
    if not np.isfinite(magnitudes.to_numpy(dtype=float)).all():
        raise ValueError("interaction magnitudes must be finite")
    if not 2 <= n_strata <= len(magnitudes):
        raise ValueError("n_strata must be between 2 and the number of conditions")
    labels = ["low", "high"] if n_strata == 2 else [f"q{i + 1}" for i in range(n_strata)]
    ordered = pd.DataFrame(
        {"condition": magnitudes.index.astype(str), "magnitude": magnitudes.to_numpy()}
    ).sort_values(["magnitude", "condition"], kind="stable")
    assignments = np.empty(len(ordered), dtype=object)
    for stratum, positions in enumerate(np.array_split(np.arange(len(ordered)), n_strata)):
        assignments[positions] = labels[stratum]
    result = pd.Series(assignments, index=ordered["condition"], name="interaction_stratum")
    return result.reindex(magnitudes.index.astype(str))


def crossfit_interaction_analysis(
    genes: Sequence[str],
    *,
    n_panels: int = 2,
    seed: int = 42,
) -> tuple[tuple[str, ...], ...]:
    """Create disjoint deterministic gene panels for cross-fitted sensitivity checks."""

    names = np.asarray([str(gene) for gene in genes], dtype=object)
    if names.size < n_panels or len(set(names)) != names.size:
        raise ValueError("genes must be unique and sufficient for all panels")
    if n_panels < 2:
        raise ValueError("n_panels must be at least two")
    ordered = np.sort(names)
    rng = np.random.default_rng(seed)
    shuffled = ordered[rng.permutation(len(ordered))]
    return tuple(tuple(panel.tolist()) for panel in np.array_split(shuffled, n_panels))

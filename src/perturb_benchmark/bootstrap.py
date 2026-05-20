"""Paired condition-level bootstrap uncertainty."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    iterations: int
    n_conditions: int
    seed: int


def paired_condition_bootstrap(
    values: pd.DataFrame,
    model_a_column: str,
    model_b_column: str,
    *,
    condition_column: str = "condition",
    iterations: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap the mean paired A-minus-B difference over test conditions."""

    required = {condition_column, model_a_column, model_b_column}
    if missing := required.difference(values.columns):
        raise ValueError(f"bootstrap frame is missing columns: {sorted(missing)}")
    if values.empty or values[condition_column].duplicated().any():
        raise ValueError("bootstrap requires exactly one paired value per condition")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    paired = values[[model_a_column, model_b_column]].to_numpy(dtype=float)
    if not np.isfinite(paired).all():
        raise ValueError("paired bootstrap values must be finite")
    differences = paired[:, 0] - paired[:, 1]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(iterations, len(differences)))
    draws = differences[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(draws, [tail, 1 - tail])
    return BootstrapInterval(
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        iterations=iterations,
        n_conditions=len(differences),
        seed=seed,
    )

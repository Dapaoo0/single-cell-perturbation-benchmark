"""Condition-level expression means and matched-control deltas."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from perturb_benchmark.perturbation import parse_condition


@dataclass(frozen=True)
class PseudobulkResult:
    """Aligned condition and replicate response matrices."""

    means: pd.DataFrame
    deltas: pd.DataFrame
    replicate_means: pd.DataFrame
    replicate_deltas: pd.DataFrame
    metadata: pd.DataFrame
    scale: str


def _row_mean(expression: Any, positions: np.ndarray) -> np.ndarray:
    subset = expression[positions]
    if sparse.issparse(subset):
        return np.asarray(subset.mean(axis=0), dtype=float).ravel()
    return np.asarray(subset, dtype=float).mean(axis=0)


def build_pseudobulk(
    expression: Any,
    obs: pd.DataFrame,
    genes: list[str] | tuple[str, ...],
    condition_col: str,
    batch_col: str | None,
    *,
    control_token: str = "ctrl",
    scale: str = "processed-expression",
) -> PseudobulkResult:
    """Build canonical means and subtract controls within each retained batch."""

    if expression.shape != (len(obs), len(genes)):
        raise ValueError("expression shape must match observations and genes")
    if condition_col not in obs:
        raise ValueError(f"condition column {condition_col!r} is missing")
    if batch_col is not None and batch_col not in obs:
        raise ValueError(f"batch column {batch_col!r} is missing")
    if len(set(genes)) != len(genes):
        raise ValueError("gene identifiers must be unique")

    cells = pd.DataFrame(
        {
            "position": np.arange(len(obs)),
            "condition": [
                parse_condition(value, control_token=control_token).canonical
                for value in obs[condition_col].astype(str)
            ],
            "batch": obs[batch_col].astype(str).to_numpy()
            if batch_col is not None
            else "global",
        }
    )
    rows: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    for (condition, batch), group in cells.groupby(
        ["condition", "batch"], sort=True, observed=True
    ):
        keys.append((condition, batch))
        rows.append(_row_mean(expression, group["position"].to_numpy()))

    index = pd.MultiIndex.from_tuples(keys, names=["condition", "batch"])
    replicate_means = pd.DataFrame(rows, index=index, columns=list(genes), dtype=float)
    if not np.isfinite(replicate_means.to_numpy()).all():
        raise ValueError("pseudo-bulk means contain nonfinite values")

    delta_rows: list[np.ndarray] = []
    for condition, batch in replicate_means.index:
        control_key = (control_token, batch)
        if control_key not in replicate_means.index:
            raise ValueError(f"matched control is missing for batch {batch!r}")
        delta_rows.append(
            replicate_means.loc[(condition, batch)].to_numpy()
            - replicate_means.loc[control_key].to_numpy()
        )
    replicate_deltas = pd.DataFrame(
        delta_rows,
        index=replicate_means.index.copy(),
        columns=list(genes),
    )
    means = replicate_means.groupby(level="condition", sort=True).mean()
    deltas = replicate_deltas.groupby(level="condition", sort=True).mean()
    metadata = (
        cells.groupby("condition", sort=True, observed=True)
        .agg(n_cells=("position", "size"), n_batches=("batch", "nunique"))
        .reset_index()
    )
    return PseudobulkResult(
        means=means,
        deltas=deltas,
        replicate_means=replicate_means,
        replicate_deltas=replicate_deltas,
        metadata=metadata,
        scale=scale,
    )

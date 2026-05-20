"""Fail-closed validation and scale evidence for AnnData inputs."""

from dataclasses import asdict, dataclass
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class AnnDataSummary:
    """JSON-safe evidence collected before preprocessing."""

    n_cells: int
    n_genes: int
    obs_columns: tuple[str, ...]
    var_columns: tuple[str, ...]
    layer_names: tuple[str, ...]
    has_raw: bool
    matrix_dtype: str
    matrix_kind: str
    zero_fraction: float
    minimum: float
    maximum: float
    quantiles: tuple[float, float, float, float, float]
    integer_like: bool
    nonnegative: bool
    scale_evidence: str
    condition_column: str
    n_conditions: int
    control_token: str
    has_control: bool
    candidate_batch_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary to JSON-safe primitives."""

        return asdict(self)


def _sample_values(matrix: Any, max_values: int = 1_000_000) -> np.ndarray:
    n_rows, n_columns = matrix.shape
    rows = min(n_rows, max(1, max_values // max(1, n_columns)))
    if sparse.issparse(matrix):
        return np.asarray(matrix[:rows].toarray(), dtype=float).ravel()
    return np.asarray(matrix[:rows], dtype=float).ravel()


def inspect_anndata(
    adata: ad.AnnData,
    condition_column: str,
    control_token: str,
) -> AnnDataSummary:
    """Validate essential AnnData invariants and report scale evidence."""

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData must contain cells and genes")
    if adata.var_names.has_duplicates:
        raise ValueError("AnnData contains duplicate gene identifiers")
    if condition_column not in adata.obs:
        raise ValueError(f"condition column {condition_column!r} is missing")
    conditions = adata.obs[condition_column].astype(str)
    has_control = bool((conditions == control_token).any())
    if not has_control:
        raise ValueError(f"control token {control_token!r} is absent")

    values = _sample_values(adata.X)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("expression matrix contains no finite validation sample")
    integer_like = bool(np.allclose(values, np.rint(values), atol=1e-6))
    nonnegative = bool(np.min(values) >= 0)
    if integer_like and nonnegative:
        scale_evidence = "count-like"
    elif nonnegative:
        scale_evidence = "nonnegative-continuous; normalization state requires provenance"
    else:
        scale_evidence = "signed-continuous; likely scaled or centered"

    total = adata.n_obs * adata.n_vars
    if sparse.issparse(adata.X):
        zero_fraction = 1.0 - (adata.X.nnz / total)
        matrix_kind = "sparse"
    else:
        zero_fraction = float(np.count_nonzero(np.asarray(adata.X) == 0) / total)
        matrix_kind = "dense"

    batch_terms = ("batch", "gemgroup", "replicate", "sample", "library")
    candidates = tuple(
        column
        for column in map(str, adata.obs.columns)
        if any(term in column.casefold() for term in batch_terms)
    )
    quantiles = tuple(float(value) for value in np.quantile(values, [0, 0.25, 0.5, 0.75, 1]))
    return AnnDataSummary(
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        obs_columns=tuple(map(str, adata.obs.columns)),
        var_columns=tuple(map(str, adata.var.columns)),
        layer_names=tuple(map(str, adata.layers.keys())),
        has_raw=adata.raw is not None,
        matrix_dtype=str(adata.X.dtype),
        matrix_kind=matrix_kind,
        zero_fraction=float(zero_fraction),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        quantiles=quantiles,
        integer_like=integer_like,
        nonnegative=nonnegative,
        scale_evidence=scale_evidence,
        condition_column=condition_column,
        n_conditions=int(conditions.nunique()),
        control_token=control_token,
        has_control=has_control,
        candidate_batch_columns=candidates,
    )


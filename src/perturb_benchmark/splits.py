"""Deterministic held-out double-perturbation folds."""

import numpy as np
import pandas as pd

REQUIRED_METADATA_COLUMNS = {
    "condition_canonical",
    "perturbation_type",
    "gene_a",
    "gene_b",
}


def _eligible_doubles(metadata: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"perturbation metadata is missing columns: {sorted(missing)}")
    if metadata["condition_canonical"].duplicated().any():
        raise ValueError("canonical conditions must be unique")
    singles = set(
        metadata.loc[metadata["perturbation_type"] == "single", "gene_a"].dropna().astype(str)
    )
    doubles = metadata.loc[metadata["perturbation_type"] == "double"].copy()
    if doubles[["gene_a", "gene_b"]].isna().any().any():
        raise ValueError("double perturbations require two component genes")
    doubles["gene_a"] = doubles["gene_a"].astype(str)
    doubles["gene_b"] = doubles["gene_b"].astype(str)
    ordered = doubles.apply(lambda row: tuple(sorted((row["gene_a"], row["gene_b"]))), axis=1)
    expected = ordered.map(lambda genes: "+".join(genes))
    if not expected.equals(doubles["condition_canonical"].astype(str)):
        raise ValueError("double perturbations are not canonically ordered")
    eligible = doubles[
        doubles["gene_a"].isin(singles) & doubles["gene_b"].isin(singles)
    ].copy()
    return eligible.sort_values("condition_canonical").reset_index(drop=True)


def create_combination_folds(
    metadata: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Assign each eligible double to exactly one balanced test fold."""

    eligible = _eligible_doubles(metadata)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if len(eligible) < n_folds:
        raise ValueError("eligible double perturbations must cover every fold")
    order = np.random.default_rng(seed).permutation(len(eligible))
    fold_by_row = np.empty(len(eligible), dtype=int)
    fold_by_row[order] = np.arange(len(eligible)) % n_folds
    split = pd.DataFrame(
        {
            "condition": eligible["condition_canonical"].astype(str),
            "gene_a": eligible["gene_a"].astype(str),
            "gene_b": eligible["gene_b"].astype(str),
            "fold": fold_by_row,
        }
    ).sort_values(["fold", "condition"], ignore_index=True)
    validate_folds(split, metadata, n_folds=n_folds)
    return split


def validate_folds(split: pd.DataFrame, metadata: pd.DataFrame, n_folds: int) -> None:
    """Fail if fold membership violates the frozen split design."""

    required = {"condition", "gene_a", "gene_b", "fold"}
    missing = required.difference(split.columns)
    if missing:
        raise ValueError(f"split is missing columns: {sorted(missing)}")
    if split["condition"].duplicated().any():
        raise ValueError("each eligible double must occur in exactly one test fold")
    eligible = _eligible_doubles(metadata)
    expected = set(eligible["condition_canonical"].astype(str))
    observed = set(split["condition"].astype(str))
    if observed != expected:
        raise ValueError("split must cover every eligible double exactly once")
    folds = set(split["fold"].astype(int))
    if folds != set(range(n_folds)):
        raise ValueError("split fold identifiers must cover zero through n_folds - 1")
    singles = set(
        metadata.loc[metadata["perturbation_type"] == "single", "gene_a"].dropna().astype(str)
    )
    components = set(split["gene_a"].astype(str)) | set(split["gene_b"].astype(str))
    if not components.issubset(singles):
        raise ValueError("every test component must have a training single perturbation")
    counts = split.groupby("fold").size()
    if int(counts.max() - counts.min()) > 1:
        raise ValueError("test folds must be balanced by condition count")

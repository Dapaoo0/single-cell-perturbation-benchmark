"""Transparent biological and regularized linear baselines."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def predict_no_change(n_conditions: int, n_genes: int) -> np.ndarray:
    """Predict zero perturbation-induced expression change."""

    return np.zeros((n_conditions, n_genes), dtype=float)


def predict_perturbed_mean(
    training_deltas: pd.DataFrame,
    test_conditions: Sequence[str],
) -> np.ndarray:
    """Repeat the mean training response for every test condition."""

    if training_deltas.empty:
        raise ValueError("training deltas cannot be empty")
    mean = training_deltas.to_numpy(dtype=float).mean(axis=0)
    return np.repeat(mean[None, :], len(test_conditions), axis=0)


def _training_single(
    gene: str,
    training_singles: Mapping[str, np.ndarray],
) -> np.ndarray:
    try:
        response = np.asarray(training_singles[gene], dtype=float)
    except KeyError as error:
        raise ValueError(f"training single is missing for gene {gene!r}") from error
    if response.ndim != 1 or not np.isfinite(response).all():
        raise ValueError(f"training single for gene {gene!r} must be one finite vector")
    return response


def predict_additive_sum(
    gene_a: str,
    gene_b: str,
    training_singles: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Add two measured training-single responses."""

    a = _training_single(gene_a, training_singles)
    b = _training_single(gene_b, training_singles)
    if a.shape != b.shape:
        raise ValueError("training single response vectors must have equal shape")
    return a + b


def predict_matching_mean(
    gene_a: str,
    gene_b: str,
    training_singles: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Average two measured training-single responses."""

    return predict_additive_sum(gene_a, gene_b, training_singles) / 2.0


@dataclass(frozen=True)
class CalibratedAdditiveModel:
    """Global train-only calibration of the additive-sum response."""

    slope: float
    intercept: float

    def predict(
        self,
        gene_a: str,
        gene_b: str,
        training_singles: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        raw = predict_additive_sum(gene_a, gene_b, training_singles)
        return self.slope * raw + self.intercept


def fit_calibrated_additive(
    training_pairs: pd.DataFrame,
    training_targets: pd.DataFrame,
    training_singles: Mapping[str, np.ndarray],
) -> CalibratedAdditiveModel:
    """Fit one slope and intercept using training doubles only."""

    if len(training_pairs) != len(training_targets) or training_pairs.empty:
        raise ValueError("training pairs and targets must be non-empty and aligned")
    raw = np.vstack(
        [
            predict_additive_sum(str(row.gene_a), str(row.gene_b), training_singles)
            for row in training_pairs.itertuples(index=False)
        ]
    )
    target = training_targets.to_numpy(dtype=float)
    if raw.shape != target.shape or not np.isfinite(target).all():
        raise ValueError("training additive predictions and targets must be finite and aligned")
    design = np.column_stack([raw.ravel(), np.ones(raw.size)])
    slope, intercept = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
    return CalibratedAdditiveModel(float(slope), float(intercept))


def _component_genes(pairs: pd.DataFrame) -> tuple[str, ...]:
    required = {"gene_a", "gene_b"}
    if missing := required.difference(pairs.columns):
        raise ValueError(f"perturbation pairs are missing columns: {sorted(missing)}")
    values: set[str] = set()
    for value in pd.concat([pairs["gene_a"], pairs["gene_b"]], ignore_index=True):
        if pd.notna(value):
            values.add(str(value))
    if not values:
        raise ValueError("perturbation pairs contain no component genes")
    return tuple(sorted(values))


def _encode_pairs(pairs: pd.DataFrame, genes: Sequence[str]) -> np.ndarray:
    lookup = {gene: position for position, gene in enumerate(genes)}
    encoded = np.zeros((len(pairs), len(genes)), dtype=float)
    for row_index, row in enumerate(pairs.itertuples(index=False)):
        components = [row.gene_a]
        if pd.notna(row.gene_b):
            components.append(row.gene_b)
        for component in components:
            gene = str(component)
            if gene not in lookup:
                raise ValueError(f"component gene {gene!r} was not observed in Ridge training")
            encoded[row_index, lookup[gene]] += 1.0
    return encoded


@dataclass(frozen=True)
class RidgeBaseline:
    """Multi-output Ridge model with a frozen perturbation vocabulary."""

    model: Ridge
    genes: tuple[str, ...]
    alpha: float

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(_encode_pairs(pairs, self.genes)), dtype=float)


def fit_ridge_nested(
    training_pairs: pd.DataFrame,
    training_targets: pd.DataFrame,
    *,
    alphas: Iterable[float],
    seed: int,
) -> RidgeBaseline:
    """Select Ridge alpha using condition-level inner folds only."""

    if len(training_pairs) != len(training_targets) or len(training_pairs) < 3:
        raise ValueError("Ridge requires at least three aligned training perturbations")
    candidates = tuple(sorted({float(alpha) for alpha in alphas}))
    if not candidates or any(alpha <= 0 for alpha in candidates):
        raise ValueError("Ridge alphas must be positive")
    genes = _component_genes(training_pairs)
    design = _encode_pairs(training_pairs, genes)
    targets = training_targets.to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Ridge training targets must be finite")

    folds = KFold(n_splits=min(5, len(training_pairs)), shuffle=True, random_state=seed)
    losses: dict[float, float] = {}
    for alpha in candidates:
        fold_losses: list[float] = []
        for train_index, validation_index in folds.split(design):
            model = Ridge(alpha=alpha).fit(design[train_index], targets[train_index])
            predicted = model.predict(design[validation_index])
            fold_losses.append(float(np.mean((predicted - targets[validation_index]) ** 2)))
        losses[alpha] = float(np.mean(fold_losses))
    selected = min(candidates, key=lambda alpha: (losses[alpha], alpha))
    model = Ridge(alpha=selected).fit(design, targets)
    return RidgeBaseline(model=model, genes=genes, alpha=selected)

"""Condition-level response metrics with explicit undefined correlations."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from perturb_benchmark.predictions import validate_predictions


@dataclass(frozen=True)
class VectorMetrics:
    pearson: float
    spearman: float
    mae: float
    rmse: float
    undefined_reason: str | None


def vector_metrics(true: np.ndarray, predicted: np.ndarray) -> VectorMetrics:
    """Evaluate two aligned vectors without replacing undefined correlations."""

    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if true.ndim != 1 or predicted.shape != true.shape or true.size < 2:
        raise ValueError("metric vectors must be aligned one-dimensional arrays of length >= 2")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("metric vectors must be finite")
    true_constant = bool(np.ptp(true) == 0)
    predicted_constant = bool(np.ptp(predicted) == 0)
    if true_constant and predicted_constant:
        reason = "constant_target_and_prediction"
    elif true_constant:
        reason = "constant_target"
    elif predicted_constant:
        reason = "constant_prediction"
    else:
        reason = None
    if reason is None:
        pearson = float(np.corrcoef(true, predicted)[0, 1])
        spearman = float(spearmanr(true, predicted).statistic)
    else:
        pearson = float("nan")
        spearman = float("nan")
    error = predicted - true
    return VectorMetrics(
        pearson=pearson,
        spearman=spearman,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        undefined_reason=reason,
    )


def condition_metrics(
    predictions: pd.DataFrame,
    *,
    training_mean_by_fold: dict[int, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute raw and train-mean-centered metrics for every condition and model."""

    validate_predictions(predictions)
    group_columns = ["model", "fold", "replicate", "condition"]
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, observed=True):
        ordered = group.sort_values("gene")
        raw = vector_metrics(
            ordered["true_delta"].to_numpy(dtype=float),
            ordered["pred_delta"].to_numpy(dtype=float),
        )
        model, fold, replicate, condition = keys
        row: dict[str, Any] = {
            "model": model,
            "fold": int(fold),
            "replicate": replicate,
            "condition": condition,
            **asdict(raw),
        }
        if training_mean_by_fold is None:
            row.update(
                centered_pearson=float("nan"),
                centered_spearman=float("nan"),
                centered_undefined_reason="training_mean_not_supplied",
            )
        else:
            try:
                training_mean = training_mean_by_fold[int(fold)].reindex(ordered["gene"])
            except KeyError as error:
                raise ValueError(f"training mean is missing for fold {fold}") from error
            if training_mean.isna().any():
                raise ValueError(f"training mean genes are incomplete for fold {fold}")
            centered = vector_metrics(
                ordered["true_delta"].to_numpy(dtype=float) - training_mean.to_numpy(),
                ordered["pred_delta"].to_numpy(dtype=float) - training_mean.to_numpy(),
            )
            row.update(
                centered_pearson=centered.pearson,
                centered_spearman=centered.spearman,
                centered_undefined_reason=centered.undefined_reason,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired condition-level metrics without hiding undefined values."""

    required = {"model", "pearson", "spearman", "mae", "rmse", "centered_pearson"}
    if missing := required.difference(metrics.columns):
        raise ValueError(f"condition metrics are missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []

    def finite_median(series: pd.Series) -> float:
        values = series.dropna()
        return float(values.median()) if not values.empty else float("nan")

    for model, group in metrics.groupby("model", sort=True, observed=True):
        rows.append(
            {
                "model": model,
                "n_conditions": len(group),
                "undefined_pearson": int(group["pearson"].isna().sum()),
                "undefined_spearman": int(group["spearman"].isna().sum()),
                "median_pearson": finite_median(group["pearson"]),
                "median_spearman": finite_median(group["spearman"]),
                "median_mae": finite_median(group["mae"]),
                "median_rmse": finite_median(group["rmse"]),
                "median_centered_pearson": finite_median(group["centered_pearson"]),
            }
        )
    return pd.DataFrame(rows)

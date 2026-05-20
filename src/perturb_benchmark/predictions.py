"""Common long-format prediction contract and pairing."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = (
    "condition",
    "fold",
    "replicate",
    "gene",
    "true_delta",
    "pred_delta",
    "model",
)
KEY_COLUMNS = ("condition", "fold", "replicate", "gene")


def validate_predictions(frame: pd.DataFrame) -> None:
    """Fail closed when a prediction artifact violates its semantic contract."""

    missing = set(PREDICTION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("prediction frame cannot be empty")
    if frame[[*KEY_COLUMNS, "model"]].isna().any().any():
        raise ValueError("prediction semantic keys cannot be null")
    if frame.duplicated([*KEY_COLUMNS, "model"]).any():
        raise ValueError("duplicate prediction key")
    numeric = frame[["true_delta", "pred_delta"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("prediction values must be finite")


def pair_model_predictions(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Align model predictions by semantic keys rather than row position."""

    if len(frames) < 2:
        raise ValueError("at least two model prediction frames are required")
    prepared: list[tuple[str, pd.DataFrame]] = []
    reference_keys: set[tuple[object, ...]] | None = None
    reference_truth: pd.DataFrame | None = None
    for frame in frames:
        validate_predictions(frame)
        models = frame["model"].astype(str).unique()
        if len(models) != 1:
            raise ValueError("each prediction frame must contain exactly one model")
        ordered = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        key_set = set(ordered[list(KEY_COLUMNS)].itertuples(index=False, name=None))
        if reference_keys is None:
            reference_keys = key_set
            reference_truth = ordered[[*KEY_COLUMNS, "true_delta"]].copy()
        elif key_set != reference_keys:
            raise ValueError("all models must use identical test keys")
        elif not np.allclose(
            ordered["true_delta"].to_numpy(dtype=float),
            reference_truth["true_delta"].to_numpy(dtype=float),
        ):
            raise ValueError("all models must use identical test targets")
        prepared.append((models[0], ordered))

    assert reference_truth is not None
    paired = reference_truth
    for model, ordered in sorted(prepared, key=lambda item: item[0]):
        column = f"pred_{model}"
        if column in paired:
            raise ValueError(f"duplicate model name {model!r}")
        paired[column] = ordered["pred_delta"].to_numpy(dtype=float)
    return paired

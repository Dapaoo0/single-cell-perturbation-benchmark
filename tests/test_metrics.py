import numpy as np
import pandas as pd

from perturb_benchmark.metrics import condition_metrics, summarize_models, vector_metrics


def test_constant_prediction_correlation_is_nan() -> None:
    result = vector_metrics(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

    assert np.isnan(result.pearson)
    assert np.isnan(result.spearman)
    assert result.undefined_reason == "constant_prediction"
    assert result.mae == 1.5
    assert result.rmse == np.sqrt(2.5)


def test_condition_metrics_include_train_centered_response() -> None:
    frame = pd.DataFrame(
        {
            "condition": ["A+B"] * 3,
            "fold": [0] * 3,
            "replicate": ["global"] * 3,
            "gene": ["G1", "G2", "G3"],
            "true_delta": [1.0, 3.0, 2.0],
            "pred_delta": [1.2, 2.5, 1.8],
            "model": ["ridge"] * 3,
        }
    )
    training_mean = {0: pd.Series({"G1": 0.5, "G2": 1.0, "G3": 1.0})}

    metrics = condition_metrics(frame, training_mean_by_fold=training_mean)

    assert metrics.loc[0, "model"] == "ridge"
    assert np.isfinite(metrics.loc[0, "pearson"])
    assert np.isfinite(metrics.loc[0, "centered_pearson"])


def test_model_summary_counts_undefined_correlations() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["zero", "zero", "ridge"],
            "pearson": [np.nan, np.nan, 0.5],
            "spearman": [np.nan, np.nan, 0.4],
            "mae": [1.0, 2.0, 0.5],
            "rmse": [1.2, 2.2, 0.7],
            "centered_pearson": [np.nan, np.nan, 0.3],
        }
    )

    summary = summarize_models(metrics).set_index("model")

    assert summary.loc["zero", "undefined_pearson"] == 2
    assert summary.loc["zero", "median_pearson"] != 0
    assert summary.loc["ridge", "median_mae"] == 0.5

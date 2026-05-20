import numpy as np
import pandas as pd
import pytest

from perturb_benchmark.baselines import (
    fit_calibrated_additive,
    fit_ridge_nested,
    predict_additive_sum,
    predict_matching_mean,
    predict_no_change,
    predict_perturbed_mean,
)


def test_fixed_baseline_formulas() -> None:
    singles = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 2.0])}

    np.testing.assert_allclose(predict_additive_sum("A", "B", singles), [1.0, 2.0])
    np.testing.assert_allclose(predict_matching_mean("A", "B", singles), [0.5, 1.0])
    np.testing.assert_allclose(predict_no_change(2, 2), np.zeros((2, 2)))


def test_additive_rejects_missing_training_single() -> None:
    with pytest.raises(ValueError, match="training single"):
        predict_additive_sum("A", "B", {"A": np.array([1.0])})


def test_perturbed_mean_uses_training_conditions_only() -> None:
    training = pd.DataFrame([[1.0, 3.0], [3.0, 5.0]], index=["A", "B"])

    prediction = predict_perturbed_mean(training, ["A+B", "A+C"])

    np.testing.assert_allclose(prediction, [[2.0, 4.0], [2.0, 4.0]])


def test_calibrated_additive_learns_global_slope_and_intercept() -> None:
    singles = {
        "A": np.array([1.0, 2.0]),
        "B": np.array([2.0, 1.0]),
        "C": np.array([0.0, 1.0]),
    }
    pairs = pd.DataFrame({"gene_a": ["A", "A"], "gene_b": ["B", "C"]})
    raw = np.vstack(
        [
            predict_additive_sum("A", "B", singles),
            predict_additive_sum("A", "C", singles),
        ]
    )
    targets = pd.DataFrame(2.0 * raw + 0.5)

    model = fit_calibrated_additive(pairs, targets, singles)

    assert model.slope == pytest.approx(2.0)
    assert model.intercept == pytest.approx(0.5)
    np.testing.assert_allclose(model.predict("B", "C", singles), 2.0 * np.array([2.0, 2.0]) + 0.5)


def test_ridge_nested_returns_aligned_multioutput_predictions() -> None:
    pairs = pd.DataFrame(
        {
            "gene_a": ["A", "B", "C", "A", "A", "B"],
            "gene_b": [None, None, None, "B", "C", "C"],
        }
    )
    targets = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 3.0],
        ],
        columns=["G1", "G2"],
    )

    model = fit_ridge_nested(pairs, targets, alphas=[0.01, 0.1, 1.0], seed=42)
    predicted = model.predict(pd.DataFrame({"gene_a": ["A"], "gene_b": ["B"]}))

    assert model.alpha in {0.01, 0.1, 1.0}
    assert predicted.shape == (1, 2)
    assert np.isfinite(predicted).all()

import pandas as pd
import pytest

from perturb_benchmark.predictions import pair_model_predictions, validate_predictions


def _frame(model: str, condition: str = "A+B") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": [condition, condition],
            "fold": [0, 0],
            "replicate": ["global", "global"],
            "gene": ["G1", "G2"],
            "true_delta": [1.0, 2.0],
            "pred_delta": [0.8, 2.2],
            "model": [model, model],
        }
    )


def test_prediction_contract_rejects_duplicate_semantic_key() -> None:
    frame = pd.concat([_frame("ridge"), _frame("ridge").iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate prediction key"):
        validate_predictions(frame)


def test_pairing_rejects_different_test_sets() -> None:
    with pytest.raises(ValueError, match="identical test keys"):
        pair_model_predictions([_frame("ridge"), _frame("additive", condition="A+C")])


def test_pairing_uses_semantic_keys_not_row_order() -> None:
    ridge = _frame("ridge")
    additive = _frame("additive").iloc[::-1].reset_index(drop=True)

    paired = pair_model_predictions([ridge, additive])

    assert list(paired["gene"]) == ["G1", "G2"]
    assert {"pred_additive", "pred_ridge"}.issubset(paired.columns)
    assert paired["true_delta"].tolist() == [1.0, 2.0]

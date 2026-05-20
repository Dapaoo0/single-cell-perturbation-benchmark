import pandas as pd
import pytest

from perturb_benchmark.splits import create_combination_folds, validate_folds


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition_canonical": ["ctrl", "A", "B", "C", "A+B", "A+C", "B+C", "A+D"],
            "perturbation_type": [
                "control",
                "single",
                "single",
                "single",
                "double",
                "double",
                "double",
                "double",
            ],
            "gene_a": [None, "A", "B", "C", "A", "A", "B", "A"],
            "gene_b": [None, None, None, None, "B", "C", "C", "D"],
        }
    )


def test_seen_components_unseen_combination_split_is_complete_and_balanced() -> None:
    metadata = _metadata()

    split = create_combination_folds(metadata, n_folds=3, seed=42)

    assert set(split["condition"]) == {"A+B", "A+C", "B+C"}
    assert split["condition"].is_unique
    assert split.groupby("fold").size().max() - split.groupby("fold").size().min() <= 1
    validate_folds(split, metadata, n_folds=3)


def test_split_is_deterministic() -> None:
    first = create_combination_folds(_metadata(), n_folds=3, seed=42)
    second = create_combination_folds(_metadata(), n_folds=3, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_validation_rejects_duplicate_test_condition() -> None:
    metadata = _metadata()
    split = create_combination_folds(metadata, n_folds=3, seed=42)
    invalid = pd.concat([split, split.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="exactly one test fold"):
        validate_folds(invalid, metadata, n_folds=3)

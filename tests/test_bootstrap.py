import pandas as pd

from perturb_benchmark.bootstrap import paired_condition_bootstrap


def test_paired_bootstrap_is_reproducible_and_condition_level() -> None:
    values = pd.DataFrame(
        {
            "condition": ["A+B", "A+C", "B+C"],
            "model_a": [1.0, 3.0, 5.0],
            "model_b": [2.0, 2.0, 2.0],
        }
    )
    first = paired_condition_bootstrap(
        values, "model_a", "model_b", iterations=200, seed=42
    )
    second = paired_condition_bootstrap(
        values, "model_a", "model_b", iterations=200, seed=42
    )
    assert first == second
    assert first.estimate == 1.0
    assert first.iterations == 200
    assert first.n_conditions == 3


def test_paired_bootstrap_rejects_duplicate_conditions() -> None:
    values = pd.DataFrame(
        {"condition": ["A+B", "A+B"], "a": [1.0, 2.0], "b": [0.0, 0.0]}
    )
    try:
        paired_condition_bootstrap(values, "a", "b", iterations=10, seed=1)
    except ValueError as error:
        assert "one paired value" in str(error)
    else:
        raise AssertionError("duplicate conditions must be rejected")

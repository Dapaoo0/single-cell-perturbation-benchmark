import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from perturb_benchmark.pseudobulk import build_pseudobulk


@pytest.mark.parametrize("as_sparse", [False, True])
def test_global_control_delta_matches_toy_example(as_sparse: bool) -> None:
    values = np.array([[1, 1], [1, 3], [3, 2], [5, 4]], dtype=float)
    expression = sparse.csr_matrix(values) if as_sparse else values
    obs = pd.DataFrame({"condition": ["ctrl", "ctrl", "A+ctrl", "ctrl+A"]})

    result = build_pseudobulk(expression, obs, ["G1", "G2"], "condition", None)

    np.testing.assert_allclose(result.means.loc["ctrl"], [1.0, 2.0])
    np.testing.assert_allclose(result.means.loc["A"], [4.0, 3.0])
    np.testing.assert_allclose(result.deltas.loc["A"], [3.0, 1.0])
    np.testing.assert_allclose(result.deltas.loc["ctrl"], [0.0, 0.0])


def test_batch_matching_precedes_equal_replicate_aggregation() -> None:
    expression = np.array([[1], [10], [3], [14]], dtype=float)
    obs = pd.DataFrame(
        {
            "condition": ["ctrl", "ctrl", "A", "A"],
            "batch": ["x", "y", "x", "y"],
        }
    )

    result = build_pseudobulk(expression, obs, ["G1"], "condition", "batch")

    assert result.replicate_deltas.loc[("A", "x"), "G1"] == 2
    assert result.replicate_deltas.loc[("A", "y"), "G1"] == 4
    assert result.deltas.loc["A", "G1"] == 3


def test_missing_matched_control_fails_closed() -> None:
    expression = np.array([[1], [3]], dtype=float)
    obs = pd.DataFrame(
        {"condition": ["ctrl", "A"], "batch": ["x", "y"]}
    )

    with pytest.raises(ValueError, match="matched control"):
        build_pseudobulk(expression, obs, ["G1"], "condition", "batch")

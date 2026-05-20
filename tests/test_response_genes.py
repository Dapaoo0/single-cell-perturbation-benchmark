import numpy as np
import pytest

from perturb_benchmark.response_genes import direction_accuracy, topk_recovery


def test_topk_overlap() -> None:
    assert topk_recovery(
        np.array([3.0, 2.0, 0.0]), np.array([4.0, 0.0, 1.0]), k=1
    ) == 1.0


def test_topk_uses_gene_names_to_break_ties() -> None:
    genes = np.array(["Z", "A", "B"])
    assert topk_recovery(
        np.array([2.0, 2.0, 0.0]),
        np.array([2.0, 0.0, 2.0]),
        k=1,
        genes=genes,
    ) == 0.0


def test_direction_accuracy_ignores_zero_truth() -> None:
    score = direction_accuracy(np.array([2.0, -1.0, 0.0]), np.array([1.0, 3.0, 4.0]))
    assert score == 0.5


def test_topk_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="k"):
        topk_recovery(np.array([1.0]), np.array([1.0]), k=2)

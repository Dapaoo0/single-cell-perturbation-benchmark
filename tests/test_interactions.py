import numpy as np
import pandas as pd

from perturb_benchmark.interactions import (
    assign_interaction_strata,
    crossfit_interaction_analysis,
    interaction_residual,
)


def test_interaction_residual() -> None:
    observed = np.array([3.0, 4.0])
    a = np.array([1.0, 1.0])
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(interaction_residual(observed, a, b), [1.0, 1.0])


def test_interaction_strata_are_assigned_once_from_observed_magnitude() -> None:
    magnitudes = pd.Series([1.0, 2.0, 3.0, 4.0], index=["d", "c", "b", "a"])
    strata = assign_interaction_strata(magnitudes, n_strata=2)
    assert strata.to_dict() == {"d": "low", "c": "low", "b": "high", "a": "high"}


def test_crossfit_panels_are_disjoint_and_deterministic() -> None:
    genes = ["G3", "G1", "G4", "G2"]
    first = crossfit_interaction_analysis(genes, n_panels=2, seed=42)
    second = crossfit_interaction_analysis(genes, n_panels=2, seed=42)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert sorted(first[0] + first[1]) == sorted(genes)

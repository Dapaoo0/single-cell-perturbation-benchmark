import pandas as pd
import pytest

from perturb_benchmark.perturbation import build_condition_metadata, parse_condition


@pytest.mark.parametrize(
    ("label", "kind", "genes", "canonical"),
    [
        ("ctrl", "control", (), "ctrl"),
        ("A+ctrl", "single", ("A",), "A"),
        ("ctrl+A", "single", ("A",), "A"),
        (" B + A ", "double", ("A", "B"), "A+B"),
    ],
)
def test_parse_condition(label: str, kind: str, genes: tuple[str, ...], canonical: str) -> None:
    parsed = parse_condition(label)

    assert (parsed.kind, parsed.genes, parsed.canonical) == (kind, genes, canonical)


@pytest.mark.parametrize("label", ["A+B+C", "A+A", "+", "A++B", ""])
def test_parser_rejects_malformed_condition(label: str) -> None:
    with pytest.raises(ValueError):
        parse_condition(label)


def test_build_condition_metadata_preserves_original_labels_and_counts_batches() -> None:
    labels = pd.Series(["ctrl", "A+ctrl", "ctrl+A", "B+A", "A+B"], name="condition")
    batches = pd.Series(["x", "x", "y", "x", "y"], name="batch")

    metadata = build_condition_metadata(labels, batches)

    single = metadata.loc[metadata["condition_canonical"] == "A"].iloc[0]
    double = metadata.loc[metadata["condition_canonical"] == "A+B"].iloc[0]
    assert single["condition_original"] == "A+ctrl|ctrl+A"
    assert single["n_cells"] == 2
    assert single["n_batches"] == 2
    assert double["gene_a"] == "A"
    assert double["gene_b"] == "B"
    assert double["n_cells"] == 2

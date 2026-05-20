"""Canonical parsing and aggregation of perturbation labels."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

PerturbationKind = Literal["control", "single", "double"]


@dataclass(frozen=True)
class Perturbation:
    """One canonical control, single, or double perturbation."""

    kind: PerturbationKind
    genes: tuple[str, ...]
    canonical: str


def parse_condition(label: str, control_token: str = "ctrl") -> Perturbation:
    """Parse a condition label and normalize component ordering."""

    if not isinstance(label, str) or not label.strip():
        raise ValueError("condition label must be a non-empty string")
    raw_parts = label.split("+")
    parts = [part.strip() for part in raw_parts]
    if any(not part for part in parts):
        raise ValueError(f"malformed condition label: {label!r}")

    genes = [part for part in parts if part != control_token]
    if len(genes) != len(set(genes)):
        raise ValueError(f"duplicate perturbed gene in condition: {label!r}")
    if len(genes) > 2:
        raise ValueError("conditions must contain one or two perturbed genes")
    if not genes:
        if any(part != control_token for part in parts):
            raise ValueError(f"malformed control condition: {label!r}")
        return Perturbation(kind="control", genes=(), canonical=control_token)

    ordered = tuple(sorted(genes))
    kind: PerturbationKind = "single" if len(ordered) == 1 else "double"
    return Perturbation(kind=kind, genes=ordered, canonical="+".join(ordered))


def build_condition_metadata(
    labels: pd.Series,
    batches: pd.Series | None = None,
    control_token: str = "ctrl",
) -> pd.DataFrame:
    """Aggregate canonical condition metadata from cell-level labels."""

    if batches is not None and len(labels) != len(batches):
        raise ValueError("labels and batches must have equal length")

    records: list[dict[str, object]] = []
    for position, original in enumerate(labels.tolist()):
        parsed = parse_condition(original, control_token=control_token)
        records.append(
            {
                "condition_original": original.strip(),
                "condition_canonical": parsed.canonical,
                "perturbation_type": parsed.kind,
                "gene_a": parsed.genes[0] if parsed.genes else None,
                "gene_b": parsed.genes[1] if len(parsed.genes) == 2 else None,
                "batch": batches.iloc[position] if batches is not None else "global",
            }
        )

    cells = pd.DataFrame.from_records(records)
    if cells.empty:
        raise ValueError("condition metadata cannot be built from zero cells")

    rows: list[dict[str, object]] = []
    for canonical, group in cells.groupby("condition_canonical", sort=True, observed=True):
        first = group.iloc[0]
        rows.append(
            {
                "condition_original": "|".join(sorted(group["condition_original"].unique())),
                "condition_canonical": canonical,
                "perturbation_type": first["perturbation_type"],
                "gene_a": first["gene_a"],
                "gene_b": first["gene_b"],
                "n_cells": len(group),
                "n_batches": int(group["batch"].nunique()),
            }
        )
    return pd.DataFrame(rows)

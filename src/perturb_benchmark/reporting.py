"""Artifact-driven English scientific reporting."""

import json
from pathlib import Path

import pandas as pd


def render_result_sentence(
    cpu_summary: dict[str, float], scgpt_summary: float | None
) -> str:
    """Render a bounded headline that cannot invent a missing scGPT result."""

    finite = {model: value for model, value in cpu_summary.items() if pd.notna(value)}
    if not finite:
        cpu_clause = "No CPU correlation estimate was defined."
    else:
        best_model, best_value = max(finite.items(), key=lambda item: (item[1], item[0]))
        cpu_clause = (
            f"The best CPU baseline by median condition-level Pearson correlation was "
            f"{best_model} ({best_value:.3f})."
        )
    if scgpt_summary is None:
        return f"{cpu_clause} scGPT was not run, so no comparative scGPT claim is made."
    relation = (
        "higher"
        if scgpt_summary > max(finite.values(), default=float("-inf"))
        else "not higher"
    )
    return (
        f"{cpu_clause} The validated scGPT estimate was {scgpt_summary:.3f}, "
        f"which was {relation}."
    )


def build_reports(project_root: Path | str) -> Path:
    """Build a concise results report exclusively from saved result artifacts."""

    root = Path(project_root)
    summary_path = root / "results/metrics/model_summary.csv"
    feasibility_path = root / "results/metadata/scgpt_feasibility.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"model summary is missing: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {"model", "median_pearson", "median_mae"}
    if missing := required.difference(summary.columns):
        raise ValueError(f"model summary is missing columns: {sorted(missing)}")
    cpu = dict(zip(summary["model"].astype(str), summary["median_pearson"], strict=True))
    feasibility = (
        json.loads(feasibility_path.read_text(encoding="utf-8"))
        if feasibility_path.is_file()
        else {"status": "not_checked"}
    )
    headline = render_result_sentence(cpu, scgpt_summary=None)
    table = summary[["model", "median_pearson", "median_mae"]].copy()
    table["median_pearson"] = table["median_pearson"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.3f}"
    )
    table["median_mae"] = table["median_mae"].map(lambda value: f"{value:.3f}")
    table_rows = ["| model | median Pearson | median MAE |", "|---|---:|---:|"]
    table_rows.extend(
        f"| {row.model} | {row.median_pearson} | {row.median_mae} |"
        for row in table.itertuples(index=False)
    )
    table_markdown = "\n".join(table_rows)
    content = f"""# Results

## Current completion state

{headline}

scGPT feasibility status: `{feasibility.get("status", "unknown")}`.

## CPU baseline summary

{table_markdown}

Each test row is a held-out double-gene CRISPRa combination whose component single-gene
perturbations remain available to training. Correlation for the no-change baseline is `N/A` by
definition because its prediction vector is constant; it is not replaced by zero.

## Interpretation boundary

Interaction magnitude is reported as a descriptive difficulty variable. Because the interaction
residual is also the additive model's error vector, associations between interaction strength and
gain over additivity are mathematically coupled and are not evidence of mechanistic understanding.
"""
    output = root / "docs/RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output

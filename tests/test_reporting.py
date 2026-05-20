from pathlib import Path

import pandas as pd

from perturb_benchmark.reporting import build_reports, render_result_sentence
from perturb_benchmark.visualization import make_figures


def test_missing_scgpt_results_are_labeled_not_run() -> None:
    sentence = render_result_sentence(cpu_summary={"ridge": 0.4}, scgpt_summary=None)
    assert "scGPT was not run" in sentence
    assert "outperformed" not in sentence


def test_reports_and_figures_are_artifact_driven(tmp_path: Path) -> None:
    (tmp_path / "results/metrics").mkdir(parents=True)
    (tmp_path / "results/tables").mkdir(parents=True)
    (tmp_path / "results/metadata").mkdir(parents=True)
    (tmp_path / "results/predictions").mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["no_change", "ridge"],
            "median_pearson": [float("nan"), 0.4],
            "median_mae": [1.0, 0.5],
        }
    ).to_csv(tmp_path / "results/metrics/model_summary.csv", index=False)
    pd.DataFrame(
        {
            "condition": ["A+B", "A+B"],
            "model": ["no_change", "ridge"],
            "topk_recovery": [0.0, 0.5],
            "direction_accuracy": [0.0, 0.5],
        }
    ).to_csv(tmp_path / "results/tables/response_gene_metrics.csv", index=False)
    pd.DataFrame(
        {
            "condition": ["A+B", "A+B"],
            "model": ["no_change", "ridge"],
            "interaction_magnitude": [1.0, 1.0],
            "interaction_stratum": ["high", "high"],
            "rmse": [1.2, 0.6],
        }
    ).to_csv(tmp_path / "results/tables/interaction_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "condition": ["ctrl", "A", "B", "A+B"],
            "G1": [0.0, 1.0, 0.0, 1.2],
            "G2": [0.0, 0.0, 1.0, 1.1],
        }
    ).to_csv(tmp_path / "results/predictions/pseudobulk_deltas.csv.gz", index=False)
    (tmp_path / "results/metadata/scgpt_feasibility.json").write_text(
        '{"status":"blocked_full_training"}', encoding="utf-8"
    )
    report = build_reports(tmp_path)
    figures = make_figures(tmp_path)
    assert "scGPT was not run" in report.read_text(encoding="utf-8")
    assert figures and all(path.stat().st_size > 0 for path in figures)
    assert (tmp_path / "figures/response_pca.png").is_file()

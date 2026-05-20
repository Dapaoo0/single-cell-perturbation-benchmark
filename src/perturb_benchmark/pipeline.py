"""Deterministic orchestration for the CPU perturbation benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from perturb_benchmark.baselines import (
    fit_calibrated_additive,
    fit_ridge_nested,
    predict_additive_sum,
    predict_matching_mean,
    predict_no_change,
    predict_perturbed_mean,
)
from perturb_benchmark.bootstrap import paired_condition_bootstrap
from perturb_benchmark.config import load_config
from perturb_benchmark.interactions import (
    assign_interaction_strata,
    crossfit_interaction_analysis,
    interaction_residual,
)
from perturb_benchmark.metrics import condition_metrics, summarize_models
from perturb_benchmark.perturbation import build_condition_metadata, parse_condition
from perturb_benchmark.predictions import pair_model_predictions, validate_predictions
from perturb_benchmark.provenance import sha256_file
from perturb_benchmark.pseudobulk import build_pseudobulk
from perturb_benchmark.response_genes import direction_accuracy, topk_recovery
from perturb_benchmark.splits import create_combination_folds

CPU_MODELS = (
    "no_change",
    "perturbed_mean",
    "matching_mean",
    "additive_sum",
    "calibrated_additive",
    "ridge",
)


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RunManifest:
    status: str
    dataset: str
    seed: int
    n_test_conditions: int
    n_genes: int
    models: tuple[str, ...]
    artifacts: tuple[ArtifactRecord, ...]


def build_synthetic_anndata() -> ad.AnnData:
    """Construct a small deterministic perturbation dataset with known responses."""

    genes = ["G1", "G2", "G3", "G4", "G5"]
    baseline = np.array([2.0, 1.0, 3.0, 2.0, 1.0])
    singles = {
        "A": np.array([1.0, 0.2, 0.0, 0.0, -0.1]),
        "B": np.array([0.0, 1.2, 0.1, 0.0, 0.0]),
        "C": np.array([0.0, 0.0, 1.1, 0.2, 0.0]),
        "D": np.array([0.1, 0.0, 0.0, 1.0, 0.3]),
    }
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    conditions: list[tuple[str, np.ndarray]] = [("ctrl", np.zeros(len(genes)))]
    conditions.extend((f"{gene}+ctrl", delta) for gene, delta in singles.items())
    for index, (gene_a, gene_b) in enumerate(pairs):
        interaction = np.zeros(len(genes))
        interaction[index % len(genes)] = 0.25 * (index + 1)
        conditions.append((f"{gene_a}+{gene_b}", singles[gene_a] + singles[gene_b] + interaction))

    offsets = np.array([-0.02, 0.0, 0.02])[:, None] * np.array([1, -1, 1, -1, 1])[None, :]
    matrix: list[np.ndarray] = []
    labels: list[str] = []
    for label, delta in conditions:
        matrix.extend(baseline + delta + offsets)
        labels.extend([label] * len(offsets))
    return ad.AnnData(
        X=np.asarray(matrix, dtype=np.float32),
        obs=pd.DataFrame({"condition": labels}, index=[f"cell_{i}" for i in range(len(labels))]),
        var=pd.DataFrame({"gene_name": genes}, index=genes),
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def _prediction_frame(
    conditions: pd.DataFrame,
    targets: pd.DataFrame,
    predicted: np.ndarray,
    *,
    fold: int,
    model: str,
) -> pd.DataFrame:
    condition_names = conditions["condition"].astype(str).tolist()
    genes = targets.columns.astype(str).tolist()
    if predicted.shape != targets.shape:
        raise ValueError(f"{model} predictions do not match target shape")
    return pd.DataFrame(
        {
            "condition": np.repeat(condition_names, len(genes)),
            "fold": fold,
            "replicate": "global",
            "gene": np.tile(genes, len(condition_names)),
            "true_delta": targets.to_numpy(dtype=float).ravel(),
            "pred_delta": np.asarray(predicted, dtype=float).ravel(),
            "model": model,
        }
    )


def _artifact(root: Path, path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        path=path.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def run_cpu_pipeline(
    config_path: Path | str,
    project_root: Path | str,
    offline: bool = False,
) -> RunManifest:
    """Run all CPU baselines and write validated checksummed scientific artifacts."""

    config = load_config(Path(config_path))
    root = Path(project_root).resolve()
    source_path = root / config.paths.raw_anndata
    if offline:
        adata = build_synthetic_anndata()
    else:
        if not source_path.is_file():
            raise FileNotFoundError(f"AnnData source does not exist: {source_path}")
        adata = ad.read_h5ad(source_path)

    condition_col = config.dataset.condition_column
    batch_col = config.dataset.batch_column
    labels = adata.obs[condition_col].astype(str)
    batches = adata.obs[batch_col] if batch_col is not None else None
    metadata = build_condition_metadata(labels, batches, config.dataset.control_token)
    retained = metadata.loc[
        metadata["n_cells"] >= config.quality.min_cells_per_condition,
        "condition_canonical",
    ]
    canonical_cells = labels.map(
        lambda value: parse_condition(value, config.dataset.control_token).canonical
    )
    keep = canonical_cells.isin(set(retained))
    if not keep.any():
        raise ValueError("no conditions pass the configured cell-count threshold")
    adata = adata[keep.to_numpy()].copy()
    metadata = build_condition_metadata(
        adata.obs[condition_col].astype(str),
        adata.obs[batch_col] if batch_col is not None else None,
        config.dataset.control_token,
    )
    genes = adata.var_names.astype(str).tolist()
    pseudobulk = build_pseudobulk(
        adata.X,
        adata.obs,
        genes,
        condition_col,
        batch_col,
        control_token=config.dataset.control_token,
        scale="GEARS processed adata.X" if not offline else "synthetic processed expression",
    )
    folds = create_combination_folds(metadata, config.split.n_folds, config.random_seed)

    singles = metadata.loc[metadata["perturbation_type"] == "single"].copy()
    single_vectors = {
        str(row.gene_a): pseudobulk.deltas.loc[str(row.condition_canonical)].to_numpy(dtype=float)
        for row in singles.itertuples(index=False)
    }
    prediction_frames: list[pd.DataFrame] = []
    training_means: dict[int, pd.Series] = {}
    for fold in range(config.split.n_folds):
        test_pairs = (
            folds.loc[folds["fold"] == fold].sort_values("condition").reset_index(drop=True)
        )
        train_pairs = (
            folds.loc[folds["fold"] != fold].sort_values("condition").reset_index(drop=True)
        )
        test_targets = pseudobulk.deltas.loc[test_pairs["condition"]]
        train_double_targets = pseudobulk.deltas.loc[train_pairs["condition"]]
        training_means[fold] = train_double_targets.mean(axis=0)

        additive = np.vstack(
            [
                predict_additive_sum(row.gene_a, row.gene_b, single_vectors)
                for row in test_pairs.itertuples(index=False)
            ]
        )
        matching = np.vstack(
            [
                predict_matching_mean(row.gene_a, row.gene_b, single_vectors)
                for row in test_pairs.itertuples(index=False)
            ]
        )
        calibrated_model = fit_calibrated_additive(
            train_pairs, train_double_targets, single_vectors
        )
        calibrated = np.vstack(
            [
                calibrated_model.predict(row.gene_a, row.gene_b, single_vectors)
                for row in test_pairs.itertuples(index=False)
            ]
        )
        ridge_pairs = pd.concat(
            [
                singles[["gene_a", "gene_b"]],
                train_pairs[["gene_a", "gene_b"]],
            ],
            ignore_index=True,
        )
        ridge_targets = pd.concat(
            [
                pseudobulk.deltas.loc[singles["condition_canonical"]],
                train_double_targets,
            ],
            ignore_index=True,
        )
        ridge = fit_ridge_nested(
            ridge_pairs,
            ridge_targets,
            alphas=[0.01, 0.1, 1, 10, 100],
            seed=config.random_seed + fold,
        ).predict(test_pairs)
        model_predictions = {
            "no_change": predict_no_change(len(test_pairs), len(genes)),
            "perturbed_mean": predict_perturbed_mean(train_double_targets, test_pairs["condition"]),
            "matching_mean": matching,
            "additive_sum": additive,
            "calibrated_additive": calibrated,
            "ridge": ridge,
        }
        fold_frames = [
            _prediction_frame(
                test_pairs,
                test_targets,
                model_predictions[model],
                fold=fold,
                model=model,
            )
            for model in CPU_MODELS
        ]
        pair_model_predictions(fold_frames)
        prediction_frames.extend(fold_frames)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    validate_predictions(predictions)
    metrics = condition_metrics(predictions, training_mean_by_fold=training_means)
    summary = summarize_models(metrics)

    metric_pivot = metrics.pivot(
        index="condition", columns="model", values="mae"
    ).reset_index()
    bootstrap_rows = []
    for model in CPU_MODELS:
        if model == "additive_sum":
            continue
        interval = paired_condition_bootstrap(
            metric_pivot,
            model,
            "additive_sum",
            iterations=config.bootstrap.iterations,
            seed=config.random_seed,
        )
        bootstrap_rows.append({"model": model, "reference": "additive_sum", **asdict(interval)})
    bootstrap = pd.DataFrame(bootstrap_rows)

    response_rows = []
    k = min(20, len(genes))
    for (condition, model), group in predictions.groupby(["condition", "model"], sort=True):
        ordered = group.sort_values("gene")
        response_rows.append(
            {
                "condition": condition,
                "model": model,
                "k": k,
                "topk_recovery": topk_recovery(
                    ordered["true_delta"].to_numpy(),
                    ordered["pred_delta"].to_numpy(),
                    k=k,
                    genes=ordered["gene"].to_numpy(),
                ),
                "direction_accuracy": direction_accuracy(
                    ordered["true_delta"].to_numpy(), ordered["pred_delta"].to_numpy()
                ),
            }
        )
    response = pd.DataFrame(response_rows)

    interaction_rows = []
    for row in folds.sort_values("condition").itertuples(index=False):
        residual = interaction_residual(
            pseudobulk.deltas.loc[row.condition].to_numpy(),
            single_vectors[row.gene_a],
            single_vectors[row.gene_b],
        )
        interaction_rows.append(
            {
                "condition": row.condition,
                "interaction_magnitude": float(np.sqrt(np.mean(residual**2))),
            }
        )
    interactions = pd.DataFrame(interaction_rows).set_index("condition")
    n_strata = min(3, len(interactions))
    interactions["interaction_stratum"] = assign_interaction_strata(
        interactions["interaction_magnitude"], n_strata=n_strata
    )
    interactions = interactions.reset_index().merge(
        metrics[["condition", "model", "rmse"]], on="condition", validate="one_to_many"
    )
    interactions["interpretation"] = "coupled_descriptive_difficulty"

    panels = crossfit_interaction_analysis(genes, n_panels=2, seed=config.random_seed)
    panel_frame = pd.DataFrame(
        [(panel_index, gene) for panel_index, panel in enumerate(panels) for gene in panel],
        columns=["panel", "gene"],
    )

    paths = {
        "metadata": root / config.paths.processed_dir / "condition_metadata.csv",
        "folds": root / config.paths.processed_dir / "combination_folds.csv",
        "predictions": root / config.paths.results_dir / "predictions" / "cpu_predictions.csv.gz",
        "metrics": root / config.paths.results_dir / "metrics" / "condition_metrics.csv",
        "summary": root / config.paths.results_dir / "metrics" / "model_summary.csv",
        "bootstrap": root / config.paths.results_dir / "bootstrap" / "paired_mae.csv",
        "response": root / config.paths.results_dir / "tables" / "response_gene_metrics.csv",
        "interactions": root / config.paths.results_dir / "tables" / "interaction_diagnostics.csv",
        "panels": root / config.paths.results_dir / "metadata" / "crossfit_gene_panels.csv",
    }
    _write_frame(metadata, paths["metadata"])
    _write_frame(folds, paths["folds"])
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        paths["predictions"],
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
        float_format="%.10g",
    )
    _write_frame(metrics, paths["metrics"])
    _write_frame(summary, paths["summary"])
    _write_frame(bootstrap, paths["bootstrap"])
    _write_frame(response, paths["response"])
    _write_frame(interactions, paths["interactions"])
    _write_frame(panel_frame, paths["panels"])
    artifacts = tuple(_artifact(root, path) for path in paths.values())
    manifest = RunManifest(
        status="cpu_complete_scgpt_pending",
        dataset=config.dataset.name,
        seed=config.random_seed,
        n_test_conditions=len(folds),
        n_genes=len(genes),
        models=CPU_MODELS,
        artifacts=artifacts,
    )
    manifest_path = root / config.paths.metadata_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest

# Single-Cell Perturbation Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-aware, reproducible Norman Perturb-seq benchmark that compares scGPT with strong non-neural baselines on held-out double-gene CRISPRa combinations.

**Architecture:** A Python 3.11 package owns all scientific logic and typed tabular contracts. Thin scripts orchestrate data acquisition, CPU analysis, scGPT execution, and reporting; notebooks only render saved artifacts. A deterministic offline fixture exercises the complete CPU path, while scGPT has a separate pinned environment and a hard feasibility manifest.

**Tech Stack:** Python 3.11, uv, NumPy, pandas, SciPy, scikit-learn, AnnData/Scanpy, PyArrow, Matplotlib, Seaborn, PyYAML, Pydantic, pytest, Ruff, Jupyter, scGPT, GEARS, PyTorch.

**Spec:** `docs/superpowers/specs/2026-08-28-single-cell-perturbation-benchmark-design.md`

## Global Constraints

- The biological system is Norman CRISPR activation/overexpression, not knockout.
- The primary split is deterministic five-fold `seen-components / unseen-combination` validation.
- Test-condition cells, pseudo-bulk targets, and outcomes never enter model fitting, hyperparameter selection, or gene selection.
- Every model uses identical ordered test conditions, genes, target scale, and common prediction schema.
- No-change correlations remain undefined and are never imputed to zero.
- Interaction strength is a difficulty descriptor, not independent evidence that scGPT understands genetic interactions.
- Full scGPT claims require real predictions from a pinned environment; no simulated predictions may be labeled scGPT.
- Large raw, interim, model, and prediction artifacts stay out of Git; checksums, small summaries, figures, and frozen splits are tracked.

---

### Task 1: Reproducible package skeleton and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `config/project.yaml`
- Create: `src/perturb_benchmark/__init__.py`
- Create: `src/perturb_benchmark/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `ProjectConfig`, `load_config(path: Path) -> ProjectConfig`, and a locked Python 3.11 CPU environment.

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path
from perturb_benchmark.config import load_config

def test_load_config_freezes_primary_design() -> None:
    cfg = load_config(Path("config/project.yaml"))
    assert cfg.dataset.geo_accession == "GSE133344"
    assert cfg.split.name == "seen-components-unseen-combination"
    assert cfg.split.n_folds == 5
    assert cfg.bootstrap.iterations == 2000
    assert cfg.random_seed == 42
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `uv run pytest tests/test_config.py -v`

Expected: FAIL because the package and environment do not exist.

- [ ] **Step 3: Create the package and exact configuration models**

Implement frozen Pydantic models for dataset, QC, split, bootstrap, gene set, and paths. Configure Ruff and pytest in `pyproject.toml`; pin Python to `>=3.11,<3.12`; include CPU analysis and dev dependencies; generate `uv.lock` using `uv lock --python 3.11`.

- [ ] **Step 4: Verify configuration and lock**

Run: `uv sync --locked --extra dev && uv run pytest tests/test_config.py -v && uv run ruff check .`

Expected: PASS with no lock drift or lint errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore LICENSE CITATION.cff config src tests
git commit -m "chore: initialize perturbation benchmark"
```

### Task 2: Canonical perturbation parser and metadata

**Files:**
- Create: `src/perturb_benchmark/perturbation.py`
- Create: `tests/test_perturbation.py`

**Interfaces:**
- Produces: `Perturbation`, `parse_condition(label: str) -> Perturbation`, and `build_condition_metadata(labels: Series, batches: Series | None) -> DataFrame`.

- [ ] **Step 1: Write parser tests**

```python
import pytest
from perturb_benchmark.perturbation import parse_condition

@pytest.mark.parametrize(
    ("label", "kind", "genes", "canonical"),
    [
        ("ctrl", "control", (), "ctrl"),
        ("A+ctrl", "single", ("A",), "A"),
        ("ctrl+A", "single", ("A",), "A"),
        ("B+A", "double", ("A", "B"), "A+B"),
    ],
)
def test_parse_condition(label, kind, genes, canonical) -> None:
    parsed = parse_condition(label)
    assert (parsed.kind, parsed.genes, parsed.canonical) == (kind, genes, canonical)

def test_parser_rejects_three_gene_condition() -> None:
    with pytest.raises(ValueError, match="one or two perturbed genes"):
        parse_condition("A+B+C")
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_perturbation.py -v`

Expected: FAIL because `perturbation.py` is absent.

- [ ] **Step 3: Implement immutable parsing and metadata aggregation**

Normalize whitespace, recognize the configured control token case-sensitively, discard one control component, reject empty/duplicate/three-gene labels, sort gene tuples, and aggregate cell and batch counts without losing original labels.

- [ ] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_perturbation.py -v && uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark/perturbation.py tests/test_perturbation.py
git commit -m "feat: canonicalize perturbation conditions"
```

### Task 3: AnnData validation, scale evidence, and provenance

**Files:**
- Create: `src/perturb_benchmark/provenance.py`
- Create: `src/perturb_benchmark/data_validation.py`
- Create: `scripts/download_norman.py`
- Create: `scripts/validate_anndata.py`
- Create: `data/README.md`
- Create: `docs/DATA_PROVENANCE.md`
- Create: `docs/DATA_DICTIONARY.md`
- Create: `tests/test_data_validation.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`, `inspect_anndata(adata, condition_column: str, control_token: str) -> AnnDataSummary`, and JSON provenance/summary artifacts.

- [ ] **Step 1: Write synthetic AnnData validation tests**

```python
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from perturb_benchmark.data_validation import inspect_anndata

def test_validation_reports_schema_and_control() -> None:
    adata = ad.AnnData(
        X=np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
        obs=pd.DataFrame({"condition": ["ctrl", "A+ctrl"]}),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    summary = inspect_anndata(adata, "condition", "ctrl")
    assert summary.n_cells == 2
    assert summary.n_genes == 2
    assert summary.has_control is True

def test_validation_rejects_duplicate_gene_ids() -> None:
    adata = ad.AnnData(np.ones((2, 2)), var=pd.DataFrame(index=["G1", "G1"]))
    with pytest.raises(ValueError, match="duplicate gene"):
        inspect_anndata(adata, "condition", "ctrl")
```

- [ ] **Step 2: Verify focused tests fail**

Run: `uv run pytest tests/test_data_validation.py -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement download and validation gates**

The downloader accepts an explicit configured URL, streams to a temporary file, rejects zero bytes, atomically moves the result, and records URL, timestamp, byte count, and SHA256. The validator reports layers, raw availability, dtype, sparsity, value quantiles, integer-likeness, condition schema, controls, and candidate batch columns; it records evidence but never guesses a normalization transform.

- [ ] **Step 4: Verify tests and script help**

Run: `uv run pytest tests/test_data_validation.py -v && uv run python scripts/download_norman.py --help && uv run python scripts/validate_anndata.py --help`

Expected: PASS and both CLIs display explicit inputs and outputs.

- [ ] **Step 5: Commit**

```bash
git add src scripts data/README.md docs/DATA_PROVENANCE.md docs/DATA_DICTIONARY.md tests
git commit -m "data: add Norman provenance and AnnData validation"
```

### Task 4: Batch-aware pseudo-bulk and delta construction

**Files:**
- Create: `src/perturb_benchmark/pseudobulk.py`
- Create: `tests/test_pseudobulk.py`

**Interfaces:**
- Produces: `build_pseudobulk(expression, obs, genes, condition_col, batch_col) -> PseudobulkResult`; result contains condition means, matched-control deltas, metadata, and scale label.

- [ ] **Step 1: Write exact pseudo-bulk tests**

```python
import numpy as np
import pandas as pd
from perturb_benchmark.pseudobulk import build_pseudobulk

def test_global_control_delta_matches_toy_example() -> None:
    x = np.array([[1, 1], [1, 3], [3, 2], [5, 4]], dtype=float)
    obs = pd.DataFrame({"condition": ["ctrl", "ctrl", "A", "A"]})
    result = build_pseudobulk(x, obs, ["G1", "G2"], "condition", None)
    np.testing.assert_allclose(result.deltas.loc["A"], [3.0, 1.0])

def test_batch_matching_precedes_aggregation() -> None:
    x = np.array([[1], [10], [3], [14]], dtype=float)
    obs = pd.DataFrame({"condition": ["ctrl", "ctrl", "A", "A"], "batch": ["x", "y", "x", "y"]})
    result = build_pseudobulk(x, obs, ["G1"], "condition", "batch")
    assert result.replicate_deltas.loc[("A", "x"), "G1"] == 2
    assert result.replicate_deltas.loc[("A", "y"), "G1"] == 4
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest tests/test_pseudobulk.py -v`

Expected: FAIL on missing implementation.

- [ ] **Step 3: Implement sparse/dense aggregation and matched controls**

Preserve the declared expression scale, support NumPy and SciPy sparse matrices, require a control in every used batch, subtract controls before averaging replicate deltas, reject NaNs, and emit deterministic gene ordering.

- [ ] **Step 4: Verify focused and full tests**

Run: `uv run pytest tests/test_pseudobulk.py -v && uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark/pseudobulk.py tests/test_pseudobulk.py
git commit -m "feat: build batch-aware perturbation deltas"
```

### Task 5: Frozen unseen-combination folds and leakage assertions

**Files:**
- Create: `src/perturb_benchmark/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Produces: `create_combination_folds(metadata: DataFrame, n_folds: int, seed: int) -> DataFrame` and `validate_fold(split, metadata) -> None`.

- [ ] **Step 1: Write split eligibility and disjointness tests**

```python
import pandas as pd
from perturb_benchmark.splits import create_combination_folds, validate_fold

def test_seen_components_unseen_combination_split() -> None:
    metadata = pd.DataFrame({
        "condition": ["A", "B", "C", "A+B", "A+C", "B+C"],
        "kind": ["single", "single", "single", "double", "double", "double"],
        "gene_a": ["A", "B", "C", "A", "A", "B"],
        "gene_b": [None, None, None, "B", "C", "C"],
    })
    split = create_combination_folds(metadata, n_folds=3, seed=42)
    assert set(split["condition"]) == {"A+B", "A+C", "B+C"}
    assert split["condition"].is_unique
    validate_fold(split, metadata)
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_splits.py -v`

Expected: FAIL because split functions are absent.

- [ ] **Step 3: Implement deterministic balanced edge assignment**

Filter doubles whose component singles exist, sort canonical labels before seeded assignment, balance fold sizes, save one test fold per double, and validate component availability, uniqueness, full coverage, and absence of swapped duplicates.

- [ ] **Step 4: Verify tests**

Run: `uv run pytest tests/test_splits.py -v && uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark/splits.py tests/test_splits.py
git commit -m "feat: freeze unseen-combination folds"
```

### Task 6: Strong biological and linear baselines

**Files:**
- Create: `src/perturb_benchmark/baselines.py`
- Create: `tests/test_baselines.py`

**Interfaces:**
- Produces: `predict_no_change`, `predict_perturbed_mean`, `predict_matching_mean`, `predict_additive_sum`, `fit_calibrated_additive`, and `fit_ridge_nested` returning matrices aligned to requested condition and gene order.

- [ ] **Step 1: Write exact additive and train-isolation tests**

```python
import numpy as np
import pytest
from perturb_benchmark.baselines import predict_additive_sum, predict_matching_mean

def test_additive_formulas() -> None:
    singles = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 2.0])}
    np.testing.assert_allclose(predict_additive_sum("A", "B", singles), [1.0, 2.0])
    np.testing.assert_allclose(predict_matching_mean("A", "B", singles), [0.5, 1.0])

def test_additive_rejects_missing_training_single() -> None:
    with pytest.raises(ValueError, match="training single"):
        predict_additive_sum("A", "B", {"A": np.array([1.0])})
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_baselines.py -v`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement baselines with nested tuning**

Fit calibration only on training doubles using predictions derived from training singles. Tune Ridge alpha over `[0.01, 0.1, 1, 10, 100]` with condition-level inner folds. Never expose outer-test conditions to calibration, alpha selection, or response means.

- [ ] **Step 4: Verify baseline tests and determinism**

Run: `uv run pytest tests/test_baselines.py -v && uv run pytest -q`

Expected: PASS with identical predictions across repeated seeded calls.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark/baselines.py tests/test_baselines.py
git commit -m "feat: add strong perturbation baselines"
```

### Task 7: Prediction contract and leakage-aware metrics

**Files:**
- Create: `src/perturb_benchmark/predictions.py`
- Create: `src/perturb_benchmark/metrics.py`
- Create: `tests/test_predictions.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `validate_predictions(frame: DataFrame) -> None`, `pair_model_predictions(frames) -> DataFrame`, `condition_metrics(frame, training_mean) -> DataFrame`, and `summarize_models(metrics) -> DataFrame`.

- [ ] **Step 1: Write contract and undefined-correlation tests**

```python
import numpy as np
from perturb_benchmark.metrics import vector_metrics

def test_constant_prediction_correlation_is_nan() -> None:
    result = vector_metrics(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert np.isnan(result.pearson)
    assert np.isnan(result.spearman)
    assert result.mae == 1.5
```

Add prediction tests that reject duplicate `(condition, fold, replicate, gene, model)` keys, missing paired genes, different test conditions, nonfinite predictions, and reordered gene identities.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_predictions.py tests/test_metrics.py -v`

Expected: FAIL on absent modules.

- [ ] **Step 3: Implement common validation and metrics**

Compute Pearson, Spearman, MAE, RMSE, centered-response metrics using only the fold's training mean, and explicit undefined-reason columns. Pair frames by semantic keys rather than row position.

- [ ] **Step 4: Verify focused and full tests**

Run: `uv run pytest tests/test_predictions.py tests/test_metrics.py -v && uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark/predictions.py src/perturb_benchmark/metrics.py tests
git commit -m "feat: validate paired predictions and metrics"
```

### Task 8: Responsive genes, bootstrap, and non-circular interaction reporting

**Files:**
- Create: `src/perturb_benchmark/response_genes.py`
- Create: `src/perturb_benchmark/bootstrap.py`
- Create: `src/perturb_benchmark/interactions.py`
- Create: `tests/test_response_genes.py`
- Create: `tests/test_bootstrap.py`
- Create: `tests/test_interactions.py`

**Interfaces:**
- Produces: `topk_recovery`, `direction_accuracy`, `paired_condition_bootstrap`, `interaction_residual`, `assign_interaction_strata`, and `crossfit_interaction_analysis`.

- [ ] **Step 1: Write biological-analysis tests**

```python
import numpy as np
from perturb_benchmark.interactions import interaction_residual
from perturb_benchmark.response_genes import topk_recovery

def test_interaction_residual() -> None:
    observed = np.array([3.0, 4.0])
    a = np.array([1.0, 1.0])
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(interaction_residual(observed, a, b), [1.0, 1.0])

def test_topk_overlap() -> None:
    assert topk_recovery(np.array([3.0, 2.0, 0.0]), np.array([4.0, 0.0, 1.0]), k=1) == 1.0
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_response_genes.py tests/test_bootstrap.py tests/test_interactions.py -v`

Expected: FAIL on missing modules.

- [ ] **Step 3: Implement deterministic analyses**

Use stable tie-breaking by gene name, condition-level paired resampling with 2,000 fixed-seed replicates, train-independent test strata computed once from observed interaction magnitude, and disjoint deterministic gene panels for the cross-fitted sensitivity analysis. Output language labels gain correlations as coupled/descriptive.

- [ ] **Step 4: Verify tests and bootstrap reproducibility**

Run: `uv run pytest tests/test_response_genes.py tests/test_bootstrap.py tests/test_interactions.py -v && uv run pytest -q`

Expected: PASS; two calls with seed 42 produce identical intervals.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark tests
git commit -m "analysis: add uncertainty and interaction diagnostics"
```

### Task 9: CPU orchestration and offline end-to-end fixture

**Files:**
- Create: `src/perturb_benchmark/pipeline.py`
- Create: `scripts/run_cpu_pipeline.py`
- Create: `tests/fixtures/synthetic_config.yaml`
- Create: `tests/fixtures/build_synthetic_anndata.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `run_cpu_pipeline(config_path: Path, project_root: Path, offline: bool) -> RunManifest` and a manifest containing artifact paths, sizes, and SHA256 values.

- [ ] **Step 1: Write the end-to-end fixture test**

```python
from perturb_benchmark.pipeline import run_cpu_pipeline

def test_offline_pipeline_produces_valid_manifest(tmp_path) -> None:
    manifest = run_cpu_pipeline(
        config_path="tests/fixtures/synthetic_config.yaml",
        project_root=tmp_path,
        offline=True,
    )
    assert manifest.n_test_conditions > 0
    assert manifest.models == ["no_change", "perturbed_mean", "matching_mean", "additive_sum", "calibrated_additive", "ridge"]
    assert all(item.sha256 and item.bytes > 0 for item in manifest.artifacts)
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_pipeline.py -v`

Expected: FAIL because the orchestrator is absent.

- [ ] **Step 3: Implement the thin orchestrator**

Build or load the fixture AnnData, validate it, construct metadata and pseudo-bulk deltas, freeze folds, run every CPU baseline, validate paired predictions, compute metrics/bootstrap/response/interaction outputs, and write a checksummed manifest. Fail before partial success claims if any required artifact is invalid.

- [ ] **Step 4: Verify offline reproducibility**

Run: `uv run pytest tests/test_pipeline.py -v && uv run python scripts/run_cpu_pipeline.py --config tests/fixtures/synthetic_config.yaml --project-root .test-run --offline`

Expected: PASS and a manifest with identical scientific output checksums on two clean runs.

- [ ] **Step 5: Commit**

```bash
git add src/perturb_benchmark/pipeline.py scripts tests/fixtures tests/test_pipeline.py
git commit -m "feat: add reproducible CPU benchmark pipeline"
```

### Task 10: Pinned scGPT feasibility adapter

**Files:**
- Create: `environment-scgpt.yml`
- Create: `config/scgpt.yaml`
- Create: `src/perturb_benchmark/scgpt_contract.py`
- Create: `scripts/check_scgpt_environment.py`
- Create: `scripts/run_scgpt.py`
- Create: `docs/SCGPT_SETUP.md`
- Create: `tests/test_scgpt_contract.py`

**Interfaces:**
- Produces: `ScGPTManifest`, `validate_scgpt_manifest(path)`, `validate_scgpt_predictions(frame, expected_keys)`, and a CLI that refuses to run without the exact checkpoint checksum and compatible environment.

- [ ] **Step 1: Write manifest-gate tests**

```python
import pytest
from perturb_benchmark.scgpt_contract import ScGPTManifest

def test_manifest_requires_real_checkpoint_checksum() -> None:
    with pytest.raises(ValueError, match="64-character SHA256"):
        ScGPTManifest(
            scgpt_commit="abc1234",
            gears_commit="def5678",
            checkpoint_sha256="unknown",
            execution_device="cpu",
            prediction_mode="full-perturbation",
        )
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_scgpt_contract.py -v`

Expected: FAIL on absent contract.

- [ ] **Step 3: Implement feasibility and output gates**

Pin exact source commits and dependency versions only after a minimal import/data-loader probe succeeds. Record checkpoint hash, vocabulary hash, device, package versions, gene overlap, control sampling seed/count, fold, and mode. `run_scgpt.py` must abort if CUDA is required but absent, if the manifest is incomplete, or if output keys differ from frozen CPU targets. Embedding mode writes model label `scgpt_embedding`, never `scgpt`.

- [ ] **Step 4: Run CPU-verifiable checks**

Run: `uv run pytest tests/test_scgpt_contract.py -v && uv run python scripts/check_scgpt_environment.py --report results/metadata/scgpt_feasibility.json`

Expected on this workstation: tests PASS; environment report records no CUDA and a truthful `blocked_full_training` status without fabricating predictions.

- [ ] **Step 5: Commit**

```bash
git add environment-scgpt.yml config/scgpt.yaml src/perturb_benchmark/scgpt_contract.py scripts docs/SCGPT_SETUP.md tests/test_scgpt_contract.py
git commit -m "feat: gate scGPT execution with pinned provenance"
```

### Task 11: Figures, notebooks, and scientific reports

**Files:**
- Create: `src/perturb_benchmark/visualization.py`
- Create: `src/perturb_benchmark/reporting.py`
- Create: `scripts/make_figures.py`
- Create: `scripts/build_reports.py`
- Create: `notebooks/01_data_validation.ipynb`
- Create: `notebooks/02_ood_benchmark.ipynb`
- Create: `notebooks/03_interaction_failures.ipynb`
- Create: `docs/METHODS.md`
- Create: `docs/RESULTS.md`
- Create: `docs/LIMITATIONS.md`
- Create: `docs/REPRODUCIBILITY.md`
- Create: `docs/DECISIONS.md`
- Create: `README.md`
- Create: `tests/test_reporting.py`

**Interfaces:**
- Consumes: only validated saved metadata, predictions, metrics, bootstrap, response-gene, and interaction artifacts.
- Produces: deterministic PNG/SVG figures, executed notebooks, result tables, and English documentation that never invents missing scGPT results.

- [ ] **Step 1: Write report traceability tests**

```python
from perturb_benchmark.reporting import render_result_sentence

def test_missing_scgpt_results_are_labeled_not_run() -> None:
    sentence = render_result_sentence(cpu_summary={"ridge": 0.4}, scgpt_summary=None)
    assert "scGPT was not run" in sentence
    assert "outperformed" not in sentence
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_reporting.py -v`

Expected: FAIL on absent reporting module.

- [ ] **Step 3: Implement artifact-driven reporting**

Generate QC counts, response PCA, magnitude distribution, model metrics, paired bootstrap, top-K recovery, interaction-stratified errors, and failure heatmaps. Keep No-change correlation visibly `N/A`. Generate only three notebooks to avoid duplicated logic. README headline depends on manifest status and is replaced with an explicit not-run statement until real scGPT predictions validate.

- [ ] **Step 4: Execute and validate reports on fixture outputs**

Run: `uv run python scripts/make_figures.py --project-root .test-run && uv run python scripts/build_reports.py --project-root .test-run && uv run pytest tests/test_reporting.py -v`

Expected: PASS; every rendered numeric claim traces to a checksummed result file.

- [ ] **Step 5: Commit**

```bash
git add src scripts notebooks docs README.md tests/test_reporting.py
git commit -m "docs: add traceable benchmark narrative"
```

### Task 12: CI, real-data CPU run, and final verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `docs/PROJECT_LOG.md`
- Modify: `README.md`
- Modify: `docs/RESULTS.md`
- Modify: `docs/DATA_PROVENANCE.md`

**Interfaces:**
- Produces: a Linux CI run, real-data provenance and CPU results, final run manifest, and a clean reproducibility audit.

- [ ] **Step 1: Add CI with the exact offline gates**

Configure Ubuntu with Python 3.11 and uv; run `uv lock --check`, `uv sync --locked --extra dev`, `uv run ruff check .`, `uv run pytest -q`, and the offline fixture pipeline.

- [ ] **Step 2: Run real-data acquisition and schema inspection**

Run:

```bash
uv run python scripts/download_norman.py --config config/project.yaml
uv run python scripts/validate_anndata.py --config config/project.yaml
```

Expected: nonempty source artifact, checksum, exact schema summary, condition parser coverage, and an explicit expression-scale decision before preprocessing continues.

- [ ] **Step 3: Run the real CPU pipeline**

Run: `uv run python scripts/run_cpu_pipeline.py --config config/project.yaml --project-root .`

Expected: all eligible held-out doubles covered exactly once, all six CPU models paired on identical keys, 2,000 bootstrap replicates, and a valid artifact manifest.

- [ ] **Step 4: Run final scientific and engineering assertions**

Run:

```bash
uv lock --check
uv sync --locked --extra dev
uv run ruff check .
uv run pytest -q
uv run python scripts/run_cpu_pipeline.py --config tests/fixtures/synthetic_config.yaml --project-root .test-run --offline
git diff --check
git status --short --branch
```

Expected: every command passes and the worktree contains only intentionally tracked real summary/report changes.

- [ ] **Step 5: Record truthful completion state and commit**

If real scGPT predictions are unavailable because no GPU execution environment has been supplied, mark the repository `CPU benchmark complete; scGPT full-model run pending GPU` and do not fill comparative scGPT claims. If real predictions exist and validate, build the paired reports and mark the full benchmark complete.

```bash
git add .github docs README.md results/metadata results/metrics results/bootstrap results/tables figures
git commit -m "analysis: report leakage-aware Norman benchmark"
```

## Execution checkpoints

1. After Task 3: review actual Norman schema and expression scale before pseudo-bulk choices are frozen.
2. After Task 9: review the complete offline CPU artifact contract.
3. After Task 10: decide whether a compatible GPU execution environment exists for full scGPT.
4. After Task 12: publish only the scientifically supported completion state.
